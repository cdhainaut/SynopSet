from __future__ import annotations
import argparse, glob
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter

from ..io import open_normalize, normalize_longitudes, export_grib_from_ds
from ..gridtime import choose_target_grid, temporal_align, interp_to, circular_mean_deg


def _gaussian_infill_scalar(da: xr.DataArray, sigma: float) -> xr.DataArray:
    """Infill 2D spatial NaN via masked gaussian; opéré pour chaque time step."""
    vals = da.values  # shape (T,Y,X)
    filled = np.empty_like(vals)
    for i in range(vals.shape[0]):
        A = vals[i]
        mask = np.isnan(A)
        if not np.any(mask):
            filled[i] = A
            continue
        # masked gaussian normalize
        sm = gaussian_filter(np.where(mask, 0.0, A), sigma=sigma)
        norm = gaussian_filter((~mask).astype(float), sigma=sigma)
        filled[i] = np.where(mask, sm / (norm + 1e-5), A)
    out = xr.DataArray(
        filled, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name
    )
    return out


def _gaussian_infill_angle_deg(da_deg: xr.DataArray, sigma: float) -> xr.DataArray:
    """Infill pour angles en degrés [0,360) via filtrage de sin/cos (évite le wrap)."""
    vals = da_deg.values
    filled = np.empty_like(vals)
    for i in range(vals.shape[0]):
        Adeg = vals[i]
        mask = np.isnan(Adeg)
        if not np.any(mask):
            filled[i] = Adeg
            continue
        Arad = np.deg2rad(Adeg % 360.0)
        S = np.where(mask, 0.0, np.sin(Arad))
        C = np.where(mask, 0.0, np.cos(Arad))
        Sm = gaussian_filter(S, sigma=sigma)
        Cm = gaussian_filter(C, sigma=sigma)
        norm = gaussian_filter((~mask).astype(float), sigma=sigma)
        S_hat = np.where(mask, Sm / (norm + 1e-5), S)
        C_hat = np.where(mask, Cm / (norm + 1e-5), C)
        Ahat = (np.degrees(np.arctan2(S_hat, C_hat)) + 360.0) % 360.0
        filled[i] = np.where(mask, Ahat, Adeg)
    out = xr.DataArray(
        filled,
        dims=da_deg.dims,
        coords=da_deg.coords,
        attrs=da_deg.attrs,
        name=da_deg.name,
    )
    return out


def infill_dataset(
    ds: xr.Dataset,
    wave_scalars=("swh", "mwp"),
    wave_angles=("mwd",),
    sigma: float = 1.5,
) -> xr.Dataset:
    """Applique l'infill aux variables si présentes, retourne un nouveau Dataset."""
    out = ds.copy()
    for v in wave_scalars:
        if v in out:
            out[v] = _gaussian_infill_scalar(out[v], sigma=sigma)
    for v in wave_angles:
        if v in out:
            out[v] = _gaussian_infill_angle_deg(out[v], sigma=sigma)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate N GRIB/NetCDF onto a common grid/time and export a single GRIB2."
    )
    ap.add_argument(
        "--in",
        dest="inputs",
        action="append",
        required=True,
        help="Input files or glob patterns. Can be repeated.",
    )
    ap.add_argument(
        "--lon-range",
        choices=["negpos", "pos360"],
        default="negpos",
        help="Normalize longitudes.",
    )
    ap.add_argument(
        "--target-grid",
        choices=["first", "finer"],
        default="finer",
        help="Spatial target grid.",
    )
    ap.add_argument(
        "--time-mode",
        choices=["intersection", "freq"],
        default="intersection",
        help="Temporal alignment mode.",
    )
    ap.add_argument(
        "--time-freq",
        type=str,
        default=None,
        help="e.g. '6H','12H','1D' (used if --time-mode=freq).",
    )
    ap.add_argument(
        "--resample-average",
        action="store_true",
        help="Use temporal averaging on freq grid (angles handled separately).",
    )
    ap.add_argument(
        "--angle-vars",
        type=str,
        default="mwd,wind_direction,dir",
        help="Comma list of angular variables in degrees.",
    )

    ap.add_argument(
        "--infill",
        action="store_true",
        help="Activer l'infill spatial pour vagues (swh/mwp/mwd).",
    )
    ap.add_argument(
        "--infill-sigma",
        type=float,
        default=1.5,
        help="Sigma du filtre gaussien (pixels).",
    )
    ap.add_argument(
        "--infill-vars",
        type=str,
        default="swh,mwp",
        help="Scalaires vagues à infiller (séparés par des virgules).",
    )
    ap.add_argument(
        "--infill-angles",
        type=str,
        default="mwd",
        help="Angles vagues (degrés) à infiller (séparés par des virgules).",
    )

    ap.add_argument("--out", type=str, default="merged.grib2")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths += glob.glob(pat)
    if not paths:
        raise SystemExit("No inputs matched.")
    dsets = []
    for p in paths:
        ds = open_normalize(p)
        print(ds)
        ds = normalize_longitudes(ds, target=args.lon_range)
        dsets.append(ds)

    aligned = temporal_align(dsets, mode=args.time_mode, freq=args.time_freq)
    lat_t, lon_t = choose_target_grid(aligned, how=args.target_grid, ref=None)
    interp_all = [interp_to(ds, lat_t, lon_t) for ds in aligned]

    if args.infill:
        wave_scalars = tuple(
            [s.strip() for s in args.infill_vars.split(",") if s.strip()]
        )
        wave_angles = tuple(
            [s.strip() for s in args.infill_angles.split(",") if s.strip()]
        )
        interp_all = [
            infill_dataset(
                ds_i,
                wave_scalars=wave_scalars,
                wave_angles=wave_angles,
                sigma=args.infill_sigma,
            )
            for ds_i in interp_all
        ]

    # If resample-average with freq: average numeric vars and circular-mean angle vars
    if args.time_mode == "freq" and args.resample_average and args.time_freq:
        hint = set(v.strip().lower() for v in args.angle_vars.split(",") if v.strip())
        merged_vars = {}
        # Merge after averaging each ds; variables from later files can overwrite earlier (same name)
        for ds in interp_all:
            parts = []
            angles = []
            numerics = []
            for v in ds.data_vars:
                da = ds[v]
                name = v.lower()
                is_angle = (name in hint) or (
                    "degree" in str(da.attrs.get("units", "")).lower()
                )
                if is_angle:
                    m = da.resample(time=args.time_freq).map(
                        lambda x: circular_mean_deg(x, dim="time")
                    )
                    angles.append(m.rename(v))
                else:
                    numerics.append(v)
            if numerics:
                parts.append(
                    ds[numerics].resample(time=args.time_freq).mean(keep_attrs=True)
                )
            if angles:
                parts.append(xr.merge(angles))
            ds_avg = xr.merge(parts).sortby(["latitude", "longitude", "time"])
            for v in ds_avg.data_vars:
                merged_vars[v] = ds_avg[v]
        merged = xr.Dataset(merged_vars).sortby(["latitude", "longitude", "time"])
    else:
        # simple merge on aligned/interp coordinates
        merged_vars = {}
        for ds in interp_all:
            for v in ds.data_vars:
                merged_vars[v] = ds[v]
        merged = xr.Dataset(merged_vars).sortby(["latitude", "longitude", "time"])

    export_grib_from_ds(merged, Path(args.out))


if __name__ == "__main__":
    main()

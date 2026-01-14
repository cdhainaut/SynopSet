#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def deg_to_uv_from(direction_deg):
    th = np.deg2rad(direction_deg)
    u = -np.sin(th)  # +est
    v = -np.cos(th)  # +nord
    return u, v


def parse_range(s):
    """'a:b' -> (a, b) en float ; None si vide."""
    if s is None:
        return None
    a, b = s.split(":")
    return float(a), float(b)


def animate(
    path_nc: Path,
    out_path: Path,
    tmax=None,
    tstep=1,
    lat_range=None,
    lon_range=None,
    qstep=None,
    dpi=140,
):
    ds = xr.open_dataset(path_nc)

    # --- Subsets légers (on découpe avant de charger en mémoire) ---
    if lat_range:
        a, b = lat_range
        ds = ds.sel(latitude=slice(a, b))
    if lon_range:
        a, b = lon_range
        ds = ds.sel(longitude=slice(a, b))

    if tmax is not None:
        ds = ds.isel(time=slice(0, tmax))
    if tstep and tstep > 1:
        ds = ds.isel(time=slice(0, None, tstep))

    # --- Variables (dataset connu) ---
    u10 = ds["u10"].values  # (t, y, x)
    v10 = ds["v10"].values
    swh = ds["swh"].values
    mwd = ds["mwd"].values  # degrés (coming-from)

    time = ds["time"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    nt, ny, nx = swh.shape
    Lon, Lat = np.meshgrid(lon, lat)

    ws = np.hypot(u10, v10)

    # Flèches de houle : direction seule (norme fixée)
    uw_raw, vw_raw = deg_to_uv_from(mwd)
    wave_scale = 0.8
    uw = uw_raw * wave_scale
    vw = vw_raw * wave_scale

    # Sous-échantillonnage quiver (lisibilité)
    if qstep is None:
        sy = max(1, ny // 30)
        sx = max(1, nx // 30)
    else:
        sy = sx = max(1, qstep)
    sly, slx = slice(0, ny, sy), slice(0, nx, sx)
    Lon_q = Lon[sly, slx]
    Lat_q = Lat[sly, slx]

    # Bornes fixes
    vmin_w, vmax_w = np.nanpercentile(ws, [1, 99])
    vmin_h, vmax_h = np.nanpercentile(swh, [1, 99])

    # --- Figure ---
    plt.close("all")
    fig, (ax_wind, ax_wave) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )

    im_wind = ax_wind.pcolormesh(
        Lon, Lat, ws[0], shading="auto", vmin=vmin_w, vmax=vmax_w
    )
    cb1 = fig.colorbar(im_wind, ax=ax_wind, fraction=0.046, pad=0.04)
    cb1.set_label("Wind speed (m/s)")
    q_wind = ax_wind.quiver(
        Lon_q, Lat_q, u10[0][sly, slx], v10[0][sly, slx], pivot="mid", scale=None
    )
    ax_wind.set_title("Vent 10 m — vitesse & direction")
    ax_wind.set_xlabel("Longitude")
    ax_wind.set_ylabel("Latitude")

    im_wave = ax_wave.pcolormesh(
        Lon, Lat, swh[0], shading="auto", vmin=vmin_h, vmax=vmax_h
    )
    cb2 = fig.colorbar(im_wave, ax=ax_wave, fraction=0.046, pad=0.04)
    cb2.set_label("SWH (m)")
    q_wave = ax_wave.quiver(
        Lon_q, Lat_q, uw[0][sly, slx], vw[0][sly, slx], pivot="mid", scale=None
    )
    ax_wave.set_title("Vagues — SWH & direction")
    ax_wave.set_xlabel("Longitude")
    ax_wave.set_ylabel("Latitude")

    suptitle = fig.suptitle(f"{np.datetime_as_string(time[0], unit='h')}", fontsize=11)

    def update(i):
        # coords 1D + shading='auto' -> set_array attend (ny*nx)
        im_wind.set_array(ws[i].ravel())
        im_wave.set_array(swh[i].ravel())
        q_wind.set_UVC(u10[i][sly, slx], v10[i][sly, slx])
        q_wave.set_UVC(uw[i][sly, slx], vw[i][sly, slx])
        suptitle.set_text(f"{np.datetime_as_string(time[i], unit='h')}")
        return []

    anim = FuncAnimation(fig, update, frames=nt, interval=300, blit=False)

    # --- Export ---
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    if out_path.suffix.lower() in (".mp4", ".m4v"):
        try:
            writer = FFMpegWriter(fps=4, bitrate=2400)
        except Exception:
            writer = None
    if writer is None and out_path.suffix.lower() == ".gif":
        try:
            writer = PillowWriter(fps=4)
        except Exception:
            writer = None
    if writer is None:
        try:
            writer = FFMpegWriter(fps=4, bitrate=2400)
            out_path = out_path.with_suffix(".mp4")
        except Exception:
            writer = PillowWriter(fps=4)
            out_path = out_path.with_suffix(".gif")

    anim.save(out_path, writer=writer, dpi=dpi)
    print(f"[OK] Animation -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", default="merged_12H.nc")
    ap.add_argument("--out", dest="output", default="anim_wind_wave.mp4")
    ap.add_argument(
        "--tmax",
        type=int,
        default=None,
        help="Nombre de pas de temps à garder depuis t=0",
    )
    ap.add_argument(
        "--tstep", type=int, default=1, help="Stride temporel (1 image sur K)"
    )
    ap.add_argument(
        "--lat", type=str, default=None, help="Fenêtre latitude 'min:max' (ex: 30:50)"
    )
    ap.add_argument(
        "--lon", type=str, default=None, help="Fenêtre longitude 'min:max' (ex: -20:10)"
    )
    ap.add_argument(
        "--qstep", type=int, default=None, help="Sous-échantillonnage des flèches"
    )
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    lat_range = parse_range(args.lat)
    lon_range = parse_range(args.lon)

    animate(
        args.input,
        args.output,
        tmax=args.tmax,
        tstep=args.tstep,
        lat_range=lat_range,
        lon_range=lon_range,
        qstep=args.qstep,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

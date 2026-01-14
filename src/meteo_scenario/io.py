from __future__ import annotations
from pathlib import Path
import shutil, tempfile, os
import numpy as np
import xarray as xr


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def open_normalize(path: str | Path) -> xr.Dataset:
    """Open a GRIB/NetCDF with xarray, harmonize dims to (time, latitude, longitude)."""
    ds = xr.open_dataset(path, chunks={"time": 50, "latitude": 200, "longitude": 200})
    if "time" not in ds.dims:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        elif "forecast_time" in ds.coords:
            ds = ds.rename({"forecast_time": "time"})
        else:
            raise ValueError("No time dimension found.")
    lat = "latitude" if "latitude" in ds.dims else ("lat" if "lat" in ds.dims else None)
    lon = (
        "longitude" if "longitude" in ds.dims else ("lon" if "lon" in ds.dims else None)
    )
    if lat is None or lon is None:
        raise ValueError("Missing latitude/longitude dims.")
    ds = ds.rename({lat: "latitude", lon: "longitude"}).sortby(
        ["latitude", "longitude", "time"]
    )
    return ds


def normalize_longitudes(ds: xr.Dataset, target: str = "negpos") -> xr.Dataset:
    """Normalize longitude to [-180,180] ('negpos') or [0,360) ('pos360')."""
    lon = ds["longitude"].values
    if target == "negpos":
        lon2 = ((lon + 180.0) % 360.0) - 180.0
    else:
        lon2 = lon % 360.0
    order = np.argsort(lon2)
    ds = ds.assign_coords(longitude=(("longitude",), lon2)).isel(longitude=order)
    return ds.sortby(["longitude", "latitude", "time"])


def export_grib_from_ds(ds: xr.Dataset, out_grib: Path) -> bool:
    """
    Export to GRIB2 via CDO; fallback NetCDF if CDO not available.
    Returns True if GRIB2 success, False otherwise (NetCDF written).
    """
    out_grib = Path(out_grib)
    out_grib.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="met_scen_"))
    tmp_nc = tmpdir / "tmp.nc"
    ds.to_netcdf(tmp_nc)
    if have("cdo"):
        cmd = f'cdo -f grb2 copy "{tmp_nc}" "{out_grib}"'
        ret = os.system(cmd)
        ok = (ret == 0) and out_grib.exists()
        try:
            tmp_nc.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass
        if ok:
            print(f"[OK] GRIB2 → {out_grib}")
            return True
        print(f"[WARN] CDO failed ({ret}). Kept NetCDF fallback.")
    else:
        print("[WARN] 'cdo' not found. Keeping NetCDF and printing conversion hint.")
    fallback_nc = out_grib.with_suffix(".nc")
    ds.to_netcdf(fallback_nc)
    print(f"[OK] NetCDF → {fallback_nc}")
    print(f"→ Later: cdo -f grb2 copy '{fallback_nc}' '{out_grib}'")
    return False

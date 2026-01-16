from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr


def choose_target_grid(
    dsets: List[xr.Dataset], how: str = "finer", ref: xr.Dataset | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    if ref is not None:
        return ref["latitude"].values, ref["longitude"].values
    lats = [d["latitude"].values for d in dsets]
    lons = [d["longitude"].values for d in dsets]
    if how == "first":
        return lats[0], lons[0]

    def res(a):
        return np.min(np.abs(np.diff(a))) if a.size > 1 else np.inf

    dlat = min(res(a) for a in lats)
    dlon = min(res(a) for a in lons)
    lat_min = max(a.min() for a in lats)
    lat_max = min(a.max() for a in lats)
    lon_min = max(a.min() for a in lons)
    lon_max = min(a.max() for a in lons)
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError("No overlapping spatial extent across inputs.")
    # keep most common orientation for latitude
    descending = sum(int(a[0] > a[-1]) for a in lats) > (len(lats) // 2)
    lat_t = np.arange(lat_min, lat_max + 1e-9, dlat)
    if descending:
        lat_t = lat_t[::-1]
    lon_t = np.arange(lon_min, lon_max + 1e-9, dlon)
    return lat_t, lon_t


def temporal_align(
    dsets: List[xr.Dataset], mode: str = "intersection", freq: Optional[str] = None
) -> List[xr.Dataset]:
    if mode == "intersection":
        times = [pd.DatetimeIndex(ds["time"].values) for ds in dsets]
        common = times[0].values
        for t in times[1:]:
            common = np.intersect1d(common, t.values)
        if common.size == 0:
            raise ValueError("No common timestamps across inputs.")
        return [ds.sel(time=common) for ds in dsets]
    if mode == "freq":
        if not freq:
            raise ValueError("Provide freq when mode='freq'")
        t0 = max(pd.DatetimeIndex(ds["time"].values).min() for ds in dsets)
        t1 = min(pd.DatetimeIndex(ds["time"].values).max() for ds in dsets)
        if t0 >= t1:
            raise ValueError("No overlapping time span across inputs.")
        target_t = pd.date_range(t0, t1, freq=freq)
        return [ds.interp(time=target_t) for ds in dsets]
    raise ValueError("Unknown temporal mode.")


def interp_to(ds: xr.Dataset, lat_t: np.ndarray, lon_t: np.ndarray) -> xr.Dataset:
    return ds.interp(latitude=lat_t, longitude=lon_t)


def circular_mean_deg(angles_deg: xr.DataArray, dim="time") -> xr.DataArray:
    ang = np.deg2rad(angles_deg)
    s, c = np.sin(ang), np.cos(ang)
    mean_ang = np.arctan2(s.mean(dim=dim), c.mean(dim=dim))
    return (np.rad2deg(mean_ang) + 360.0) % 360.0

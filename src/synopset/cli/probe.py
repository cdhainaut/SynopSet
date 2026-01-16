from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from ..io import open_normalize
from ..windows import build_windows
from ..plotting import probe_panels


def main():
    ap = argparse.ArgumentParser(
        description="Probe-point diagnostic plots per cluster (u10 & v10)."
    )
    ap.add_argument(
        "grib", type=str, help="Path to GRIB/NetCDF with cluster info (or provide CSV)."
    )
    ap.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Optional CSV with columns time,cluster_id.",
    )
    ap.add_argument("--probe-lat", type=float, required=True)
    ap.add_argument("--probe-lon", type=float, required=True)
    ap.add_argument(
        "--window-hours", type=int, default=72, help="Must match clustering setup."
    )
    ap.add_argument(
        "--stride-hours", type=int, default=24, help="Must match clustering setup."
    )
    ap.add_argument("--plot-out", type=str, default="probe.png")
    args = ap.parse_args()

    ds = open_normalize(args.grib).sortby(["latitude", "longitude", "time"])
    times = pd.DatetimeIndex(ds["time"].values)

    # Rebuild windows to match clustering (best-effort if CSV provided)
    wins = build_windows(times, args.window_hours, args.stride_hours)
    if not wins:
        raise ValueError("No window fits with given window/stride.")

    # cluster labels per window: try from dataset; else derive from time labels by majority
    # Minimal approach: if ds has cluster_id per time-step, rebuild per-window labels via majority voting on that window.
    if "cluster_id" in ds:
        # read time labels from any spatial point (assumed constant)
        try:
            lab_t = ds["cluster_id"].isel(latitude=0, longitude=0).values.astype(int)
        except Exception:
            lab_t = ds["cluster_id"].values.astype(int)

        # Window labels = majority over that window
        labels_seq = np.zeros(len(wins), dtype=int)
        for i, (s, e) in enumerate(wins):
            vals, counts = np.unique(lab_t[s : e + 1], return_counts=True)
            labels_seq[i] = int(vals[np.argmax(counts)])
    else:
        # CSV fallback (align exact times)
        if args.labels_csv is None:
            raise ValueError("No cluster_id in DS and no --labels-csv provided.")
        lab = pd.read_csv(args.labels_csv)
        lab["time"] = pd.to_datetime(lab["time"]).values.astype("datetime64[ns]")
        times_dt = ds["time"].to_pandas().astype("datetime64[ns]")
        merged = pd.DataFrame({"time": times_dt}).merge(lab, on="time", how="left")
        if merged["cluster_id"].isna().any():
            print("[WARN] Some timestamps have no cluster_id in CSV; filling with -1.")
        lab_t = merged["cluster_id"].fillna(-1).astype(int).to_numpy()
        labels_seq = np.zeros(len(wins), dtype=int)
        for i, (s, e) in enumerate(wins):
            vals, counts = np.unique(lab_t[s : e + 1], return_counts=True)
            labels_seq[i] = int(vals[np.argmax(counts)])

    # crude medoid estimate from time labels: pick the window whose center time is most frequent per cluster
    # (for visualization only; real medoids are exported by meteo-reduce)
    centers = {}
    k = int(np.max(labels_seq)) + 1
    for c in range(k):
        idx = np.where(labels_seq == c)[0]
        if idx.size == 0:
            continue
        centers[c] = int(idx[len(idx) // 2])

    probe_panels(
        ds,
        wins,
        labels_seq,
        centers,
        args.probe_lat,
        args.probe_lon,
        out_png=args.plot_out,
    )


if __name__ == "__main__":
    main()

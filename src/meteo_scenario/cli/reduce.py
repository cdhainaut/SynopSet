from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

from ..io import open_normalize, export_grib_from_ds
from ..windows import build_windows
from ..clustering import cluster_sequences_dtw  # DTW path keeps existing impl
from ..export import majority_vote_labels, attach_time_labels_scores


# -----------------------------
# Streaming flatten (time, F)
# -----------------------------
def iter_time_batches_flat(
    ds_std: xr.Dataset, vars_list: List[str], batch_t: int
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Iterate over time in blocks; yields (s, e, Xb) with Xb shape (e-s, F) float32.
    Memory: one time batch in RAM.
    """
    if batch_t <= 0:
        raise ValueError("--batch-time must be >= 1")

    T = int(ds_std.sizes["time"])
    ny = int(ds_std.sizes["latitude"])
    nx = int(ds_std.sizes["longitude"])
    n_pix = ny * nx
    n_vars = len(vars_list)
    F = n_vars * n_pix

    for s in range(0, T, batch_t):
        e = min(T, s + batch_t)
        n = e - s
        Xb = np.empty((n, F), dtype=np.float32)

        for j, v in enumerate(vars_list):
            # (n, ny, nx) -> (n, n_pix)
            A = (
                ds_std[v]
                .isel(time=slice(s, e))
                .transpose("time", "latitude", "longitude")
                .to_numpy()
            )
            if A.dtype != np.float32:
                A = A.astype(np.float32, copy=False)
            Xb[:, j * n_pix : (j + 1) * n_pix] = A.reshape(n, n_pix)

        yield s, e, Xb


# -----------------------------
# Incremental PCA (robust)
# -----------------------------
def fit_ipca_stream(
    ds_std: xr.Dataset,
    vars_list: List[str],
    batch_t: int,
    n_components: int,
    fit_sample_rate: float,
    seed: int,
) -> IncrementalPCA:
    """
    Fit IncrementalPCA in streaming mode.
    Handles small batches via buffering and supports temporal subsampling.
    """
    if n_components <= 0:
        raise ValueError("--components must be >= 1")

    ny = int(ds_std.sizes["latitude"])
    nx = int(ds_std.sizes["longitude"])
    F = len(vars_list) * ny * nx
    comp = min(int(n_components), int(F))

    ipca = IncrementalPCA(n_components=comp)

    r = float(fit_sample_rate)
    if not (0.0 < r <= 1.0):
        r = 1.0
    rng = np.random.default_rng(seed)

    buffer: List[np.ndarray] = []
    buf_rows = 0
    total_fit_rows = 0

    for _, _, Xb in iter_time_batches_flat(ds_std, vars_list, batch_t):
        if r < 1.0:
            mask = rng.random(Xb.shape[0]) < r
            if not mask.any():
                continue
            Xb = Xb[mask]

        if Xb.shape[0] == 0:
            continue

        buffer.append(Xb)
        buf_rows += Xb.shape[0]
        total_fit_rows += Xb.shape[0]

        # sklearn requires n_samples >= n_components per partial_fit call
        if buf_rows >= comp:
            Xbuf = np.concatenate(buffer, axis=0)
            ipca.partial_fit(Xbuf)
            buffer.clear()
            buf_rows = 0

    if ipca.n_samples_seen_ is None or int(ipca.n_samples_seen_) == 0:
        raise ValueError(
            "IPCA fit got 0 samples. Check --fit-sample-rate, dataset length, and filters."
        )
    if total_fit_rows < comp:
        raise ValueError(
            f"Not enough samples to fit IPCA: got {total_fit_rows} < components {comp}. "
            "Increase time range, --fit-sample-rate, or reduce --components."
        )

    return ipca


def transform_to_memmap(
    ds_std: xr.Dataset,
    vars_list: List[str],
    batch_t: int,
    out_path: Path,
    use_pca: bool,
    ipca: IncrementalPCA | None,
    T: int,
    d: int,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(out_path, dtype="float32", mode="w+", shape=(T, d))
    for s, e, Xb in iter_time_batches_flat(ds_std, vars_list, batch_t):
        if use_pca:
            assert ipca is not None
            Z = ipca.transform(Xb)
        else:
            Z = Xb
        mm[s:e, :] = Z.astype(np.float32, copy=False)
    del mm  # flush to disk


# -----------------------------
# Euclid clustering (streaming)
# -----------------------------
def _iter_window_vectors(
    steps_embed: np.memmap,
    wins: List[Tuple[int, int]],
    win_len: int,
    d: int,
    batch_w: int,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Yield (w0, w1, X) where X is (w1-w0, win_len*d) float32.
    Copies only one batch of windows at a time.
    """
    N = len(wins)
    p = win_len * d
    for w0 in range(0, N, batch_w):
        w1 = min(N, w0 + batch_w)
        n = w1 - w0
        X = np.empty((n, p), dtype=np.float32)
        for i in range(n):
            s, e = wins[w0 + i]
            X[i, :] = steps_embed[s : e + 1, :].reshape(-1)
        yield w0, w1, X


def cluster_windows_euclid_stream(
    steps_embed: np.memmap,
    wins: List[Tuple[int, int]],
    k: int,
    seed: int,
    batch_windows: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """
    Streaming MiniBatchKMeans on window vectors.
    Returns: labels (N,), centers (k,p), window_score (N,), medoid_win_idx (dict cluster->window index).
    """
    if k <= 0:
        raise ValueError("--clusters must be >= 1")
    if not wins:
        raise ValueError("No windows to cluster.")

    win_len = int(wins[0][1] - wins[0][0] + 1)
    d = int(steps_embed.shape[1])
    p = win_len * d

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        n_init="auto",
        batch_size=min(1024, batch_windows),
    )

    # ---- fit (buffer so first partial_fit sees enough samples) ----
    fit_buf: List[np.ndarray] = []
    fit_rows = 0
    for _, _, X in _iter_window_vectors(steps_embed, wins, win_len, d, batch_windows):
        fit_buf.append(X)
        fit_rows += X.shape[0]
        if fit_rows >= k:
            km.partial_fit(np.concatenate(fit_buf, axis=0))
            fit_buf.clear()
            fit_rows = 0
        else:
            continue

    if fit_buf:
        km.partial_fit(np.concatenate(fit_buf, axis=0))

    centers = km.cluster_centers_  # (k, p)

    # ---- predict + distances (store per-window) ----
    N = len(wins)
    labels = np.empty(N, dtype=np.int32)
    d2 = np.empty(N, dtype=np.float32)  # squared distance to assigned center

    # per-cluster min/max distance + medoid
    d2_min = np.full(k, np.inf, dtype=np.float64)
    d2_max = np.full(k, -np.inf, dtype=np.float64)
    best_d2 = np.full(k, np.inf, dtype=np.float64)
    medoid_win_idx: dict[int, int] = {}

    for w0, w1, X in _iter_window_vectors(steps_embed, wins, win_len, d, batch_windows):
        labs = km.predict(X).astype(np.int32, copy=False)
        C = centers[labs]  # (n, p)
        diff = X - C
        # squared L2 per row
        dd = np.einsum("ij,ij->i", diff, diff).astype(np.float32, copy=False)

        labels[w0:w1] = labs
        d2[w0:w1] = dd

        for i in range(w1 - w0):
            c = int(labs[i])
            v = float(dd[i])
            if v < d2_min[c]:
                d2_min[c] = v
            if v > d2_max[c]:
                d2_max[c] = v
            if v < best_d2[c]:
                best_d2[c] = v
                medoid_win_idx[c] = int(w0 + i)

    # ---- window_score in [0,1] (same logic as old code, using squared distances) ----
    window_score = np.zeros(N, dtype=np.float32)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        mn = float(d2_min[c])
        mx = float(d2_max[c])
        if mx > mn:
            window_score[idx] = 1.0 - (d2[idx] - mn) / (mx - mn)
        else:
            window_score[idx] = 1.0

    return labels, centers, window_score, medoid_win_idx


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reduce weather scenarios: streaming PCA -> window clustering -> weights + medoids + time votes."
    )
    ap.add_argument("grib", type=str, help="Path to merged GRIB/NetCDF")

    # Vars + preprocessing
    ap.add_argument(
        "--vars", type=str, default="u10,v10", help="Comma-separated variables."
    )
    ap.add_argument(
        "--coarsen", type=int, default=0, help="Spatial coarsen factor (0=off)."
    )
    ap.add_argument(
        "--time-agg",
        type=str,
        default="",
        help="Resample frequency (e.g., '6H'); empty=off.",
    )

    # Windows + clustering
    ap.add_argument("--window-hours", type=int, default=72)
    ap.add_argument("--stride-hours", type=int, default=24)
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--seq-metric", choices=["euclid", "dtw"], default="euclid")
    ap.add_argument(
        "--pam-iters", type=int, default=8, help="Only for DTW (PAM refinements)."
    )
    ap.add_argument(
        "--batch-windows",
        type=int,
        default=2048,
        help="Batch size for Euclid window vector streaming.",
    )

    # PCA
    ap.add_argument("--use-pca", dest="use_pca", action="store_true", default=True)
    ap.add_argument(
        "--no-pca",
        dest="use_pca",
        action="store_false",
        help="Strongly discouraged (very large embedding).",
    )
    ap.add_argument(
        "--uce-pca",
        dest="use_pca",
        action="store_true",
        help="Deprecated alias of --use-pca.",
    )
    ap.add_argument("--components", type=int, default=15)
    ap.add_argument(
        "--batch-time",
        type=int,
        default=512,
        help="Time batch size for streaming flatten/PCA.",
    )
    ap.add_argument(
        "--fit-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of timesteps used for IPCA fit (0<r<=1).",
    )
    ap.add_argument("--seed", type=int, default=42)

    # Outputs
    ap.add_argument("--out", type=str, default="reduced_out")
    ap.add_argument("--memmap-name", type=str, default="steps_embed.memmap")
    ap.add_argument(
        "--save-windows",
        action="store_true",
        help="Optional: dump per-window assignments CSV.",
    )

    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Load + variable selection ---
    ds = open_normalize(args.grib)
    keep = [v.strip() for v in args.vars.split(",") if v.strip()]
    for v in keep:
        if v not in ds:
            raise KeyError(f"Var '{v}' not in dataset. Available: {list(ds.data_vars)}")
    ds = ds[keep].sortby(["latitude", "longitude", "time"])

    # Optional: spatial coarsen / time resample
    if args.coarsen and int(args.coarsen) > 1:
        ds = ds.coarsen(
            latitude=int(args.coarsen), longitude=int(args.coarsen), boundary="trim"
        ).mean()
    if args.time_agg:
        ds = ds.resample(time=str(args.time_agg)).mean()

    # Standardize over time (anomalies) + downcast
    mean_t = ds.mean("time", skipna=True)
    std_t = ds.std("time", skipna=True)
    std_t = xr.where(std_t == 0, 1.0, std_t)
    ds_std = ((ds - mean_t) / std_t).astype("float32").fillna(0.0)

    times = pd.DatetimeIndex(ds_std["time"].values)
    T = int(ds_std.sizes["time"])
    vars_list = list(ds_std.data_vars)

    # Decide embedding size
    ny = int(ds_std.sizes["latitude"])
    nx = int(ds_std.sizes["longitude"])
    F = len(vars_list) * ny * nx
    use_pca = bool(args.use_pca)
    d = min(int(args.components), F) if use_pca else F

    print(ds_std)
    print(
        f"[INFO] T={T} | grid={ny}x{nx} | vars={len(vars_list)} | F={F} | embedding d={d}"
    )

    ipca = None
    if use_pca:
        print("[INFO] Fitting IncrementalPCA (streaming)...")
        ipca = fit_ipca_stream(
            ds_std=ds_std,
            vars_list=vars_list,
            batch_t=int(args.batch_time),
            n_components=int(args.components),
            fit_sample_rate=float(args.fit_sample_rate),
            seed=int(args.seed),
        )

    # Transform -> memmap
    steps_path = OUT / str(args.memmap_name)
    print(f"[INFO] Writing embedding memmap: {steps_path}")
    transform_to_memmap(
        ds_std=ds_std,
        vars_list=vars_list,
        batch_t=int(args.batch_time),
        out_path=steps_path,
        use_pca=use_pca,
        ipca=ipca,
        T=T,
        d=d,
    )

    # Reopen memmap read-only (zero-copy)
    steps_embed = np.memmap(steps_path, dtype="float32", mode="r", shape=(T, d))

    # Windows
    print("[INFO] Building windows...")
    wins = build_windows(times, int(args.window_hours), int(args.stride_hours))
    if not wins:
        raise ValueError("No window fits: enlarge time range or reduce --window-hours.")
    win_bounds = np.asarray(wins, dtype=int)

    # Clustering
    print(
        f"[INFO] Clustering {len(wins)} windows with metric='{args.seq_metric}' (k={int(args.clusters)})"
    )
    k = int(args.clusters)

    if args.seq_metric == "euclid":
        labels_seq, centers, window_score, medoid_win_idx = (
            cluster_windows_euclid_stream(
                steps_embed=steps_embed,
                wins=wins,
                k=k,
                seed=int(args.seed),
                batch_windows=int(args.batch_windows),
            )
        )
    else:
        # DTW path: WARNING - O(N^2) memory/time. Keep original implementation.
        seq_list = [steps_embed[s : e + 1, :] for (s, e) in wins]  # views on memmap
        labels_seq, medoid_win_idx, window_score = cluster_sequences_dtw(
            seq_list, k, pam_iters=int(args.pam_iters)
        )
        labels_seq = np.asarray(labels_seq, dtype=np.int32)
        window_score = np.asarray(window_score, dtype=float)

    # Weights per cluster (+ weighted by window quality)
    counts = np.bincount(labels_seq, minlength=k).astype(int)
    weight_frac = (
        counts / counts.sum() if counts.sum() > 0 else np.zeros(k, dtype=float)
    )
    wcounts = np.bincount(labels_seq, weights=window_score, minlength=k)
    wfrac = wcounts / wcounts.sum() if wcounts.sum() > 0 else np.zeros(k, dtype=float)

    # Cluster summary (+ medoid window metadata)
    rows = []
    for c in range(k):
        mw = int(medoid_win_idx.get(c, -1))
        if mw >= 0:
            si, ei = win_bounds[mw]
            st = pd.Timestamp(times[si])
            et = pd.Timestamp(times[ei])
            ms = float(window_score[mw])
        else:
            si = ei = -1
            st = et = pd.NaT
            ms = float("nan")

        rows.append(
            {
                "cluster": c,
                "weight_count": int(counts[c]),
                "weight_frac": float(weight_frac[c]),
                "weight_count_weighted": float(np.round(wcounts[c], 3)),
                "weight_frac_weighted": float(np.round(wfrac[c], 6)),
                "medoid_window_index": mw,
                "medoid_score": ms,
                "medoid_start_time": st,
                "medoid_end_time": et,
            }
        )

    pd.DataFrame(rows).sort_values("cluster").to_csv(
        OUT / "cluster_summary.csv", index=False
    )

    # Time vote labels
    labels_time, time_vote_frac = majority_vote_labels(wins, labels_seq, T=len(times))
    pd.DataFrame(
        {"time": times, "cluster_id": labels_time, "vote_frac": time_vote_frac}
    ).to_csv(OUT / "time_cluster_map.csv", index=False)

    # Attach labels back to dataset and export
    ds_with = attach_time_labels_scores(ds, labels_time, time_vote_frac)
    export_grib_from_ds(ds_with, OUT / "original_with_cluster_id.grib2")

    # Optional: per-window dump
    if args.save_windows:
        wrows = []
        for w, (si, ei) in enumerate(win_bounds):
            wrows.append(
                {
                    "window_index": int(w),
                    "start_index": int(si),
                    "end_index": int(ei),
                    "start_time": pd.Timestamp(times[si]),
                    "end_time": pd.Timestamp(times[ei]),
                    "cluster": int(labels_seq[w]),
                    "window_score": float(window_score[w]),
                }
            )
        pd.DataFrame(wrows).sort_values(
            ["cluster", "window_score"], ascending=[True, False]
        ).to_csv(OUT / "windows_assignments.csv", index=False)

    print(f"[DONE] Outputs in: {OUT}")
    print(" - cluster_summary.csv  (weights + medoids)")
    print(" - time_cluster_map.csv (vote_frac)")
    print(" - original_with_cluster_id.grib2")
    print(f" - {args.memmap_name} (embedding memmap)")
    if args.save_windows:
        print(" - windows_assignments.csv (optional)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..clustering import cluster_sequences_dtw, cluster_windows_euclid_stream
from ..export import attach_time_labels_scores, export_medoids, majority_vote_labels
from ..io import (
    export_grib_from_ds,
    fit_ipca_stream,
    open_normalize,
    transform_to_memmap,
    standardize_over_time,
)
from ..windows import build_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce weather scenarios: standardize -> (optional) streaming PCA -> "
            "window clustering (euclid or DTW) -> weights + medoids + time vote map"
        )
    )
    parser.add_argument("grib", type=str, help="Path to merged GRIB/NetCDF")

    # Variables + preprocessing
    parser.add_argument(
        "--vars", type=str, default="u10,v10", help="Comma-separated variables."
    )
    parser.add_argument(
        "--coarsen", type=int, default=0, help="Spatial coarsen factor (0=off)."
    )
    parser.add_argument(
        "--time-agg",
        type=str,
        default="",
        help="Resample frequency (e.g., '6H'); empty=off.",
    )

    # Windows + clustering
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument(
        "--seq-metric",
        choices=["euclid", "dtw"],
        default="euclid",
        help=(
            "'euclid' is scalable (streaming MiniBatchKMeans). "
            "'dtw' is more robust to time shifts but O(N^2) in windows."
        ),
    )
    parser.add_argument(
        "--pam-iters",
        type=int,
        default=8,
        help="Only for DTW (PAM refinements).",
    )
    parser.add_argument(
        "--batch-windows",
        type=int,
        default=2048,
        help="Batch size for Euclid window vector streaming.",
    )

    # PCA
    parser.add_argument("--use-pca", dest="use_pca", action="store_true", default=True)
    parser.add_argument(
        "--no-pca",
        dest="use_pca",
        action="store_false",
        help="Strongly discouraged (very large embedding).",
    )
    parser.add_argument(
        "--uce-pca",
        dest="use_pca",
        action="store_true",
        help="Deprecated alias of --use-pca.",
    )
    parser.add_argument("--components", type=int, default=15)
    parser.add_argument(
        "--batch-time",
        type=int,
        default=512,
        help="Time batch size for streaming flatten/PCA.",
    )
    parser.add_argument(
        "--fit-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of timesteps used for IPCA fit (0<r<=1).",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Medoid export
    parser.add_argument(
        "--export-medoids", dest="export_medoids", action="store_true", default=True
    )
    parser.add_argument(
        "--no-export-medoids", dest="export_medoids", action="store_false"
    )
    parser.add_argument(
        "--medoid-format", choices=["grib", "nc", "both"], default="both"
    )
    parser.add_argument(
        "--medoid-dir",
        type=str,
        default="medoids",
        help="Subdirectory under --out",
    )
    parser.add_argument(
        "--medoid-name-time",
        action="store_true",
        help="Include start/end timestamps in medoid filenames.",
    )

    # Outputs
    parser.add_argument("--out", type=str, default="reduced_out")
    parser.add_argument("--memmap-name", type=str, default="steps_embed.memmap")
    parser.add_argument(
        "--save-windows",
        action="store_true",
        help="Optional: dump per-window assignments CSV.",
    )

    args = parser.parse_args()

    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load + variable selection
    # ------------------------------------------------------------------
    dataset_original = open_normalize(path=args.grib)
    variable_names = [
        name.strip() for name in str(args.vars).split(",") if name.strip()
    ]

    for variable_name in variable_names:
        if variable_name not in dataset_original:
            raise KeyError(
                f"Var '{variable_name}' not in dataset. Available: {list(dataset_original.data_vars)}"
            )

    dataset_original = dataset_original[variable_names].sortby(
        ["latitude", "longitude", "time"]
    )

    # Optional: spatial coarsen / time resample
    if int(args.coarsen) > 1:
        dataset_original = dataset_original.coarsen(
            latitude=int(args.coarsen),
            longitude=int(args.coarsen),
            boundary="trim",
        ).mean()

    if str(args.time_agg):
        dataset_original = dataset_original.resample(time=str(args.time_agg)).mean()

    # ------------------------------------------------------------------
    # Standardize + prepare dimensions
    # ------------------------------------------------------------------
    dataset_standardized = standardize_over_time(ds=dataset_original)

    time_index = pd.DatetimeIndex(dataset_standardized["time"].values)
    num_timesteps = int(dataset_standardized.sizes["time"])

    num_latitude = int(dataset_standardized.sizes["latitude"])
    num_longitude = int(dataset_standardized.sizes["longitude"])
    num_features = (
        len(list(dataset_standardized.data_vars)) * num_latitude * num_longitude
    )

    use_pca = bool(args.use_pca)
    embedding_dim = (
        min(int(args.components), int(num_features)) if use_pca else int(num_features)
    )

    print(dataset_standardized)
    print(
        "[INFO] "
        f"T={num_timesteps} | grid={num_latitude}x{num_longitude} | vars={len(dataset_standardized.data_vars)} "
        f"| features={num_features} | embedding_dim={embedding_dim}"
    )

    # ------------------------------------------------------------------
    # Fit streaming PCA (optional)
    # ------------------------------------------------------------------
    ipca = None
    if use_pca:
        print("[INFO] Fitting IncrementalPCA (streaming)...")
        ipca = fit_ipca_stream(
            ds_standardized=dataset_standardized,
            variable_names=list(dataset_standardized.data_vars),
            time_batch_size=int(args.batch_time),
            requested_components=int(args.components),
            fit_sample_rate=float(args.fit_sample_rate),
            seed=int(args.seed),
        )

    # ------------------------------------------------------------------
    # Transform -> memmap embedding
    # ------------------------------------------------------------------
    memmap_path = output_directory / str(args.memmap_name)
    print(f"[INFO] Writing embedding memmap: {memmap_path}")

    transform_to_memmap(
        ds_standardized=dataset_standardized,
        variable_names=list(dataset_standardized.data_vars),
        time_batch_size=int(args.batch_time),
        memmap_path=memmap_path,
        use_pca=use_pca,
        ipca=ipca,
        num_timesteps=num_timesteps,
        embedding_dim=embedding_dim,
    )

    embedding_memmap = np.memmap(
        memmap_path,
        dtype="float32",
        mode="r",
        shape=(num_timesteps, embedding_dim),
    )

    # ------------------------------------------------------------------
    # Build windows
    # ------------------------------------------------------------------
    print("[INFO] Building windows...")
    windows = build_windows(
        times=time_index,
        window_hours=int(args.window_hours),
        stride_hours=int(args.stride_hours),
    )
    if not windows:
        raise ValueError("No window fits: enlarge time range or reduce --window-hours.")
    window_bounds = np.asarray(windows, dtype=int)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    num_clusters = int(args.clusters)
    print(
        f"[INFO] Clustering {len(windows)} windows with metric='{args.seq_metric}' (k={num_clusters})"
    )

    if str(args.seq_metric) == "euclid":
        labels_per_window, cluster_centers, window_score, medoid_window_index = (
            cluster_windows_euclid_stream(
                embedding_memmap=embedding_memmap,
                windows=windows,
                num_clusters=num_clusters,
                random_seed=int(args.seed),
                window_batch_size=int(args.batch_windows),
            )
        )
        _ = cluster_centers  # centers are saved only for debugging currently
    else:
        # DTW path: WARNING - O(N^2) memory/time due to pairwise distance matrix.
        sequences = [
            embedding_memmap[window_start_index : window_end_index + 1, :]
            for (window_start_index, window_end_index) in windows
        ]
        labels_per_window, medoid_window_index, window_score = cluster_sequences_dtw(
            seq_list=sequences,
            k=num_clusters,
            pam_iters=int(args.pam_iters),
        )
        labels_per_window = np.asarray(labels_per_window, dtype=np.int32)
        window_score = np.asarray(window_score, dtype=float)

    # ------------------------------------------------------------------
    # Export medoids (optional)
    # ------------------------------------------------------------------
    if bool(args.export_medoids):
        medoid_output_directory = output_directory / str(args.medoid_dir)
        print(
            f"[INFO] Exporting medoids to: {medoid_output_directory} (format={args.medoid_format})"
        )
        export_medoids(
            ds_original=dataset_original,
            windows=windows,
            medoid_window_index=medoid_window_index,
            window_score=window_score,
            output_directory=medoid_output_directory,
            output_format=str(args.medoid_format),
            include_time_in_filename=bool(args.medoid_name_time),
        )

    # ------------------------------------------------------------------
    # Cluster weights + summary
    # ------------------------------------------------------------------
    counts_per_cluster = np.bincount(labels_per_window, minlength=num_clusters).astype(
        int
    )
    weight_fraction = (
        counts_per_cluster / counts_per_cluster.sum()
        if counts_per_cluster.sum() > 0
        else np.zeros(num_clusters)
    )

    weighted_counts = np.bincount(
        labels_per_window,
        weights=window_score,
        minlength=num_clusters,
    )
    weighted_fraction = (
        weighted_counts / weighted_counts.sum()
        if weighted_counts.sum() > 0
        else np.zeros(num_clusters)
    )

    cluster_rows: list[dict] = []
    for cluster_id in range(num_clusters):
        medoid_idx = int(medoid_window_index.get(cluster_id, -1))
        if medoid_idx >= 0:
            medoid_start_index, medoid_end_index = window_bounds[medoid_idx]
            medoid_start_time = pd.Timestamp(time_index[int(medoid_start_index)])
            medoid_end_time = pd.Timestamp(time_index[int(medoid_end_index)])
            medoid_score = float(window_score[medoid_idx])
        else:
            medoid_start_index = -1
            medoid_end_index = -1
            medoid_start_time = pd.NaT
            medoid_end_time = pd.NaT
            medoid_score = float("nan")

        cluster_rows.append(
            {
                "cluster": int(cluster_id),
                "weight_count": int(counts_per_cluster[cluster_id]),
                "weight_frac": float(weight_fraction[cluster_id]),
                "weight_count_weighted": float(
                    np.round(weighted_counts[cluster_id], 3)
                ),
                "weight_frac_weighted": float(
                    np.round(weighted_fraction[cluster_id], 6)
                ),
                "medoid_window_index": int(medoid_idx),
                "medoid_score": float(medoid_score),
                "medoid_start_index": int(medoid_start_index),
                "medoid_end_index": int(medoid_end_index),
                "medoid_start_time": medoid_start_time,
                "medoid_end_time": medoid_end_time,
            }
        )

    pd.DataFrame(cluster_rows).sort_values("cluster").to_csv(
        output_directory / "cluster_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Time vote labels
    # ------------------------------------------------------------------
    labels_per_time, vote_fraction = majority_vote_labels(
        windows=windows,
        window_labels=labels_per_window,
        num_timesteps=len(time_index),
    )
    pd.DataFrame(
        {
            "time": time_index,
            "cluster_id": labels_per_time,
            "vote_frac": vote_fraction,
        }
    ).to_csv(output_directory / "time_cluster_map.csv", index=False)

    # Attach labels back to dataset and export
    ds_labeled = attach_time_labels_scores(
        ds=dataset_original,
        labels_time=labels_per_time,
        time_score=vote_fraction,
    )
    export_grib_from_ds(
        ds=ds_labeled, out_grib=output_directory / "original_with_cluster_id.grib2"
    )

    # Optional: per-window dump
    if bool(args.save_windows):
        window_rows: list[dict] = []
        for window_index, (window_start_index, window_end_index) in enumerate(
            window_bounds
        ):
            window_rows.append(
                {
                    "window_index": int(window_index),
                    "start_index": int(window_start_index),
                    "end_index": int(window_end_index),
                    "start_time": pd.Timestamp(time_index[int(window_start_index)]),
                    "end_time": pd.Timestamp(time_index[int(window_end_index)]),
                    "cluster": int(labels_per_window[window_index]),
                    "window_score": float(window_score[window_index]),
                }
            )
        pd.DataFrame(window_rows).sort_values(
            ["cluster", "window_score"],
            ascending=[True, False],
        ).to_csv(output_directory / "windows_assignments.csv", index=False)

    # ------------------------------------------------------------------
    # Final recap
    # ------------------------------------------------------------------
    print(f"[DONE] Outputs in: {output_directory}")
    print(" - cluster_summary.csv  (weights + medoids)")
    print(" - time_cluster_map.csv (vote_frac)")
    print(" - original_with_cluster_id.grib2")
    print(f" - {args.memmap_name} (embedding memmap)")
    if bool(args.export_medoids):
        print(f" - {args.medoid_dir}/... (medoid exports)")
    if bool(args.save_windows):
        print(" - windows_assignments.csv (optional)")


if __name__ == "__main__":
    main()

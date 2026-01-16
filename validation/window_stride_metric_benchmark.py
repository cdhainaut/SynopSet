from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from meteo_scenario.clustering import (
    cluster_sequences_dtw,
    cluster_sequences_euclid,
    dtw_distance,
    medoid_indices_from_centroids,
)
from meteo_scenario.io import (
    fit_ipca_stream,
    open_normalize,
    standardize_over_time,
    transform_to_memmap,
)
from meteo_scenario.windows import build_windows


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def parse_comma_list(*, text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def ensure_variables_present(*, dataset: xr.Dataset, variable_names: List[str]) -> None:
    missing = [name for name in variable_names if name not in dataset.data_vars]
    if missing:
        raise KeyError(
            "Missing variables in dataset: "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(map(str, dataset.data_vars))
        )


def compute_time_step_hours(*, time_index: pd.DatetimeIndex) -> float:
    if len(time_index) < 2:
        raise ValueError("Dataset must have at least 2 timesteps.")
    return float((time_index[1] - time_index[0]).total_seconds()) / 3600.0


def build_windowed_sequences(
    *,
    embedding_array: np.ndarray,
    windows: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """Slice the embedding (time, dim) into per-window sequences (window_len, dim)."""
    sequences: List[np.ndarray] = []
    for start_index, end_index_inclusive in windows:
        sequences.append(
            embedding_array[int(start_index) : int(end_index_inclusive) + 1, :]
        )
    return sequences


def compute_pc1_peak_offset_hours_per_window(
    *,
    windowed_sequences: List[np.ndarray],
    time_step_hours: float,
) -> np.ndarray:
    """
    Routing-oriented timing proxy.

    We use the index of the maximum magnitude of PC1 within each window:
        peak_index = argmax_t |PC1(t)|
        offset_hours = peak_index * time_step_hours

    This is NOT a physical variable. It's a compact proxy to detect whether clusters
    mix windows with very different internal timing.
    """
    peak_offsets_hours = np.zeros(len(windowed_sequences), dtype=float)
    for window_index, sequence in enumerate(windowed_sequences):
        if sequence.size == 0 or sequence.shape[1] == 0:
            peak_offsets_hours[window_index] = 0.0
            continue
        pc1 = sequence[:, 0]
        peak_index = int(np.argmax(np.abs(pc1)))
        peak_offsets_hours[window_index] = float(peak_index) * float(time_step_hours)
    return peak_offsets_hours


def compute_distance_per_step_to_representative(
    *,
    metric: str,
    windowed_sequences: List[np.ndarray],
    labels_per_window: np.ndarray,
    representative_window_index_by_cluster: Dict[int, int],
) -> np.ndarray:
    """
    Comparable representativeness metric for Euclid and DTW.

    For each window, measure distance to the representative (medoid-like) window of its cluster.

    - metric="euclid":
        average per-step L2 distance in embedding space:
            mean_t ||x_t - m_t||_2

    - metric="dtw":
        normalized DTW distance:
            DTW(x, m) / L

    Returns:
        distances_per_step: array (num_windows,)
    """
    if not windowed_sequences:
        return np.asarray([], dtype=float)

    window_length_steps = int(windowed_sequences[0].shape[0])
    distances_per_step = np.full(len(windowed_sequences), np.nan, dtype=float)

    for window_index, sequence in enumerate(windowed_sequences):
        cluster_id = int(labels_per_window[window_index])
        representative_index = int(
            representative_window_index_by_cluster.get(cluster_id, -1)
        )
        if representative_index < 0:
            continue

        representative_sequence = windowed_sequences[representative_index]

        if metric == "euclid":
            # (L, d) -> per-step L2 norms (L,) -> mean
            per_step_norm = np.linalg.norm(sequence - representative_sequence, axis=1)
            distances_per_step[window_index] = float(np.mean(per_step_norm))
        elif metric == "dtw":
            # dtw_distance already uses per-step Euclidean costs; normalize by length
            distances_per_step[window_index] = float(
                dtw_distance(sequence, representative_sequence)
                / float(window_length_steps)
            )
        else:
            raise ValueError("metric must be 'euclid' or 'dtw'")

    return distances_per_step


def normalized_entropy(*, probabilities: np.ndarray) -> float:
    """
    Normalized entropy in [0, 1]:
        H = -sum(p log p)
        H_norm = H / log(K)
    """
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return float("nan")
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    k = int(probabilities.size)
    if k <= 1:
        return 0.0
    return float(entropy / math.log(k))


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def summarize_clusters(
    *,
    labels_per_window: np.ndarray,
    window_score: np.ndarray,
    distance_per_step: np.ndarray,
    windows: List[Tuple[int, int]],
    time_index: pd.DatetimeIndex,
    pc1_peak_offsets_hours: np.ndarray,
    representative_window_index_by_cluster: Dict[int, int],
) -> pd.DataFrame:
    """
    Build a cluster-level table with routing-relevant diagnostics.
    """
    rows: List[dict] = []
    num_clusters = int(labels_per_window.max()) + 1 if labels_per_window.size else 0

    for cluster_id in range(num_clusters):
        window_indices = np.where(labels_per_window == cluster_id)[0]
        if window_indices.size == 0:
            continue

        count = int(window_indices.size)
        sum_window_score = float(window_score[window_indices].sum())

        pc1_peak_mean = float(np.mean(pc1_peak_offsets_hours[window_indices]))
        pc1_peak_std = float(np.std(pc1_peak_offsets_hours[window_indices]))

        # Distance-to-representative stats (comparable across metrics)
        dist_values = distance_per_step[window_indices]
        dist_mean = float(np.nanmean(dist_values))
        dist_p50 = float(np.nanpercentile(dist_values, 50))
        dist_p90 = float(np.nanpercentile(dist_values, 90))

        representative_index = int(
            representative_window_index_by_cluster.get(cluster_id, -1)
        )
        if representative_index >= 0:
            start_i, end_i = windows[representative_index]
            representative_start_time = pd.Timestamp(time_index[int(start_i)])
            representative_end_time = pd.Timestamp(time_index[int(end_i)])
            representative_score = float(window_score[representative_index])
        else:
            representative_start_time = pd.NaT
            representative_end_time = pd.NaT
            representative_score = float("nan")

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(count),
                "sum_window_score": float(sum_window_score),
                "pc1_peak_offset_hours_mean": float(pc1_peak_mean),
                "pc1_peak_offset_hours_std": float(pc1_peak_std),
                "distance_per_step_mean": float(dist_mean),
                "distance_per_step_p50": float(dist_p50),
                "distance_per_step_p90": float(dist_p90),
                "representative_window_index": int(representative_index),
                "representative_start_time": representative_start_time,
                "representative_end_time": representative_end_time,
                "representative_score": float(representative_score),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["weight_frac_count"] = df["count"] / float(df["count"].sum())
    df["weight_frac_score"] = df["sum_window_score"] / float(
        df["sum_window_score"].sum() or 1.0
    )
    df = df.sort_values("cluster_id")
    return df


def compute_global_metrics_from_cluster_table(
    *,
    cluster_table: pd.DataFrame,
    small_cluster_threshold: int,
) -> dict:
    """
    Produce global routing-oriented metrics (comparable Euclid vs DTW).

    Metrics included:
    - timing coherence:
        - timing_std_weighted_mean
        - timing_std_p90_clusters
    - representativeness:
        - distance_per_step_mean
        - distance_per_step_p90
    - fragmentation:
        - min_cluster_size
        - small_cluster_fraction
    - weight distribution:
        - weight_entropy_count_norm
        - weight_entropy_score_norm
    """
    if cluster_table.empty:
        return {
            "timing_std_weighted_mean": float("nan"),
            "timing_std_p90_clusters": float("nan"),
            "distance_per_step_mean": float("nan"),
            "distance_per_step_p90": float("nan"),
            "min_cluster_size": 0,
            "small_cluster_fraction": float("nan"),
            "weight_entropy_count_norm": float("nan"),
            "weight_entropy_score_norm": float("nan"),
        }

    counts = cluster_table["count"].to_numpy(dtype=float)
    weight_count = cluster_table["weight_frac_count"].to_numpy(dtype=float)
    weight_score = cluster_table["weight_frac_score"].to_numpy(dtype=float)

    timing_std = cluster_table["pc1_peak_offset_hours_std"].to_numpy(dtype=float)
    distance_mean = cluster_table["distance_per_step_mean"].to_numpy(dtype=float)

    # Weighted mean of timing stds (weighted by cluster size)
    timing_std_weighted_mean = float(np.sum(weight_count * timing_std))

    # p90 across clusters (unweighted, because it's "worst-ish cluster")
    timing_std_p90_clusters = float(np.nanpercentile(timing_std, 90))

    # Representative distance: also give global mean (weighted by cluster size) + p90 across clusters
    distance_per_step_mean = float(np.sum(weight_count * distance_mean))
    distance_per_step_p90 = float(np.nanpercentile(distance_mean, 90))

    min_cluster_size = int(np.min(counts)) if counts.size else 0
    small_cluster_fraction = float(np.mean(counts < float(small_cluster_threshold)))

    weight_entropy_count_norm = float(normalized_entropy(probabilities=weight_count))
    weight_entropy_score_norm = float(normalized_entropy(probabilities=weight_score))

    return {
        "timing_std_weighted_mean": timing_std_weighted_mean,
        "timing_std_p90_clusters": timing_std_p90_clusters,
        "distance_per_step_mean": distance_per_step_mean,
        "distance_per_step_p90": distance_per_step_p90,
        "min_cluster_size": min_cluster_size,
        "small_cluster_fraction": small_cluster_fraction,
        "weight_entropy_count_norm": weight_entropy_count_norm,
        "weight_entropy_score_norm": weight_entropy_score_norm,
    }


# --------------------------------------------------------------------------------------
# Core runner
# --------------------------------------------------------------------------------------
def run_one_config(
    *,
    logger: logging.Logger,
    config_name: str,
    metric: str,
    stride_hours: int,
    window_hours: int,
    clusters: int,
    dtw_pam_iters: int,
    dtw_max_windows: int,
    embedding_memmap: np.memmap,
    embedding_dim_used: int,
    time_index: pd.DatetimeIndex,
    time_step_hours: float,
    out_dir: Path,
    small_cluster_threshold: int,
) -> dict:
    overall_start = time.perf_counter()

    logger.info(
        "[%s] Start | metric=%s | window_hours=%d | stride_hours=%d | clusters=%d",
        config_name,
        metric,
        int(window_hours),
        int(stride_hours),
        int(clusters),
    )

    # ---- Build windows ----
    window_build_start = time.perf_counter()
    windows = build_windows(
        times=time_index,
        window_hours=int(window_hours),
        stride_hours=int(stride_hours),
    )
    window_build_end = time.perf_counter()

    if not windows:
        raise ValueError("No windows produced. Check window/stride vs dataset length.")

    num_windows = int(len(windows))
    window_length_steps = int(windows[0][1] - windows[0][0] + 1)

    logger.info(
        "[%s] Windows built: N=%d | window_len_steps=%d (%.2f days) | build_time=%.3fs",
        config_name,
        int(num_windows),
        int(window_length_steps),
        (float(window_length_steps) * float(time_step_hours)) / 24.0,
        float(window_build_end - window_build_start),
    )

    if metric == "dtw" and num_windows > int(dtw_max_windows):
        raise ValueError(
            f"[{config_name}] DTW refused: num_windows={num_windows} > dtw_max_windows={int(dtw_max_windows)}"
        )

    # ---- Build sequences (windowed embeddings) ----
    seq_build_start = time.perf_counter()
    embedding_view = embedding_memmap[:, : int(embedding_dim_used)]
    windowed_sequences = build_windowed_sequences(
        embedding_array=embedding_view,
        windows=windows,
    )
    pc1_peak_offsets_hours = compute_pc1_peak_offset_hours_per_window(
        windowed_sequences=windowed_sequences,
        time_step_hours=float(time_step_hours),
    )
    seq_build_end = time.perf_counter()

    logger.info(
        "[%s] Sequences built | time=%.3fs",
        config_name,
        float(seq_build_end - seq_build_start),
    )

    # ---- Clustering ----
    clustering_start = time.perf_counter()

    centers_flat: np.ndarray | None = None
    representative_window_index_by_cluster: Dict[int, int]

    if metric == "euclid":
        labels, centers_flat, window_score = cluster_sequences_euclid(
            seq_list=windowed_sequences,
            k=int(clusters),
        )
        labels_per_window = np.asarray(labels, dtype=int)
        window_score = np.asarray(window_score, dtype=float)

        representative_window_index_by_cluster = medoid_indices_from_centroids(
            seq_list=windowed_sequences,
            labels=labels_per_window,
            centers=np.asarray(centers_flat, dtype=float),
        )

    elif metric == "dtw":
        labels, representative_window_index_by_cluster, window_score = (
            cluster_sequences_dtw(
                seq_list=windowed_sequences,
                k=int(clusters),
                pam_iters=int(dtw_pam_iters),
            )
        )
        labels_per_window = np.asarray(labels, dtype=int)
        window_score = np.asarray(window_score, dtype=float)

    else:
        raise ValueError("metric must be 'euclid' or 'dtw'")

    clustering_end = time.perf_counter()

    logger.info(
        "[%s] Clustering done | time=%.3fs",
        config_name,
        float(clustering_end - clustering_start),
    )

    # ---- Comparable representativeness metric (distance to representative) ----
    metrics_start = time.perf_counter()
    distance_per_step = compute_distance_per_step_to_representative(
        metric=str(metric),
        windowed_sequences=windowed_sequences,
        labels_per_window=labels_per_window,
        representative_window_index_by_cluster=representative_window_index_by_cluster,
    )

    cluster_table = summarize_clusters(
        labels_per_window=labels_per_window,
        window_score=window_score,
        distance_per_step=distance_per_step,
        windows=windows,
        time_index=time_index,
        pc1_peak_offsets_hours=pc1_peak_offsets_hours,
        representative_window_index_by_cluster=representative_window_index_by_cluster,
    )

    clusters_csv_path = out_dir / f"{config_name}_clusters.csv"
    cluster_table.to_csv(clusters_csv_path, index=False)

    global_metrics = compute_global_metrics_from_cluster_table(
        cluster_table=cluster_table,
        small_cluster_threshold=int(small_cluster_threshold),
    )

    metrics_end = time.perf_counter()

    overall_end = time.perf_counter()

    logger.info(
        "[%s] Metrics done | time=%.3fs | wrote=%s",
        config_name,
        float(metrics_end - metrics_start),
        str(clusters_csv_path),
    )

    return {
        "config": str(config_name),
        "metric": str(metric),
        "window_hours": int(window_hours),
        "stride_hours": int(stride_hours),
        "clusters": int(clusters),
        "num_windows": int(num_windows),
        "window_length_steps": int(window_length_steps),
        "embedding_dim_used": int(embedding_dim_used),
        "time_window_build_sec": float(window_build_end - window_build_start),
        "time_sequence_build_sec": float(seq_build_end - seq_build_start),
        "time_clustering_sec": float(clustering_end - clustering_start),
        "time_metrics_sec": float(metrics_end - metrics_start),
        "time_total_sec": float(overall_end - overall_start),
        "clusters_csv": str(clusters_csv_path),
        **global_metrics,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stride/metric trade-offs for routing-oriented scenario reduction."
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to GRIB/NetCDF dataset."
    )
    parser.add_argument(
        "--vars",
        type=str,
        default="u10,v10,mwd,mwp,swh",
        help="Comma-separated variables.",
    )

    parser.add_argument("--window-hours", type=int, default=170)
    parser.add_argument("--clusters", type=int, default=6)

    parser.add_argument(
        "--run",
        choices=["A", "B", "both"],
        default="both",
        help="Which configuration(s) to run. Use A to skip config_B (e.g. avoid DTW).",
    )

    # Compare two configs A and B
    parser.add_argument(
        "--config-a-metric", choices=["euclid", "dtw"], default="euclid"
    )
    parser.add_argument("--config-a-stride-hours", type=int, default=24)

    parser.add_argument("--config-b-metric", choices=["euclid", "dtw"], default="dtw")
    parser.add_argument("--config-b-stride-hours", type=int, default=48)

    # PCA / embedding
    parser.add_argument("--max-components", type=int, default=80)
    parser.add_argument("--time-batch-size", type=int, default=300)
    parser.add_argument("--fit-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # DTW guard
    parser.add_argument("--dtw-pam-iters", type=int, default=8)
    parser.add_argument("--dtw-max-windows", type=int, default=1200)

    # Diagnostics / logging
    parser.add_argument("--small-cluster-threshold", type=int, default=10)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Verbosity level.",
    )

    parser.add_argument("--out-dir", type=str, default="stride_metric_tradeoff_out")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("tradeoff_benchmark")

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    variable_names = parse_comma_list(text=str(args.vars))

    logger.info("Loading dataset: %s", str(args.input))
    dataset_original = open_normalize(path=Path(args.input))
    ensure_variables_present(dataset=dataset_original, variable_names=variable_names)

    dataset_selected = dataset_original[variable_names].sortby(
        ["latitude", "longitude", "time"]
    )

    logger.info("Standardizing over time (float32, NaNs->0)...")
    dataset_standardized = standardize_over_time(ds=dataset_selected)

    time_index = pd.DatetimeIndex(dataset_standardized["time"].values)
    time_step_hours = compute_time_step_hours(time_index=time_index)
    num_timesteps = int(dataset_standardized.sizes["time"])

    logger.info(
        "Dataset ready | T=%d | dt=%.3fh | vars=%d | grid=%dx%d",
        int(num_timesteps),
        float(time_step_hours),
        int(len(dataset_standardized.data_vars)),
        int(dataset_standardized.sizes["latitude"]),
        int(dataset_standardized.sizes["longitude"]),
    )

    # ---- Fit ONE IPCA ----
    logger.info(
        "Fitting IncrementalPCA (streaming) | max_components=%d | batch_time=%d | fit_sample_rate=%.3f",
        int(args.max_components),
        int(args.time_batch_size),
        float(args.fit_sample_rate),
    )
    ipca_start = time.perf_counter()
    ipca = fit_ipca_stream(
        ds_standardized=dataset_standardized,
        variable_names=list(dataset_standardized.data_vars),
        time_batch_size=int(args.time_batch_size),
        requested_components=int(args.max_components),
        fit_sample_rate=float(args.fit_sample_rate),
        seed=int(args.seed),
    )
    ipca_end = time.perf_counter()
    embedding_dim = int(getattr(ipca, "n_components", int(args.max_components)))
    logger.info(
        "IPCA fitted | embedding_dim=%d | time=%.3fs",
        int(embedding_dim),
        float(ipca_end - ipca_start),
    )

    # ---- Transform ONCE -> memmap ----
    memmap_path = (
        output_directory / f"embedding_T{num_timesteps}_D{embedding_dim}.memmap"
    )
    logger.info("Transforming to memmap: %s", str(memmap_path))
    transform_start = time.perf_counter()
    transform_to_memmap(
        ds_standardized=dataset_standardized,
        variable_names=list(dataset_standardized.data_vars),
        time_batch_size=int(args.time_batch_size),
        memmap_path=memmap_path,
        use_pca=True,
        ipca=ipca,
        num_timesteps=int(num_timesteps),
        embedding_dim=int(embedding_dim),
    )
    transform_end = time.perf_counter()
    logger.info("Memmap written | time=%.3fs", float(transform_end - transform_start))

    embedding_memmap = np.memmap(
        memmap_path,
        dtype="float32",
        mode="r",
        shape=(int(num_timesteps), int(embedding_dim)),
    )

    summary_rows: List[dict] = []

    if str(args.run) in ("A", "both"):
        summary_rows.append(
            run_one_config(
                logger=logger,
                config_name="config_A",
                metric=str(args.config_a_metric),
                stride_hours=int(args.config_a_stride_hours),
                window_hours=int(args.window_hours),
                clusters=int(args.clusters),
                dtw_pam_iters=int(args.dtw_pam_iters),
                dtw_max_windows=int(args.dtw_max_windows),
                embedding_memmap=embedding_memmap,
                embedding_dim_used=int(embedding_dim),
                time_index=time_index,
                time_step_hours=float(time_step_hours),
                out_dir=output_directory,
                small_cluster_threshold=int(args.small_cluster_threshold),
            )
        )

    if str(args.run) in ("B", "both"):
        summary_rows.append(
            run_one_config(
                logger=logger,
                config_name="config_B",
                metric=str(args.config_b_metric),
                stride_hours=int(args.config_b_stride_hours),
                window_hours=int(args.window_hours),
                clusters=int(args.clusters),
                dtw_pam_iters=int(args.dtw_pam_iters),
                dtw_max_windows=int(args.dtw_max_windows),
                embedding_memmap=embedding_memmap,
                embedding_dim_used=int(embedding_dim),
                time_index=time_index,
                time_step_hours=float(time_step_hours),
                out_dir=output_directory,
                small_cluster_threshold=int(args.small_cluster_threshold),
            )
        )

    if not summary_rows:
        raise ValueError("Nothing to run. Use --run A, --run B, or --run both.")

    summary_table = pd.DataFrame(summary_rows)
    summary_csv_path = output_directory / "tradeoff_summary.csv"
    summary_table.to_csv(summary_csv_path, index=False)

    logger.info("Summary CSV written: %s", str(summary_csv_path))
    print("\n[OK] Summary ->", summary_csv_path)
    print(summary_table.to_string(index=False))
    print("\nDetails per config:")
    if str(args.run) in ("A", "both"):
        print(" - config_A_clusters.csv")
    if str(args.run) in ("B", "both"):
        print(" - config_B_clusters.csv")


if __name__ == "__main__":
    main()

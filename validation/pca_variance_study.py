from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from meteo_scenario.io import (
    fit_ipca_stream,
    open_normalize,
    standardize_over_time,
)
from meteo_scenario.windows import build_windows


def _parse_comma_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_component_grid(component_grid_text: str) -> list[int]:
    """Parse a comma-separated list of component counts."""
    component_counts: list[int] = []
    for token in _parse_comma_list(component_grid_text):
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --component-grid: '{token}'") from exc
        if value <= 0:
            raise ValueError("All values in --component-grid must be >= 1")
        component_counts.append(value)

    if not component_counts:
        raise ValueError("--component-grid produced an empty list")

    # Ensure unique + sorted for stable plots
    component_counts = sorted(set(component_counts))
    return component_counts


def _ensure_variables_present(
    *, dataset: xr.Dataset, variable_names: list[str]
) -> None:
    missing = [name for name in variable_names if name not in dataset.data_vars]
    if missing:
        raise KeyError(
            "Missing variables in dataset: "
            # + ", ".join(missing)
            + ". Available: "
            + ", ".join(map(str, dataset.data_vars))
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Study PCA component count impact on clustering stability. "
            "Fits ONE streaming IncrementalPCA, transforms ONCE to a memmap, then evaluates "
            "Euclidean vs DTW clustering stability across a component grid (ARI metric)."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="merged.nc",
        help="Path to merged GRIB/NetCDF dataset.",
    )

    parser.add_argument(
        "--vars",
        type=str,
        default="u10,v10,mwd,mwp,swh",
        help="Comma-separated variables to include.",
    )

    parser.add_argument(
        "--component-grid",
        type=str,
        default="5,20,40,80,120,150,200",
        help="Comma-separated list of PCA component counts to evaluate.",
    )

    parser.add_argument(
        "--max-components",
        type=int,
        default=200,
        help=(
            "Maximum number of PCA components to fit/transform. "
            "Must be >= max(--component-grid)."
        ),
    )

    parser.add_argument(
        "--time-batch-size",
        type=int,
        default=300,
        help="Time batch size used for streaming flatten/PCA fit/transform.",
    )

    parser.add_argument(
        "--fit-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of timesteps used for IPCA fit (0<r<=1).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for IPCA fitting subsampling.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="pca_variance_study_out",
        help="Output directory for memmap and plots.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    variable_names = _parse_comma_list(args.vars)
    component_grid = _parse_component_grid(args.component_grid)

    if int(args.max_components) < max(component_grid):
        raise ValueError(
            f"--max-components={int(args.max_components)} must be >= max(--component-grid)={max(component_grid)}"
        )

    # ------------------------------------------------------------------
    # Load + standardize (same logic as reduce.py, via shared function)
    # ------------------------------------------------------------------
    dataset_original = open_normalize(path=input_path)
    _ensure_variables_present(dataset=dataset_original, variable_names=variable_names)

    dataset_selected = dataset_original[variable_names].sortby(
        ["latitude", "longitude", "time"]
    )
    dataset_standardized = standardize_over_time(ds=dataset_selected)

    print("[INFO] Dataset loaded and standardized")
    print(dataset_standardized)

    # ------------------------------------------------------------------
    # Fit ONE streaming IPCA with max components
    # ------------------------------------------------------------------
    print(
        f"[INFO] Fitting IncrementalPCA with max_components={int(args.max_components)} (streaming)..."
    )

    ipca = fit_ipca_stream(
        ds_standardized=dataset_standardized,
        variable_names=list(dataset_standardized.data_vars),
        time_batch_size=int(args.time_batch_size),
        requested_components=int(args.max_components),
        fit_sample_rate=float(args.fit_sample_rate),
        seed=int(args.seed),
    )

    explained_variance_ratio = np.asarray(ipca.explained_variance_ratio_, dtype=float)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # ------------------------------------------------------------------
    # Evaluate clustering stability across component counts
    # ------------------------------------------------------------------
    cumulative_variance_at_components: list[float] = []

    for num_components in component_grid:
        print(f"\n[INFO] Evaluating n_components={int(num_components)}")

        cumulative_variance_at_components.append(
            float(cumulative_explained_variance[int(num_components) - 1])
        )

    # ------------------------------------------------------------------
    # Save study results (CSV + plots)
    # ------------------------------------------------------------------
    results_table = pd.DataFrame(
        {
            "n_components": component_grid,
            "cumulative_explained_variance": cumulative_variance_at_components,
        }
    )

    results_csv_path = output_directory / "pca_variance_study_results.csv"
    results_table.to_csv(results_csv_path, index=False)
    print(f"\n[OK] Results CSV → {results_csv_path}")

    # Plot: variance explained
    variance_plot_path = output_directory / "pca_variance_explained.png"
    plt.figure()
    plt.plot(component_grid, cumulative_variance_at_components, marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True)
    plt.savefig(variance_plot_path, dpi=200)
    print(f"[OK] Plot → {variance_plot_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()

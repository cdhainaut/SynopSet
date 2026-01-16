from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .io import export_grib_from_ds


def majority_vote_labels(
    *,
    windows: List[Tuple[int, int]],
    window_labels: np.ndarray,
    num_timesteps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-window labels into per-timestep labels by majority vote.

    Parameters
    ----------
    windows:
        Window list as ``(start_index, end_index_inclusive)``.
    window_labels:
        Array of shape ``(num_windows,)``.
    num_timesteps:
        Length of the time axis.

    Returns
    -------
    labels_per_time, vote_fraction:
        Arrays of shape ``(num_timesteps,)``.
        ``vote_fraction[t]`` is in ``[0, 1]`` and measures how dominant the winning
        label was among windows covering time step ``t``.
    """

    per_time_votes: List[List[int]] = [[] for _ in range(int(num_timesteps))]
    for window_index, (start_index, end_index_inclusive) in enumerate(windows):
        cluster_id = int(window_labels[window_index])
        for time_index in range(int(start_index), int(end_index_inclusive) + 1):
            per_time_votes[time_index].append(cluster_id)

    labels_per_time = np.zeros(int(num_timesteps), dtype=int)
    vote_fraction = np.zeros(int(num_timesteps), dtype=float)

    for time_index, votes in enumerate(per_time_votes):
        if not votes:
            labels_per_time[time_index] = -1
            vote_fraction[time_index] = 0.0
            continue

        values, counts = np.unique(np.asarray(votes), return_counts=True)
        winning_index = int(np.argmax(counts))
        labels_per_time[time_index] = int(values[winning_index])
        vote_fraction[time_index] = float(counts[winning_index]) / float(len(votes))

    return labels_per_time, vote_fraction


def attach_time_labels_scores(
    ds: xr.Dataset,
    labels_time: np.ndarray,
    time_score: np.ndarray,
    lat_name="latitude",
    lon_name="longitude",
) -> xr.Dataset:
    T = ds.sizes["time"]
    lab = xr.DataArray(
        np.broadcast_to(
            labels_time[:, None, None], (T, ds.sizes[lat_name], ds.sizes[lon_name])
        ),
        coords={"time": ds["time"], lat_name: ds[lat_name], lon_name: ds[lon_name]},
        dims=("time", lat_name, lon_name),
        name="cluster_id",
        attrs={"long_name": "sequence cluster label (per time-step)", "units": "1"},
    )
    sc = xr.DataArray(
        np.broadcast_to(
            time_score[:, None, None], (T, ds.sizes[lat_name], ds.sizes[lon_name])
        ),
        coords={"time": ds["time"], lat_name: ds[lat_name], lon_name: ds[lon_name]},
        dims=("time", lat_name, lon_name),
        name="cluster_score",
        attrs={"long_name": "majority-vote strength (0..1)", "units": "1"},
    )
    out = ds.copy()
    out["cluster_id"] = lab
    out["cluster_score"] = sc
    return out


def export_medoids(
    *,
    ds_original: xr.Dataset,
    windows: List[Tuple[int, int]],
    medoid_window_index: Dict[int, int],
    window_score: np.ndarray,
    output_directory: Path,
    output_format: str = "grib",
    include_time_in_filename: bool = False,
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
) -> pd.DataFrame:
    """Export one file per cluster medoid as a time slice of the original dataset.

    Parameters
    ----------
    ds_original:
        Original dataset (not standardized) to export.

    windows:
        Window list as ``(start_index, end_index_inclusive)``.

    medoid_window_index:
        Dict ``{cluster_id: window_index}``.

    window_score:
        Per-window representativeness score in ``[0, 1]``.

    output_format:
        - ``"grib"``: export GRIB2 (with NetCDF fallback if CDO is missing).
        - ``"nc"``: export NetCDF.
        - ``"both"``: write both GRIB2 and NetCDF.

    include_time_in_filename:
        If True, filenames include start/end timestamps.

    Returns
    -------
    pandas.DataFrame
        Summary table saved as ``scenarios_sequences_summary.csv``.
    """

    fmt = str(output_format).lower().strip()
    if fmt not in {"grib", "nc", "both"}:
        raise ValueError("output_format must be one of: 'grib', 'nc', 'both'")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    time_index = pd.DatetimeIndex(ds_original["time"].values)

    rows: List[dict] = []
    grib_success_count = 0

    for cluster_id, window_index in sorted(
        medoid_window_index.items(), key=lambda item: item[0]
    ):
        start_index, end_index_inclusive = windows[int(window_index)]

        start_time = pd.Timestamp(time_index[int(start_index)])
        end_time = pd.Timestamp(time_index[int(end_index_inclusive)])

        rows.append(
            {
                "cluster": int(cluster_id),
                "score": float(window_score[int(window_index)]),
                "window_index": int(window_index),
                "start_index": int(start_index),
                "end_index": int(end_index_inclusive),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }
        )

        ds_medoid = ds_original.isel(
            time=slice(int(start_index), int(end_index_inclusive) + 1)
        ).copy()

        # Tag cluster metadata (handy downstream)
        ds_medoid.attrs = dict(ds_medoid.attrs)
        ds_medoid.attrs.update(
            {
                "cluster_id": int(cluster_id),
                "medoid_window_index": int(window_index),
                "medoid_start_index": int(start_index),
                "medoid_end_index": int(end_index_inclusive),
            }
        )

        # Optionally embed cluster_id/score as fields (broadcast over space)
        time_length = int(ds_medoid.sizes["time"])
        cluster_vector = np.full(time_length, int(cluster_id), dtype=int)
        score_vector = np.full(
            time_length, float(window_score[int(window_index)]), dtype=float
        )

        cluster_da = xr.DataArray(
            np.broadcast_to(
                cluster_vector[:, None, None],
                (
                    time_length,
                    ds_medoid.sizes[latitude_dim],
                    ds_medoid.sizes[longitude_dim],
                ),
            ),
            coords={
                "time": ds_medoid["time"],
                latitude_dim: ds_medoid[latitude_dim],
                longitude_dim: ds_medoid[longitude_dim],
            },
            dims=("time", latitude_dim, longitude_dim),
            name="cluster_id",
            attrs={"long_name": "sequence cluster label (window)", "units": "1"},
        )
        score_da = xr.DataArray(
            np.broadcast_to(
                score_vector[:, None, None],
                (
                    time_length,
                    ds_medoid.sizes[latitude_dim],
                    ds_medoid.sizes[longitude_dim],
                ),
            ),
            coords={
                "time": ds_medoid["time"],
                latitude_dim: ds_medoid[latitude_dim],
                longitude_dim: ds_medoid[longitude_dim],
            },
            dims=("time", latitude_dim, longitude_dim),
            name="cluster_score",
            attrs={"long_name": "representativeness score (0..1)", "units": "1"},
        )

        ds_medoid["cluster_id"] = cluster_da
        ds_medoid["cluster_score"] = score_da

        if include_time_in_filename:
            base_name = f"scenario_seq_cluster{int(cluster_id):02d}_{start_time:%Y%m%d%H}_{end_time:%Y%m%d%H}"
        else:
            base_name = f"scenario_seq_cluster{int(cluster_id)}"

        if fmt in {"grib", "both"}:
            out_grib = output_directory / f"{base_name}.grib2"
            if export_grib_from_ds(ds=ds_medoid, out_grib=out_grib):
                grib_success_count += 1

        if fmt in {"nc", "both"}:
            out_nc = output_directory / f"{base_name}.nc"
            ds_medoid.to_netcdf(out_nc)
            print(f"[OK] NetCDF → {out_nc}")

    df = pd.DataFrame(rows)
    df.to_csv(output_directory / "scenarios_sequences_summary.csv", index=False)
    if fmt in {"grib", "both"}:
        print(
            f"[OK] Windows GRIB2 (with score): {grib_success_count}/{len(medoid_window_index)}"
        )
    return df

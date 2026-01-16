from __future__ import annotations

from typing import Iterator, List, Tuple
from pathlib import Path
import shutil, tempfile, os

import numpy as np
import xarray as xr

from sklearn.decomposition import IncrementalPCA


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def open_normalize(path: str | Path) -> xr.Dataset:
    """Open a GRIB/NetCDF with xarray, harmonize dims to (time, latitude, longitude)."""
    ds = xr.open_dataset(path, chunks={"time": 200, "latitude": 200, "longitude": 200})
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


def standardize_over_time(*, ds: xr.Dataset) -> xr.Dataset:
    """Standardize each variable over the time axis.

    This converts each variable into an anomaly-like representation:

    .. math::
        z(t, y, x) = \frac{v(t, y, x) - \mu(y, x)}{\sigma(y, x)}

    where :math:`\mu` and :math:`\sigma` are computed over the ``time`` dimension.

    Notes
    -----
    - Any zero standard deviation is replaced by 1.0 to avoid division by zero.
    - NaNs are filled with 0.0 after standardization.
    - Output is ``float32``.
    """

    mean_over_time = ds.mean(dim="time", skipna=True)
    std_over_time = ds.std(dim="time", skipna=True)
    std_over_time = xr.where(std_over_time == 0, 1.0, std_over_time)

    return ((ds - mean_over_time) / std_over_time).astype("float32").fillna(0.0)


# -----------------------------
# Streaming flatten (time, features)
# -----------------------------
def iter_time_batches_flat(
    *,
    ds_standardized: xr.Dataset,
    variable_names: List[str],
    time_batch_size: int,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Iterate over the time axis and yield flattened feature batches.

    Parameters
    ----------
    ds_standardized:
        Dataset containing standardized variables with dims (time, latitude, longitude).

    variable_names:
        Variables to include. The flattening order is:
        ``[var0 pixels..., var1 pixels..., ...]``.

    time_batch_size:
        Number of consecutive time steps per yielded batch.

    Yields
    ------
    (start_time_index, end_time_index_exclusive, batch_matrix)
        ``batch_matrix`` has shape ``(batch_len, num_features)`` and dtype float32.

    Notes
    -----
    - Memory stays bounded: only one time batch is held in RAM.
    - ``ds_standardized`` should already be float32 for best performance.
    """

    if time_batch_size <= 0:
        raise ValueError("time_batch_size must be >= 1")

    num_timesteps = int(ds_standardized.sizes["time"])
    num_lat = int(ds_standardized.sizes["latitude"])
    num_lon = int(ds_standardized.sizes["longitude"])
    num_pixels = num_lat * num_lon

    num_variables = len(variable_names)
    num_features = num_variables * num_pixels

    for start_time_index in range(0, num_timesteps, int(time_batch_size)):
        end_time_index_exclusive = min(
            num_timesteps, start_time_index + int(time_batch_size)
        )
        batch_len = end_time_index_exclusive - start_time_index

        batch_matrix = np.empty((batch_len, num_features), dtype=np.float32)

        for variable_index, variable_name in enumerate(variable_names):
            variable_block = (
                ds_standardized[variable_name]
                .isel(time=slice(start_time_index, end_time_index_exclusive))
                .transpose("time", "latitude", "longitude")
                .to_numpy()
            )
            if variable_block.dtype != np.float32:
                variable_block = variable_block.astype(np.float32, copy=False)

            feature_start = variable_index * num_pixels
            feature_end = (variable_index + 1) * num_pixels
            batch_matrix[:, feature_start:feature_end] = variable_block.reshape(
                batch_len, num_pixels
            )

        yield start_time_index, end_time_index_exclusive, batch_matrix


# -----------------------------
# Incremental PCA (robust streaming)
# -----------------------------
def fit_ipca_stream(
    *,
    ds_standardized: xr.Dataset,
    variable_names: List[str],
    time_batch_size: int,
    requested_components: int,
    fit_sample_rate: float,
    seed: int,
) -> IncrementalPCA:
    """Fit an IncrementalPCA model in streaming mode.

    Why this exists
    ---------------
    Flattening (time, lat, lon, vars) can be huge. IncrementalPCA lets us fit a
    PCA basis using batches to keep memory bounded.

    Important sklearn constraint
    ----------------------------
    Each ``partial_fit`` call must see at least ``n_components`` samples.
    We therefore buffer batches until we have enough rows.

    Parameters
    ----------
    requested_components:
        Target number of PCA components. The actual number is clipped to
        ``min(requested_components, num_features)``.

    fit_sample_rate:
        If < 1.0, randomly subsample time steps during fitting (useful for speed).

    Returns
    -------
    IncrementalPCA
        Fitted model.
    """

    if requested_components <= 0:
        raise ValueError("requested_components must be >= 1")

    num_lat = int(ds_standardized.sizes["latitude"])
    num_lon = int(ds_standardized.sizes["longitude"])
    num_features = len(variable_names) * num_lat * num_lon

    num_components = min(int(requested_components), int(num_features))
    ipca = IncrementalPCA(n_components=num_components)

    sample_rate = float(fit_sample_rate)
    if not (0.0 < sample_rate <= 1.0):
        sample_rate = 1.0

    rng = np.random.default_rng(int(seed))

    buffered_batches: List[np.ndarray] = []
    buffered_rows = 0
    total_selected_rows = 0

    for _, _, batch_matrix in iter_time_batches_flat(
        ds_standardized=ds_standardized,
        variable_names=variable_names,
        time_batch_size=int(time_batch_size),
    ):
        if sample_rate < 1.0:
            selection_mask = rng.random(batch_matrix.shape[0]) < sample_rate
            if not selection_mask.any():
                continue
            batch_matrix = batch_matrix[selection_mask]

        if batch_matrix.shape[0] == 0:
            continue

        buffered_batches.append(batch_matrix)
        buffered_rows += int(batch_matrix.shape[0])
        total_selected_rows += int(batch_matrix.shape[0])

        if buffered_rows >= num_components:
            stacked = np.concatenate(buffered_batches, axis=0)
            ipca.partial_fit(stacked)
            buffered_batches.clear()
            buffered_rows = 0

    if buffered_batches:
        stacked = np.concatenate(buffered_batches, axis=0)
        if stacked.shape[0] >= num_components:
            ipca.partial_fit(stacked)

    if ipca.n_samples_seen_ is None or int(ipca.n_samples_seen_) == 0:
        raise ValueError(
            "IPCA fit got 0 samples. Check fit_sample_rate, dataset length, and filters."
        )
    if total_selected_rows < num_components:
        raise ValueError(
            f"Not enough samples to fit IPCA: got {total_selected_rows} < components {num_components}. "
            "Increase time range, fit_sample_rate, or reduce requested_components."
        )

    return ipca


def transform_to_memmap(
    *,
    ds_standardized: xr.Dataset,
    variable_names: List[str],
    time_batch_size: int,
    memmap_path: Path,
    use_pca: bool,
    ipca: IncrementalPCA | None,
    num_timesteps: int,
    embedding_dim: int,
) -> None:
    """Write per-timestep embeddings to a memory-mapped file.

    Parameters
    ----------
    memmap_path:
        Output path for the memmap file.
    use_pca:
        If True, apply ``ipca.transform`` to each batch.
    ipca:
        Fitted IncrementalPCA (required when use_pca=True).
    embedding_dim:
        Number of columns in the output memmap (PCA components or raw features).
    """

    memmap_path = Path(memmap_path)
    memmap_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_memmap = np.memmap(
        memmap_path,
        dtype="float32",
        mode="w+",
        shape=(int(num_timesteps), int(embedding_dim)),
    )

    for (
        start_time_index,
        end_time_index_exclusive,
        batch_matrix,
    ) in iter_time_batches_flat(
        ds_standardized=ds_standardized,
        variable_names=variable_names,
        time_batch_size=int(time_batch_size),
    ):
        if bool(use_pca):
            if ipca is None:
                raise ValueError("use_pca=True but ipca is None")
            transformed = ipca.transform(batch_matrix)
        else:
            transformed = batch_matrix

        embedding_memmap[start_time_index:end_time_index_exclusive, :] = (
            transformed.astype(
                np.float32,
                copy=False,
            )
        )

    # Flush to disk.
    del embedding_memmap

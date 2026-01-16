from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


def build_windows(
    times: pd.DatetimeIndex, window_hours: int, stride_hours: int
) -> List[Tuple[int, int]]:
    """Build sliding windows over a regular time index.

    Notes
    -----
    - Windows are returned as ``(start_index, end_index)`` with **inclusive** end.
    - The function assumes the time step is (approximately) constant and infers it
      from the first two timestamps.
    """

    if len(times) < 2:
        return []

    time_step_hours = float((times[1] - times[0]).total_seconds()) / 3600.0
    if time_step_hours <= 0:
        raise ValueError("Invalid time axis (non-increasing timestamps).")

    window_length_steps = max(1, int(round(float(window_hours) / time_step_hours)))
    stride_length_steps = max(1, int(round(float(stride_hours) / time_step_hours)))

    windows: List[Tuple[int, int]] = []
    for start_index in range(
        0, len(times) - window_length_steps + 1, stride_length_steps
    ):
        end_index_inclusive = start_index + window_length_steps - 1
        windows.append((start_index, end_index_inclusive))

    return windows


def iter_window_vectors(
    *,
    embedding_memmap: np.memmap,
    windows: List[Tuple[int, int]],
    window_batch_size: int,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Iterate over windowed time-series as flat feature vectors.

    Parameters
    ----------
    embedding_memmap:
        Memory-mapped array of shape ``(num_timesteps, embedding_dim)``.
        Each row is the embedding for one time step (e.g., PCA components).

    windows:
        List of ``(start_index, end_index_inclusive)`` windows on the *time* axis.
        All windows are expected to have the same length.

    window_batch_size:
        Number of windows to materialize per batch.

    Yields
    ------
    (start_window_index, end_window_index_exclusive, window_matrix)
        ``window_matrix`` has shape ``(batch_windows, window_length * embedding_dim)``.

    Notes
    -----
    This function keeps RAM usage bounded by ``window_batch_size``.
    """

    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be >= 1")
    if not windows:
        return

    num_windows = len(windows)
    embedding_dim = int(embedding_memmap.shape[1])
    window_length = int(windows[0][1] - windows[0][0] + 1)
    window_vector_size = window_length * embedding_dim

    for start_window_index in range(0, num_windows, window_batch_size):
        end_window_index_exclusive = min(
            num_windows, start_window_index + window_batch_size
        )
        batch_window_count = end_window_index_exclusive - start_window_index

        window_matrix = np.empty(
            (batch_window_count, window_vector_size), dtype=np.float32
        )

        for local_index in range(batch_window_count):
            window_start_index, window_end_index_inclusive = windows[
                start_window_index + local_index
            ]
            window_slice = embedding_memmap[
                window_start_index : window_end_index_inclusive + 1, :
            ]
            window_matrix[local_index, :] = window_slice.reshape(-1)

        yield start_window_index, end_window_index_exclusive, window_matrix

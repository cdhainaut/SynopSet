from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .windows import iter_window_vectors


# ---------- DTW multivariée (euclid per step) ----------
def dtw_distance(seqA: np.ndarray, seqB: np.ndarray) -> float:
    """Compute a multivariate DTW distance between two sequences.

    Parameters
    ----------
    seqA, seqB:
        Arrays of shape ``(time, embedding_dim)``.

    Returns
    -------
    float
        The DTW alignment cost using an L2 cost per step.

    Notes
    -----
    - Complexity is ``O(L_A * L_B)`` in time and memory.
    - This implementation is intentionally simple/transparent.
    """
    La, Lb = seqA.shape[0], seqB.shape[0]
    D = np.full((La + 1, Lb + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, La + 1):
        ai = seqA[i - 1]
        for j in range(1, Lb + 1):
            bj = seqB[j - 1]
            cost = float(np.linalg.norm(ai - bj))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[La, Lb])


def pairwise_dtw(seq_list: List[np.ndarray]) -> np.ndarray:
    N = len(seq_list)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = dtw_distance(seq_list[i], seq_list[j])
    return D


# ---------- Medoids: init farthest-first + PAM swaps ----------
def medoids_from_D(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    centers = [int(np.argmin(D.sum(axis=1)))]
    while len(centers) < k:
        dmin = np.min(D[:, centers], axis=1)
        nxt = int(np.argmax(dmin))
        if nxt in centers:
            break
        centers.append(nxt)
    labels = np.argmin(D[:, centers], axis=1).astype(int)
    return np.array(centers), labels


def pam_refine(
    D: np.ndarray, centers_in: np.ndarray, iters: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    centers = list(int(c) for c in centers_in)

    def assign_labels(D, centers):
        return np.argmin(D[:, centers], axis=1).astype(int)

    def total_cost(D, centers, labels):
        c = np.array(centers)
        return float(D[np.arange(D.shape[0]), c[labels]].sum())

    labels = assign_labels(D, centers)
    best_cost = total_cost(D, centers, labels)
    k = len(centers)
    N = D.shape[0]
    for _ in range(iters):
        improved = False
        for ci in range(k):
            for cand in range(N):
                if cand in centers:
                    continue
                new_centers = centers.copy()
                new_centers[ci] = cand
                new_labels = assign_labels(D, new_centers)
                new_cost = total_cost(D, new_centers, new_labels)
                if new_cost + 1e-9 < best_cost:
                    centers, labels, best_cost = new_centers, new_labels, new_cost
                    improved = True
        if not improved:
            break
    return np.array(centers), labels


# ---------- Euclidean (MiniBatchKMeans on flattened sequences) ----------
def cluster_sequences_euclid(
    seq_list: List[np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy (non-streaming) Euclidean clustering for small problems.

    This is kept mainly for debugging and comparisons.

    Returns
    -------
    labels:
        Array of shape ``(num_windows,)``.
    centers:
        KMeans centroids in the flattened window space.
    window_score:
        Per-window representativeness score in ``[0, 1]`` (1 = closest to its centroid).
    """
    seq_flat = np.array([seq.ravel() for seq in seq_list])  # (N, L*d)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=512)
    labels = km.fit_predict(seq_flat)
    centers = km.cluster_centers_
    d_to_center = np.linalg.norm(seq_flat - centers[labels], axis=1)

    window_score = np.zeros(len(seq_flat), dtype=float)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        d = d_to_center[idx]
        d_min, d_max = float(d.min()), float(d.max())
        if d_max > d_min:
            sc = 1.0 - (d - d_min) / (d_max - d_min)
        else:
            sc = np.ones_like(d)
        window_score[idx] = sc
    return labels, centers, window_score


def medoid_indices_from_centroids(
    seq_list: List[np.ndarray], labels: np.ndarray, centers: np.ndarray
) -> Dict[int, int]:
    """Return medoid index per cluster as closest window to centroid in flatten space."""
    seq_flat = np.array([s.ravel() for s in seq_list])
    med = {}
    k = centers.shape[0]
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        d = np.linalg.norm(seq_flat[idx] - centers[c][None, :], axis=1)
        med[c] = int(idx[np.argmin(d)])
    return med


# ---------- DTW + PAM ----------
def cluster_sequences_dtw(
    seq_list: List[np.ndarray], k: int, pam_iters: int = 8
) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    """
    Returns labels, medoid_win_idx (dict cluster->index), window_score (0..1).
    """
    D = pairwise_dtw(seq_list)
    centers0, labels = medoids_from_D(D, k)
    centersA, labels = pam_refine(D, centers0, iters=pam_iters)
    medoid_win_idx = {c: int(centersA[c]) for c in range(k)}

    # score = 1 - normalized distance to medoid (per cluster)
    N = len(seq_list)
    window_score = np.zeros(N, dtype=float)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        ref = medoid_win_idx[c]
        d = D[idx, ref]
        d_min, d_max = float(d.min()), float(d.max())
        if d_max > d_min:
            sc = 1.0 - (d - d_min) / (d_max - d_min)
        else:
            sc = np.ones_like(d)
        window_score[idx] = sc
    return labels, medoid_win_idx, window_score


def cluster_windows_euclid_stream(
    *,
    embedding_memmap: np.memmap,
    windows: List[Tuple[int, int]],
    num_clusters: int,
    random_seed: int,
    window_batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """Cluster windows using streaming MiniBatchKMeans on flattened window vectors.

    Parameters
    ----------
    embedding_memmap:
        Memory-mapped array of shape ``(num_timesteps, embedding_dim)``.

    windows:
        List of ``(start_index, end_index_inclusive)`` windows on the time axis.

    num_clusters:
        Number of clusters (k).

    random_seed:
        Seed for MiniBatchKMeans reproducibility.

    window_batch_size:
        Number of windows to materialize per batch.

    Returns
    -------
    labels_per_window:
        Array of shape ``(num_windows,)``.
    cluster_centers:
        KMeans centroids in flattened window space.
    window_score:
        Per-window representativeness score in ``[0, 1]`` within each cluster.
    medoid_window_index:
        Dict ``{cluster_id: window_index}`` where the medoid is the closest window
        to the cluster centroid (in squared Euclidean distance).

    Notes
    -----
    - This is *not* DTW: it compares windows point-by-point.
    - Streaming keeps RAM usage bounded even for many windows.
    """

    if num_clusters <= 0:
        raise ValueError("num_clusters must be >= 1")
    if not windows:
        raise ValueError("No windows to cluster.")
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be >= 1")

    # Infer flatten dimensionality.
    embedding_dim = int(embedding_memmap.shape[1])
    window_length = int(windows[0][1] - windows[0][0] + 1)
    window_vector_size = window_length * embedding_dim

    kmeans = MiniBatchKMeans(
        n_clusters=int(num_clusters),
        random_state=int(random_seed),
        n_init="auto",
        batch_size=min(1024, int(window_batch_size)),
    )

    # ---- Fit (ensure first partial_fit sees enough samples) ----
    fit_batches: List[np.ndarray] = []
    buffered_window_count = 0

    for _, _, window_matrix in iter_window_vectors(
        embedding_memmap=embedding_memmap,
        windows=windows,
        window_batch_size=int(window_batch_size),
    ):
        if window_matrix.shape[1] != window_vector_size:
            raise ValueError("Inconsistent window length detected.")

        fit_batches.append(window_matrix)
        buffered_window_count += int(window_matrix.shape[0])

        if buffered_window_count >= int(num_clusters):
            kmeans.partial_fit(np.concatenate(fit_batches, axis=0))
            fit_batches.clear()
            buffered_window_count = 0

    if fit_batches:
        kmeans.partial_fit(np.concatenate(fit_batches, axis=0))

    cluster_centers = kmeans.cluster_centers_  # (k, window_vector_size)

    # ---- Predict + squared distances (store per-window) ----
    num_windows = len(windows)
    labels_per_window = np.empty(num_windows, dtype=np.int32)
    squared_distance_to_center = np.empty(num_windows, dtype=np.float32)

    squared_distance_min = np.full(int(num_clusters), np.inf, dtype=np.float64)
    squared_distance_max = np.full(int(num_clusters), -np.inf, dtype=np.float64)
    best_squared_distance = np.full(int(num_clusters), np.inf, dtype=np.float64)
    medoid_window_index: Dict[int, int] = {}

    for (
        start_window_index,
        end_window_index_exclusive,
        window_matrix,
    ) in iter_window_vectors(
        embedding_memmap=embedding_memmap,
        windows=windows,
        window_batch_size=int(window_batch_size),
    ):
        predicted_labels = kmeans.predict(window_matrix).astype(np.int32, copy=False)

        assigned_centers = cluster_centers[predicted_labels]
        difference = window_matrix - assigned_centers
        batch_squared_dist = np.einsum("ij,ij->i", difference, difference).astype(
            np.float32, copy=False
        )

        labels_per_window[start_window_index:end_window_index_exclusive] = (
            predicted_labels
        )
        squared_distance_to_center[start_window_index:end_window_index_exclusive] = (
            batch_squared_dist
        )

        for local_index in range(end_window_index_exclusive - start_window_index):
            cluster_id = int(predicted_labels[local_index])
            value = float(batch_squared_dist[local_index])

            if value < squared_distance_min[cluster_id]:
                squared_distance_min[cluster_id] = value
            if value > squared_distance_max[cluster_id]:
                squared_distance_max[cluster_id] = value
            if value < best_squared_distance[cluster_id]:
                best_squared_distance[cluster_id] = value
                medoid_window_index[cluster_id] = int(start_window_index + local_index)

    # ---- Convert distances to representativeness score in [0, 1] per cluster ----
    window_score = np.zeros(num_windows, dtype=np.float32)
    for cluster_id in range(int(num_clusters)):
        indices = np.where(labels_per_window == cluster_id)[0]
        if indices.size == 0:
            continue

        mn = float(squared_distance_min[cluster_id])
        mx = float(squared_distance_max[cluster_id])
        if mx > mn:
            window_score[indices] = 1.0 - (squared_distance_to_center[indices] - mn) / (
                mx - mn
            )
        else:
            window_score[indices] = 1.0

    return labels_per_window, cluster_centers, window_score, medoid_window_index

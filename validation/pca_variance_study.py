import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_rand_score

from meteo_scenario.windows import build_windows
from meteo_scenario.clustering import cluster_sequences_euclid, cluster_sequences_dtw

import xarray as xr
import pandas as pd


# -------------------------------------------------------------------
# Helper: flatten dataset as (time, features) WITHOUT loading into RAM
# -------------------------------------------------------------------
def ds_to_2d_lazy(ds_std, vars_list):
    """
    Returns a Dask-backed 2D array (time, features)
    by stacking the (var, lat, lon) dims.
    """
    arr = ds_std[vars_list].to_array()  # (var, time, lat, lon)
    da2 = arr.stack(features=("variable", "latitude", "longitude")).transpose(
        "time", "features"
    )
    return da2


def fit_incremental_pca(da2, batch_size, n_comp):
    ipca = IncrementalPCA(n_components=n_comp)

    T = da2.sizes["time"]
    buffer = []

    for start in range(0, T, batch_size):
        stop = min(start + batch_size, T)
        Xb = da2[start:stop].compute()

        buffer.append(Xb)

        # empile les batchs dans le buffer
        Xbuf = np.vstack(buffer)

        # Vérifie si on a assez de samples pour partial_fit
        if Xbuf.shape[0] >= n_comp:
            ipca.partial_fit(Xbuf)
            buffer = []  # reset

    return ipca


def transform_incremental(da2, ipca, batch_size):
    T = da2.sizes["time"]
    embed_full = np.zeros((T, ipca.n_components), dtype=np.float32)

    for start in range(0, T, batch_size):
        stop = min(start + batch_size, T)
        Xb = da2[start:stop].compute()
        embed_full[start:stop] = ipca.transform(Xb)

    return embed_full


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    PATH = "merged.nc"

    # ---------------------------------------------------------------
    # Load dataset chunked (avoids RAM blow)
    # ---------------------------------------------------------------
    ds = xr.open_dataset(PATH, chunks={"time": 200, "latitude": 200, "longitude": 200})

    keep = ["u10", "v10", "mwd", "mwp", "swh"]
    ds = ds[keep]

    # Standardize
    mean_t = ds.mean("time")
    std_t = xr.where(ds.std("time") == 0, 1.0, ds.std("time"))

    ds_std = ((ds - mean_t) / std_t).astype("float32").fillna(0.0)

    vars_list = keep
    times = pd.DatetimeIndex(ds_std.time.values)

    # ---------------------------------------------------------------
    # Build windows for clustering
    # ---------------------------------------------------------------
    wins = build_windows(times, 170, 24)
    k = 6  # or 4, 8…

    # ---------------------------------------------------------------
    # Flatten: (time, features) - DASK LAZY OPERATION
    # ---------------------------------------------------------------
    da2 = ds_to_2d_lazy(ds_std, vars_list)
    T = da2.sizes["time"]
    print("Flattened shape:", da2)

    # ---------------------------------------------------------------
    # Fit ONE PCA with max components
    # ---------------------------------------------------------------
    n_components_max = 200
    batch_size = 300  # safe: loads 200 timesteps at a time

    print(f"Fitting IncrementalPCA with {n_components_max} components...")
    ipca = fit_incremental_pca(da2, batch_size, n_components_max)

    # Full explained variance
    explained_full = ipca.explained_variance_ratio_
    cumexp = np.cumsum(explained_full)

    # ---------------------------------------------------------------
    # Transform ONCE to get embed_full(time, n_components_max)
    # ---------------------------------------------------------------
    print("Transforming full dataset through PCA...")
    embed_full = transform_incremental(da2, ipca, batch_size)

    # ---------------------------------------------------------------
    # Component grid for plots
    # ---------------------------------------------------------------
    component_grid = [5, 20, 40, 80, 120, 150, 200]

    ARI_euclid = []
    ARI_dtw = []
    ref_e, ref_d = None, None
    variance_list = []

    for nc in component_grid:
        print(f"\nProcessing n_components = {nc}")
        variance_list.append(float(cumexp[nc - 1]))

        embed_nc = embed_full[:, :nc]

        # Clustering sequences
        seq_list = [embed_nc[s : e + 1] for (s, e) in wins]

        # Euclid
        labs_e, _, _ = cluster_sequences_euclid(seq_list, k)
        labs_e = np.array(labs_e)
        if ref_e is None:
            ref_e = labs_e
            ARI_euclid.append(1.0)
        else:
            ARI_euclid.append(adjusted_rand_score(ref_e, labs_e))

        # DTW
        labs_d, _, _ = cluster_sequences_dtw(seq_list, k)
        labs_d = np.array(labs_d)
        if ref_d is None:
            ref_d = labs_d
            ARI_dtw.append(1.0)
        else:
            ARI_dtw.append(adjusted_rand_score(ref_d, labs_d))

    # ---------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------
    plt.figure()
    plt.plot(component_grid, ARI_euclid, marker="o", label="Euclid")
    plt.plot(component_grid, ARI_dtw, marker="o", label="DTW")
    plt.xlabel("n_components")
    plt.ylabel("ARI")
    plt.title("Cluster stability (Euclid vs DTW)")
    plt.grid(True)
    plt.legend()
    plt.savefig("pca_cluster_stability_euclid_vs_dtw.png", dpi=200)

    plt.figure()
    plt.plot(component_grid, variance_list, marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Variance explained")
    plt.grid(True)
    plt.savefig("pca_variance.png", dpi=200)

    print("\nDone.")


if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from meteo_cluster.clustering import pairwise_dtw, cluster_sequences_euclid, cluster_sequences_dtw
from meteo_cluster.windows import build_windows
import xarray as xr
import pandas as pd


def pairwise_euclid(seq_list):
    flat = [s.ravel() for s in seq_list]
    return squareform(pdist(flat, metric="euclidean"))


def corr_upper(A, B):
    iu = np.triu_indices_from(A, k=1)
    return np.corrcoef(A[iu], B[iu])[0, 1]


def main():
    PATH = "ton_fichier.nc"
    ds = xr.open_dataset(PATH)
    keep = ["u10", "v10"]
    ds = ds[keep]

    mean_t = ds.mean("time")
    std_t = ds.std("time")
    ds_std = ((ds - mean_t) / std_t).astype("float32")

    times = pd.DatetimeIndex(ds_std.time.values)
    embed = ds_std.to_array().transpose("time", "variable", "latitude", "longitude").values
    embed = embed.reshape(len(times), -1)

    stride_grid = [1, 3, 6, 12, 24]
    corr_results = []

    for stride in stride_grid:
        print(f"Stride={stride}h")

        wins = build_windows(times, 72, stride)
        seq_list = [embed[s:e+1] for (s, e) in wins]

        D_e = pairwise_euclid(seq_list)
        D_d = pairwise_dtw(seq_list)

        c = corr_upper(D_e, D_d)
        corr_results.append(c)
        print(f"Corr(Euclid, DTW) = {c:.3f}")

    # Figure
    plt.figure()
    plt.plot(stride_grid, corr_results, marker="o")
    plt.xlabel("Stride (h)")
    plt.ylabel("Corr(Euclid, DTW)")
    plt.title("Convergence Euclidienne → DTW selon stride")
    plt.grid()
    plt.savefig("stride_vs_metric.png")


if __name__ == "__main__":
    main()

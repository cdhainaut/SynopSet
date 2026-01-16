# Validation Studies

This folder contains scripts used for scientific validation, trade-off exploration, and reproducible experiments.
These scripts are **not** part of the main package API. They exist for reproducibility and publication-quality validation.

---

## Scripts

### `pca_variance_study.py`
Studies how the **number of PCA components** impacts the representation.

**What it does**
- Fits **one** streaming `IncrementalPCA` (IPCA) up to `--max-components`
- Transforms the dataset **once** into a `memmap` embedding `(time, max_components)`
- Produces:
  - cumulative variance explained vs number of components

> Note: clustering-stability metrics (e.g. ARI) are not the primary goal here; the core purpose is to understand the PCA compression trade-off for a given dataset sample.

---

### `window_stride_metric_benchmark.py` (recommended)
Benchmarks the routing-oriented trade-off between:
- **window length** (`--window-hours`)
- **stride** (`--config-*-stride-hours`)
- **distance metric** (`--config-*-metric`: `euclid` vs `dtw`)

**Purpose**  
Extract **representative windows (medoids/representatives)** and **cluster weights** for use in routing optimization, while quantifying:
- feasibility / runtime
- representativeness (how well representatives summarize windows)
- timing coherence inside clusters (routing-critical)

---

## Why window / stride / metric matter for routing

In routing, a window is typically interpreted as **starting now (t0) and evolving over the horizon**.

- If a major event occurs at **t0+48h** vs **t0+72h**, routing decisions can change a lot.
  Euclidean distance (time-aligned) naturally distinguishes these cases.

- DTW (Dynamic Time Warping) can align similar shapes even if they are shifted inside the window.
  This can be useful to group “same pattern, shifted timing”, **but** it may also **wash out timing information**
  that is physically important for routing (event timing relative to departure).

So the trade-off is:

- **Euclidean**: respects event timing relative to departure, scalable  
- **DTW**: robust to phase shifts, but expensive and may mix different timings

---

## Computational cost overview

Let:
- `T` = number of timesteps
- `L` = window length in timesteps (from `window_hours`)
- `S` = stride in timesteps (from `stride_hours`)
- `N` ≈ number of windows
- `d` = embedding dimension (PCA components)

Number of windows:\
$$
N \approx 1 + \left\lfloor \frac{T - L}{S} \right\rfloor
$$

### Euclidean clustering (scalable)

Windows are represented as sequences (length `L`, dimension `d`) and clustered in a time-aligned space.
A typical cost scales roughly like:\
$$
\text{Cost} \approx O(N \cdot L \cdot d \cdot I)
$$

where `I` is an iteration factor (algorithm-dependent).

**Main drivers:** `N` (stride), `L` (window length), `d` (PCA dimension).

### DTW + medoids (expensive)

DTW distance between two windows scales roughly as:\
$$
\text{Pairwise DTW cost} \approx O(L^2 \cdot d)
$$

and medoid/PAM-like approaches require many pairwise comparisons:\
$$
\text{Total DTW+PAM cost} \approx O(N^2 \cdot L^2 \cdot d)
$$

**Main driver:** `N^2` — so small stride becomes quickly infeasible for DTW.

This is why the benchmark includes a safety guard: `--dtw-max-windows`.

---

## What window length changes (quality)

- **Too short windows:** capture snapshots but miss evolution (front passage, veer/shift, transitions).
- **Too long windows:** mix multiple regimes; clusters become less “pure” and representatives less meaningful.

A good starting point: **`window_hours` ≈ routing decision horizon**, or slightly larger.

---

## What stride changes (weights and redundancy)

- **Small stride (high overlap):**
  - Pros: better coverage of possible departures (t0 shifts)
  - Cons: many near-duplicate windows (strong temporal correlation)

- **Large stride (low overlap):**
  - Pros: fewer windows → faster; weights can be less correlated
  - Cons: lower temporal resolution (can miss short transitions)

For routing, small stride is often desirable to capture the departure time precisely, **but** DTW may become infeasible.

---

## What the benchmark measures (routing-oriented)

The benchmark is not centered on label stability (ARI). Instead it measures:

1) **Runtime and feasibility**
- number of windows (`num_windows`)
- clustering time (`time_clustering_sec`) and total time (`time_total_sec`)

2) **Representativeness (comparable Euclid vs DTW)**  
Instead of mixing incompatible scales (centroid L2 vs DTW), we report a **distance-to-representative per time step**.

- **Euclid** (time-aligned representative):\
$$
d_{\text{euclid-per-step}}(x,m)=\frac{1}{L}\sum_{t=1}^{L}\left\|x_t-m_t\right\|_2
$$

- **DTW** (warped representative):\
$$
d_{\text{dtw-per-step}}(x,m)=\frac{\mathrm{DTW}(x,m)}{L}
$$

This yields comparable magnitudes across metrics.

3) **Timing coherence (routing-critical proxy)**  
A practical timing proxy is computed by:
- finding the time offset of the maximum magnitude of **PC1** inside each window
- measuring dispersion **inside clusters**

High timing dispersion suggests clusters may mix windows where key events occur at different times relative to departure.

> Note: PC1 peak timing is a proxy. For higher confidence, you can replace it with a physical timing proxy (e.g. peak SWH or peak wind in a corridor).

---

## `window_stride_metric_benchmark.py` arguments

### Required
- `--input` : path to GRIB/NetCDF dataset

### Core modeling
- `--vars` : comma-separated variable list
- `--window-hours` : window duration in hours
- `--clusters` : number of clusters `k`
- `--max-components` : PCA embedding dimension (upper bound for streaming IPCA)

### Two-config comparison
- `--run {A,B,both}` : run only config A, only B, or both
- `--config-a-metric {euclid,dtw}` / `--config-a-stride-hours`
- `--config-b-metric {euclid,dtw}` / `--config-b-stride-hours`

### PCA streaming controls
- `--time-batch-size`
- `--fit-sample-rate`
- `--seed`

### DTW controls / safety guard
- `--dtw-pam-iters`
- `--dtw-max-windows`

### Diagnostics
- `--small-cluster-threshold` : threshold used to count “tiny clusters”
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

### Output
- `--out-dir`

---

## How to run

### PCA variance study
```bash
python pca_variance_study.py \
  --input merged.nc \
  --vars u10,v10,mwd,mwp,swh \
  --max-components 200 \
  --out-dir pca_variance_study_out
```

### Trade-off benchmark (Euclid ONLY, fast)
```bash
python window_stride_metric_benchmark.py \
  --input merged.nc \
  --vars u10,v10,mwd,mwp,swh \
  --window-hours 170 \
  --clusters 20 \
  --run A \
  --config-a-metric euclid --config-a-stride-hours 12 \
  --max-components 100 \
  --log-level INFO \
  --out-dir tradeoff_out_euclid
```

### Trade-off benchmark (Euclid vs DTW)
```bash
python window_stride_metric_benchmark.py \
  --input merged.nc \
  --vars u10,v10,mwd,mwp,swh \
  --window-hours 170 \
  --clusters 12 \
  --run both \
  --config-a-metric euclid --config-a-stride-hours 12 \
  --config-b-metric dtw    --config-b-stride-hours 48 \
  --max-components 100 \
  --dtw-pam-iters 8 \
  --dtw-max-windows 300 \
  --log-level INFO \
  --out-dir tradeoff_out_euclid_vs_dtw
```

---

## Outputs

### `pca_variance_study.py`
- `pca_variance_explained.png`
- `pca_variance_study_results.csv` (variance explained per component count)

### `window_stride_metric_benchmark.py`
- `tradeoff_summary.csv` (one row per config)
- `config_A_clusters.csv` / `config_B_clusters.csv` (cluster-by-cluster diagnostics)
- embedding memmap (for reproducibility): `embedding_T{T}_D{D}.memmap`

---

## Output fields explained

### `tradeoff_summary.csv` fields (one row per config)

**Identity / setup**
- `config` : config name (`config_A`, `config_B`)
- `metric` : clustering metric (`euclid` or `dtw`)
- `window_hours` : window duration (hours)
- `stride_hours` : stride between windows (hours)
- `clusters` : number of clusters (k)
- `num_windows` : number of windows created
- `window_length_steps` : number of timesteps inside each window
- `embedding_dim_used` : number of PCA components used

**Timing / profiling**
- `time_window_build_sec` : time to build windows list
- `time_sequence_build_sec` : time to slice the embedding into window sequences
- `time_clustering_sec` : clustering time (dominant for DTW)
- `time_metrics_sec` : time to compute metrics and write cluster CSV
- `time_total_sec` : total time per config
- `clusters_csv` : path to per-cluster CSV

**Routing-oriented quality metrics**
- `timing_std_weighted_mean` : weighted mean of within-cluster timing dispersion.  
  Lower is better if you want consistent event timing relative to departure.
- `timing_std_p90_clusters` : 90th percentile of timing dispersion across clusters.  
  High value indicates some clusters are “timing-mixed”.

- `distance_per_step_mean` : weighted mean distance-to-representative per time step.  
  Comparable between Euclid and DTW; lower means representatives summarize better.
- `distance_per_step_p90` : 90th percentile of per-cluster mean distance-to-representative.  
  Identifies “poorly represented” clusters.

- `min_cluster_size` : smallest cluster size (windows)
- `small_cluster_fraction` : fraction of clusters smaller than `--small-cluster-threshold`
- `weight_entropy_count_norm` : normalized entropy of weights by count (0..1)
- `weight_entropy_score_norm` : normalized entropy of weights by score (0..1)

### `config_X_clusters.csv` fields (one row per cluster)

**Weights**
- `cluster_id` : cluster id
- `count` : number of windows in cluster (weight basis)
- `sum_window_score` : sum of within-cluster centrality scores
- `weight_frac_count` : `count / sum(count)` (natural weight for uniform departure sampling at stride)
- `weight_frac_score` : `sum_window_score / sum(sum_window_score)` (quality-weighted alternative)

**Timing proxy**
- `pc1_peak_offset_hours_mean` : mean timing of max |PC1| inside windows
- `pc1_peak_offset_hours_std` : dispersion of this timing inside the cluster

**Representativeness (comparable Euclid/DTW)**
- `distance_per_step_mean` : mean distance-to-representative per timestep inside the cluster
- `distance_per_step_p50` : median per-window distance-to-representative
- `distance_per_step_p90` : p90 per-window distance-to-representative (tail / outliers)

**Representative window**
- `representative_window_index` : window index chosen as representative.  
  Euclid: closest-to-centroid window; DTW: medoid window.
- `representative_start_time`, `representative_end_time` : representative window time range
- `representative_score` : within-cluster centrality score of the representative (often ~1.0 by construction)

---

## Final recommendation (best validation)

The best validation is routing-based:

1. Run routing on a subset of raw windows → estimate distribution of objective/cost
2. Run routing on representatives (medoids) with cluster weights
3. Compare expectation and key quantiles (p50, p90)

If those match reasonably, the reduced scenario set is fit for optimization.

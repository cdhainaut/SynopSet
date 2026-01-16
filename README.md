# SynopSet
Reducing weather scenario sets for efficient and robust performance optimisation of wind-assisted ships.

## Summary
**SynopSet** is an open-source Python package that builds **reduced, representative sets of meteorological scenarios** from GRIB/NetCDF archives.
It is designed for studies where coupling **weather variability** with **routing / performance simulation / optimisation** becomes computationally expensive.

The core idea is to:
1. harmonise datasets (grid + time),
2. build sliding **time windows**,
3. embed each timestep (optional streaming PCA),
4. cluster windows (Euclidean or DTW-based),
5. export **representative windows** (medoids/representatives) and **cluster weights** for downstream optimisation.

## Scientific context
Optimising wind-assisted ship performance under realistic meteorology often requires running routing/performance models across large ensembles (multi-month to multi-year).
Brute-force evaluation is statistically robust, but quickly becomes prohibitive when embedded in design optimisation, operational routing, or fleet studies.

This repository implements a **synoptic-scale scenario reduction** workflow based on clustering meteorological windowed sequences (e.g. 10 m wind, significant wave height).
It can be used to approximate the full ensemble using a small set of representative windows plus weights.

## Method overview
- Data harmonisation: consistent grid, time axis, variable naming.
- Windowing: generate overlapping or non-overlapping time windows.
- Embedding:
  - raw flattening (baseline, large dimensionality), or
  - streaming `IncrementalPCA` (recommended for large grids / many variables).
- Clustering:
  - **Euclidean** distance (time-aligned, scalable; good when event timing relative to departure matters),
  - **DTW + medoids** (robust to phase shifts; more expensive; can mix different event timings inside a window).
- Outputs:
  - representative windows (exportable as GRIB2 and/or NetCDF),
  - per-window labels and per-time labels (majority vote),
  - cluster weights (count-based and/or quality-weighted).

## Repository structure
```text
.
├─ src/
│  └─ meteo_scenario/
│     ├─ io.py              # I/O (open, normalize, export)
│     ├─ gridtime.py        # Spatial/temporal alignment helpers
│     ├─ windows.py         # Sliding window generation
│     ├─ clustering.py      # Euclidean + DTW/PAM clustering utilities
│     ├─ export.py          # Label mapping, summaries, dataset export
│     ├─ plotting.py        # Maps & diagnostics (optional)
│     └─ cli/
│        ├─ aggregate.py    # meteo-aggregate
│        ├─ reduce.py       # meteo-reduce
│        ├─ plot_map.py     # meteo-plot-map
│        └─ probe.py        # meteo-probe
├─ validation/
│  ├─ README.md             # Validation notes (publication-oriented)
│  ├─ pca_variance_study.py
│  └─ window_stride_metric_benchmark.py
└─ examples/
   ├─ 01_merge_wind_wave.sh
   ├─ 02_reduce_sequences.sh
   └─ 03_probe_plots.sh
```

## Typical workflow

### 1) Merge heterogeneous meteorological datasets
```bash
meteo-aggregate   --in "gribs/wind_2020.grib"   --in "gribs/wave_2020.grib"   --target-grid finer   --time-mode freq --time-freq 12H   --out merged_12H.grib2
```

### 2) Reduce to representative synoptic scenarios
```bash
meteo-reduce merged_12H.grib2   --vars u10,v10   --window-hours 72   --stride-hours 24   --clusters 6   --seq-metric euclid   --out reduced/
```

To use DTW (note: cost increases quickly with the number of windows):
```bash
meteo-reduce merged_12H.grib2   --vars u10,v10   --window-hours 72   --stride-hours 48   --clusters 6   --seq-metric dtw   --out reduced_dtw/
```

### 3) Visualise and validate
```bash
meteo-plot-map reduced/original_with_cluster_id.grib2

meteo-probe reduced/original_with_cluster_id.grib2   --probe-lat 45.0 --probe-lon -20.0   --plot-out probe.png
```

## Validation studies
See `validation/README.md` for:
- PCA variance study (choose `--max-components` rationally),
- window/stride/metric benchmark (Euclid vs DTW trade-off for routing use).

These scripts are **not part of the public package API**; they exist for reproducibility and publication-quality figures/tables.

## Installation
```bash
git clone https://github.com/cdhainaut/synopset.git
cd synopset
pip install -e .
```

## Research scope
This tool supports studies on:
- scenario reduction for performance prediction of wind-assisted ships,
- design–routing coupling under uncertain meteorological forcing,
- fleet/mission-level optimisation,
- uncertainty quantification and robust decision-making for maritime decarbonisation.

## Citation
If you use this package in your research, please cite:

```bibtex
@inproceedings{dhainaut2025meteo,
  title   = {Reducing Weather Scenario Sets for Efficient and Robust Performance Optimisation of Wind-Assisted Ships},
  author  = {Dhainaut, Charles and Sacher, Mathieu},
  booktitle = {7th Innov'Sail Symposium},
  year    = {2026},
  address = {Göteborg, Sweden},
  organization = {ENSTA Bretagne, IRDL}
}
```

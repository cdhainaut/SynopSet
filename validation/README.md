# Validation Studies

This folder contains scripts used for scientific validation and experiments.

- pca_variance_study.py  
  Tests multiple PCA component counts and produces:
  * variance explained curve
  * clustering stability vs components (ARI)

- stride_metric_study.py  
  Compares Euclidean vs DTW distance matrices for multiple strides.
  Produces:
  * correlation(Euclid, DTW) vs stride
  * diagnostic figures

This code is NOT part of the main package and is only used for reproducibility
and publication-quality validation.

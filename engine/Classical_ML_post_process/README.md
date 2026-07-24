# Classical multivariate anomaly detection — PC methods

Detecting anomalies from **multiple time series at once** using
principal-component (PC) methods, following the multivariate anomaly detection
described in the EGADS paper and its companions.

> N. Laptev, S. Amizadeh, I. Flint, *"Generic and Scalable Framework for
> Automated Time-series Anomaly Detection"* (**EGADS**), KDD 2015,
> DOI 10.1145/2783258.2788611.

EGADS detects three anomaly classes (§3): outliers, change points, and
**anomalous time-series** — the multivariate case (§3.3), where a series is
flagged by clustering a population on time-series features and measuring each
member's deviation from the cluster centroids, thresholded by the §4.1 rules.

Two complementary, paper-grounded PC detectors are implemented. Neither is
made-up: each step maps to a cited method, and every threshold is fit on the
training span only.

## Modules

| File | What it does | Grounding |
|---|---|---|
| `egads_features.py` | EGADS **Table-1** per-series features: trend/seasonality strength (STL), spectral entropy, dominant periodicity, autocorrelation, Teräsvirta nonlinearity, skewness, kurtosis, Hurst (R/S), Lyapunov. | EGADS Table 1; ref [29] Wang–Smith-Miles–Hyndman |
| `egads_pc_cluster.py` | `PCClusterAnomalyModel`: standardise → **PCA** → **k-means** clustering → **centroid-deviation** metric (intra/inter-cluster) → **K·σ** (three-sigma) and **density (LOF)** thresholds. | EGADS §3.3 + §4.1 |
| `egads_detector.py` | `AnomalousSeriesDetector`: slide windows over a multivariate series (bag-of-windows), build the joint per-channel feature matrix, flag windows, expand to a per-timestamp score; `evaluate` = precision/recall/F1 (EGADS §6.3 metric). | EGADS §3.3 |
| `pca_subspace.py` | `PCASubspaceDetector`: PCA **normal / residual subspace** with **Hotelling T²** (excursion along the coupled modes) and **SPE / Q-statistic** (correlation-structure break); parametric control limits + EGADS §4.1 rules. | Lakhina–Crovella–Diot, SIGCOMM 2004; Jackson–Mudholkar, Technometrics 1979; Hotelling T² |
| `extra_data.py` | Loaders + labels: NOAA/CPC **ONI El Niño/La Niña episodes** on the monthly ENSO grid, and the **daily SST marine-heat-wave (MHW)** record + event catalogue. Reuses `retrieve_data.prepare_compound`. | — |
| `retrieve_data.py` | (pre-existing) leak-free ENSO SST / SOI / PDO monthly loaders + compound co-exceedance labels. | — |

## Which detector for which anomaly

- **Co-exceedance / level anomalies** (several drivers excited together — the
  rare compound ENSO event, marine heat waves): `PCASubspaceDetector` **T²**.
  On the compound label (base rate 0.04) it reaches ROC-AUC **0.96** and
  F1 ≈ **0.44** at the parametric limit; on daily MHWs ROC-AUC **0.87**.
- **Structural / regime anomalies** (a series whose *dynamics* are unusual):
  the EGADS feature-clustering detector. Its Table-1 features are deliberately
  *level-invariant*, so it triages structurally-odd windows but cannot see a
  pure level co-exceedance — the two methods are complementary, which is the
  EGADS thesis that no single model wins every use-case (§6.3).

## Demonstration

`demo_pc_clustering_detection.ipynb` runs both detectors end-to-end:
Part A ENSO compound co-exceedance (subspace T²/SPE), Part B EGADS feature
clustering on the same channels, Part C daily marine heat waves. Run it with the
`quantathon-qrc` kernel (venv:
`QRC_main_stack/QRC_code_stack/.venv`).

```bash
python -c "import retrieve_data, extra_data, pca_subspace, egads_detector"  # smoke import
jupyter nbconvert --to notebook --execute --inplace demo_pc_clustering_detection.ipynb
```

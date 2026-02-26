# Citi Bike Demand Forecasting & A/B Testing Framework
## Living Project Plan — Progress, Decisions & Hurdles

---

## Project Story

Citi Bike's biggest operational problem is **rebalancing** — bikes pile up at some stations and run out at others. This project builds a demand forecasting model, then designs and simulates an A/B test comparing the current rebalancing strategy vs an ML-optimized one. We measure dock availability, ride wait times, and revenue impact.

**Target audience:** Data science portfolio — demonstrates a distinct skill set from causal inference work (Medicaid project).

---

## Skills Showcased

| Area | Methods |
|---|---|
| Time series forecasting | Prophet, ARIMA, XGBoost |
| A/B testing & experiment design | Power analysis, sequential testing, Bayesian A/B |
| Clustering & segmentation | K-means, DBSCAN |
| Anomaly detection | Isolation Forest, Z-score flagging |
| Geospatial ML | Spatial features, distance-based modeling, Folium heatmaps |

---

## Project Phases

### Phase 1 — Setup & Data Acquisition
- [ ] Project scaffold (directories, requirements, .gitignore)
- [ ] Download Citi Bike trip data (publicly available via S3)
- [ ] Initial EDA notebook

### Phase 2 — Station Clustering & Segmentation
- [ ] Feature engineering per station (hourly demand patterns, location)
- [ ] K-means clustering → commuter / tourist / recreational segments
- [ ] DBSCAN for geospatial density clustering
- [ ] Visualize on Folium map

### Phase 3 — Demand Forecasting
- [ ] Build Prophet model (station-level, hourly)
- [ ] Build ARIMA baseline
- [ ] Build XGBoost with lag features + weather + calendar features
- [ ] Model comparison (MAE, RMSE, MAPE per station cluster)

### Phase 4 — Anomaly Detection
- [ ] Flag unusual demand spikes (events, weather, outages)
- [ ] Isolation Forest on residuals
- [ ] Z-score / IQR flagging per station-hour

### Phase 5 — A/B Test Design & Simulation
- [ ] Define treatment: ML-optimized rebalancing schedule vs current heuristic
- [ ] Unit of randomization: station cluster (cluster-level A/B)
- [ ] Power analysis — minimum detectable effect, required sample size
- [ ] Simulate A/B test outcomes (synthetic control data)
- [ ] Frequentist analysis (t-test, Mann-Whitney U)
- [ ] Sequential testing (alpha spending / group sequential)
- [ ] Bayesian A/B test (PyMC, posterior on uplift)
- [ ] Measure: dock availability rate, ride wait time, revenue per station

### Phase 6 — Geospatial Analysis
- [ ] Spatial features: distance to subway, population density, POIs
- [ ] Heatmap of demand by hour/day
- [ ] Rebalancing route optimization sketch

### Phase 7 — Synthesis & Write-up
- [ ] Final notebook tying story together
- [ ] README with results summary
- [ ] Portfolio-ready visualizations

---

## Current Status

**Date started:** 2026-02-25
**Current phase:** Phase 1 — Setup
**Branch:** `feature/project-setup`

---

## Commit Log (Summary)

| Commit | What |
|---|---|
| (pending) | Initial scaffold — project structure, plan.md, requirements |

---

## Hurdles & Solutions

> This section is updated as we hit problems. Every hurdle is documented with root cause + solution so future work doesn't repeat the same debugging.

### H001 — (template)
**Date:**
**Problem:**
**Root cause:**
**Solution:**
**Files affected:**

---

## Key Decisions

| Decision | Rationale |
|---|---|
| Cluster-level A/B randomization | Station-level would have interference (SUTVA violations) — bikes move between stations |
| XGBoost over LSTM | Simpler, faster, more interpretable for portfolio; LSTM overkill without GPU |
| Prophet as baseline | Handles seasonality + holidays out-of-box; good explainability for stakeholders |
| PyMC for Bayesian A/B | Industry standard, well-documented, integrates with Python stack |

---

## Data Sources

- **Citi Bike trip data:** https://citibikenyc.com/system-data (public S3 bucket)
- **Weather:** NOAA / Open-Meteo API (free, no key needed)
- **NYC POI / Subway stations:** NYC Open Data

---

## File Structure

```
demand-forecasting-experiment-design/
├── plan.md                        # This file — living doc
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
├── .gitignore
├── data/
│   ├── raw/                       # Raw Citi Bike trip CSVs
│   └── processed/                 # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_forecasting.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_ab_testing.ipynb
│   └── 06_geospatial.ipynb
├── src/
│   ├── data/
│   │   ├── download.py            # Citi Bike S3 download helpers
│   │   └── preprocess.py          # Cleaning & feature engineering
│   ├── models/
│   │   ├── forecasting.py         # Prophet, ARIMA, XGBoost wrappers
│   │   ├── clustering.py          # K-means, DBSCAN
│   │   └── anomaly.py             # Anomaly detection
│   ├── experiments/
│   │   ├── ab_test.py             # A/B test design & simulation
│   │   ├── power_analysis.py      # Power / sample size calculations
│   │   └── bayesian_ab.py         # Bayesian A/B with PyMC
│   └── visualization/
│       ├── maps.py                # Folium geospatial maps
│       └── plots.py               # Matplotlib / Plotly charts
└── tests/
    └── test_ab_test.py
```


# Citi Bike Demand Forecasting & A/B Testing Framework
### Optimizing Station Rebalancing with ML

---

## Overview

Citi Bike's biggest operational challenge is **rebalancing** — bikes pile up at popular destination stations while origin stations run dry. This project attacks that problem end-to-end:

1. **Forecast demand** per station per hour using Prophet, ARIMA, and XGBoost
2. **Segment stations** by usage pattern (commuter, tourist, recreational) with K-means and DBSCAN
3. **Detect anomalies** — unusual demand spikes from events, weather, or outages
4. **Design and simulate an A/B test** comparing the current rebalancing heuristic vs an ML-optimized schedule
5. **Geospatial analysis** — heatmaps, spatial features, proximity to transit

---

## Skills Demonstrated

- Time series forecasting (Prophet, ARIMA, XGBoost)
- A/B test design: power analysis, sequential testing, Bayesian A/B (PyMC)
- Clustering & segmentation (K-means, DBSCAN)
- Anomaly detection (Isolation Forest)
- Geospatial ML (Folium, spatial features)

---

## Data

Publicly available Citi Bike trip data from the [Citi Bike System Data](https://citibikenyc.com/system-data) page (AWS S3).

---

## Project Structure

See `plan.md` for the full phase-by-phase plan, key decisions, and a log of every hurdle encountered with its solution.

---

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Results (updated as phases complete)

_Coming soon._

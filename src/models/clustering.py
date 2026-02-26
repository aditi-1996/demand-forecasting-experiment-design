"""
Station clustering: K-means and DBSCAN.
Groups stations into usage segments (commuter / tourist / recreational).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def build_station_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-station feature matrix for clustering.

    Features:
    - Normalized demand profile by hour of day (shape, not magnitude)
    - AM peak intensity (7–9am vs daily avg)
    - PM peak intensity (5–7pm vs daily avg)
    - Weekday / weekend demand ratio
    - log total demand
    - lat, lng
    """
    # --- Hourly demand profile (shape only, normalized per station) ---
    profile = (
        hourly.groupby(["station_id", "hour_of_day"])["trip_count"]
        .mean()
        .unstack(fill_value=0)
    )
    # Normalize each station so rows sum to 1 (demand *shape*, not volume)
    row_sums = profile.sum(axis=1).replace(0, 1)
    profile_norm = profile.div(row_sums, axis=0)
    profile_norm.columns = [f"h{c:02d}" for c in profile_norm.columns]

    # --- Peak intensity features ---
    avg_by_hour = hourly.groupby(["station_id", "hour_of_day"])["trip_count"].mean().unstack(fill_value=0)
    daily_avg = avg_by_hour.mean(axis=1).replace(0, 1)
    am_peak = avg_by_hour[[7, 8, 9]].mean(axis=1) / daily_avg
    pm_peak = avg_by_hour[[17, 18, 19]].mean(axis=1) / daily_avg

    # --- Weekday vs weekend ratio ---
    weekday_demand = hourly[hourly["is_weekend"] == 0].groupby("station_id")["trip_count"].mean()
    weekend_demand = hourly[hourly["is_weekend"] == 1].groupby("station_id")["trip_count"].mean()
    wkday_wkend_ratio = (weekday_demand / weekend_demand.replace(0, np.nan)).fillna(1)

    # --- Log total demand ---
    total_demand = hourly.groupby("station_id")["trip_count"].sum()
    log_demand = np.log1p(total_demand)

    # --- Location ---
    coords = hourly.groupby("station_id")[["lat", "lng"]].median()

    # --- Assemble ---
    features = profile_norm.copy()
    features["am_peak"] = am_peak
    features["pm_peak"] = pm_peak
    features["wkday_wkend_ratio"] = wkday_wkend_ratio
    features["log_demand"] = log_demand
    features["lat"] = coords["lat"]
    features["lng"] = coords["lng"]
    features = features.dropna()
    return features


def kmeans_elbow(features: pd.DataFrame, k_range=range(2, 10), random_state=42):
    """Return inertia and silhouette scores for a range of K values."""
    X = StandardScaler().fit_transform(features)
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=min(5000, len(X))))
    return list(k_range), inertias, silhouettes


def fit_kmeans(features: pd.DataFrame, n_clusters: int, random_state=42):
    """Fit K-means and return (labels array, scaler, model)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, scaler, km


def fit_dbscan(features: pd.DataFrame, eps=0.006, min_samples=5):
    """
    DBSCAN on lat/lng only (geospatial density clusters).
    eps ~ 0.006 degrees ≈ 600m at NYC latitude.
    Returns labels (-1 = noise).
    """
    coords = features[["lat", "lng"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)
    return labels


CLUSTER_NAMES = {
    0: "Commuter",
    1: "Tourist",
    2: "Recreational",
    3: "Low-activity",
}

"""
Anomaly detection on station-level hourly demand.
Methods:
  - Z-score flagging (per station, rolling window)
  - IQR flagging (per station)
  - Isolation Forest (system-wide feature matrix)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Z-score flagging
# ---------------------------------------------------------------------------

def zscore_anomalies(hourly: pd.DataFrame, window: int = 168, threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag hours where a station's demand deviates > `threshold` std devs
    from its rolling mean (window = 168h = 1 week by default).

    Returns the input df with added columns:
      rolling_mean, rolling_std, zscore, zscore_anomaly (bool)
    """
    df = hourly.sort_values(['station_id', 'hour']).copy()
    grp = df.groupby('station_id')['trip_count']
    df['rolling_mean'] = grp.transform(lambda x: x.rolling(window, min_periods=24).mean())
    df['rolling_std']  = grp.transform(lambda x: x.rolling(window, min_periods=24).std())
    df['zscore'] = (df['trip_count'] - df['rolling_mean']) / df['rolling_std'].replace(0, np.nan)
    df['zscore_anomaly'] = df['zscore'].abs() > threshold
    return df


# ---------------------------------------------------------------------------
# IQR flagging
# ---------------------------------------------------------------------------

def iqr_anomalies(hourly: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Flag hours where demand falls outside [Q1 - k*IQR, Q3 + k*IQR]
    computed per station × hour-of-day (so 8am Monday is compared to
    all 8am Mondays, not the full series).

    Returns df with added column: iqr_anomaly (bool)
    """
    df = hourly.copy()
    # Group by station + hour_of_day to get expected range per slot
    grp = df.groupby(['station_id', 'hour_of_day'])['trip_count']
    q1  = grp.transform('quantile', 0.25)
    q3  = grp.transform('quantile', 0.75)
    iqr = q3 - q1
    df['iqr_anomaly'] = (
        (df['trip_count'] < q1 - multiplier * iqr) |
        (df['trip_count'] > q3 + multiplier * iqr)
    )
    return df


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def build_if_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix for Isolation Forest.
    One row per station-hour with:
      trip_count, hour_of_day, day_of_week, is_weekend,
      lag_1h, lag_24h, lag_168h, rolling_mean_24h
    """
    df = hourly.sort_values(['station_id', 'hour']).copy()
    grp = df.groupby('station_id')['trip_count']
    df['lag_1h']          = grp.transform(lambda x: x.shift(1))
    df['lag_24h']         = grp.transform(lambda x: x.shift(24))
    df['lag_168h']        = grp.transform(lambda x: x.shift(168))
    df['rolling_mean_24h'] = grp.transform(lambda x: x.shift(1).rolling(24).mean())
    return df.dropna(subset=['lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h'])


IF_FEATURE_COLS = [
    'trip_count', 'hour_of_day', 'day_of_week', 'is_weekend',
    'lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h',
]


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42):
    """
    Fit Isolation Forest on IF_FEATURE_COLS.
    Returns (model, scaler, df_with_scores).
    Adds columns: if_score (anomaly score), if_anomaly (bool).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[IF_FEATURE_COLS])
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    df = df.copy()
    df['if_score']   = model.fit_predict(X)          # -1 = anomaly, 1 = normal
    df['if_anomaly'] = df['if_score'] == -1
    return model, scaler, df

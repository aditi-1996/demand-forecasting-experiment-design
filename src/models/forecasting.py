"""
Time series forecasting wrappers: Prophet, ARIMA, XGBoost.
Each model follows the same interface:
    fit(train_df) -> model
    predict(model, future_df) -> forecast_df
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_station_series(hourly: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Return hourly trip_count series for one station, sorted and gap-filled."""
    s = (
        hourly[hourly["station_id"] == station_id]
        .set_index("hour")[["trip_count", "hour_of_day", "day_of_week", "is_weekend", "month"]]
        .sort_index()
    )
    # Fill any missing hours with 0
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full_idx, fill_value=0)
    s.index.name = "hour"
    # Re-derive time features after reindex
    s["hour_of_day"] = s.index.hour
    s["day_of_week"] = s.index.dayofweek
    s["is_weekend"] = s["day_of_week"].isin([5, 6]).astype(int)
    s["month"] = s.index.month
    return s


def train_test_split(series: pd.DataFrame, cutoff: str):
    """Split on a date string, e.g. '2024-03-01'."""
    cutoff_dt = pd.Timestamp(cutoff)
    train = series[series.index < cutoff_dt]
    test = series[series.index >= cutoff_dt]
    return train, test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE (ignores zeros in true for MAPE)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def fit_prophet(train: pd.DataFrame, daily_seasonality=True, weekly_seasonality=True):
    """Fit Prophet on a station series. train must have DatetimeIndex + trip_count."""
    from prophet import Prophet
    df_prophet = train[["trip_count"]].reset_index().rename(
        columns={"hour": "ds", "index": "ds", "trip_count": "y"}
    )
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        uncertainty_samples=0,    # skip uncertainty for speed
    )
    model.add_country_holidays(country_name="US")
    model.fit(df_prophet)
    return model


def predict_prophet(model, test: pd.DataFrame) -> np.ndarray:
    """Forecast for the hours in test's index. Returns array of yhat."""
    future = pd.DataFrame({"ds": test.index})
    forecast = model.predict(future)
    return forecast["yhat"].clip(lower=0).values


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def fit_arima(train: pd.DataFrame, order=(2, 1, 2), seasonal_order=(1, 0, 1, 24)):
    """Fit SARIMA on trip_count. Uses statsmodels SARIMAX."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        train["trip_count"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return result


def predict_arima(result, steps: int) -> np.ndarray:
    """Forecast `steps` hours ahead from end of training data."""
    forecast = result.forecast(steps=steps)
    return np.clip(forecast.values, 0, None)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]   # 1w lag captures weekly seasonality


def make_lag_features(series: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-mean features to a station series."""
    df = series.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df["trip_count"].shift(lag)
    df["roll_mean_24h"] = df["trip_count"].shift(1).rolling(24).mean()
    df["roll_mean_168h"] = df["trip_count"].shift(1).rolling(168).mean()
    return df.dropna()


FEATURE_COLS = (
    ["hour_of_day", "day_of_week", "is_weekend", "month"]
    + [f"lag_{l}h" for l in LAG_HOURS]
    + ["roll_mean_24h", "roll_mean_168h"]
)


def fit_xgboost(train_feats: pd.DataFrame):
    """Fit XGBoost regressor on pre-computed lag features."""
    import xgboost as xgb
    X = train_feats[FEATURE_COLS]
    y = train_feats["trip_count"]
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_xgboost(model, test_feats: pd.DataFrame) -> np.ndarray:
    """Predict on test lag features."""
    X = test_feats[FEATURE_COLS]
    return np.clip(model.predict(X), 0, None)

"""
Data cleaning and feature engineering for Citi Bike trip data.
Produces station-level hourly demand time series saved to data/processed/.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names across Citi Bike schema versions
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Parse start time
    time_col = "started_at" if "started_at" in df.columns else "starttime"
    df["started_at"] = pd.to_datetime(df[time_col])
    # Drop nulls in key columns
    df = df.dropna(subset=["started_at", "start_station_id"])
    return df


def build_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips to station × hour level."""
    df["hour"] = df["started_at"].dt.floor("H")
    station_col = "start_station_id"
    demand = (
        df.groupby([station_col, "hour"])
        .size()
        .reset_index(name="trip_count")
        .rename(columns={station_col: "station_id"})
    )
    return demand


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"] = df["hour"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def run_pipeline(filepath: Path) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw(filepath)
    df = clean(df)
    demand = build_hourly_demand(df)
    demand = add_time_features(demand)
    out = PROCESSED_DIR / (filepath.stem + "_hourly.parquet")
    demand.to_parquet(out, index=False)
    print(f"Saved {len(demand):,} rows → {out}")
    return out

"""
Data cleaning and feature engineering for Citi Bike trip data.
Handles zip files containing multiple CSVs (2024+ format).
Produces station-level hourly demand time series saved to data/processed/.
"""
import zipfile
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DTYPES = {
    "ride_id": "str",
    "rideable_type": "category",
    "start_station_name": "str",
    "start_station_id": "str",
    "end_station_name": "str",
    "end_station_id": "str",
    "member_casual": "category",
}


def load_zip(filepath: Path) -> pd.DataFrame:
    """Load all CSVs from a zip file and concatenate."""
    frames = []
    with zipfile.ZipFile(filepath) as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        for csv_file in csv_files:
            with z.open(csv_file) as f:
                frames.append(pd.read_csv(f, dtype=DTYPES, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    # Drop rows missing station info or timestamps
    df = df.dropna(subset=["started_at", "start_station_id", "start_lat", "start_lng"])
    # Drop obvious bad durations (< 1 min or > 24 hrs)
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 1440)]
    return df.reset_index(drop=True)


def build_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips to station × hour level with location info."""
    df = df.copy()
    df["hour"] = df["started_at"].dt.floor("h")
    # Pre-compute binary columns — avoids slow per-group lambdas
    df["is_member"] = (df["member_casual"] == "member").astype("int8")
    df["is_casual"] = (df["member_casual"] == "casual").astype("int8")
    df["is_classic"] = (df["rideable_type"] == "classic_bike").astype("int8")
    df["is_electric"] = (df["rideable_type"] == "electric_bike").astype("int8")
    # Carry lat/lng as median per station (stable across trips)
    station_coords = (
        df.groupby("start_station_id")[["start_lat", "start_lng"]]
        .median()
        .rename(columns={"start_lat": "lat", "start_lng": "lng"})
    )
    demand = (
        df.groupby(["start_station_id", "hour"])
        .agg(
            trip_count=("ride_id", "count"),
            member_trips=("is_member", "sum"),
            casual_trips=("is_casual", "sum"),
            classic_trips=("is_classic", "sum"),
            electric_trips=("is_electric", "sum"),
        )
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
        .merge(station_coords, left_on="station_id", right_index=True, how="left")
    )
    return demand


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"] = df["hour"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def run_pipeline(filepath: Path) -> Path:
    """Full pipeline: load zip → clean → hourly demand → save parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {filepath.name} ...")
    df = load_zip(filepath)
    print(f"  Raw rows: {len(df):,}")
    df = clean(df)
    print(f"  After cleaning: {len(df):,}")
    demand = build_hourly_demand(df)
    demand = add_time_features(demand)
    out = PROCESSED_DIR / (filepath.stem.replace("-citibike-tripdata", "") + "_hourly.parquet")
    demand.to_parquet(out, index=False)
    print(f"  Saved {len(demand):,} station-hour rows → {out}")
    return out


if __name__ == "__main__":
    for zip_path in sorted(RAW_DIR.glob("*-citibike-tripdata.zip")):
        run_pipeline(zip_path)

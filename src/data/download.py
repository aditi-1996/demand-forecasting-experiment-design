"""
Download Citi Bike trip data from the public S3 bucket.

Usage:
    python -m src.data.download --year 2023 --months 1 2 3
"""
import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://s3.amazonaws.com/tripdata"
RAW_DIR = Path("data/raw")


def build_filename(year: int, month: int) -> str:
    """Return the S3 filename for a given year/month."""
    return f"{year}{month:02d}-citibike-tripdata.csv.zip"


def download_month(year: int, month: int, overwrite: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = build_filename(year, month)
    dest = RAW_DIR / filename
    if dest.exists() and not overwrite:
        print(f"Already exists, skipping: {dest}")
        return dest
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for m in args.months:
        download_month(args.year, m, overwrite=args.overwrite)

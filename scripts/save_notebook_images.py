"""
Extract embedded PNG outputs from all notebooks and save to images/.
Run from repo root: python scripts/save_notebook_images.py
"""

import json
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
IMAGES_DIR = REPO_ROOT / "images"

# Ordered image names per notebook (must match cell output order in each notebook)
NAMES = {
    # 9 embedded PNGs (geographic scatter + top-5 line were not rendered inline)
    "01_eda": [
        "01_ridetype_dist",
        "01_usertype_dist",
        "01_duration_hist",
        "01_daily_trips_ts",
        "01_hourly_avg_bar",
        "01_demand_heatmap_hourday",
        "01_member_casual_hourly_bar",
        "01_station_trips_hist",
        "01_station_trips_log_hist",
    ],
    # 4 embedded PNGs (geographic K-means/DBSCAN scatter not rendered inline)
    "02_clustering": [
        "02_demand_profiles_sample",
        "02_elbow_inertia",
        "02_silhouette_k",
        "02_cluster_profiles_grid",
    ],
    # 5 embedded PNGs (all-models overlay not rendered inline)
    "03_forecasting": [
        "03_train_test_split_ts",
        "03_prophet_forecast_grid",
        "03_sarima_forecast_grid",
        "03_xgb_feature_importance",
        "03_mape_comparison_bar",
    ],
    # 5 embedded PNGs
    "04_anomaly_detection": [
        "04_zscore_anomalies_ts",
        "04_iqr_rate_hour_bar",
        "04_iqr_rate_cluster_bar",
        "04_isolation_forest_ts",
        "04_consensus_anomalies_daily_bar",
    ],
    # 4 embedded PNGs (Bayesian posterior plots not rendered inline)
    "05_ab_testing": [
        "05_power_curve",
        "05_availability_dist_hist",
        "05_daily_mean_ts",
        "05_sequential_test_line",
    ],
    # 3 embedded PNGs (hourly-by-zone line not rendered inline)
    "06_geospatial": [
        "06_demand_vs_distance_scatter",
        "06_trip_share_zone_bar",
        "06_avg_trips_zone_bar",
    ],
    # 07_synthesis has no embedded PNGs until the notebook is re-run with data
    "07_synthesis": [],
}


def extract_images_from_notebook(nb_path: Path, names: list[str]) -> int:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    img_index = 0
    saved = 0

    for cell in nb.get("cells", []):
        outputs = cell.get("outputs", [])
        for output in outputs:
            # PNG data lives in output["data"]["image/png"]
            data = output.get("data", {})
            png_b64 = data.get("image/png")
            if png_b64 is None:
                continue

            if img_index >= len(names):
                print(
                    f"  WARNING: more images than names in {nb_path.name} "
                    f"(extra index {img_index}) — saving as {nb_path.stem}_extra_{img_index}"
                )
                out_name = f"{nb_path.stem}_extra_{img_index}.png"
            else:
                out_name = f"{names[img_index]}.png"

            out_path = IMAGES_DIR / out_name
            # base64 may be a list of strings (chunked) or a single string
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            out_path.write_bytes(base64.b64decode(png_b64))
            print(f"  Saved: {out_name}")
            img_index += 1
            saved += 1

    if img_index < len(names):
        print(
            f"  WARNING: expected {len(names)} images but only found {img_index} "
            f"in {nb_path.name}. Re-run the notebook to generate outputs."
        )

    return saved


def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    total = 0

    for nb_stem, names in NAMES.items():
        nb_path = NOTEBOOKS_DIR / f"{nb_stem}.ipynb"
        if not nb_path.exists():
            print(f"SKIP (not found): {nb_path.name}")
            continue
        print(f"\n{nb_path.name}")
        count = extract_images_from_notebook(nb_path, names)
        total += count

    print(f"\nDone — {total} image(s) saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()

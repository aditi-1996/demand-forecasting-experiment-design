"""
Geospatial visualizations using Folium.
- Demand heatmap (HeatMap plugin)
- Cluster map with A/B group overlay
- Hour-of-day animated heatmap
"""
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime


NYC_CENTER = [40.7282, -73.9942]


def demand_heatmap(station_summary: pd.DataFrame, title: str = "Station Demand Heatmap") -> folium.Map:
    """
    HeatMap of total demand across all stations.
    station_summary must have: lat, lng, total_trips
    """
    m = folium.Map(location=NYC_CENTER, zoom_start=12, tiles="CartoDB positron")

    heat_data = (
        station_summary[["lat", "lng", "total_trips"]]
        .dropna()
        .values
        .tolist()
    )
    HeatMap(heat_data, radius=12, blur=15, max_zoom=13).add_to(m)

    folium.map.Marker(
        NYC_CENTER,
        icon=folium.DivIcon(html=f"<div style='font-size:14px;font-weight:bold'>{title}</div>"),
    ).add_to(m)
    return m


def cluster_ab_map(
    station_clusters: pd.DataFrame,
    cluster_assignments: dict,
    cluster_names: dict,
) -> folium.Map:
    """
    Map showing each station coloured by:
      - Shape/outline: K-means cluster (commuter / tourist / etc.)
      - Fill colour: A/B group (treatment=coral, control=steelblue)

    station_clusters must have: station_id, lat, lng, kmeans_cluster, cluster_name
    cluster_assignments: {cluster_id: 'treatment'|'control'}
    """
    CLUSTER_COLORS = {0: "blue", 1: "red", 2: "green", 3: "gray", 4: "purple"}
    GROUP_FILL = {"treatment": "#e8785a", "control": "#4682b4"}

    m = folium.Map(location=NYC_CENTER, zoom_start=12, tiles="CartoDB positron")

    for _, row in station_clusters.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lng"]):
            continue
        cid = int(row["kmeans_cluster"])
        group = cluster_assignments.get(cid, "control")
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=5,
            color=CLUSTER_COLORS.get(cid, "black"),
            fill=True,
            fill_color=GROUP_FILL[group],
            fill_opacity=0.75,
            weight=1.5,
            popup=folium.Popup(
                f"<b>{row['station_id']}</b><br>"
                f"Cluster: {cluster_names.get(cid, cid)}<br>"
                f"A/B Group: <b>{group}</b>",
                max_width=180,
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;background:white;
                padding:10px;border-radius:6px;border:1px solid #ccc;font-size:12px">
      <b>A/B Group</b><br>
      <span style='color:#e8785a'>&#9679;</span> Treatment<br>
      <span style='color:#4682b4'>&#9679;</span> Control<br>
      <br><b>Cluster (outline)</b><br>
      <span style='color:blue'>&#9675;</span> Commuter<br>
      <span style='color:red'>&#9675;</span> Tourist<br>
      <span style='color:green'>&#9675;</span> Recreational<br>
      <span style='color:gray'>&#9675;</span> Low-activity
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def hourly_heatmap_animation(
    hourly: pd.DataFrame,
    station_summary: pd.DataFrame,
    hours: list = None,
) -> folium.Map:
    """
    Animated heatmap stepping through hours of the day (averaged over all days).
    hours: list of ints 0-23 to include (default: all 24)
    """
    if hours is None:
        hours = list(range(24))

    coords = station_summary.set_index("station_id")[["lat", "lng"]]
    avg_by_hour = (
        hourly.groupby(["station_id", "hour_of_day"])["trip_count"]
        .mean()
        .reset_index()
        .merge(coords.reset_index(), on="station_id", how="left")
        .dropna(subset=["lat", "lng"])
    )

    # Build list of frames (one per hour)
    max_val = avg_by_hour["trip_count"].max()
    heat_data = []
    time_index = []
    for h in hours:
        frame = avg_by_hour[avg_by_hour["hour_of_day"] == h]
        heat_data.append(
            [[row["lat"], row["lng"], row["trip_count"] / max_val]
             for _, row in frame.iterrows()]
        )
        time_index.append(f"{h:02d}:00")

    m = folium.Map(location=NYC_CENTER, zoom_start=12, tiles="CartoDB positron")
    HeatMapWithTime(
        heat_data,
        index=time_index,
        radius=12,
        blur=15,
        max_opacity=0.8,
        auto_play=True,
        display_index=True,
    ).add_to(m)
    return m

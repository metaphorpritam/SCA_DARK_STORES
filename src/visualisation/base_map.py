import pandas as pd
import folium
from folium.plugins import HeatMap

# -----------------------------
# LOAD DATA
# -----------------------------
customers = pd.read_parquet("data/master_df_v2.parquet")
centroids = pd.read_csv("data/dark_store_candidates.csv")

# Clean customer coordinates
customers = customers[["customer_lat", "customer_lon"]].dropna()

# -----------------------------
# CREATE MAP
# -----------------------------
m = folium.Map(
    location=[customers["customer_lat"].mean(), customers["customer_lon"].mean()],
    zoom_start=10,
)

# -----------------------------
# ADD HEATMAP (Demand Density)
# -----------------------------
heat_data = customers[["customer_lat", "customer_lon"]].values.tolist()

HeatMap(heat_data, radius=8, blur=12, min_opacity=0.3).add_to(m)

# -----------------------------
# ADD CUSTOMER POINTS (sample for speed)
# -----------------------------
for _, row in customers.sample(1000, random_state=42).iterrows():
    folium.CircleMarker(
        location=[row["customer_lat"], row["customer_lon"]],
        radius=2,
        color="blue",
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

# -----------------------------
# ADD DARK STORE MARKERS
# -----------------------------
for _, row in centroids.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.Icon(color="red", icon="star"),
        popup="Dark Store",
    ).add_to(m)

# -----------------------------
# ADD LEGEND (IMPORTANT)
# -----------------------------
legend_html = """
<div style="
position: fixed; 
bottom: 50px; left: 50px; width: 220px; height: 120px; 
background-color: white; z-index:9999; font-size:14px;
border:2px solid grey; padding: 10px;
">
<b>Legend</b><br>
<span style="color:red;">★</span> Dark Store<br>
<span style="color:blue;">●</span> Customer Location<br>
Heatmap → Demand Density
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# -----------------------------
# ADD LAYER CONTROL (optional but good)
# -----------------------------
folium.LayerControl().add_to(m)

# -----------------------------
# SAVE MAP
# -----------------------------
m.save("outputs/dark_store_map.html")

print("✅ dark_store_map.html created successfully!")

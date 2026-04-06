import pandas as pd
import folium
from folium.plugins import HeatMap

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_parquet("data/master_df_v3.parquet")

# High return probability customers
df = df[df["return_prob"] > 0.3]

# -----------------------------
# CREATE MAP
# -----------------------------
m = folium.Map(
    location=[df["customer_lat"].mean(), df["customer_lon"].mean()], zoom_start=10
)

# -----------------------------
# HEATMAP
# -----------------------------
heat_data = df[["customer_lat", "customer_lon"]].dropna().values.tolist()

HeatMap(heat_data, radius=10, blur=15, min_opacity=0.4).add_to(m)

# -----------------------------
# SAVE
# -----------------------------
m.save("outputs/return_heatmap.html")

print("✅ return_heatmap.html created")

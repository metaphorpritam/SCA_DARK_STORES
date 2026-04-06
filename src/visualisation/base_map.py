import folium
import os

# 1. Ensure the outputs folder exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# 2. Centering the map on São Paulo (Target Region)
# Coordinates: -23.5505, -46.6333
sp_map = folium.Map(
    location=[-23.5505, -46.6333], zoom_start=11, tiles="cartodbpositron"
)

# 3. Save the map
sp_map.save("outputs/base_map.html")

print("Success! Base map saved in 'outputs/base_map.html'")

import pandas as pd
import folium

# Load data
customers = pd.read_parquet("data/master_df_v2.parquet")
centroids = pd.read_csv("data/dark_store_candidates.csv")

# Map
m = folium.Map(
    location=[customers["customer_lat"].mean(), customers["customer_lon"].mean()],
    zoom_start=10,
)

# Customers (sample)
for _, row in customers.sample(1000).iterrows():
    folium.CircleMarker(
        location=[row["customer_lat"], row["customer_lon"]],
        radius=2,
        color="blue",
        fill=True,
    ).add_to(m)

# Dark Stores (IMPORTANT)
for _, row in centroids.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.Icon(color="red", icon="star"),
        popup="Dark Store",
    ).add_to(m)

# Save updated map
m.save("outputs/dark_store_map.html")

print("✅ dark_store_map.html created")

import json
import folium
import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------
stores = pd.read_csv("data/dark_store_candidates.csv")

with open("outputs/forward_routes.json") as f:
    routes = json.load(f)

# -----------------------------
# CREATE MAP
# -----------------------------
m = folium.Map(location=[stores["lat"].mean(), stores["lon"].mean()], zoom_start=10)

# -----------------------------
# ADD DARK STORES
# -----------------------------
for _, row in stores.iterrows():
    folium.Marker(
        [row["lat"], row["lon"]],
        icon=folium.Icon(color="red", icon="star"),
        popup="Dark Store",
    ).add_to(m)

# -----------------------------
# ADD ROUTES
# -----------------------------
colors = [
    "blue",
    "green",
    "purple",
    "orange",
    "black",
    "pink",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkred",
]

for i, zone in enumerate(routes):

    if "routes" not in zone:
        continue

    for route in zone["routes"]:
        coords = route.get("coordinates", [])

        if len(coords) > 1:
            folium.PolyLine(
                coords, color=colors[i % len(colors)], weight=3, opacity=0.8
            ).add_to(m)

# -----------------------------
# SAVE
# -----------------------------
m.save("outputs/forward_routes_map.html")

print("✅ forward_routes_map.html created")

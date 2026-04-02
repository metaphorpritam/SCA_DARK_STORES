import folium
import os

# 1. Ensure the outputs folder exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# 2. Centering the map on São Paulo (Target Region)
# Coordinates: -23.5505, -46.6333
sp_map = folium.Map(location=[-23.5505, -46.6333], zoom_start=11, tiles="cartodbpositron")

# 3. Save the map
sp_map.save("outputs/base_map.html")

print("Success! Base map saved in 'outputs/base_map.html'")
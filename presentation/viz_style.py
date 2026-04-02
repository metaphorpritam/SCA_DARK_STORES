import plotly.io as pio
import plotly.express as px

# Define Team Palette (Professional Logistics Theme)
team_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Set as default template
pio.templates["dark_store_theme"] = pio.templates["plotly_white"]
pio.templates["dark_store_theme"].layout.colorway = team_colors

print("✅ Plotly template configured with team colors.")
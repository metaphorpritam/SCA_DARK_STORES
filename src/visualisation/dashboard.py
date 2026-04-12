import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

print("\n📊 Creating Day 5 Dashboard...\n")

# =========================
# LOAD REQUIRED DATA
# =========================
forward_zone = pd.read_csv("outputs/forward_kpi_by_zone.csv")
master_df = pd.read_parquet("data/master_df_v3.parquet")

# =========================
# CHART 1 — COST PER ZONE
# =========================
fig1 = px.bar(forward_zone, x="zone_id", y="routing_cost_R$", title="Cost per Zone")

# =========================
# CHART 2 — DISTANCE PER ZONE
# =========================
fig2 = px.bar(
    forward_zone, x="zone_id", y="total_distance_km", title="Distance per Zone"
)

# =========================
# CHART 3 — STOPS PER ZONE
# =========================
fig3 = px.bar(forward_zone, x="zone_id", y="num_stops", title="Stops per Zone")

# =========================
# CHART 4 — RETURN PROBABILITY
# =========================
fig4 = px.histogram(
    master_df, x="return_prob", nbins=30, title="Return Probability Distribution"
)

# =========================
# CREATE DASHBOARD
# =========================
dashboard = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Cost per Zone",
        "Distance per Zone",
        "Stops per Zone",
        "Return Probability",
    ),
)

dashboard.add_trace(fig1.data[0], row=1, col=1)
dashboard.add_trace(fig2.data[0], row=1, col=2)
dashboard.add_trace(fig3.data[0], row=2, col=1)
dashboard.add_trace(fig4.data[0], row=2, col=2)

dashboard.update_layout(
    title="Supply Chain Dashboard — Day 5", height=800, showlegend=False
)

# =========================
# SAVE OUTPUT
# =========================
dashboard.write_html("outputs/dashboard_draft.html")

print("✅ Day 5 dashboard created → outputs/dashboard_draft.html")

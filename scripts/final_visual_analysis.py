"""
Module: final_visual_analysis.py
Stage:  Post-Pipeline Visualisation (Day 7)

Runs AFTER the full pipeline is complete. Reads all outputs and generates
publication-ready maps + charts for the report and presentation.

DEPENDS ON:
    data/master_df_v3.parquet
    data/dark_stores_final.csv
    outputs/forward_routes.csv
    outputs/reverse_routes.csv
    outputs/forward_kpi_summary.csv
    outputs/reverse_kpi_summary.csv
    outputs/hybrid_kpi_summary.csv
    outputs/scenario_results_table.csv
    outputs/combined_kpi_report.csv
    outputs/baseline_kpis_naive.csv

OUTPUT:
    visualisations/dark_store_map.html         — Folium: dark stores + customer clusters + coverage circles
    visualisations/forward_routes_map.html      — Folium: forward delivery routes per zone
    visualisations/return_heatmap.html          — Folium: return probability heatmap
    outputs/scenario_comparison.png
    outputs/sdvrp_savings.png
    outputs/cost_breakdown.png
    outputs/coverage_by_zone.png
    outputs/vehicle_utilisation.png
    outputs/naive_vs_optimised.png

INTERFACE:
    build_dark_store_map(master_df, dark_stores, vis_dir)  -> folium.Map
    build_routes_map(master_df, dark_stores, routes_df, vis_dir)  -> folium.Map
    build_return_heatmap(master_df, dark_stores, vis_dir)  -> folium.Map
    plot_scenario_comparison(scenario_df, chart_dir)  -> None
    plot_sdvrp_savings(combined_kpi, chart_dir)  -> None
    plot_cost_breakdown(fwd_kpi, rev_kpi, hybrid_kpi, chart_dir)  -> None
    plot_coverage_by_zone(master_df, dark_stores, chart_dir)  -> None
    plot_vehicle_utilisation(fwd_kpi, rev_kpi, chart_dir)  -> None
    plot_naive_vs_optimised(naive_kpi, fwd_kpi, chart_dir)  -> None
    run(data_dir, out_dir, vis_dir)  -> None
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ZONE_COLOURS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
]

FIG_BG = "#fafafa"
GRID_ALPHA = 0.3

plt.rcParams.update(
    {
        "figure.facecolor": FIG_BG,
        "axes.facecolor": "#ffffff",
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


# ---------------------------------------------------------------------------
# 1. Dark Store Map (Folium)
# ---------------------------------------------------------------------------


def build_dark_store_map(
    master_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    vis_dir: str | Path = "visualisations",
):
    """
    Folium map with:
      - Customer points coloured by zone
      - Dark store markers (star icons)
      - 5 km coverage circles around each dark store
      - LayerControl toggle
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        print("[WARN] folium not installed — skipping dark_store_map")
        return None

    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    centre_lat = master_df["customer_lat"].median()
    centre_lon = master_df["customer_lon"].median()
    m = folium.Map(
        location=[centre_lat, centre_lon], zoom_start=11, tiles="CartoDB positron"
    )

    # Customer points by zone (sampled for performance)
    sample = master_df.sample(n=min(3000, len(master_df)), random_state=42)
    for _, row in sample.iterrows():
        zid = int(row["dark_store_id"]) if pd.notna(row.get("dark_store_id")) else 0
        colour = ZONE_COLOURS[zid % len(ZONE_COLOURS)]
        folium.CircleMarker(
            location=[row["customer_lat"], row["customer_lon"]],
            radius=2,
            color=colour,
            fill=True,
            fill_opacity=0.5,
            weight=0.5,
        ).add_to(m)

    # Dark stores + coverage circles
    for _, store in dark_stores.iterrows():
        sid = int(store["dark_store_id"])
        colour = ZONE_COLOURS[sid % len(ZONE_COLOURS)]

        # 5 km coverage circle
        folium.Circle(
            location=[store["lat"], store["lon"]],
            radius=5000,
            color=colour,
            fill=True,
            fill_opacity=0.08,
            weight=1.5,
            dash_array="5",
            popup=f"Zone {sid} — 5 km coverage",
        ).add_to(m)

        # Dark store marker
        folium.Marker(
            location=[store["lat"], store["lon"]],
            popup=f"Dark Store {sid}",
            tooltip=f"DS-{sid}",
            icon=folium.Icon(color="darkblue", icon="warehouse", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    out_path = vis_dir / "dark_store_map.html"
    m.save(str(out_path))
    print(f"[vis] dark_store_map.html saved → {out_path}")
    return m


# ---------------------------------------------------------------------------
# 2. Forward Routes Map (Folium)
# ---------------------------------------------------------------------------


def build_routes_map(
    master_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    routes_df: pd.DataFrame,
    vis_dir: str | Path = "visualisations",
):
    """
    Folium map with forward delivery route polylines per zone.
    Each vehicle route is a coloured line from depot through stops.
    """
    try:
        import folium
    except ImportError:
        print("[WARN] folium not installed — skipping forward_routes_map")
        return None

    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    centre_lat = master_df["customer_lat"].median()
    centre_lon = master_df["customer_lon"].median()
    m = folium.Map(
        location=[centre_lat, centre_lon], zoom_start=11, tiles="CartoDB positron"
    )

    # Dark store markers
    for _, store in dark_stores.iterrows():
        sid = int(store["dark_store_id"])
        folium.Marker(
            location=[store["lat"], store["lon"]],
            tooltip=f"DS-{sid}",
            icon=folium.Icon(color="darkblue", icon="warehouse", prefix="fa"),
        ).add_to(m)

    # Route polylines
    if "zone_id" in routes_df.columns and "vehicle_id" in routes_df.columns:
        for (zid, vid), group in routes_df.groupby(["zone_id", "vehicle_id"]):
            coords = group[["lat", "lon"]].values.tolist()
            if len(coords) < 2:
                continue
            colour = ZONE_COLOURS[int(zid) % len(ZONE_COLOURS)]
            folium.PolyLine(
                coords,
                color=colour,
                weight=2.5,
                opacity=0.7,
                tooltip=f"Zone {int(zid)} — Vehicle {int(vid)}",
            ).add_to(m)

    folium.LayerControl().add_to(m)

    out_path = vis_dir / "forward_routes_map.html"
    m.save(str(out_path))
    print(f"[vis] forward_routes_map.html saved → {out_path}")
    return m


# ---------------------------------------------------------------------------
# 3. Return Heatmap (Folium)
# ---------------------------------------------------------------------------


def build_return_heatmap(
    master_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    vis_dir: str | Path = "visualisations",
):
    """
    Folium heatmap of return probability across the metro area.
    Higher return_prob → hotter areas.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("[WARN] folium not installed — skipping return_heatmap")
        return None

    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    centre_lat = master_df["customer_lat"].median()
    centre_lon = master_df["customer_lon"].median()
    m = folium.Map(
        location=[centre_lat, centre_lon], zoom_start=11, tiles="CartoDB dark_matter"
    )

    # Heatmap weighted by return_prob
    if "return_prob" in master_df.columns:
        heat_data = master_df[["customer_lat", "customer_lon", "return_prob"]].dropna()
        HeatMap(
            heat_data.values.tolist(),
            radius=12,
            blur=15,
            min_opacity=0.3,
            name="Return probability",
            gradient={
                0.2: "blue",
                0.4: "lime",
                0.6: "yellow",
                0.8: "orange",
                1.0: "red",
            },
        ).add_to(m)

    # Dark store markers
    for _, store in dark_stores.iterrows():
        sid = int(store["dark_store_id"])
        folium.Marker(
            location=[store["lat"], store["lon"]],
            tooltip=f"DS-{sid}",
            icon=folium.Icon(color="white", icon="warehouse", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    out_path = vis_dir / "return_heatmap.html"
    m.save(str(out_path))
    print(f"[vis] return_heatmap.html saved → {out_path}")
    return m


# ---------------------------------------------------------------------------
# 4. Scenario Comparison Chart
# ---------------------------------------------------------------------------


def plot_scenario_comparison(
    scenario_df: pd.DataFrame,
    chart_dir: str | Path = "outputs",
) -> None:
    """Grouped bar chart: 3 scenarios × 4 KPIs."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Scenario Analysis — A (Base) vs B (+30% Surge) vs C (2× Returns)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    metrics = [
        ("total_routing_cost_R$", "Total Routing Cost (R$)", "#2196F3"),
        ("total_distance_km", "Total Distance (km)", "#4CAF50"),
        ("n_vehicles", "Fleet Size (vehicles)", "#FF9800"),
        ("return_efficiency_pct", "Return Efficiency (%)", "#9C27B0"),
    ]

    scenarios = scenario_df["scenario"].tolist()
    x = np.arange(len(scenarios))
    width = 0.5

    for ax, (col, title, colour) in zip(axes.flat, metrics):
        vals = scenario_df[col].values
        bars = ax.bar(x, vals, width, color=colour, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Scenario {s}" for s in scenarios])
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = chart_dir / "scenario_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] scenario_comparison.png saved")


# ---------------------------------------------------------------------------
# 5. SDVRP Savings Chart
# ---------------------------------------------------------------------------


def plot_sdvrp_savings(
    combined_kpi: pd.DataFrame,
    chart_dir: str | Path = "outputs",
) -> None:
    """Horizontal bar chart: per-zone SDVRP cost saving (% and R$)."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    if "saving_pct" not in combined_kpi.columns:
        print("[vis] No SDVRP savings data — skipping sdvrp_savings.png")
        return

    df = combined_kpi.sort_values("saving_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    zones = [f"Zone {int(z)}" for z in df["zone_id"]]
    savings = df["saving_pct"].values
    colours = plt.cm.RdYlGn(savings / savings.max())

    bars = ax.barh(zones, savings, color=colours, edgecolor="white", height=0.6)
    ax.set_xlabel("SDVRP Saving (%)")
    ax.set_title(
        "SDVRP Hybrid Savings by Zone\n(Separate Forward+Reverse → Integrated)",
        fontweight="bold",
    )
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=4)
    ax.set_xlim(0, max(savings) * 1.15)

    # Add total saving annotation
    total_saving = combined_kpi["saving_R$"].sum()
    avg_pct = combined_kpi["saving_pct"].mean()
    ax.annotate(
        f"Total saving: R${total_saving:,.0f}  (avg {avg_pct:.1f}%)",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#4CAF50"),
    )

    plt.tight_layout()
    out = chart_dir / "sdvrp_savings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] sdvrp_savings.png saved")


# ---------------------------------------------------------------------------
# 6. Cost Breakdown Stacked Bar
# ---------------------------------------------------------------------------


def plot_cost_breakdown(
    fwd_kpi: pd.DataFrame,
    rev_kpi: pd.DataFrame,
    hybrid_kpi: pd.DataFrame | None,
    chart_dir: str | Path = "outputs",
) -> None:
    """Stacked bar: forward vs reverse cost per zone, with hybrid overlay."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    zones = fwd_kpi["zone_id"].values
    fwd_costs = fwd_kpi["routing_cost_R$"].values
    rev_costs = rev_kpi["routing_cost_R$"].values

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(zones))
    width = 0.35

    ax.bar(
        x - width / 2,
        fwd_costs,
        width,
        label="Forward (delivery)",
        color="#2196F3",
        alpha=0.85,
    )
    ax.bar(
        x - width / 2,
        rev_costs,
        width,
        bottom=fwd_costs,
        label="Reverse (pickup)",
        color="#FF9800",
        alpha=0.85,
    )

    if hybrid_kpi is not None and "routing_cost_R$" in hybrid_kpi.columns:
        hyb_costs = hybrid_kpi["routing_cost_R$"].values
        ax.bar(
            x + width / 2,
            hyb_costs,
            width,
            label="SDVRP hybrid",
            color="#4CAF50",
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Routing Cost (R$)")
    ax.set_title("Cost Breakdown: Separate vs Hybrid (per zone)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(z)}" for z in zones])
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = chart_dir / "cost_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] cost_breakdown.png saved")


# ---------------------------------------------------------------------------
# 7. Coverage by Zone
# ---------------------------------------------------------------------------


def plot_coverage_by_zone(
    master_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    chart_dir: str | Path = "outputs",
    radius_km: float = 5.0,
) -> None:
    """Bar chart: % customers within 5 km of their dark store, per zone."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    store_lookup = dark_stores.set_index("dark_store_id")[["lat", "lon"]]
    df = master_df.dropna(subset=["dark_store_id"]).copy()
    df["dark_store_id"] = df["dark_store_id"].astype(int)

    # Approximate distance to assigned dark store
    df = df.merge(
        store_lookup.rename(columns={"lat": "store_lat", "lon": "store_lon"}),
        left_on="dark_store_id",
        right_index=True,
        how="left",
    )
    df["dist_to_store_km"] = np.sqrt(
        ((df["customer_lat"] - df["store_lat"]) * 111.0) ** 2
        + ((df["customer_lon"] - df["store_lon"]) * 92.0) ** 2
    )

    coverage = (
        df.groupby("dark_store_id")
        .apply(lambda g: (g["dist_to_store_km"] <= radius_km).mean() * 100)
        .reset_index(name="coverage_pct")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    zones = coverage["dark_store_id"].values
    pcts = coverage["coverage_pct"].values
    colours = [
        "#4CAF50" if p >= 70 else "#FF9800" if p >= 50 else "#f44336" for p in pcts
    ]

    bars = ax.bar(zones, pcts, color=colours, edgecolor="white", width=0.6)
    ax.axhline(y=70, color="#333", linestyle="--", linewidth=1, label="70% target")
    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(
        f"Customer Coverage within {radius_km} km of Dark Store", fontweight="bold"
    )
    ax.set_xticks(zones)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=3)
    ax.set_ylim(0, 105)
    ax.legend()

    overall = (df["dist_to_store_km"] <= radius_km).mean() * 100
    ax.annotate(
        f"Overall: {overall:.1f}%",
        xy=(0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#2196F3"),
    )

    plt.tight_layout()
    out = chart_dir / "coverage_by_zone.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] coverage_by_zone.png saved")


# ---------------------------------------------------------------------------
# 8. Vehicle Utilisation
# ---------------------------------------------------------------------------


def plot_vehicle_utilisation(
    fwd_kpi: pd.DataFrame,
    rev_kpi: pd.DataFrame,
    chart_dir: str | Path = "outputs",
) -> None:
    """Grouped bar: vehicles per zone (forward vs reverse)."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    zones = fwd_kpi["zone_id"].values
    fwd_veh = fwd_kpi["n_vehicles_used"].values
    rev_veh = rev_kpi["n_vehicles_used"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(zones))
    width = 0.35

    ax.bar(x - width / 2, fwd_veh, width, label="Forward", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, rev_veh, width, label="Reverse", color="#FF9800", alpha=0.85)

    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Vehicles Used")
    ax.set_title("Vehicle Utilisation by Zone", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(z)}" for z in zones])
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = chart_dir / "vehicle_utilisation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] vehicle_utilisation.png saved")


# ---------------------------------------------------------------------------
# 9. Naive vs Optimised
# ---------------------------------------------------------------------------


def plot_naive_vs_optimised(
    naive_kpi: pd.DataFrame,
    fwd_kpi: pd.DataFrame,
    chart_dir: str | Path = "outputs",
) -> None:
    """Bar chart comparing naive individual delivery vs optimised VRP cost."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    naive_total = naive_kpi["naive_routing_cost_R$"].iloc[0]
    naive_avg = naive_kpi["avg_naive_dist_km"].iloc[0]
    n_naive = naive_kpi["n_customers"].iloc[0]

    vrp_total = fwd_kpi["routing_cost_R$"].sum()
    vrp_dist = fwd_kpi["total_dist_km"].sum()
    n_vrp_stops = fwd_kpi.get(
        "n_customers", fwd_kpi.get("n_vehicles_used", pd.Series([0]))
    ).sum()

    naive_per_order = naive_total / max(n_naive, 1)
    vrp_per_order = vrp_total / max(825, 1)  # 825 sampled deliveries

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Total cost
    labels = ["Naive\n(individual trips)", "VRP Optimised\n(consolidated routes)"]
    vals = [naive_total, vrp_total]
    colours = ["#f44336", "#4CAF50"]
    bars = ax1.bar(labels, vals, color=colours, width=0.5, edgecolor="white")
    ax1.set_ylabel("Total Routing Cost (R$)")
    ax1.set_title("Total Cost: Naive vs Optimised", fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    for bar, val in zip(bars, vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            f"R${val:,.0f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

    # Panel 2: Per-order cost
    vals2 = [naive_per_order, vrp_per_order]
    bars2 = ax2.bar(labels, vals2, color=colours, width=0.5, edgecolor="white")
    ax2.set_ylabel("Cost per Delivery (R$)")
    ax2.set_title("Per-Delivery Cost", fontweight="bold")
    for bar, val in zip(bars2, vals2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            f"R${val:,.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

    reduction = (1 - vrp_total / naive_total) * 100
    fig.suptitle(
        f"Route Consolidation Reduces Cost by {reduction:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    out = chart_dir / "naive_vs_optimised.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] naive_vs_optimised.png saved")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    data_dir: str | Path = "data",
    out_dir: str | Path = "outputs",
    vis_dir: str | Path = "visualisations",
) -> None:
    """Generate all final visualisations from completed pipeline outputs."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    vis_dir = Path(vis_dir)
    chart_dir = out_dir
    chart_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FINAL VISUAL ANALYSIS")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading pipeline outputs...")
    master_df = pd.read_parquet(data_dir / "master_df_v3.parquet")
    dark_stores = pd.read_csv(data_dir / "dark_stores_final.csv")
    print(f"  master_df: {len(master_df):,} rows | {len(dark_stores)} dark stores")

    fwd_routes = pd.read_csv(out_dir / "forward_routes.csv")
    fwd_kpi = pd.read_csv(out_dir / "forward_kpi_summary.csv")
    rev_kpi = pd.read_csv(out_dir / "reverse_kpi_summary.csv")
    print(f"  Forward routes: {len(fwd_routes):,} stops")

    hybrid_kpi = None
    hybrid_path = out_dir / "hybrid_kpi_summary.csv"
    if hybrid_path.exists():
        hybrid_kpi = pd.read_csv(hybrid_path)
        print(f"  Hybrid KPI: {len(hybrid_kpi)} zones")

    scenario_df = pd.read_csv(out_dir / "scenario_results_table.csv")
    combined_kpi = pd.read_csv(out_dir / "combined_kpi_report.csv")
    naive_kpi = pd.read_csv(out_dir / "baseline_kpis_naive.csv")
    print(f"  Scenarios: {len(scenario_df)} | Combined KPI: {len(combined_kpi)} zones")

    # ── Folium maps ───────────────────────────────────────────────
    print("\n[2/4] Building Folium maps...")
    build_dark_store_map(master_df, dark_stores, vis_dir)
    build_routes_map(master_df, dark_stores, fwd_routes, vis_dir)
    build_return_heatmap(master_df, dark_stores, vis_dir)

    # ── Matplotlib charts ─────────────────────────────────────────
    print("\n[3/4] Generating charts...")
    plot_scenario_comparison(scenario_df, chart_dir)
    plot_sdvrp_savings(combined_kpi, chart_dir)
    plot_cost_breakdown(fwd_kpi, rev_kpi, hybrid_kpi, chart_dir)
    plot_coverage_by_zone(master_df, dark_stores, chart_dir)
    plot_vehicle_utilisation(fwd_kpi, rev_kpi, chart_dir)
    plot_naive_vs_optimised(naive_kpi, fwd_kpi, chart_dir)

    # ── Summary ───────────────────────────────────────────────────
    print("\n[4/4] Done.")
    print("\n" + "=" * 60)
    print("  VISUAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Maps   : {vis_dir}/")
    print(f"    • dark_store_map.html")
    print(f"    • forward_routes_map.html")
    print(f"    • return_heatmap.html")
    print(f"  Charts : {chart_dir}/")
    print(f"    • scenario_comparison.png")
    print(f"    • sdvrp_savings.png")
    print(f"    • cost_breakdown.png")
    print(f"    • coverage_by_zone.png")
    print(f"    • vehicle_utilisation.png")
    print(f"    • naive_vs_optimised.png")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()

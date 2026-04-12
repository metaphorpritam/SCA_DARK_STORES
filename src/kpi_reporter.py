"""
Module: kpi_reporter.py
Stage:  Day 6 — Combined KPI Report + SDVRP Zone Priority Ranking

Merges forward, reverse, and (optionally) hybrid SDVRP KPIs into a single
per-zone report, computes savings from SDVRP hybridisation, and ranks zones
by SDVRP saving potential (highest cost-reduction opportunity first).

INPUT
-----
outputs/forward_kpi_summary.csv   — one row per zone
    Columns: zone_id, n_customers, n_vehicles_used, total_dist_km,
             routing_cost_R$, max_route_km, min_route_km

outputs/reverse_kpi_summary.csv   — one row per zone
    Same schema (n_pickups instead of n_customers).

outputs/hybrid_kpi_summary.csv    — optional; written by run_all_zones_sdvrp()
    Columns: zone_id, n_deliveries, n_pickups, n_vehicles_used, total_dist_km,
             routing_cost_R$, separate_cost_R$, saving_R$, saving_pct,
             max_route_km, min_route_km

OUTPUT
------
outputs/combined_kpi_report.csv   — per-zone full KPI table
    Columns (always present):
        zone_id
        fwd_customers, fwd_vehicles, fwd_dist_km, fwd_cost_R$
        rev_pickups,   rev_vehicles, rev_dist_km, rev_cost_R$
        combined_vehicles, combined_dist_km, separate_cost_R$
        sdvrp_priority_rank            (1 = highest saving opportunity)
    Columns (present only when hybrid_kpi_summary.csv exists):
        hybrid_vehicles, hybrid_dist_km, hybrid_cost_R$
        saving_R$, saving_pct, sdvrp_solved

INTERFACE
---------
    load_forward(path)   -> pd.DataFrame
    load_reverse(path)   -> pd.DataFrame
    load_hybrid(path)    -> pd.DataFrame | None        # None if file absent
    build_combined(fwd, rev, hybrid) -> pd.DataFrame
    zone_priority_ranking(df)        -> pd.DataFrame   # adds sdvrp_priority_rank
    print_summary(df)                -> None
    run(fwd_path, rev_path, hybrid_path, out_path) -> pd.DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# File paths (defaults)
# ---------------------------------------------------------------------------
DEFAULT_FWD_PATH = "outputs/forward_kpi_summary.csv"
DEFAULT_REV_PATH = "outputs/reverse_kpi_summary.csv"
DEFAULT_HYBRID_PATH = "outputs/hybrid_kpi_summary.csv"
DEFAULT_OUT_PATH = "outputs/combined_kpi_report.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_forward(path: str | Path = DEFAULT_FWD_PATH) -> pd.DataFrame:
    """
    Load forward_kpi_summary.csv.

    Expected columns: zone_id, n_customers, n_vehicles_used, total_dist_km,
                      routing_cost_R$, max_route_km, min_route_km
    Returns a clean DataFrame with canonical fwd_ prefixed columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[kpi_reporter] Forward KPI not found: {path}")
    df = pd.read_csv(path)
    rename = {
        "n_customers": "fwd_customers",
        "n_vehicles_used": "fwd_vehicles",
        "total_dist_km": "fwd_dist_km",
        "routing_cost_R$": "fwd_cost_R$",
        "max_route_km": "fwd_max_route_km",
        "min_route_km": "fwd_min_route_km",
    }
    df = df.rename(columns=rename)
    print(f"[kpi_reporter] Forward KPI loaded: {len(df)} zones  ({path.name})")
    return df


def load_reverse(path: str | Path = DEFAULT_REV_PATH) -> pd.DataFrame:
    """
    Load reverse_kpi_summary.csv.

    Expected columns: zone_id, n_pickups, n_vehicles_used, total_dist_km,
                      routing_cost_R$, max_route_km, min_route_km
    Returns a clean DataFrame with canonical rev_ prefixed columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[kpi_reporter] Reverse KPI not found: {path}")
    df = pd.read_csv(path)
    rename = {
        "n_pickups": "rev_pickups",
        "n_vehicles_used": "rev_vehicles",
        "total_dist_km": "rev_dist_km",
        "routing_cost_R$": "rev_cost_R$",
        "max_route_km": "rev_max_route_km",
        "min_route_km": "rev_min_route_km",
    }
    df = df.rename(columns=rename)
    print(f"[kpi_reporter] Reverse KPI loaded: {len(df)} zones  ({path.name})")
    return df


def load_hybrid(path: str | Path = DEFAULT_HYBRID_PATH) -> pd.DataFrame | None:
    """
    Load hybrid_kpi_summary.csv if it exists; return None otherwise.

    Expected columns: zone_id, n_deliveries, n_pickups, n_vehicles_used,
                      total_dist_km, routing_cost_R$, separate_cost_R$,
                      saving_R$, saving_pct, max_route_km, min_route_km
    Returns a clean DataFrame with canonical hybrid_ prefixed columns.
    """
    path = Path(path)
    if not path.exists():
        print(
            f"[kpi_reporter] Hybrid KPI not found ({path.name}) — "
            "SDVRP columns will be omitted.  Run run_all_zones_sdvrp() first."
        )
        return None
    df = pd.read_csv(path)
    rename = {
        "n_vehicles_used": "hybrid_vehicles",
        "total_dist_km": "hybrid_dist_km",
        "routing_cost_R$": "hybrid_cost_R$",
    }
    df = df.rename(columns=rename)
    df["sdvrp_solved"] = True
    print(f"[kpi_reporter] Hybrid KPI loaded: {len(df)} zones  ({path.name})")
    return df


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_combined(
    fwd: pd.DataFrame,
    rev: pd.DataFrame,
    hybrid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge forward + reverse KPIs on zone_id and compute joint metrics.

    Joint metrics (always computed):
        combined_vehicles  = fwd_vehicles + rev_vehicles
        combined_dist_km   = fwd_dist_km  + rev_dist_km
        separate_cost_R$   = fwd_cost_R$  + rev_cost_R$

    SDVRP metrics (only if hybrid is not None):
        hybrid_vehicles, hybrid_dist_km, hybrid_cost_R$
        saving_R$   = separate_cost_R$ - hybrid_cost_R$
        saving_pct  = saving_R$ / separate_cost_R$ × 100
        sdvrp_solved (bool)

    Parameters
    ----------
    fwd    : output of load_forward()
    rev    : output of load_reverse()
    hybrid : output of load_hybrid() or None

    Returns
    -------
    pd.DataFrame — one row per zone, sorted by zone_id
    """
    df = pd.merge(fwd, rev, on="zone_id", how="outer").sort_values("zone_id")
    df = df.fillna(0)

    df["combined_vehicles"] = (df["fwd_vehicles"] + df["rev_vehicles"]).astype(int)
    df["combined_dist_km"] = (df["fwd_dist_km"] + df["rev_dist_km"]).round(3)
    df["separate_cost_R$"] = (df["fwd_cost_R$"] + df["rev_cost_R$"]).round(2)

    if hybrid is not None:
        # Bring in hybrid columns; zones not in hybrid remain NaN → sdvrp_solved=False
        hybrid_cols = [
            "zone_id",
            "hybrid_vehicles",
            "hybrid_dist_km",
            "hybrid_cost_R$",
            "saving_R$",
            "saving_pct",
            "sdvrp_solved",
        ]
        hybrid_sub = hybrid[[c for c in hybrid_cols if c in hybrid.columns]]
        df = pd.merge(df, hybrid_sub, on="zone_id", how="left")
        df["sdvrp_solved"] = df["sdvrp_solved"].fillna(False)

        # Recompute saving from our canonical separate cost if hybrid provided
        # its own separate_cost_R$ column, use ours for consistency
        if "hybrid_cost_R$" in df.columns:
            df["saving_R$"] = (df["separate_cost_R$"] - df["hybrid_cost_R$"]).round(2)
            df["saving_pct"] = (
                df["saving_R$"] / df["separate_cost_R$"].replace(0, 1) * 100
            ).round(1)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zone priority ranking
# ---------------------------------------------------------------------------


def zone_priority_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sdvrp_priority_rank column — zones ranked by SDVRP saving opportunity.

    Priority is determined by separate_cost_R$ (descending): the zone with
    the highest combined fwd+rev cost has the most to gain from hybridisation
    and should be the first target for SDVRP deployment.

    If saving_R$ is available (hybrid solved), ranking is re-ordered by
    saving_R$ descending (actual measured saving takes precedence).

    Returns the input DataFrame with sdvrp_priority_rank added (in-place copy).
    """
    df = df.copy()
    if "saving_R$" in df.columns and df["saving_R$"].notna().any():
        rank_col = "saving_R$"
        print("[kpi_reporter] Priority ranking by actual saving_R$ (SDVRP solved)")
    else:
        rank_col = "separate_cost_R$"
        print("[kpi_reporter] Priority ranking by separate_cost_R$ (SDVRP not yet run)")
    df["sdvrp_priority_rank"] = (
        df[rank_col].rank(method="first", ascending=False).astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise combined KPI summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  COMBINED KPI REPORT SUMMARY")
    print(sep)

    total_fwd_cost = df["fwd_cost_R$"].sum()
    total_rev_cost = df["rev_cost_R$"].sum()
    total_separate = df["separate_cost_R$"].sum()
    total_fwd_dist = df["fwd_dist_km"].sum()
    total_rev_dist = df["rev_dist_km"].sum()

    print(f"  Zones            : {len(df)}")
    print(f"  Forward cost     : R$ {total_fwd_cost:,.2f}  ({total_fwd_dist:,.2f} km)")
    print(f"  Reverse cost     : R$ {total_rev_cost:,.2f}  ({total_rev_dist:,.2f} km)")
    print(f"  Separate total   : R$ {total_separate:,.2f}")

    if "saving_R$" in df.columns and df["saving_R$"].notna().any():
        total_saving = df["saving_R$"].dropna().sum()
        avg_pct = df["saving_pct"].dropna().mean()
        print(f"\n  SDVRP hybrid saving : R$ {total_saving:,.2f}  (avg {avg_pct:.1f}%)")
        print(f"  Post-SDVRP total    : R$ {total_separate - total_saving:,.2f}")

    print(
        f"\n  {'Zone':>5}  {'FwdCost':>10}  {'RevCost':>10}  "
        f"{'SepCost':>10}  {'HybCost':>10}  {'Saving%':>8}  {'Rank':>5}"
    )
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")
    for _, row in df.sort_values("zone_id").iterrows():
        hyb = (
            f"R${row['hybrid_cost_R$']:>8.2f}"
            if "hybrid_cost_R$" in df.columns and row.get("sdvrp_solved")
            else f"{'—':>10}"
        )
        sav = (
            f"{row['saving_pct']:>7.1f}%"
            if "saving_pct" in df.columns and row.get("sdvrp_solved")
            else f"{'—':>8}"
        )
        print(
            f"  {int(row['zone_id']):>5}  "
            f"R${row['fwd_cost_R$']:>8.2f}  "
            f"R${row['rev_cost_R$']:>8.2f}  "
            f"R${row['separate_cost_R$']:>8.2f}  "
            f"{hyb}  {sav}  "
            f"{int(row['sdvrp_priority_rank']):>5}"
        )

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    fwd_path: str | Path = DEFAULT_FWD_PATH,
    rev_path: str | Path = DEFAULT_REV_PATH,
    hybrid_path: str | Path = DEFAULT_HYBRID_PATH,
    out_path: str | Path = DEFAULT_OUT_PATH,
) -> pd.DataFrame:
    """
    Full pipeline: load → merge → rank → save → print.

    Returns the combined KPI DataFrame (also saved to out_path).
    """
    fwd = load_forward(fwd_path)
    rev = load_reverse(rev_path)
    hybrid = load_hybrid(hybrid_path)

    df = build_combined(fwd, rev, hybrid)
    df = zone_priority_ranking(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"[kpi_reporter] Saved → {out_path}  ({len(df)} zones, {len(df.columns)} cols)"
    )

    print_summary(df)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run()
    print(result.to_string(index=False))

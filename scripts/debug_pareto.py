#!/usr/bin/env python3
"""
debug_pareto.py — Standalone Pareto analysis debug script
==========================================================
From: Vybhav
Usage:
    uv run python scripts/debug_pareto.py
    uv run python scripts/debug_pareto.py --gamma 5.0 --delta 100.0
    uv run python scripts/debug_pareto.py --fwd outputs/forward_routes.csv \
                                           --rev outputs/reverse_routes.csv

What this script does
---------------------
1. Loads forward + reverse routes (real CSVs or synthetic fallback)
2. Builds return_probs from a Beta(2,15) draw (mean ≈ 0.12)
3. Runs pareto_sweep() with verbose intermediate prints
4. Prints a full breakdown:
   - Per-vehicle cost tables (fwd + rev, cheapest-first)
   - Prefix sums
   - All (k_fwd, k_rev) combinations with C_fwd, C_rev, T_pen, Z
   - Dominance matrix
   - Pareto front + knee point
5. Saves outputs/pareto_results.csv and outputs/pareto_debug.png
"""

import sys
import argparse
from pathlib import Path

# ── Make project root importable regardless of cwd ──────────────────────────
_here = Path(__file__).resolve().parent  # scripts/
_root = _here.parent  # project root
sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.joint_optimizer import DEFAULT_GAMMA, DEFAULT_DELTA


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pareto sweep debug script")
    p.add_argument(
        "--fwd",
        default=str(_root / "outputs/forward_routes.csv"),
        help="Path to forward_routes.csv",
    )
    p.add_argument(
        "--rev",
        default=str(_root / "outputs/reverse_routes.csv"),
        help="Path to reverse_routes.csv",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"T_pen weight (default: {DEFAULT_GAMMA})",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help=f"Per-vehicle cost (default: {DEFAULT_DELTA})",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="RNG seed for synthetic return_probs"
    )
    p.add_argument(
        "--return-mean",
        type=float,
        default=None,
        help="Force a specific return probability mean (overrides Beta draw)",
    )
    p.add_argument(
        "--out",
        default=str(_root / "outputs/pareto_results_debug.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--plot",
        default=str(_root / "outputs/pareto_debug.png"),
        help="Output PNG path",
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _sep(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def load_routes(fwd_path: str, rev_path: str):
    fwd_p, rev_p = Path(fwd_path), Path(rev_path)

    if fwd_p.exists() and rev_p.exists():
        fwd = pd.read_csv(fwd_p)
        rev = pd.read_csv(rev_p)
        print(
            f"[load] forward_routes: {len(fwd)} rows, "
            f"{fwd['vehicle_id'].nunique()} vehicles — {fwd_p}"
        )
        print(
            f"[load] reverse_routes: {len(rev)} rows, "
            f"{rev['vehicle_id'].nunique()} vehicles — {rev_p}"
        )
    else:
        print("[load] ⚠️  Route CSVs not found — using synthetic data")
        # 6 fwd vehicles, 2 rev vehicles (mirrors SP pilot instance)
        fwd = pd.DataFrame(
            {
                "vehicle_id": [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3,
                "cumulative_distance_km": [
                    5,
                    10,
                    15,  # v0 — cheapest
                    8,
                    16,
                    24,  # v1
                    10,
                    20,
                    30,  # v2
                    12,
                    24,
                    36,  # v3
                    15,
                    30,
                    45,  # v4
                    20,
                    40,
                    60,  # v5 — most expensive
                ],
            }
        )
        rev = pd.DataFrame(
            {
                "vehicle_id": [0] * 3 + [1] * 3,
                "cumulative_distance_km": [4, 8, 12, 18, 36, 54],
            }
        )
        print(
            f"[load] synthetic fwd: {fwd['vehicle_id'].nunique()} vehicles  "
            f"rev: {rev['vehicle_id'].nunique()} vehicles"
        )

    return fwd, rev


def build_return_probs(
    fwd: pd.DataFrame, seed: int, forced_mean: float | None
) -> pd.Series:
    n_stops = fwd["vehicle_id"].nunique() * 10
    rng = np.random.default_rng(seed)
    if forced_mean is not None:
        probs = pd.Series(np.full(n_stops, forced_mean))
        print(f"[probs] forced constant mean={forced_mean:.3f}  n={n_stops}")
    else:
        probs = pd.Series(rng.beta(2, 15, size=n_stops).astype(float))
        print(
            f"[probs] Beta(2,15) draw  n={len(probs)}  "
            f"sum={probs.sum():.2f}  mean={probs.mean():.4f}  "
            f"std={probs.std():.4f}"
        )
    return probs


# ────────────────────────────────────────────────────────────────────────────
# Verbose sweep (mirrors pareto_sweep internals, step-by-step)
# ────────────────────────────────────────────────────────────────────────────
def run_debug_sweep(
    fwd: pd.DataFrame,
    rev: pd.DataFrame,
    return_probs: pd.Series,
    gamma: float,
    delta: float,
    out_csv: str,
    out_png: str,
) -> pd.DataFrame:

    # ── 1. Per-vehicle costs ─────────────────────────────────────────────────
    _sep("STEP 1 — Per-vehicle max cumulative distance (route cost proxy)")

    fwd_costs = fwd.groupby("vehicle_id")["cumulative_distance_km"].max().sort_values()
    rev_costs = rev.groupby("vehicle_id")["cumulative_distance_km"].max().sort_values()

    print("\nForward vehicles (cheapest first):")
    for vid, cost in fwd_costs.items():
        print(f"  v{vid:>3}: {cost:.3f} km")

    print("\nReverse vehicles (cheapest first):")
    for vid, cost in rev_costs.items():
        print(f"  v{vid:>3}: {cost:.3f} km")

    fwd_arr = fwd_costs.values
    rev_arr = rev_costs.values
    n_fwd, n_rev = len(fwd_arr), len(rev_arr)

    # ── 2. Prefix sums ───────────────────────────────────────────────────────
    _sep("STEP 2 — Prefix sums  (C_fwd[k] = sum of cheapest k vehicles)")

    fwd_prefix = np.cumsum(fwd_arr)
    rev_prefix = np.cumsum(rev_arr)

    print("\nForward prefix sums:")
    for k, v in enumerate(fwd_prefix, 1):
        print(f"  k_fwd={k}: C_fwd={v:.3f} km")

    print("\nReverse prefix sums:")
    for k, v in enumerate(rev_prefix, 1):
        print(f"  k_rev={k}: C_rev={v:.3f} km")

    # ── 3. expected_returns ──────────────────────────────────────────────────
    _sep("STEP 3 — Expected returns")

    expected_returns = float(return_probs.sum()) if len(return_probs) > 0 else 0.0
    print(f"  sum(return_probs) = {expected_returns:.4f}")
    print(f"  gamma             = {gamma}")
    print(f"  T_pen formula:    gamma × E × (1 - k_rev / n_rev)")
    print(
        f"                  = {gamma} × {expected_returns:.4f} × (1 - k_rev / {n_rev})"
    )

    # ── 4. Enumerate all (k_fwd, k_rev) ─────────────────────────────────────
    _sep(f"STEP 4 — Full enumeration: {n_fwd} fwd × {n_rev} rev = {n_fwd*n_rev} combos")

    rows = []
    header = f"{'k_fwd':>6} {'k_rev':>6} {'C_fwd':>9} {'C_rev':>9} {'T_pen':>9} {'routing':>9} {'N_veh':>6} {'Z':>9}"
    print(header)
    print("-" * len(header))

    for k_fwd in range(1, n_fwd + 1):
        for k_rev in range(1, n_rev + 1):
            c_fwd = float(fwd_prefix[k_fwd - 1])
            c_rev = float(rev_prefix[k_rev - 1])
            t_pen = gamma * expected_returns * (1.0 - k_rev / n_rev)
            n_veh = k_fwd + k_rev
            routing = round(c_fwd + c_rev, 3)
            z = round(routing + t_pen + delta * n_veh, 3)
            print(
                f"{k_fwd:>6} {k_rev:>6} {c_fwd:>9.3f} {c_rev:>9.3f} "
                f"{t_pen:>9.3f} {routing:>9.3f} {n_veh:>6} {z:>9.3f}"
            )
            rows.append(
                {
                    "n_fwd_active": k_fwd,
                    "n_rev_active": k_rev,
                    "gamma": gamma,
                    "delta": delta,
                    "C_fwd": round(c_fwd, 3),
                    "C_rev": round(c_rev, 3),
                    "T_pen": round(t_pen, 6),
                    "N_veh": n_veh,
                    "total_routing_cost": routing,
                    "Z": z,
                }
            )

    df = pd.DataFrame(rows)

    # ── 5. Dominance matrix ──────────────────────────────────────────────────
    _sep("STEP 5 — Dominance check")

    c_arr = df["total_routing_cost"].values.astype(float)
    t_arr = df["T_pen"].values.astype(float)
    n = len(df)
    dominated = np.zeros(n, dtype=bool)

    print(f"\n{'Point':>6} {'routing':>9} {'T_pen':>9}  Dominated by")
    for i in range(n):
        dominators = []
        for j in range(n):
            if i == j:
                continue
            if c_arr[j] <= c_arr[i] and t_arr[j] <= t_arr[i]:
                if c_arr[j] < c_arr[i] or t_arr[j] < t_arr[i]:
                    dominated[i] = True
                    dominators.append(
                        f"(k_fwd={int(df.iloc[j]['n_fwd_active'])}, "
                        f"k_rev={int(df.iloc[j]['n_rev_active'])})"
                    )
        flag = "DOMINATED" if dominated[i] else "PARETO ✅"
        dom_str = ", ".join(dominators[:2]) if dominators else "—"
        print(
            f"({int(df.iloc[i]['n_fwd_active'])},{int(df.iloc[i]['n_rev_active'])})  "
            f"{c_arr[i]:>9.3f} {t_arr[i]:>9.3f}  [{flag}]  {dom_str}"
        )

    df["is_pareto"] = ~dominated

    # ── 6. Knee point ────────────────────────────────────────────────────────
    _sep("STEP 6 — Knee point (min normalised distance to ideal (0,0))")

    c_min, c_max = c_arr.min(), c_arr.max()
    t_min, t_max = t_arr.min(), t_arr.max()
    c_range = c_max - c_min if c_max > c_min else 1.0
    t_range = t_max - t_min if t_max > t_min else 1.0
    c_norm = (c_arr - c_min) / c_range
    t_norm = (t_arr - t_min) / t_range
    df["dist_to_ideal"] = np.sqrt(c_norm**2 + t_norm**2).round(6)
    df["is_knee"] = False

    pareto_idx = df.index[~dominated].tolist()
    if pareto_idx:
        knee_idx = df.loc[pareto_idx, "dist_to_ideal"].idxmin()
        df.loc[knee_idx, "is_knee"] = True
        kr = df.loc[knee_idx]
        print(f"\nKnee point:")
        print(f"  k_fwd={int(kr['n_fwd_active'])}  k_rev={int(kr['n_rev_active'])}")
        print(f"  C_routing = {kr['total_routing_cost']:.3f} km")
        print(f"  T_pen     = {kr['T_pen']:.6f}")
        print(f"  Z         = {kr['Z']:.3f}")
        print(f"  dist_to_ideal = {kr['dist_to_ideal']:.6f}")

    # ── 7. Pareto front summary ──────────────────────────────────────────────
    _sep("STEP 7 — Pareto front summary")

    pareto_df = df[df["is_pareto"]].sort_values("total_routing_cost")
    print(f"\n{len(pareto_df)} Pareto-optimal solution(s):\n")
    print(
        pareto_df[
            [
                "n_fwd_active",
                "n_rev_active",
                "C_fwd",
                "C_rev",
                "T_pen",
                "total_routing_cost",
                "Z",
                "is_knee",
            ]
        ].to_string(index=False)
    )

    # ── 8. Collapse warning ──────────────────────────────────────────────────
    if len(pareto_df) <= 2:
        _sep("⚠️  COLLAPSE NOTICE")
        print(
            f"""
  The Pareto front has only {len(pareto_df)} point(s).

  Possible causes:
  1. n_rev = {n_rev} is very small → T_pen can only take {n_rev} distinct values.
     Solution: use route data with more reverse vehicles.
  2. expected_returns = {expected_returns:.4f}
     If ≈ 0 → T_pen = 0 for all rows → all solutions tie on one objective.
     Solution: ensure return_probs is non-empty (check --return-mean flag).
  3. Forward routing cost is dominated by one vehicle → only k_fwd=1 survives.
     Solution: inspect per-vehicle cost ratios above.

  This is expected for the SP PILOT instance (2 rev vehicles).
  A production dataset will produce a richer Pareto front.
"""
        )

    # ── 9. Save CSV ──────────────────────────────────────────────────────────
    _sep("STEP 8 — Save")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  CSV → {out_csv}")

    # ── 10. Plot ─────────────────────────────────────────────────────────────
    _plot(df, out_png, gamma, expected_returns)
    _sep("DONE")

    return df


def _plot(
    df: pd.DataFrame, out_png: str, gamma: float, expected_returns: float
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Pareto Debug  |  γ={gamma}  E[returns]={expected_returns:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    # ── Left: scatter all points ─────────────────────────────────────────────
    ax = axes[0]
    dominated = df[~df["is_pareto"]]
    pareto = df[df["is_pareto"]].sort_values("total_routing_cost")
    knee = df[df["is_knee"]]

    ax.scatter(
        dominated["total_routing_cost"],
        dominated["T_pen"],
        color="lightgray",
        s=80,
        label="Dominated",
        zorder=2,
    )
    ax.scatter(
        pareto["total_routing_cost"],
        pareto["T_pen"],
        color="#2196F3",
        s=120,
        label="Pareto front",
        zorder=3,
    )
    if len(pareto) > 1:
        ax.plot(
            pareto["total_routing_cost"],
            pareto["T_pen"],
            color="#2196F3",
            lw=1.5,
            ls="--",
            zorder=2,
        )
    if not knee.empty:
        ax.scatter(
            knee["total_routing_cost"],
            knee["T_pen"],
            color="red",
            s=250,
            marker="*",
            label="Knee",
            zorder=5,
        )

    for _, row in df.iterrows():
        ax.annotate(
            f"({int(row['n_fwd_active'])},{int(row['n_rev_active'])})",
            (row["total_routing_cost"], row["T_pen"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color=(
                "#E53935"
                if row["is_knee"]
                else ("#2196F3" if row["is_pareto"] else "#9E9E9E")
            ),
        )

    ax.set_xlabel("Total Routing Cost  C_fwd + C_rev  (km)")
    ax.set_ylabel("Time Penalty  T_pen")
    ax.set_title("Objective Space\n(label = k_fwd, k_rev)")
    ax.legend(fontsize=9)

    # ── Right: heatmap of Z by (k_fwd, k_rev) ────────────────────────────────
    ax2 = axes[1]
    pivot = df.pivot(index="n_rev_active", columns="n_fwd_active", values="Z")
    im = ax2.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r", origin="lower")
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([f"k_fwd={c}" for c in pivot.columns], rotation=45, ha="right")
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels([f"k_rev={r}" for r in pivot.index])
    ax2.set_title("Z heatmap  (darker = better)")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="Z")

    # Annotate Pareto points on heatmap
    for _, row in df[df["is_pareto"]].iterrows():
        xi = list(pivot.columns).index(int(row["n_fwd_active"]))
        yi = list(pivot.index).index(int(row["n_rev_active"]))
        marker = "★" if row["is_knee"] else "●"
        ax2.text(
            xi,
            yi,
            marker,
            ha="center",
            va="center",
            fontsize=14,
            color="blue" if not row["is_knee"] else "red",
        )

    blue_patch = mpatches.Patch(color="blue", label="Pareto ●")
    red_patch = mpatches.Patch(color="red", label="Knee ★")
    ax2.legend(handles=[blue_patch, red_patch], fontsize=9, loc="upper left")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  PNG → {out_png}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    _sep("PARETO ANALYSIS DEBUG SCRIPT")
    print(f"  gamma        = {args.gamma}")
    print(f"  delta        = {args.delta}")
    print(f"  seed         = {args.seed}")
    print(
        f"  return_mean  = {args.return_mean if args.return_mean else 'Beta(2,15) draw'}"
    )
    print(f"  fwd CSV      = {args.fwd}")
    print(f"  rev CSV      = {args.rev}")

    fwd, rev = load_routes(args.fwd, args.rev)
    return_probs = build_return_probs(fwd, args.seed, args.return_mean)

    df = run_debug_sweep(
        fwd=fwd,
        rev=rev,
        return_probs=return_probs,
        gamma=args.gamma,
        delta=args.delta,
        out_csv=args.out,
        out_png=args.plot,
    )

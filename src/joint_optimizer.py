"""
Module: joint_optimizer.py
Stage:  Joint Forward + Reverse Logistics Optimisation (MILP via PuLP)

Objective (minimise):
    Z = α·C_fwd + β·C_rev + γ·T_pen + δ·N_veh

    C_fwd  = total forward variable routing cost (R$)
    C_rev  = total reverse variable routing cost (R$)
    T_pen  = expected return penalty (unserved reverse demand)
    N_veh  = number of vehicles deployed (forward + reverse)

INPUT:
    forward_routes_df  : pd.DataFrame  (output of route_parser.py)
        Columns: vehicle_id, stop_seq, node_idx, node_id, lat, lon,
                 cumulative_distance_km, load_after_stop
    reverse_routes_df  : pd.DataFrame  (output of route_parser.py)
        Same schema.
    return_probs       : pd.Series indexed by order_id  — float32 ∈ [0,1]
        Output of return_classifier.predict_proba()
    alpha, beta, gamma, delta : float — objective weights

OUTPUT:
    outputs/joint_optimizer_result.json
    outputs/pareto_results.csv          — ε-constraint Pareto front
    outputs/pareto_tradeoff.png         — Pareto front visualisation

INTERFACE:
    build_model(...)        -> (prob, decision_vars)
    solve(prob)             -> str
    extract_results(...)    -> dict
    run(...)                -> dict
    solve_sdvrp_hybrid(...) -> dict
    run_all_zones_sdvrp(...)-> pd.DataFrame
    z_sensitivity_sweep(...)-> pd.DataFrame
    pareto_sweep(...)       -> pd.DataFrame
    save_pareto_plot(df, output_path) -> None
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.route_parser import (
    VEHICLE_CAPACITY_G,
    VEHICLE_SPEED_KMH,
    SERVICE_TIME_MIN,
    SOLVER_TIME_LIMIT_S,
    FIXED_COST_PER_ROUTE,
    VAR_COST_PER_KM,
    build_distance_matrix,
    compute_routing_cost,
)


# ---------------------------------------------------------------------------
# Default objective weights
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 1.0  # forward variable-cost weight
DEFAULT_BETA = 0.8  # reverse variable-cost weight
DEFAULT_GAMMA = 2.0  # penalty weight for unserved returns
DEFAULT_DELTA = FIXED_COST_PER_ROUTE  # R$ fixed cost per active vehicle


# ---------------------------------------------------------------------------
# Helper — unique vehicle route costs
# ---------------------------------------------------------------------------


def _unique_vehicle_costs(routes_df: pd.DataFrame) -> pd.Series:
    """
    Return the total distance (km) per unique physical vehicle route.

    Vehicle IDs reset to 0 in each zone, so grouping by vehicle_id alone
    merges vehicles from different zones.  If zone_id is present, we group
    by (zone_id, vehicle_id) and flatten to string keys like "z0_v3".

    Returns
    -------
    pd.Series — index = unique vehicle key (str), values = max cumulative km
                sorted ascending (cheapest routes first).
    """
    if routes_df.empty:
        return pd.Series(dtype=float)

    if "zone_id" in routes_df.columns:
        costs = routes_df.groupby(["zone_id", "vehicle_id"])[
            "cumulative_distance_km"
        ].max()
        costs.index = [f"z{z}_v{v}" for z, v in costs.index]
    else:
        costs = routes_df.groupby("vehicle_id")["cumulative_distance_km"].max()
        costs.index = [f"v{v}" for v in costs.index]

    return costs.sort_values()


# ═══════════════════════════════════════════════════════════════════════════
#  MILP Joint Optimiser
# ═══════════════════════════════════════════════════════════════════════════


def build_model(
    forward_routes_df: pd.DataFrame,
    reverse_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
) -> tuple:
    """
    Construct the PuLP MILP for joint optimisation.

    Decision variables:
        u_v ∈ {0,1}  — vehicle v is active for forward routing
        w_v ∈ {0,1}  — vehicle v is active for reverse routing

    Per-vehicle costs are converted from km to R$ (variable component only)
    so the objective is in consistent monetary units.  The fixed cost per
    vehicle is captured by δ·N_veh.

    Returns
    -------
    (prob, vars_dict)
    """
    prob = pulp.LpProblem("joint_fwd_rev_optimisation", pulp.LpMinimize)

    # Per unique physical vehicle route: variable routing cost in R$
    # (groups by zone_id + vehicle_id to avoid merging across zones)
    fwd_cost_series = _unique_vehicle_costs(forward_routes_df) * VAR_COST_PER_KM
    rev_cost_series = _unique_vehicle_costs(reverse_routes_df) * VAR_COST_PER_KM

    fwd_vehicles = fwd_cost_series.index.tolist()
    rev_vehicles = rev_cost_series.index.tolist()
    fwd_cost = fwd_cost_series.to_dict()
    rev_cost = rev_cost_series.to_dict()

    # Binary activation variables
    u = {v: pulp.LpVariable(f"u_{v}", cat="Binary") for v in fwd_vehicles}
    w = {v: pulp.LpVariable(f"w_{v}", cat="Binary") for v in rev_vehicles}

    # Expected return penalty — decreases as more reverse vehicles are active
    expected_returns = float(return_probs.sum()) if not return_probs.empty else 0.0
    n_rev = max(len(rev_vehicles), 1)
    T_pen_expr = gamma * expected_returns * (1 - pulp.lpSum(w.values()) / n_rev)

    # Objective components
    C_fwd_expr = alpha * pulp.lpSum(fwd_cost[v] * u[v] for v in fwd_vehicles)
    C_rev_expr = beta * pulp.lpSum(rev_cost[v] * w[v] for v in rev_vehicles)
    N_veh_expr = delta * (pulp.lpSum(u.values()) + pulp.lpSum(w.values()))

    prob += C_fwd_expr + C_rev_expr + T_pen_expr + N_veh_expr, "total_cost"

    # Constraints: at least one active vehicle of each type (if routes exist)
    if fwd_vehicles:
        prob += pulp.lpSum(u.values()) >= 1, "min_one_fwd_vehicle"
    if rev_vehicles:
        prob += pulp.lpSum(w.values()) >= 1, "min_one_rev_vehicle"

    return prob, {
        "u": u,
        "w": w,
        "fwd_cost": fwd_cost,
        "rev_cost": rev_cost,
        "expected_returns": expected_returns,
        "gamma": gamma,
        "n_rev": n_rev,
    }


def solve(prob: pulp.LpProblem, time_limit_s: int = 60) -> str:
    """Solve the MILP; returns status string."""
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit_s)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def extract_results(
    prob: pulp.LpProblem,
    vars_dict: dict,
    forward_routes_df: pd.DataFrame,
    reverse_routes_df: pd.DataFrame,
) -> dict:
    """Parse solved MILP into a result dict."""
    u, w = vars_dict["u"], vars_dict["w"]
    fwd_cost, rev_cost = vars_dict["fwd_cost"], vars_dict["rev_cost"]

    active_fwd = [
        v for v, var in u.items() if pulp.value(var) and pulp.value(var) > 0.5
    ]
    active_rev = [
        v for v, var in w.items() if pulp.value(var) and pulp.value(var) > 0.5
    ]

    C_fwd = sum(fwd_cost.get(v, 0) for v in active_fwd)
    C_rev = sum(rev_cost.get(v, 0) for v in active_rev)
    N_veh = len(active_fwd) + len(active_rev)
    Z = pulp.value(prob.objective) or 0.0

    assignments = pd.DataFrame(
        [{"vehicle_id": v, "role": "forward", "active": v in active_fwd} for v in u]
        + [{"vehicle_id": v, "role": "reverse", "active": v in active_rev} for v in w]
    )

    gamma = vars_dict.get("gamma", DEFAULT_GAMMA)
    n_rev = vars_dict.get("n_rev", max(len(w), 1))
    T_pen = (
        gamma * vars_dict["expected_returns"] * (1.0 - len(active_rev) / max(n_rev, 1))
    )

    return {
        "Z": round(Z, 3),
        "C_fwd": round(C_fwd, 3),
        "C_rev": round(C_rev, 3),
        "T_pen": round(T_pen, 3),
        "N_veh": N_veh,
        "status": pulp.LpStatus[prob.status],
        "vehicle_assignments": assignments,
    }


def run(
    forward_routes_df: pd.DataFrame,
    reverse_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    output_path: str | Path = "outputs/joint_optimizer_result.json",
) -> dict:
    """Convenience: build → solve → extract → save result."""
    prob, vars_dict = build_model(
        forward_routes_df, reverse_routes_df, return_probs, alpha, beta, gamma, delta
    )
    status = solve(prob)
    result = extract_results(prob, vars_dict, forward_routes_df, reverse_routes_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in result.items() if k != "vehicle_assignments"}
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[INFO] Joint optimiser result → {output_path}")
    print(f"       Status={status}  Z={result['Z']}  Nveh={result['N_veh']}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  SDVRP Hybrid Solver
# ═══════════════════════════════════════════════════════════════════════════


def solve_sdvrp_hybrid(
    zone_id: int,
    fwd_zone: dict,
    rev_zone: dict,
    num_vehicles: int = 5,
    separate_cost_r: float | None = None,
    output_path: str | Path = "outputs/sdvrp_zone8_result.json",
) -> dict:
    """
    Simultaneous Delivery and Pickup VRP (SDVRP) for one zone.

    Merges forward delivery nodes and reverse pickup nodes into a single
    OR-Tools CVRPTW solve so vehicles can deliver and collect on the same trip.

    Capacity model — single "Load" dimension (correct SDVRP invariant):
        load(t) = initial_delivery_load − cumul_delivered(t) + cumul_picked(t)
        transit[i] = pickup_weight[i] − delivery_weight[i]
        fix_start_cumul_to_zero=False → OR-Tools sets start = total_delivery_wt
        Constraint: 0 ≤ Load_cumul[i] ≤ VEHICLE_CAPACITY_G for all nodes
    """
    # ---- 1. Build combined node list ---------------------------------- #
    n_del = len(fwd_zone["node_coords"]) - 1
    n_pick = len(rev_zone["node_coords"]) - 1

    depot_coords = fwd_zone["node_coords"][[0]]
    del_coords = fwd_zone["node_coords"][1:]
    pick_coords = rev_zone["node_coords"][1:]
    node_coords = np.vstack([depot_coords, del_coords, pick_coords])

    del_weights = fwd_zone["demands"][1:].tolist()
    pick_weights = rev_zone["demands"][1:].tolist()

    del_demand_arr = [0] + del_weights + [0] * n_pick
    pick_demand_arr = [0] + [0] * n_del + pick_weights

    del_tw = fwd_zone["time_windows"][1:]
    pick_tw = rev_zone["time_windows"][1:]
    all_tw = [[0, 1440]] + del_tw + pick_tw
    n_nodes = 1 + n_del + n_pick

    # ---- 2. Distance & time matrices ---------------------------------- #
    dist_matrix = build_distance_matrix(node_coords)
    speed_m_per_min = VEHICLE_SPEED_KMH * 1000 / 60
    time_matrix = np.rint(dist_matrix / speed_m_per_min).astype(int)

    # ---- 3. OR-Tools model -------------------------------------------- #
    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def time_cb(i, j):
        ni = manager.IndexToNode(i)
        return int(time_matrix[ni][manager.IndexToNode(j)]) + (
            SERVICE_TIME_MIN if ni != 0 else 0
        )

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 60, 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node_idx, (open_t, close_t) in enumerate(all_tw):
        time_dim.CumulVar(manager.NodeToIndex(node_idx)).SetRange(open_t, close_t)

    # ---- 4. Single-dimension SDVRP load model ------------------------- #
    def load_transit_cb(i):
        node = manager.IndexToNode(i)
        return int(pick_demand_arr[node]) - int(del_demand_arr[node])

    load_idx = routing.RegisterUnaryTransitCallback(load_transit_cb)
    routing.AddDimensionWithVehicleCapacity(
        load_idx, 0, [VEHICLE_CAPACITY_G] * num_vehicles, False, "Load"
    )
    load_dim = routing.GetDimensionOrDie("Load")
    for node in range(n_nodes):
        idx = manager.NodeToIndex(node)
        if idx >= 0:
            load_dim.CumulVar(idx).SetMin(0)

    # Soft disjunction — large penalty for dropped nodes
    penalty = 100_000
    for node in range(1, n_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # ---- 5. Solve ----------------------------------------------------- #
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    )
    params.time_limit.seconds = SOLVER_TIME_LIMIT_S

    assignment = routing.SolveWithParameters(params)

    if assignment is None:
        print(f"  [SDVRP] Zone {zone_id}: NO SOLUTION FOUND")
        return {"zone_id": zone_id, "solved": False, "routes": []}

    # ---- 6. Extract routes + KPIs ------------------------------------- #
    depot_id = fwd_zone.get("node_ids", ["depot"])[0]
    del_ids = list(fwd_zone.get("node_ids", [])[1:]) or [
        str(i) for i in range(1, n_del + 1)
    ]
    pick_ids = list(rev_zone.get("node_ids", [])[1:]) or [
        str(i) for i in range(1, n_pick + 1)
    ]
    all_node_ids = [depot_id] + del_ids + pick_ids
    node_types = ["depot"] + ["delivery"] * n_del + ["pickup"] * n_pick

    total_dist_m = 0.0
    n_veh_used = 0
    route_records: list[dict] = []

    for v in range(num_vehicles):
        idx = routing.Start(v)
        cum_dist_km = 0.0
        stop_seq = 0
        veh_stops: list[dict] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            nxt = assignment.Value(routing.NextVar(idx))
            veh_stops.append(
                {
                    "vehicle_id": v,
                    "stop_seq": stop_seq,
                    "node_idx": node,
                    "node_id": all_node_ids[node],
                    "lat": float(node_coords[node][0]),
                    "lon": float(node_coords[node][1]),
                    "node_type": node_types[node],
                    "cumulative_distance_km": round(cum_dist_km, 3),
                    "zone_id": zone_id,
                }
            )
            cum_dist_km += dist_matrix[node][manager.IndexToNode(nxt)] / 1000.0
            stop_seq += 1
            idx = nxt
        if cum_dist_km > 0:
            total_dist_m += cum_dist_km * 1000.0
            n_veh_used += 1
            route_records.extend(veh_stops)

    total_dist_km = total_dist_m / 1000.0
    hybrid_cost = compute_routing_cost(n_veh_used, total_dist_km)

    saving_r = round(separate_cost_r - hybrid_cost, 2) if separate_cost_r else None
    saving_pct = (
        round(saving_r / separate_cost_r * 100, 1)
        if saving_r and separate_cost_r
        else None
    )

    result = {
        "zone_id": zone_id,
        "solved": True,
        "n_deliveries": n_del,
        "n_pickups": n_pick,
        "total_dist_km": round(total_dist_km, 2),
        "n_vehicles": n_veh_used,
        "hybrid_cost_R$": round(hybrid_cost, 2),
        "separate_cost_R$": round(separate_cost_r, 2) if separate_cost_r else None,
        "saving_R$": saving_r,
        "saving_pct": saving_pct,
        "strategy": "PATH_CHEAPEST_ARC + SIMULATED_ANNEALING",
        "routes": route_records,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in result.items() if k != "routes"}
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    sep_str = (
        f"vs R${separate_cost_r:.2f} separate → {saving_pct:.1f}% saving"
        if saving_r
        else ""
    )
    print(
        f"[SDVRP] Zone {zone_id}: {n_veh_used} vehicles | "
        f"{total_dist_km:.2f} km | R${hybrid_cost:.2f}  {sep_str}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  All-Zone SDVRP Runner
# ═══════════════════════════════════════════════════════════════════════════


def run_all_zones_sdvrp(
    fwd_zones: list[dict] | dict,
    rev_zones: list[dict] | dict,
    fwd_kpi_df: pd.DataFrame,
    rev_kpi_df: pd.DataFrame,
    num_vehicles: int = 5,
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """
    Run solve_sdvrp_hybrid for every zone and write outputs.

    Accepts zone data as either a list of zone dicts (from build_vrp_nodes)
    or a dict keyed by zone_id.  Lists are converted to dicts internally.

    Writes
    ------
    <output_dir>/hybrid_routes.json
    <output_dir>/hybrid_kpi_summary.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Normalise to dict keyed by zone_id ──────────────────────────────
    if isinstance(fwd_zones, list):
        fwd_zones = {z["zone_id"]: z for z in fwd_zones}
    if isinstance(rev_zones, list):
        rev_zones = {z["zone_id"]: z for z in rev_zones}

    fwd_cost_map = fwd_kpi_df.set_index("zone_id")["routing_cost_R$"].to_dict()
    rev_cost_map = rev_kpi_df.set_index("zone_id")["routing_cost_R$"].to_dict()

    zone_ids = sorted(set(fwd_zones.keys()) & set(rev_zones.keys()))
    print(f"[SDVRP-all] Running {len(zone_ids)} zones: {zone_ids}")

    all_zone_routes: list[dict] = []
    kpi_rows: list[dict] = []

    for zone_id in zone_ids:
        fwd_zone = fwd_zones[zone_id]
        rev_zone = rev_zones[zone_id]
        if len(rev_zone.get("node_coords", [])) <= 1:
            print(f"  [SDVRP-all] Zone {zone_id}: no reverse nodes, skipping")
            continue

        separate_cost = fwd_cost_map.get(zone_id, 0.0) + rev_cost_map.get(zone_id, 0.0)
        zone_result = solve_sdvrp_hybrid(
            zone_id=zone_id,
            fwd_zone=fwd_zone,
            rev_zone=rev_zone,
            num_vehicles=num_vehicles,
            separate_cost_r=separate_cost if separate_cost > 0 else None,
            output_path=out / f"sdvrp_zone{zone_id}_result.json",
        )
        if not zone_result.get("solved"):
            continue

        zone_routes = zone_result.get("routes", [])
        veh_kms: dict[int, float] = {}
        for rec in zone_routes:
            veh_kms[rec["vehicle_id"]] = rec["cumulative_distance_km"]

        all_zone_routes.append(
            {
                "zone_id": zone_id,
                "n_vehicles": zone_result["n_vehicles"],
                "total_dist_km": zone_result["total_dist_km"],
                "routing_cost_R$": zone_result["hybrid_cost_R$"],
                "routes": zone_routes,
            }
        )
        kpi_rows.append(
            {
                "zone_id": zone_id,
                "n_deliveries": zone_result["n_deliveries"],
                "n_pickups": zone_result["n_pickups"],
                "n_vehicles_used": zone_result["n_vehicles"],
                "total_dist_km": zone_result["total_dist_km"],
                "routing_cost_R$": zone_result["hybrid_cost_R$"],
                "separate_cost_R$": zone_result.get("separate_cost_R$"),
                "saving_R$": zone_result.get("saving_R$"),
                "saving_pct": zone_result.get("saving_pct"),
                "max_route_km": max(veh_kms.values()) if veh_kms else 0.0,
                "min_route_km": min(veh_kms.values()) if veh_kms else 0.0,
            }
        )

    routes_path = out / "hybrid_routes.json"
    with open(routes_path, "w") as f:
        json.dump(all_zone_routes, f, indent=2)
    print(
        f"[SDVRP-all] hybrid_routes.json → {routes_path}  "
        f"({len(all_zone_routes)} zones)"
    )

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_path = out / "hybrid_kpi_summary.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"[SDVRP-all] hybrid_kpi_summary.csv → {kpi_path}")

    if "saving_R$" in kpi_df.columns:
        total_saving = kpi_df["saving_R$"].dropna().sum()
        print(f"[SDVRP-all] Total fleet saving vs separate: R${total_saving:.2f}")

    return kpi_df


# ═══════════════════════════════════════════════════════════════════════════
#  Z Weight Sensitivity Sweep
# ═══════════════════════════════════════════════════════════════════════════


def z_sensitivity_sweep(
    fwd_routes_df: pd.DataFrame,
    rev_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    alpha_grid: list[float] | None = None,
    beta_grid: list[float] | None = None,
    output_path: str | Path = "outputs/z_sensitivity.csv",
) -> pd.DataFrame:
    """
    Grid-search over (alpha, beta) with gamma=delta=(1-alpha-beta)/2.

    Iterates every (alpha, beta) pair where alpha + beta <= 0.9.
    For each valid pair: gamma = delta = (1 - alpha - beta) / 2.
    """
    if alpha_grid is None:
        alpha_grid = [round(i * 0.1, 1) for i in range(1, 9)]
    if beta_grid is None:
        beta_grid = alpha_grid

    combos = [(a, b) for a in alpha_grid for b in beta_grid if round(a + b, 10) <= 0.9]
    print(f"[Z-sweep] Running {len(combos)} (alpha, beta) combinations...")

    rows = []
    for i, (alpha, beta) in enumerate(combos):
        gamma = delta = round((1.0 - alpha - beta) / 2.0, 6)
        prob, vars_dict = build_model(
            fwd_routes_df,
            rev_routes_df,
            return_probs,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        solve(prob, time_limit_s=30)
        res = extract_results(prob, vars_dict, fwd_routes_df, rev_routes_df)
        rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "Z": res["Z"],
                "C_fwd": res["C_fwd"],
                "C_rev": res["C_rev"],
                "T_pen": res["T_pen"],
                "N_veh": res["N_veh"],
                "status": res["status"],
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(combos)} done")

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Z-sweep] Saved {len(df)} rows → {output_path}")

    best = df.loc[df["Z"].idxmin()]
    print(
        f"[Z-sweep] Lowest Z={best['Z']:.3f} at "
        f"α={best['alpha']} β={best['beta']} "
        f"γ={best['gamma']} δ={best['delta']}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Pareto Sweep — ε-constraint enumeration
# ═══════════════════════════════════════════════════════════════════════════


def save_pareto_plot(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/pareto_tradeoff.png",
) -> None:
    """
    Generate the Pareto front scatter + line plot.

    Axes:
        x = Total routing cost (R$)   — includes variable + fixed vehicle costs
        y = T_pen (return penalty R$)  — decreases as more reverse vehicles active
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    non_pareto = df[~df["is_pareto"]]
    pareto = df[df["is_pareto"]].sort_values("total_routing_cost")
    knee = df[df["is_knee"]]

    ax.scatter(
        non_pareto["total_routing_cost"],
        non_pareto["T_pen"],
        alpha=0.25,
        s=25,
        color="gray",
        label="Dominated",
    )
    ax.plot(
        pareto["total_routing_cost"],
        pareto["T_pen"],
        "o-",
        color="steelblue",
        markersize=8,
        linewidth=2,
        label="Pareto front",
    )
    if not knee.empty:
        ax.scatter(
            knee["total_routing_cost"],
            knee["T_pen"],
            s=250,
            marker="*",
            color="crimson",
            zorder=10,
            label="Knee point",
        )
        kr = knee.iloc[0]
        ax.annotate(
            f"  Knee: R${kr['total_routing_cost']:.0f} / T={kr['T_pen']:.1f}",
            xy=(kr["total_routing_cost"], kr["T_pen"]),
            fontsize=9,
            fontweight="bold",
            color="crimson",
        )

    ax.set_xlabel("Total Routing Cost (R$)", fontsize=12)
    ax.set_ylabel("Return Penalty  T_pen (R$)", fontsize=12)
    ax.set_title(
        "Pareto Front — Routing Cost vs Return Penalty\n"
        "(ε-constraint enumeration over active vehicle counts)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Pareto] Plot saved → {output_path}")


def pareto_sweep(
    fwd_routes_df: pd.DataFrame,
    rev_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    output_path: str | Path = "outputs/pareto_results.csv",
    plot_path: str | Path = "outputs/pareto_tradeoff.png",
) -> pd.DataFrame:
    """
    Pareto sweep using **ε-constraint enumeration**.

    Enumerates all (k_fwd, k_rev) combinations — how many forward and
    reverse vehicles to activate.  For each combination the cheapest k
    vehicles of each type are selected (optimal for fixed-route selection).
    Costs are computed analytically (no MILP per point).

    Two conflicting objectives:
        Obj 1 (min): total_routing_cost = variable_R$ + fixed_cost × N_veh
        Obj 2 (min): T_pen = γ × expected_returns × (1 − k_rev / n_rev)

    All monetary values are in R$ for consistent units.

    Returns
    -------
    pd.DataFrame with columns:
        n_fwd_active, n_rev_active, gamma, delta,
        C_fwd, C_rev, T_pen, N_veh, total_routing_cost,
        Z, dist_to_ideal, is_pareto, is_knee
    """
    # ── Per unique vehicle route: variable costs (R$, cheapest-first) ──
    fwd_cost_series = _unique_vehicle_costs(fwd_routes_df)  # km, sorted
    rev_cost_series = _unique_vehicle_costs(rev_routes_df)

    # Convert km → R$ variable cost
    fwd_costs = (fwd_cost_series * VAR_COST_PER_KM).values
    rev_costs = (rev_cost_series * VAR_COST_PER_KM).values
    n_fwd = len(fwd_costs)
    n_rev = len(rev_costs)

    if n_fwd == 0 or n_rev == 0:
        raise ValueError("[Pareto] forward or reverse routes DataFrame is empty.")

    expected_returns = float(return_probs.sum()) if len(return_probs) > 0 else 0.0

    # Prefix sums for cheapest-k variable costs
    fwd_prefix = np.cumsum(fwd_costs)
    rev_prefix = np.cumsum(rev_costs)

    print(
        f"[Pareto] ε-constraint enumeration: {n_fwd} fwd × {n_rev} rev "
        f"= {n_fwd * n_rev} combinations"
    )

    rows = []
    for k_fwd in range(1, n_fwd + 1):
        for k_rev in range(1, n_rev + 1):
            c_fwd_var = float(fwd_prefix[k_fwd - 1])
            c_rev_var = float(rev_prefix[k_rev - 1])
            n_veh = k_fwd + k_rev
            fixed = delta * n_veh
            t_pen = gamma * expected_returns * (1.0 - k_rev / n_rev)
            routing = round(c_fwd_var + c_rev_var + fixed, 3)
            z = round(routing + t_pen, 3)
            rows.append(
                {
                    "n_fwd_active": k_fwd,
                    "n_rev_active": k_rev,
                    "gamma": gamma,
                    "delta": delta,
                    "C_fwd": round(c_fwd_var, 3),
                    "C_rev": round(c_rev_var, 3),
                    "T_pen": round(t_pen, 3),
                    "N_veh": n_veh,
                    "total_routing_cost": routing,
                    "Z": z,
                }
            )

    df = pd.DataFrame(rows)

    # ── Pareto front: non-dominated on (total_routing_cost, T_pen) ──────
    def _is_pareto(costs: np.ndarray, tpens: np.ndarray) -> np.ndarray:
        n = len(costs)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if costs[j] <= costs[i] and tpens[j] <= tpens[i]:
                    if costs[j] < costs[i] or tpens[j] < tpens[i]:
                        dominated[i] = True
                        break
        return ~dominated

    c_arr = df["total_routing_cost"].values.astype(float)
    t_arr = df["T_pen"].values.astype(float)
    pareto_mask = _is_pareto(c_arr, t_arr)
    df["is_pareto"] = pareto_mask

    # ── Knee point (closest normalised distance to ideal) ────────────────
    c_min, c_max = c_arr.min(), c_arr.max()
    t_min, t_max = t_arr.min(), t_arr.max()
    c_range = c_max - c_min if c_max > c_min else 1.0
    t_range = t_max - t_min if t_max > t_min else 1.0
    c_norm = (c_arr - c_min) / c_range
    t_norm = (t_arr - t_min) / t_range
    df["dist_to_ideal"] = np.sqrt(c_norm**2 + t_norm**2).round(6)
    df["is_knee"] = False

    pareto_indices = df.index[pareto_mask].tolist()
    if pareto_indices:
        knee_idx = df.loc[pareto_indices, "dist_to_ideal"].idxmin()
        df.loc[knee_idx, "is_knee"] = True
        kr = df.loc[knee_idx]
        print(
            f"[Pareto] Knee: k_fwd={int(kr['n_fwd_active'])} "
            f"k_rev={int(kr['n_rev_active'])}  "
            f"RoutingCost=R${kr['total_routing_cost']:.2f}  "
            f"T_pen={kr['T_pen']:.3f}"
        )

    n_pareto = int(pareto_mask.sum())
    print(f"[Pareto] {n_pareto} Pareto-optimal solutions out of {len(df)} total")

    # ── Save CSV ─────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Pareto] Saved → {output_path}")

    # ── Save plot ────────────────────────────────────────────────────────
    try:
        save_pareto_plot(df, output_path=plot_path)
    except Exception as e:
        print(f"[Pareto] WARNING: Plot generation failed — {e}")
        print("         CSV was saved successfully; plot can be generated later.")

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Try loading real pipeline outputs; fall back to synthetic ────────
    fwd_path = Path("outputs/forward_routes.csv")
    rev_path = Path("outputs/reverse_routes.csv")
    master_path = Path("data/master_df_v3.parquet")

    if fwd_path.exists() and rev_path.exists() and master_path.exists():
        fwd = pd.read_csv(fwd_path)
        rev = pd.read_csv(rev_path)
        master = pd.read_parquet(master_path)
        probs = master.loc[master["return_flag"] == 1, "return_prob"]
    else:
        print("[joint_optimizer] Real data not found — using synthetic smoke test")
        fwd = pd.DataFrame(
            {
                "vehicle_id": [0, 0, 1, 1, 0, 0, 1, 1],
                "zone_id": [0, 0, 0, 0, 1, 1, 1, 1],
                "cumulative_distance_km": [10, 20, 5, 15, 8, 18, 6, 12],
            }
        )
        rev = pd.DataFrame(
            {
                "vehicle_id": [0, 0, 0, 0],
                "zone_id": [0, 0, 1, 1],
                "cumulative_distance_km": [8, 16, 5, 10],
            }
        )
        probs = pd.Series([0.3, 0.7, 0.1])

    # ── 1. Joint MILP optimiser ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  JOINT OPTIMISER — MILP")
    print("=" * 60)
    result = run(fwd, rev, probs)
    print(result)

    # ── 2. Pareto sweep ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PARETO SWEEP — ε-constraint enumeration")
    print("=" * 60)
    pareto_df = pareto_sweep(fwd, rev, probs)

    print("\n" + "=" * 60)
    print("  JOINT OPTIMIZER COMPLETE")
    print("=" * 60)

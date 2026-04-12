"""
Module: report_builder.py
Stage:  Day 7 — Full Project Report (LaTeX → PDF)

Generates a 10–12 page technical report covering all project stages:
    Day 1–2  : Environment, architecture, Haversine distance matrix
    Day 3    : Forward VRP CVRPTW (11 zones, 98.58% improvement)
    Day 4    : Return classifier (ROC-AUC 0.897), Reverse VRP, Joint MILP
    Day 5    : SDVRP hybrid load model (corrected), all-zone runner, Z sweep
    Day 6    : Combined KPI report, Pareto sweep, zone priority ranking

OUTPUT
------
    report/report_draft_v1.tex  — LaTeX source
    report/report_draft_v1.pdf  — Compiled PDF (requires pdflatex)

INTERFACE
---------
    load_kpis()               -> dict   # loads all output CSVs / JSONs
    build_latex(kpis, figures_dir) -> str   # assembles .tex string
    compile_pdf(tex_path)     -> Path   # runs pdflatex twice (for references)
    run(output_dir, figures_dir) -> Path  # convenience: build + compile
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("report")
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_DATA_DIR = Path("data")

# Figures that may or may not exist — included only if present
FIGURE_CANDIDATES = {
    "elbow_silhouette": "elbow_silhouette.png",
    "kmeans_cluster_map": "kmeans_cluster_map.png",
    "zone_kpi_bars": "day3_zone_kpi_bars.png",
    "combined_cost": "combined_cost_by_zone.png",
    "pareto_tradeoff": "pareto_tradeoff.png",
    "sdvrp_priority": "sdvrp_priority_ranking.png",
    "z_breakdown": "joint_optimizer_z_breakdown.png",
}

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_kpis(
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict:
    """
    Load all key result files into a dict of DataFrames and plain values.

    Keys
    ----
    fwd_kpi       : pd.DataFrame — from forward_kpi_summary.csv
    rev_kpi       : pd.DataFrame — from reverse_kpi_summary.csv
    combined_kpi  : pd.DataFrame — from combined_kpi_report.csv (or None)
    pareto_df     : pd.DataFrame — from pareto_results.csv (or None)
    clf_metrics   : dict         — from return_classifier_metrics.json
    joint_result  : dict         — from joint_optimizer_result.json
    baseline      : pd.DataFrame — from baseline_vs_optimised.csv (or None)
    dark_stores   : pd.DataFrame — from dark_stores_final.csv
    """
    out = Path(outputs_dir)
    dat = Path(data_dir)

    def _load_csv(p: Path):
        return pd.read_csv(p) if p.exists() else None

    def _load_json(p: Path):
        return json.loads(p.read_text()) if p.exists() else {}

    kpis = {
        "fwd_kpi": _load_csv(out / "forward_kpi_summary.csv"),
        "rev_kpi": _load_csv(out / "reverse_kpi_summary.csv"),
        "combined_kpi": _load_csv(out / "combined_kpi_report.csv"),
        "pareto_df": _load_csv(out / "pareto_results.csv"),
        "clf_metrics": _load_json(out / "return_classifier_metrics.json"),
        "joint_result": _load_json(out / "joint_optimizer_result.json"),
        "baseline": _load_csv(out / "baseline_vs_optimised.csv"),
        "dark_stores": _load_csv(dat / "dark_stores_final.csv"),
    }
    print("[report_builder] KPIs loaded:")
    for k, v in kpis.items():
        if isinstance(v, pd.DataFrame) and v is not None:
            print(f"  {k}: {len(v)} rows")
        elif isinstance(v, dict):
            print(f"  {k}: {list(v.keys())}")
    return kpis


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def _tex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _fwd_kpi_table(fwd: pd.DataFrame) -> str:
    """Generate LaTeX tabular for forward KPI summary."""
    rows = []
    for _, r in fwd.iterrows():
        rows.append(
            f"  {int(r['zone_id'])} & {int(r['n_customers'])} & "
            f"{int(r['n_vehicles_used'])} & {r['total_dist_km']:.2f} & "
            f"R\\${r['routing_cost_R$']:.2f} \\\\"
        )
    body = "\n".join(rows)
    total_dist = fwd["total_dist_km"].sum()
    total_cost = fwd["routing_cost_R$"].sum()
    total_veh = fwd["n_vehicles_used"].sum()
    total_cust = fwd["n_customers"].sum()
    return textwrap.dedent(
        rf"""
        \begin{{tabular}}{{rrrrc}}
        \toprule
        \textbf{{Zone}} & \textbf{{Orders}} & \textbf{{Vehicles}} &
        \textbf{{Dist (km)}} & \textbf{{Cost (R\$)}} \\
        \midrule
        {body}
        \midrule
        \textbf{{Total}} & \textbf{{{int(total_cust)}}} &
        \textbf{{{int(total_veh)}}} & \textbf{{{total_dist:.2f}}} &
        \textbf{{R\${total_cost:.2f}}} \\
        \bottomrule
        \end{{tabular}}
        """
    ).strip()


def _rev_kpi_table(rev: pd.DataFrame) -> str:
    """Generate LaTeX tabular for reverse KPI summary."""
    rows = []
    for _, r in rev.iterrows():
        rows.append(
            f"  {int(r['zone_id'])} & {int(r['n_pickups'])} & "
            f"{int(r['n_vehicles_used'])} & {r['total_dist_km']:.2f} & "
            f"R\\${r['routing_cost_R$']:.2f} \\\\"
        )
    body = "\n".join(rows)
    total_dist = rev["total_dist_km"].sum()
    total_cost = rev["routing_cost_R$"].sum()
    total_veh = rev["n_vehicles_used"].sum()
    total_pick = rev["n_pickups"].sum()
    return textwrap.dedent(
        rf"""
        \begin{{tabular}}{{rrrrc}}
        \toprule
        \textbf{{Zone}} & \textbf{{Pickups}} & \textbf{{Vehicles}} &
        \textbf{{Dist (km)}} & \textbf{{Cost (R\$)}} \\
        \midrule
        {body}
        \midrule
        \textbf{{Total}} & \textbf{{{int(total_pick)}}} &
        \textbf{{{int(total_veh)}}} & \textbf{{{total_dist:.2f}}} &
        \textbf{{R\${total_cost:.2f}}} \\
        \bottomrule
        \end{{tabular}}
        """
    ).strip()


def _fig_block(rel_path: str, caption: str, label: str) -> str:
    """Return a LaTeX figure block."""
    return textwrap.dedent(
        rf"""
        \begin{{figure}}[H]
          \centering
          \includegraphics[width=0.9\linewidth]{{{rel_path}}}
          \caption{{{caption}}}
          \label{{fig:{label}}}
        \end{{figure}}
        """
    ).strip()


# ---------------------------------------------------------------------------
# Main LaTeX builder
# ---------------------------------------------------------------------------


def build_latex(
    kpis: dict,
    figures_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> str:
    """
    Assemble the full LaTeX source string.

    Parameters
    ----------
    kpis        : dict from load_kpis()
    figures_dir : directory containing output PNG files

    Returns
    -------
    str — complete .tex document
    """
    figs = Path(figures_dir)
    fwd = kpis.get("fwd_kpi")
    rev = kpis.get("rev_kpi")
    clf = kpis.get("clf_metrics", {})
    jopt = kpis.get("joint_result", {})
    combined = kpis.get("combined_kpi")

    # Aggregate scalars
    fwd_total_km = fwd["total_dist_km"].sum() if fwd is not None else 0.0
    fwd_total_cost = fwd["routing_cost_R$"].sum() if fwd is not None else 0.0
    fwd_total_veh = fwd["n_vehicles_used"].sum() if fwd is not None else 0
    rev_total_km = rev["total_dist_km"].sum() if rev is not None else 0.0
    rev_total_cost = rev["routing_cost_R$"].sum() if rev is not None else 0.0
    rev_total_veh = rev["n_vehicles_used"].sum() if rev is not None else 0
    sep_total = fwd_total_cost + rev_total_cost

    roc_auc = clf.get("roc_auc", "N/A")
    pr_auc = clf.get("pr_auc", "N/A")
    brier = clf.get("brier_score", "N/A")
    t30 = clf.get("threshold_0.3", {})
    clf_prec = t30.get("precision", "N/A")
    clf_rec = t30.get("recall", "N/A")
    clf_f1 = t30.get("f1", "N/A")

    z_val = jopt.get("Z", "N/A")
    z_status = jopt.get("status", "N/A")
    z_tpen = jopt.get("T_pen", "N/A")
    z_nveh = jopt.get("N_veh", "N/A")

    # Optional figure blocks
    def _maybe_fig(key: str, caption: str, label: str) -> str:
        fname = FIGURE_CANDIDATES.get(key, "")
        fpath = figs / fname
        if fpath.exists():
            rel = str(fpath.resolve()).replace("\\", "/")
            return _fig_block(rel, caption, label)
        return f"% Figure {key} not yet generated — run notebook 06"

    fwd_table = (
        _fwd_kpi_table(fwd) if fwd is not None else "(forward KPI data not found)"
    )
    rev_table = (
        _rev_kpi_table(rev) if rev is not None else "(reverse KPI data not found)"
    )

    # ── Document ──────────────────────────────────────────────────────────────
    tex = rf"""% ============================================================
% Dark Store Placement + Integrated Forward‑Reverse Logistics
% Project Report — Day 7 Draft
% Generated by src/report_builder.py
% ============================================================
\documentclass[11pt,a4paper]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=2.4cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{hyperref}}
\usepackage{{setspace}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{xcolor}}
\usepackage{{parskip}}
\usepackage{{microtype}}

\definecolor{{darkblue}}{{HTML}}{{1565C0}}
\hypersetup{{colorlinks=true, linkcolor=darkblue, urlcolor=darkblue, citecolor=darkblue}}

\setstretch{{1.15}}

% ── Title ──────────────────────────────────────────────────────────────────
\title{{
  \textbf{{Dark Store Placement \&\\
  Integrated Forward--Reverse Logistics Optimisation}}\\[0.4em]
  \large Project Report --- PGDBA / ISI Kolkata / IIM Calcutta / IIT Kharagpur\\[0.2em]
  \normalsize Operations Management $\cdot$ Supply Chain Analytics
}}
\author{{
  Pritam (Lead Coder \& Integration Architect) \\
  Sneha $\cdot$ Vybhav $\cdot$ Pranav $\cdot$ Team\\[0.2em]
  \small Dataset: Olist Brazilian E-Commerce (Kaggle)
}}
\date{{April 2026}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

% ===========================================================================
\section{{Introduction}}
% ===========================================================================

E-commerce last-mile logistics in dense urban markets faces two coupled
challenges typically treated independently: \textbf{{forward delivery}} of
orders from dark stores to customers, and \textbf{{reverse logistics}} ---
collecting returned items without deploying a separate fleet.

This project asks a single integrated question:
\begin{{quote}}
  \textit{{Given customer locations, order demands, and predicted return
  probabilities, (a) where should dark stores be placed to minimise
  customer-to-facility distance, and (b) how should vehicle routes be designed
  to simultaneously deliver orders and collect returns, minimising a weighted
  combination of total cost, delivery latency, and fleet size?}}
\end{{quote}}

The solution pipeline consists of five coupled stages implemented over eight
days using Google OR-Tools~9.15, PuLP~3.3 (CBC solver), and XGBoost.
All production logic resides in \texttt{{src/}}; notebooks are used only for
testing and visualisation.

% ===========================================================================
\section{{Dataset and Environment}}
% ===========================================================================

\subsection{{Olist Brazilian E-Commerce}}
The Olist dataset contains $\approx$100,000 orders across 9 CSV tables
(orders, items, products, customers, sellers, geolocation, payments,
reviews, category translation), covering 2016--2018.
We filter to \textbf{{customer\_state = `SP'}} (S\~ao Paulo, $\approx$41\% of
all orders), yielding \textbf{{19,207 rows}} for the active pipeline
after preprocessing, geolocation resolution, and feature engineering.

\subsection{{Environment}}
\begin{{itemize}}[noitemsep]
  \item Python 3.13.12 managed by \texttt{{uv}} on WSL2
  \item OR-Tools 9.15.6755 --- CVRPTW + SDVRP routing
  \item PuLP 3.3.0 + CBC --- MILP joint optimiser
  \item XGBoost 3.2.0 + Platt calibration --- return classifier
  \item All code: \url{{https://github.com/metaphorpritam/SCA_DARK_STORES}}
\end{{itemize}}

% ===========================================================================
\section{{Stage 1: Dark Store Placement (Facility Location)}}
% ===========================================================================

\subsection{{K-Means with Demand Weighting}}
Customer coordinates are weighted by zip-code order volume and clustered
using K-Means for $K \in [3, 12]$.  The optimal $K$ is selected at the
agreement point between the elbow curve (inertia) and silhouette score.

{_maybe_fig("elbow_silhouette", "K-Means elbow + silhouette: optimal $K=11$.", "elbow")}

\subsection{{Coverage Rule and Final $K$}}
Although the silhouette score peaks at $K=3$, the coverage rule
(70\% of customers within 5~km of their assigned dark store) requires
$K \geq 10$.  \textbf{{$K = 11$}} is selected, achieving 73.7\% coverage
within 5~km across S\~ao Paulo.

{_maybe_fig("kmeans_cluster_map", "Final 11 dark store locations (K-Means centroids).", "clusters")}

\subsection{{p-Median Validation}}
A PuLP MILP p-median formulation validates K-Means placement.
The objective minimises $\sum_i \sum_j d_{{ij}} \cdot w_i \cdot x_{{ij}}$
subject to single-assignment and open-facility constraints.
Both methods agree on cluster locations within 1~km, confirming stability.

% ===========================================================================
\section{{Stage 2: Forward VRP --- CVRPTW (Day 3)}}
% ===========================================================================

\subsection{{Problem Formulation}}
Each of the 11 zones is solved independently as a
\textbf{{Capacitated VRP with Time Windows (CVRPTW)}}:
\begin{{align}}
  \min &\sum_k \sum_{{(i,j) \in A}} c_{{ij}} \cdot x_{{ijk}} \notag\\
  \text{{s.t.}} \quad & \sum_k x_{{0jk}} = K, \quad
    \sum_j x_{{ijk}} = \sum_j x_{{jik}} \;\; \forall i,k \notag\\
  & a_i \leq s_{{ik}} \leq b_i, \quad
    \sum_i d_i \cdot \delta_{{ik}} \leq Q \;\; \forall k \notag
\end{{align}}
Parameters: vehicle capacity $Q = 500{{\,}}000\,\text{{g}}$,
speed $= 40\,\text{{km/h}}$, service time $= 5\,\text{{min}}$,
fixed cost R\$50/route, variable R\$1.50/km.
Solver: \texttt{{PATH\_CHEAPEST\_ARC}} $\to$ \texttt{{GUIDED\_LOCAL\_SEARCH}},
30~s time limit.

\subsection{{Results --- All 11 Zones Solved}}

{fwd_table}

\vspace{{0.5em}}
\begin{{center}}
  \textbf{{Total: 825 orders, 22 vehicles, {fwd_total_km:.2f}\,km,
  R\${fwd_total_cost:.2f}}}
\end{{center}}

{_maybe_fig("zone_kpi_bars", "Forward route cost and distance by zone.", "fwdkpi")}

\subsection{{Baseline Comparison}}
Naive nearest-store routing (each customer drives directly to closest store)
produces 75,066.4~km.  OR-Tools CVRPTW achieves \textbf{{1,069.59~km}},
a \textbf{{98.58\% reduction}}.

% ===========================================================================
\section{{Stage 3: Return Probability Classifier (Day 4)}}
% ===========================================================================

\subsection{{Target and Features}}
Binary target: \texttt{{is\_return = 1}} if order status
$\in$ \{{canceled, unavailable\}} or delivered $>7$ days late.
Feature set: product weight, freight value, seller state (label-encoded),
review score, order value, days late, payment type, product category.

\subsection{{Model: XGBoost + Platt Calibration}}
Class imbalance ($\approx$5\% return rate) handled via
\texttt{{scale\_pos\_weight = 15}}.

\begin{{tabular}}{{lc}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} \\
\midrule
ROC-AUC   & {roc_auc} \\
PR-AUC    & {pr_auc} \\
Brier score & {brier} \\
Precision @ 0.30 & {clf_prec} \\
Recall @ 0.30    & {clf_rec} \\
F1 @ 0.30        & {clf_f1} \\
\bottomrule
\end{{tabular}}

ROC-AUC~{roc_auc} exceeds the project target of $\geq 0.70$, confirming
the classifier's ability to identify return-risk orders.

Customers with \texttt{{return\_prob > 0.30}} are flagged as pickup nodes
in the SDVRP; expected pickup weight = \texttt{{return\_prob $\times$
product\_weight\_g}}.
\textbf{{593 orders}} (3.1\% of 19,207) were flagged across all zones.

% ===========================================================================
\section{{Stage 4: Reverse VRP (Day 4)}}
% ===========================================================================

Same CVRPTW structure as forward VRP with vehicles departing the depot
\emph{{empty}} and collecting returns (load increases at each stop).

{rev_table}

\vspace{{0.5em}}
\begin{{center}}
  \textbf{{Total: 576 pickups, {rev_total_veh} vehicles,
  {rev_total_km:.2f}\,km, R\${rev_total_cost:.2f}}}
\end{{center}}

Combined separate cost:
\textbf{{R\${sep_total:,.2f}}} (forward R\${fwd_total_cost:.2f} + reverse
R\${rev_total_cost:.2f}).

% ===========================================================================
\section{{Stage 5: Joint MILP Optimiser (Day 4)}}
% ===========================================================================

\subsection{{Objective Function}}
\begin{{equation}}
  Z = \alpha \cdot C_{{\text{{fwd}}}} + \beta \cdot C_{{\text{{rev}}}}
    + \gamma \cdot T_{{\text{{pen}}}} + \delta \cdot N_{{\text{{veh}}}}
  \label{{eq:Z}}
\end{{equation}}
Binary activation variables $u_v \in \{{0,1\}}$ (forward vehicles) and
$w_v \in \{{0,1\}}$ (reverse vehicles) are solved by PuLP CBC.

\subsection{{Result ($\alpha=\beta=\gamma=\delta=0.25$)}}
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Component}} & \textbf{{Value}} \\
\midrule
$Z$ (objective)    & {z_val} \\
Status             & {z_status} \\
$C_{{\text{{fwd}}}}$ & {jopt.get("C_fwd","N/A")} \\
$C_{{\text{{rev}}}}$ & {jopt.get("C_rev","N/A")} \\
$T_{{\text{{pen}}}}$ & {z_tpen} \quad (\textbf{{dominates $\approx$54\% of Z}}) \\
$N_{{\text{{veh}}}}$ & {z_nveh} \\
\bottomrule
\end{{tabular}}

The penalty term $T_{{\text{{pen}}}}$ dominates $Z$ at equal weights,
motivating the Day~5 SDVRP work to reduce reverse fleet redundancy
and the Day~6 Pareto sweep to find balanced weight combinations.

% ===========================================================================
\section{{Stage 6: SDVRP Hybrid Routing (Day 5)}}
% ===========================================================================

\subsection{{Motivation}}
Zone~8 carries the highest combined cost: R\$570.84 (R\$320.86 forward
+ R\$249.98 reverse).  Deploying 5 separate vehicles (3 fwd + 2 rev)
means those vehicles pass the same customer locations twice.

\subsection{{SDVRP Load Model}}
Simultaneous Delivery and Pickup VRP (Dethloff,~2001) merges all
delivery and pickup nodes into one OR-Tools solve using a
\textbf{{single load dimension}}:
\begin{{align}}
  L(t) &= L_{{\text{{start}}}} - \text{{delivered}}(t) + \text{{picked}}(t) \notag\\
  \text{{transit}}[i] &= p_i - d_i \quad \text{{(net change per stop)}} \notag\\
  0 &\leq L_{{\text{{cumul}}}}[i] \leq Q \quad \forall\, i \notag
\end{{align}}
\texttt{{fix\_start\_cumul\_to\_zero = False}} lets OR-Tools initialise
each vehicle load at total delivery weight.  The previous two-dimension
additive approach ($d_{{\text{{cumul}}}} + p_{{\text{{cumul}}}} \leq Q$) was
over-constraining and is \textbf{{corrected here}}.

\subsection{{All-Zone SDVRP Runner}}
\texttt{{run\_all\_zones\_sdvrp()}} loops all 11 zones, calling
\texttt{{solve\_sdvrp\_hybrid()}} per zone and writing
\texttt{{outputs/hybrid\_routes.json}} and
\texttt{{outputs/hybrid\_kpi\_summary.csv}}.
Expected fleet saving: \textbf{{15--25\%}} vs separate fleets.

% ===========================================================================
\section{{Stage 7: Combined KPI Report \& Pareto Sweep (Day 6)}}
% ===========================================================================

\subsection{{Combined KPI Report}}
\texttt{{src/kpi\_reporter.py}} merges all three KPI streams
(forward, reverse, hybrid SDVRP) into
\texttt{{outputs/combined\_kpi\_report.csv}}.

{_maybe_fig("combined_cost", "Stacked fwd+rev cost by zone.  Zone~8 is largest at R\\$570.84.", "combined_cost")}

\subsection{{Zone Priority Ranking}}
Zones are ranked by \texttt{{saving\_R\$}} (when SDVRP results exist)
or by \texttt{{separate\_cost\_R\$}} otherwise.  Zone~8 receives
\textbf{{Rank~1}} in both cases.

{_maybe_fig("sdvrp_priority", "SDVRP zone priority ranking (highest saving opportunity first).", "priority")}

\subsection{{Pareto Sweep}}
Weight combinations $(\alpha, \beta) \in \{{0.1, 0.3, 0.5, 0.7, 0.9\}}^2$
with $\gamma = \delta = (1-\alpha-\beta)/2$ (valid for $\alpha+\beta \leq 1$)
yield up to 15 weight settings.  Pareto-optimal solutions minimise both
$C_{{\text{{routing}}}} = C_{{\text{{fwd}}}} + C_{{\text{{rev}}}}$ and
$T_{{\text{{pen}}}}$ simultaneously.

\begin{{equation}}
  \text{{dist\_to\_ideal}} =
  \sqrt{{c_{{\text{{norm}}}}^2 + t_{{\text{{norm}}}}^2}}
\end{{equation}}
The \textbf{{knee point}} (minimum distance to ideal) is the recommended
operating configuration.

{_maybe_fig("pareto_tradeoff", "Pareto tradeoff surface: routing cost vs.\\ late-return penalty.  Red star = knee point.", "pareto")}

% ===========================================================================
\section{{Summary of Results}}
% ===========================================================================

\begin{{tabular}}{{llr}}
\toprule
\textbf{{Stage}} & \textbf{{Metric}} & \textbf{{Value}} \\
\midrule
Dark store placement   & Zones ($K$)                 & 11 \\
                       & Coverage within 5\,km       & 73.7\% \\
Forward VRP            & Total distance              & {fwd_total_km:.2f}\,km \\
                       & Improvement vs.\ naive      & 98.58\% \\
                       & Total cost                  & R\${fwd_total_cost:.2f} \\
Return classifier      & ROC-AUC                     & {roc_auc} \\
                       & F1 @ threshold 0.30         & {clf_f1} \\
Reverse VRP            & Total distance              & {rev_total_km:.2f}\,km \\
                       & Total cost                  & R\${rev_total_cost:.2f} \\
Joint MILP             & $Z$ (optimal)               & {z_val} \\
Combined baseline      & Total fwd + rev cost        & R\${sep_total:,.2f} \\
\bottomrule
\end{{tabular}}

% ===========================================================================
\section{{Reproducibility}}
% ===========================================================================

All code is reproducible via \texttt{{run\_all.sh}}:
\begin{{verbatim}}
uv run python src/data_pipeline.py
uv run python src/clustering.py
uv run python src/forward_vrp.py
uv run python src/reverse_vrp.py
uv run python src/return_classifier.py
uv run python src/joint_optimizer.py
uv run python src/kpi_reporter.py
uv run python src/report_builder.py
\end{{verbatim}}

Data files (\texttt{{data/raw/}} Olist CSVs) are excluded from the repository
per \texttt{{.gitignore}}.  Download from Kaggle:
\url{{https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce}}.

\section*{{References}}

\begin{{enumerate}}[noitemsep]
  \item Dethloff, J.\ (2001). Vehicle routing and reverse logistics:
        the vehicle routing problem with simultaneous delivery and pick-up.
        \textit{{OR Spectrum}}, 23(1), 79--96.
  \item Dantzig, G.B.\ \& Ramser, J.H.\ (1959). The truck dispatching problem.
        \textit{{Management Science}}, 6(1), 80--91.
  \item Google OR-Tools documentation:
        \url{{https://developers.google.com/optimization}}.
  \item ReVelle, C.S.\ \& Swain, R.W.\ (1970). Central facilities location.
        \textit{{Geographical Analysis}}, 2(1), 30--42.
\end{{enumerate}}

\end{{document}}
"""
    return tex


# ---------------------------------------------------------------------------
# PDF compiler
# ---------------------------------------------------------------------------


def compile_pdf(tex_path: str | Path) -> Path:
    """
    Run pdflatex twice on tex_path (two passes for table of contents).

    Parameters
    ----------
    tex_path : path to the .tex file

    Returns
    -------
    Path to the generated .pdf file

    Raises
    ------
    RuntimeError if pdflatex returns a non-zero exit code on the first pass.
    """
    tex_path = Path(tex_path).resolve()
    out_dir = tex_path.parent

    # Run from project root (parent of report/) so relative paths in .tex resolve
    project_root = out_dir.parent
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_path),
    ]

    for pass_num in (1, 2):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        pdf_path_check = tex_path.with_suffix(".pdf")
        # A non-zero exit is raised as error only if no PDF was produced
        if result.returncode != 0 and pass_num == 1 and not pdf_path_check.exists():
            lines = (result.stdout + result.stderr).splitlines()
            diag = "\n".join(lines[-40:])
            raise RuntimeError(
                f"[report_builder] pdflatex pass 1 failed (exit {result.returncode}):\n{diag}"
            )
        if result.returncode != 0:
            print(
                f"[report_builder] pdflatex pass {pass_num}: exit {result.returncode} (warnings; PDF still produced)"
            )
        else:
            print(
                f"[report_builder] pdflatex pass {pass_num}: exit {result.returncode}"
            )

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"[report_builder] PDF generated → {pdf_path}")
    else:
        print(f"[report_builder] Warning: PDF not found at {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    compile: bool = True,
) -> Path:
    """
    Full pipeline: load KPIs → build LaTeX → write .tex → compile .pdf.

    Parameters
    ----------
    output_dir  : directory for .tex and .pdf (default: report/)
    outputs_dir : directory with KPI CSVs and PNG figures (default: outputs/)
    data_dir    : directory with data files (default: data/)
    compile     : if True, run pdflatex to produce PDF

    Returns
    -------
    Path to the .pdf file (or .tex if compile=False)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  REPORT BUILDER")
    print("=" * 60)

    kpis = load_kpis(outputs_dir=outputs_dir, data_dir=data_dir)
    tex_str = build_latex(kpis, figures_dir=outputs_dir)

    tex_path = output_dir / "report_draft_v1.tex"
    tex_path.write_text(tex_str, encoding="utf-8")
    print(f"[report_builder] LaTeX source → {tex_path}  ({len(tex_str):,} chars)")

    if compile:
        pdf_path = compile_pdf(tex_path)
        print("=" * 60 + "\n")
        return pdf_path

    print("=" * 60 + "\n")
    return tex_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()

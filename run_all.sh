#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full SCA Dark Stores Pipeline (one-shot reproducibility script)
#
# Usage:
#   bash run_all.sh          # run all stages in order
#   bash run_all.sh --check  # dry-run: print stage order and exit
#
# Requirements:
#   - uv installed and available in PATH
#   - Raw Olist CSV files in data/raw/  (excluded from git)
#   - pdflatex (TeX Live) for the report stage
#
# Output artefacts:
#   data/              — processed parquet, NumPy distance matrix, CSVs
#   outputs/           — KPI summaries, route JSONs, classifier PKL, PNGs
#   report/            — LaTeX source + compiled PDF
# =============================================================================
set -euo pipefail

PYTHON="uv run python"

# ── helpers ──────────────────────────────────────────────────────────────────

banner() {
    echo ""
    echo "======================================================================"
    echo "  STAGE $1: $2"
    echo "======================================================================"
}

run_stage() {
    local n="$1"
    local desc="$2"
    local script="$3"
    banner "$n" "$desc"
    $PYTHON "$script"
    echo "  [OK] Stage $n complete"
}

# ── dry-run mode ─────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--check" ]]; then
    echo "SCA Dark Stores — pipeline stages:"
    echo "  1  data_pipeline.py         — ingest & feature engineer Olist CSVs"
    echo "  2  haversine_matrix.py      — build Haversine distance matrix"
    echo "  3  clustering.py            — K-Means + p-Median dark store placement"
    echo "  4  forward_vrp.py           — CVRPTW forward delivery (all 11 zones)"
    echo "  5  return_classifier.py     — XGBoost return probability model"
    echo "  6  reverse_vrp.py           — reverse-logistics CVRPTW (all 11 zones)"
    echo "  7  joint_optimizer.py       — MILP joint cost optimiser + Pareto sweep"
    echo "  8  all_zones_aggregator.py  — SDVRP hybrid routes (all zones)"
    echo "  9  kpi_reporter.py          — combined KPI report + zone rankings"
    echo "  10 report_builder.py        — LaTeX report → PDF in report/"
    echo ""
    exit 0
fi

# ── pipeline ─────────────────────────────────────────────────────────────────

START_TS=$(date +%s)
echo ""
echo "Starting SCA Dark Stores full pipeline at $(date)"

run_stage  1 "Data pipeline — ingest & feature engineering"    src/data_pipeline.py
run_stage  2 "Haversine distance matrix"                        src/haversine_matrix.py
run_stage  3 "Clustering — K-Means + p-Median dark store placement" src/clustering.py
run_stage  4 "Forward VRP — CVRPTW (all 11 zones)"             src/forward_vrp.py
run_stage  5 "Return classifier — XGBoost"                      src/return_classifier.py
run_stage  6 "Reverse VRP — CVRPTW (all 11 zones)"             src/reverse_vrp.py
run_stage  7 "Joint MILP optimiser + Pareto sweep"             src/joint_optimizer.py
run_stage  8 "SDVRP hybrid routes — all zones aggregator"      src/all_zones_aggregator.py
run_stage  9 "Combined KPI report + zone priority ranking"     src/kpi_reporter.py
run_stage 10 "LaTeX report builder → report/report_draft_v1.pdf" src/report_builder.py

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

echo ""
echo "======================================================================"
echo "  ALL 10 STAGES COMPLETE"
echo "  Elapsed: ${ELAPSED}s"
echo "======================================================================"
echo ""
echo "Key outputs:"
echo "  data/dark_stores_final.csv          — 11 dark store locations"
echo "  outputs/forward_kpi_summary.csv     — forward VRP KPIs by zone"
echo "  outputs/reverse_kpi_summary.csv     — reverse VRP KPIs by zone"
echo "  outputs/return_classifier_metrics.json"
echo "  outputs/joint_optimizer_result.json"
echo "  outputs/pareto_results.csv          — Pareto sweep (15 weight combos)"
echo "  outputs/combined_kpi_report.csv     — merged KPI + zone priority ranking"
echo "  report/report_draft_v1.pdf          — final project report"
echo ""

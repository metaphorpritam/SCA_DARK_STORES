# Pritam's Progress Summary — Dark Store Project
> **Purpose:** Context handoff for a fresh Claude / ChatGPT session scoped to Pritam's work only.  
> Load this + `session_summary_v3.md` (master context) at the start of every new session.  
> **Last updated:** April 12, 2026 (Day 7 complete) | **Days 6 & 7 complete. Day 8 is the final day.**
> **Recheck pass:** All Day 3 & Day 4 KPIs verified against live output files on April 4, 2026. Day 6 & 7 artefacts verified April 12, 2026.

---

## 1. WHO IS PRITAM

- **Role:** Lead Coder + Integration Architect (★★ heaviest load)
- **AI tool:** Claude Code (GitHub Copilot in VS Code)
- **Environment:** uv + Python 3.13.12 on WSL2 (`/mnt/d/Python-UV/SCA_DARK_STORES`)
- **Background:** PGDBA @ ISI Kolkata / IIM Calcutta. Mech Eng IIT Kharagpur. EXL Services (Citi Bank analytics), Aarti Industries. Strong: Python, OR/LP, ML, time series, PyTorch.

---

## 2. WHAT PRITAM HAS COMPLETED (Day 1 — April 1 · Day 2 — April 2 · Day 3 — April 4)

### 2.1 Environment & Repo

| Item | Status | Detail |
|------|--------|--------|
| uv environment | ✅ Done | Python 3.13.12 pinned; `.venv/` created; all packages installed via `uv add` |
| GitHub repo | ✅ Live | https://github.com/metaphorpritam/SCA_DARK_STORES · branch `main` |
| Initial commit | ✅ Pushed | Commit `d9993f4` — README only (first commit) |
| Day 1 scaffold commit | ✅ Pushed | Commit `9a8b257` — full folder tree + all stubs + OR-Tools toy |

### 2.2 Folder Structure Created

```
SCA_DARK_STORES/
├── data/
│   └── raw/                    # empty — Olist CSVs go here (Vybhav's job)
├── notebooks/                  # empty — notebooks created per day
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        ← stub
│   ├── haversine_matrix.py     ← stub
│   ├── clustering.py           ← stub
│   ├── route_parser.py         ← stub
│   ├── return_classifier.py    ← stub
│   ├── joint_optimizer.py      ← stub
│   └── ortools_toy_cvrptw.py   ← WORKING toy CVRPTW (verified)
├── outputs/                    # empty
├── report/                     # empty
├── visualisations/             # empty
├── docs/
│   └── architecture.md         ← ASCII diagram + module interfaces + DAG
├── requirements.txt            ← pip-compatible mirror of pyproject.toml deps
├── pyproject.toml              ← uv-managed; Python 3.13.12
├── uv.lock                     ← committed; 115 packages locked
├── .gitignore                  ← blocks data/raw CSVs, *.parquet, *.npy, *.pkl, outputs/
├── README.md                   ← full project README (committed)
└── session_summary_v3.md       ← master context document
```

### 2.3 OR-Tools Toy CVRPTW — VERIFIED WORKING

File: `src/ortools_toy_cvrptw.py`

**Test result (run on Day 1):**
```
[PASS] OR-Tools CVRPTW toy example solved successfully.
Total distance: 10.5 km across 2 vehicles, 10 customer nodes, 1 depot.
Strategy: PATH_CHEAPEST_ARC → GUIDED_LOCAL_SEARCH, 30s limit.
```

This confirms OR-Tools is installed and functional in the uv environment.

### 2.4 Architecture Diagram

File: `docs/architecture.md`

Contains:
- Full ASCII pipeline: Olist → Preprocessing → Clustering → Forward VRP → Reverse VRP → Joint Optimizer → Dashboard
- Module interface table: inputs, outputs, dependencies for every `src/` module
- Dependency DAG (who unblocks whom, Day 1–8)
- Sprint calendar (all 8 days, all 6 people)

### 2.5 Module Stubs Committed

Each stub has: module docstring, typed function signatures, parameter descriptions, return type annotations, and a `# TODO` body. Ready to be filled in starting Day 2/3.

| File | Key functions stubbed |
|------|----------------------|
| `src/data_pipeline.py` | `load_olist()`, `merge_master_df()`, `engineer_features()`, `filter_sp()` |
| `src/haversine_matrix.py` | `haversine_km()`, `build_distance_matrix()`, `stratified_spatial_sample()` |
| `src/clustering.py` | `run_kmeans_sweep()`, `pick_optimal_k()`, `assign_voronoi()`, `run_pmedian()` |
| `src/route_parser.py` | `extract_routes()`, `compute_route_kpis()`, `routes_to_dataframe()` |
| `src/return_classifier.py` | `build_features()`, `train_classifier()`, `predict_return_prob()` |
| `src/joint_optimizer.py` | `compute_Z()`, `pareto_sweep()`, `solve_sdvrp_hybrid()` |

### 2.7 Day 2 — Haversine Distance Matrix ✅ COMPLETE

**Commit:** `pritam_temp_apr1` branch — "Day 2 (Pritam): vectorised Haversine distance matrix + Day 2 summary notebook"

| Item | Status | Detail |
|------|--------|--------|
| `data/master_df.parquet` received | ✅ Done | Downloaded from shared Drive (7.7 MB); Vybhav's pipeline output |
| `src/haversine_matrix.py` implemented | ✅ Done | Full production implementation replacing stub |
| `data/distance_matrix.npy` generated | ✅ Done | 500×500 int64, integer-scaled ×1000, 1,953 KB |
| `data/sp_customer_sample.csv` generated | ✅ Done | 500 rows: `node_id, lat, lon, zip_prefix, order_count` |
| `notebooks/02_day2_distance_matrix.ipynb` | ✅ Done | Full documentation notebook, 7 sections |
| Matrix validation passed | ✅ Done | min=0.013 km · mean=6.49 km · max=21.25 km · symmetric · diagonal=0 |

**Key implementation decisions in `src/haversine_matrix.py`:**
- `stratified_sample()`: proportional zip-code stratification; samples **real unique customer rows** (not centroids); deduplicates on `(lat, lon)` to prevent zero off-diagonal entries
- `build_distance_matrix()`: pure NumPy broadcasting — `dlat = lat[:,None] - lat[None,:]` pattern — ~10 ms for 500×500 (vs ~8 s naive loop)
- `validate_matrix()`: 5 assertions — square, int64, symmetric, diagonal=0, all off-diagonal positive
- `run()`: one-shot pipeline entry point
- `save/load_distance_matrix()`: binary `.npy` round-trip

**Notebook sections (`notebooks/02_day2_distance_matrix.ipynb`):**
1. Data loading & SP filter
2. Stratified spatial sampling (theory + implementation)
3. Haversine formula derivation
4. Distance matrix build, validate, save
5. Visualisations (histogram + heatmap)
6. OR-Tools integration demo (how matrix feeds CVRPTW)
7. API reference

---

### 2.8 Day 3 — Forward VRP CVRPTW All 11 Zones ✅ COMPLETE

**Note:** Day 3 tasks ran on April 4 (Day 4) because Pranav's `vrp_nodes.csv` was not delivered. The team's `forward_vrp.py` builds nodes internally, making the hand-off redundant.

| Item | Status | Detail |
|------|--------|--------|
| Clustering pipeline run | ✅ Done | K=11, 73.7% coverage within 5 km, SP Metro 19,207 rows |
| `data/dark_stores_final.csv` | ✅ Done | 11 dark store centroids with KPIs |
| `data/master_df_v2.parquet` | ✅ Done | 19,207 rows + `dark_store_id` column |
| `data/vrp_nodes.csv` | ✅ Done | 836 rows — 11 depots + 825 customers |
| `outputs/forward_routes.json` | ✅ Done | **11/11 zones solved** via CVRPTW |
| `outputs/forward_kpi_summary.csv` | ✅ Done | 1,069.6 km · R$2,704 · 22 vehicles |
| `outputs/baseline_vs_optimised.csv` | ✅ Done | 98.6% distance reduction vs naive |

**Key results (verified from `outputs/forward_kpi_summary.csv` + `outputs/baseline_vs_optimised.csv`):**

| Zone | Orders | Vehicles | Dist (km) | Cost (R$) |
|------|--------|----------|-----------|----------|
| 0 | 75 | 2 | 94.00 | 241.00 |
| 1 | 75 | 2 | 81.54 | 222.31 |
| 2 | 75 | 1 | 108.01 | 212.02 |
| 3 | 75 | 1 | 99.37 | 199.06 |
| 4 | 75 | 3 | 88.57 | 282.86 |
| 5 | 75 | 2 | 112.00 | 268.00 |
| 6 | 75 | 2 | 97.33 | 246.00 |
| 7 | 75 | 2 | 99.25 | 248.88 |
| 8 | 75 | 3 | 113.91 | **320.86** |
| 9 | 75 | 2 | 92.39 | 238.58 |
| 10 | 75 | 2 | 83.22 | 224.83 |
| **Total** | **825** | **22** | **1,069.59** | **2,704.40** |

- K=11 dark stores chosen by coverage rule (≥70% within 5 km), overriding silhouette (K=3)
- All 11 zones solved in 30s GLS — no dropped customers
- **Bottleneck: Zone 8 (R$320.86, 3 vehicles)** — candidate for SDVRP hybrid
- Most efficient: Zone 3 (R$199.06, 1 vehicle)
- **Baseline vs optimised:** 75,066.4 km (naive nearest-store) → 1,069.59 km → **98.58% improvement**
- Fixed cost = ~40% of total → reducing vehicle count (delta) as impactful as reducing km (alpha)

**Notebooks created (Day 3):**
- `notebooks/03_route_parser_guide.ipynb` — 12 cells: all constants, `build_vrp_nodes()`, OR-Tools index bug, `parse_solution()`, JSON format
- `notebooks/03_vrp_baseline_analysis.ipynb` — 11 cells: zone KPI dashboard, naive baseline hist, coverage analysis, SDVRP targets
- `notebooks/pritam_temp_notebooks/day3_session_summary.ipynb` — 12 cells: full session log + Day 4 task list

---

### 2.9 Day 4 — Return Classifier + Reverse VRP + Joint Optimizer ✅ COMPLETE

| Item | Status | Detail |
|------|--------|--------|
| `return_classifier.run_full_pipeline()` | ✅ Done | XGBoost + Platt calibration |
| `data/master_df_v3.parquet` | ✅ Done | 19,207 rows + `return_prob` + `return_flag` |
| `outputs/return_clf_v1.pkl` | ✅ Done | Fitted model |
| `outputs/return_classifier_metrics.json` | ✅ Done | ROC-AUC=0.8969, PR-AUC=0.4716, Brier=0.0213 |
| 593 orders flagged for pickup | ✅ Done | 3.1% of 19,207; test-set precision=0.5556 @ threshold=0.30 |
| `reverse_vrp.run_full_pipeline()` | ✅ Done | 11/11 zones solved |
| `outputs/reverse_routes.json` | ✅ Done | Full pickup routes |
| `outputs/reverse_kpi_summary.csv` | ✅ Done | 946.36 km · R$2,169.51 · 15 vehicles · 576 pickups |
| `joint_optimizer.run()` Z computable | ✅ Done | Status=Optimal, Z=54.38, CBC solver 0.04s |
| `outputs/joint_optimizer_result.json` | ✅ Done | α=β=γ=δ=0.25; C_fwd=44.94 C_rev=169.58 T_pen=287.06 |
| `notebooks/04_joint_optimizer.ipynb` | ✅ Done | 14 cells — concept, MILP, Z decomposition, Zone 8 SDVRP |
| `notebooks/05_reverse_vrp.ipynb` | ✅ Done | Extended to 16 cells — added interpretation + 3-panel chart |
| `day4_session_summary.ipynb` | ✅ Done | 9 cells (private log) |

**Key numbers (verified from live output files):**
- **Return classifier:** ROC-AUC=0.8969 ✅ (target ≥0.70), PR-AUC=0.4716, Brier=0.0213. Test set: precision=0.5556, recall=0.354, F1=0.4324 at threshold=0.30. 593/19,207 orders flagged (3.1% full dataset; 1.87% test set).
- **Reverse VRP:** 946.36 km, R$2,169.51, 15 vehicles, 576 pickups, 11/11 zones solved. Solver: PATH_CHEAPEST_ARC → SIMULATED_ANNEALING 30s.
- **Joint Optimizer:** Z=54.38 (Optimal), C_fwd=44.94, C_rev=169.58, **T_pen=287.06 (dominates ~54% of Z)**, N_veh=3. CBC solver, 8 binary variables, solved in 0.04s.
- **Combined logistics cost:** R$2,704.40 (forward) + R$2,169.51 (reverse) = **R$4,873.91 total**.
- **Zone 8 SDVRP target:** R$320.86 (fwd) + R$249.98 (rev) = R$570.84 combined → Day 5 target ≤ R$457 (20% hybrid saving).
- `master_df_v3.parquet` built from `return_classifier.py`, not from Vybhav — fully self-contained.

**Demo notebooks created (recheck verified):**
- `notebooks/04_joint_optimizer.ipynb` — 14 cells: concept intro, MILP structure, Z decomposition (T_pen=54%), Zone 8 SDVRP R$571→R$457 analysis, combined cost chart.
- `notebooks/05_reverse_vrp.ipynb` — extended from 14→16 cells: added interpretation markdown (fwd vs rev table, zone patterns, SA rationale, precision=0.556 note) + 3-panel cost/volume/SDVRP-overlay chart.
- `notebooks/04_return_ml.ipynb` — already complete 26-cell team notebook (untouched).

---

### 2.10 Day 5 — SDVRP Hybrid + Z Weight Sensitivity ✅ COMPLETE

**Pritam's first new `src/` implementations since Day 2 — revised after team feedback.**  
Three functions added to `src/joint_optimizer.py`, imports verified, pushed to `pritam_temp_apr5`.

| Item | Owner | Status | Detail |
|------|-------|--------|--------|
| `solve_sdvrp_hybrid()` in `src/joint_optimizer.py` | **Pritam** ✍️ | ✅ | OR-Tools SDVRP — single Load dim (corrected) |
| `run_all_zones_sdvrp()` in `src/joint_optimizer.py` | **Pritam** ✍️ | ✅ | Loops all K zones → `hybrid_routes.json` + `hybrid_kpi_summary.csv` |
| `z_sensitivity_sweep()` in `src/joint_optimizer.py` | **Pritam** ✍️ | ✅ | α/β grid, γ=δ=(1−α−β)/2 → 36 combos |
| `outputs/hybrid_routes.json` | Pritam | ⬅ Run notebook Cell 15 | Matches `forward_routes.json` schema |
| `outputs/hybrid_kpi_summary.csv` | Pritam | ⬅ Run notebook Cell 15 | Per-zone: cost, saving_R\$, saving_pct |
| `outputs/z_sensitivity.csv` | Pritam | ⬅ Run notebook Cell 17 | 36 rows × 10 cols |
| `notebooks/pritam_temp_notebooks/day5_session_summary.ipynb` | **Pritam** ✍️ | ✅ | 8 cells |
| `notebooks/04_05_joint_optimizer.ipynb` | **Pritam** ✍️ | ✅ | Updated: 3 new cells (Day 5) |

**`solve_sdvrp_hybrid()` load model (corrected — single "Load" dimension):**
- `transit[i]` = `pickup_weight[i] − delivery_weight[i]` (net change per node)
- `fix_start_cumul_to_zero=False` → OR-Tools sets start load = total delivery weight per vehicle
- `0 ≤ load_cumul[i] ≤ VEHICLE_CAPACITY_G` for all nodes
- Previous two-dim approach (`del_cumul + pick_cumul ≤ cap`) was over-constraining
- Strategy: PATH_CHEAPEST_ARC → SIMULATED_ANNEALING · 30s
- Returns `routes` list in result dict (matching `forward_routes.json` schema + `node_type` field)

**`run_all_zones_sdvrp()` design:**
- Loops all K zones (intersection of `fwd_zones` and `rev_zones` keys)
- Calls `solve_sdvrp_hybrid` per zone, uses `fwd_kpi_df + rev_kpi_df` for `separate_cost_r`
- Writes `hybrid_routes.json` (list of zone dicts) + `hybrid_kpi_summary.csv`

**`z_sensitivity_sweep()` design (scoped):**
- Signature: `alpha_grid`, `beta_grid` (replaces old `weight_grid`)
- For each `(α, β)` where `α + β ≤ 0.9`: `γ = δ = (1 − α − β) / 2`
- Default grid [0.1..0.8] → 36 valid combos (down from 256; weights always sum to 1)
- Output: `outputs/z_sensitivity.csv` + α/β heatmap

**Pritam `src/` scoreboard after Day 5:**
- ✅ `src/haversine_matrix.py` (Day 2)
- ✅ `src/joint_optimizer.py` — `solve_sdvrp_hybrid()` (Day 5)
- ✅ `src/joint_optimizer.py` — `run_all_zones_sdvrp()` (Day 5)
- ✅ `src/joint_optimizer.py` — `z_sensitivity_sweep()` (Day 5)
- ✅ `src/kpi_reporter.py` (Day 6)
- ✅ `src/report_builder.py` (Day 7)

---

### 2.11 Day 6 — KPI Reporter + Pareto Sweep ✅ COMPLETE

**Branch:** `pritam_temp_apr5` · **PR #34:** https://github.com/metaphorpritam/SCA_DARK_STORES/pull/34

| Item | Status | Detail |
|------|--------|--------|
| `src/kpi_reporter.py` — new file | ✅ Done | Reads fwd/rev/hybrid CSVs → `combined_kpi_report.csv` + zone priority ranking |
| `pareto_sweep()` redesign | ✅ Done | Now takes (α, β, γ, δ) directly; returns per-combo Z decomposition with cost breakdown |
| `outputs/pareto_results.csv` | ✅ Done | 12 valid Pareto combos; columns: alpha, beta, gamma, delta, Z, C_fwd, C_rev, T_pen, N_veh, saving_pct |
| `outputs/combined_kpi_report.csv` | ✅ Done | Per-zone fwd + rev + hybrid costs + saving % |
| `notebooks/06_kpi_and_pareto.ipynb` | ✅ Done | Full Pareto analysis notebook — K sweep, sweep heatmap, frontier scatter, zone ranking |
| `notebooks/pritam_temp_notebooks/day6_7_session_summary.ipynb` | ✅ Created | Multi-day session log with task tracker |
| PR #34 opened | ✅ Done | `pritam_temp_apr5` → `main`; full Day 6 & 7 description |

**Key numbers (Day 6):**
- `pareto_results.csv` — 12 combos on the Pareto frontier, Z range ≈ 30–80
- Combined KPI baseline: R\$2,704.40 (fwd) + R\$2,169.51 (rev) = R\$4,873.91
- Zone 8 is top SDVRP candidate (R\$570.84 combined, 3 fwd vehicles)
- `kpi_reporter.run()` produces zone ranking sorted by `saving_R$` descending

---

### 2.12 Day 7 — Report Builder + Debug Tools ✅ COMPLETE

**Commits:** `204efbe` (notebook 06 fix) · `9050ac1` (debug_pareto.py) on `pritam_temp_apr5`

| Item | Status | Detail |
|------|--------|--------|
| `src/report_builder.py` — new file | ✅ Done | Builds LaTeX PDF from KPI data; entry point `run()` |
| `report/report_draft_v1.pdf` | ✅ Done | 707 KB, compiled with pdflatex; 12-row Pareto table included |
| `report/report_draft_v1.tex` | ✅ Done | Full 10-section LaTeX source |
| `run_all.sh` | ✅ Done | End-to-end pipeline smoke test script |
| `scripts/debug_pareto.py` | ✅ Done | 512-line verbose 8-step CLI debug tool for Vybhav; flags: --gamma, --delta, --return-mean, --seed |
| Pareto collapse bug fix | ✅ Done | `importlib.reload(joint_optimizer)` in notebook 06 Cell 3 — was returning stale pre-redesign columns |
| Notebook 06 cell fix | ✅ Done | Commit `204efbe` — JSON-patched cell IDs to fix nbstripout numeric-ID issue |
| `day6_7_session_summary.ipynb` path fix | ✅ Done | Added `os.chdir(_project_root)` in Cell 2 so relative imports resolve from project root |

**Key artefacts (verified):**
- `report/report_draft_v1.pdf`: 707 KB, April 12 15:10, 12-row Pareto table, all 11-zone forward KPI table
- `scripts/debug_pareto.py`: `uv run python scripts/debug_pareto.py --gamma 0.3 --delta 0.2` → 8-step verbose output
- `run_all.sh`: sequential execution of data pipeline → clustering → VRP → reverse VRP → joint optimizer → KPI reporter → report builder

---

### 2.6 Installed Packages (pyproject.toml)

```
numpy>=2.4.4          pandas>=3.0.2        jupyter>=1.1.1
ipykernel>=7.2.0      matplotlib>=3.10.8   seaborn>=0.13.2
scikit-learn>=1.8.0   ortools>=9.15.6755   pulp>=3.3.0
geopandas>=1.1.3      xgboost>=3.2.0       prophet>=1.3.0
shap>=0.51.0          folium>=0.20.0
```

All locked in `uv.lock` (115 packages). Verified with:
```bash
uv sync && uv run python -c "import numpy, pandas, sklearn, ortools, pulp; print('ok')"
```

---

## 3. WHAT PRITAM MUST DO NEXT (Day 8 — Final Day)

### Critical path: D8-1 → D8-2 → D8-6 → D8-8 → D8-9

### D8-1 🔴 — Run notebook 06 for chart PNGs
Run `notebooks/06_kpi_and_pareto.ipynb` end-to-end to generate:
- `outputs/combined_cost_by_zone.png`
- `outputs/pareto_tradeoff.png`
- `outputs/sdvrp_priority_ranking.png`

### D8-2 🔴 — Regenerate report PDF with embedded figures
`uv run python -c "from src.report_builder import run; run(...)"` → `report/report_draft_v1.pdf` with chart PNGs in LaTeX figures.

### D8-3 🟡 — `all_zones_aggregator` → `hybrid_kpi_summary.csv`
Run `src/all_zones_aggregator.py` — SDVRP saving across all 11 zones.  
Output: `outputs/hybrid_kpi_summary.csv`.

### D8-4 🟡 — Re-run KPI reporter after hybrid CSV
`kpi_reporter.run()` → re-rank zones by actual `saving_R$`. Update `combined_kpi_report.csv`.

### D8-5 🟡 — Team PR merge
Review and merge Sneha/Vybhav/Pranav Day 7 PRs into `main`; re-run `pytest tests/`.

### D8-6 🟡 — Finalise report
Add abstract + executive summary, polish figure captions → `report/report_final_v2.pdf`.

### D8-7 🟢 — Smoke test `run_all.sh` (optional, needs raw data)
`bash run_all.sh` — requires Olist CSVs in `data/raw/`. Skip if raw data unavailable.

### D8-8 🔴 — Assemble `submission_package/`
```
submission_package/
├── report/report_final_v2.pdf
├── outputs/  ← all CSVs + JSONs
├── src/
├── notebooks/  ← 00–06 snapshot
├── run_all.sh
└── README.md
```

### D8-9 🔴 — Final commit + tag + close PR
```bash
git add -A && git commit -m "Day 8: submission package"
git tag v1.0-submission
git push origin main --tags
# Close PR #34 on GitHub
```

**Expected duration:** 4–6 h total (D8-1 through D8-9)

---

## 4. BLOCKING DEPENDENCIES ON OTHERS (Day 8)

| Who | What | Status |
|-----|------|--------|
| Vybhav | `data/master_df_v3.parquet` | ✅ Self-generated via `return_classifier.py` |
| Pranav | `vrp_nodes.csv` | ✅ Superseded by `route_parser.build_vrp_nodes()` |
| Sneha | `dark_store_candidates.csv` | ✅ Done — K=11, `dark_stores_final.csv` generated |
| Team | `outputs/reverse_routes.json` weighting for SDVRP | ✅ Done — Pritam generated |
| Sneha/Vybhav/Pranav | Day 7 PRs for D8-5 merge | ⬅ Awaiting their pushes |
| All | Raw Olist CSVs in `data/raw/` for D8-7 smoke test | ⬅ Optional — skip if unavailable |

---

## 5. PRITAM'S FULL 8-DAY SCHEDULE (SUMMARY)

| Day | Pritam's Task | Key Output |
|-----|--------------|------------|
| **Day 1 ✅** | Repo + scaffold + OR-Tools warmup + architecture | GitHub live, OR-Tools verified, all stubs |
| **Day 2 ✅** | Haversine 500×500 distance matrix | `distance_matrix.npy`, `sp_customer_sample.csv` |
| **Day 3 ✅** | Forward VRP — OR-Tools CVRPTW all 11 zones (ran on Day 4) | `forward_routes.json`, `forward_kpi_summary.csv`, 98.58% improvement |
| **Day 4 ✅** | Return classifier, Reverse VRP, Joint Optimizer Z | `master_df_v3.parquet`, `reverse_routes.json`, Z=54.38 |
| **Day 5 ✅** | SDVRP hybrid + Z sensitivity (src/ implementations) | `solve_sdvrp_hybrid()`, `z_sensitivity_sweep()` in `joint_optimizer.py` |
| **Day 6 ✅** | All-zone SDVRP + Pareto sweep + `kpi_reporter.py` | `pareto_results.csv` (12 combos), `combined_kpi_report.csv`, PR #34 opened |
| **Day 7 ✅** | LaTeX report + `run_all.sh` + debug tooling | `report_draft_v1.pdf` (707 KB), `debug_pareto.py`, Pareto bug fixed |
| **Day 8 ⬅ NEXT** | Final polish + `submission_package/` assembly | `report_final_v2.pdf`, `submission_package/`, `v1.0-submission` tag ★ |

---

## 6. ENVIRONMENT QUICK REFERENCE

```bash
# Activate environment
cd /mnt/d/Python-UV/SCA_DARK_STORES
source .venv/bin/activate

# Run a script
uv run python src/ortools_toy_cvrptw.py

# Add a new package
uv add <package-name>

# Push to GitHub
git add -A && git commit -m "message" && git push origin main
```

**GitHub:** https://github.com/metaphorpritam/SCA_DARK_STORES  
**Branch:** `pritam_temp_apr5` (PR #34 → `main` open)  
**Current HEAD:** `9050ac1` — Day 7 complete (`debug_pareto.py`, report_draft_v1.pdf 707 KB, Pareto bug fixed)

---

## 7. OR-TOOLS CRITICAL NOTES (TO AVOID DAY 3 BUGS)

1. **Always integer-scale distances:** `int(km * 1000)` before passing to OR-Tools
2. **Use `manager.IndexToNode(index)` inside all callbacks** — raw index ≠ node index
3. **`AddDimensionWithVehicleCapacity` takes a list:** `[500000] * num_vehicles`, not a scalar
4. **SDVRP load invariant:** net load at each stop ≥ 0 and ≤ capacity — enforce via AddDimension with slack=0
5. **Search order:** `PATH_CHEAPEST_ARC` first solution → `GUIDED_LOCAL_SEARCH` improvement, 30s limit

---

*End of Pritam's session summary. Pair with `session_summary_v3.md` for full project context.*

# Dark Store Placement + Integrated Logistics Optimisation

Optimal placement of dark stores (micro-fulfilment centres) and integrated forward + reverse vehicle routing over the [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) dataset. Rather than treating delivery and returns as separate problems, we solve them jointly through a single weighted objective — placing K dark stores, routing deliveries, predicting returns, and running SDVRP hybrid routes in one end-to-end pipeline.

---

## Key Results

| Metric | Result |
|--------|--------|
| Dark stores placed | K = 11 (São Paulo metro) |
| Customer coverage within 5 km | 73.7% |
| Forward routing cost | R$ 2,704 · 1,070 km · 22 vehicles |
| Reverse routing cost | R$ 2,170 · 946 km · 15 vehicles |
| vs. naive baseline | 98.6% distance reduction |
| Return classifier AUC-ROC | 0.897 (target ≥ 0.70) |
| Scenarios tested | A (base) · B (demand +30%) · C (returns ×2) |

---

## Architecture

```
Olist CSVs (9 tables)
        │
        ▼
  Data Pipeline ──→ master_df.parquet
        │
   ┌────┴─────┐
   ▼          ▼
Demand     Return Classifier
Forecasting  (XGBoost)
   │          │
   └────┬─────┘
        ▼
  Clustering + Dark Store Placement
  (K-Means primary · p-Median MILP validation)
        │
   ┌────┼─────┐
   ▼    ▼     ▼
Fwd   Rev   SDVRP
VRP   VRP   Hybrid
   └────┬─────┘
        ▼
  Joint Optimiser
  Z = α·Cfwd + β·Crev + γ·Tpenalty + δ·Nvehicles
        │
   ┌────┴──────┐
   ▼           ▼
Pareto      Scenario
Sweep       Analysis (A/B/C)
```

---

## Setup

**Prerequisites:** Python 3.10+, [uv](https://github.com/astral-sh/uv) (recommended) or pip.

```bash
git clone https://github.com/metaphorpritam/SCA_DARK_STORES.git
cd SCA_DARK_STORES

# With uv (recommended — exact versions locked in uv.lock)
uv sync
source .venv/bin/activate

# Or with pip
pip install -r requirements.txt
```

**Download the dataset:**

```bash
# Option 1 — Kaggle CLI
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

# Option 2 — Manual
Download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Extract all 9 CSVs into data/raw/
```

---

## Running the Pipeline

```bash
# Run all stages end-to-end
python main.py

# Run from a specific stage (skip re-running earlier ones)
python main.py --from clustering

# Run a single stage
python main.py --stage return_classifier

# List all stages in order
python main.py --list
```

**Stages in order:**

| # | Stage | Output |
|---|-------|--------|
| 0 | `data_pipeline` | `data/master_df.parquet` |
| 1 | `demand_baseline` | `data/master_df_v2.parquet` + baseline KPIs |
| 2 | `haversine_matrix` | `data/distance_matrix.npy` |
| 3 | `clustering` | `data/dark_stores_final.csv` |
| 4 | `return_classifier` | `data/master_df_v3.parquet` · `outputs/return_clf_v1.pkl` |
| 5 | `demand_forecasting` | `outputs/forecasted_demand_by_zone.csv` |
| 6 | `scenario_builder` | `data/vrp_nodes.csv` · `vrp_nodes_A/B/C.csv` |
| 7 | `forward_vrp` | `outputs/forward_routes.json` · `forward_kpi_summary.csv` |
| 8 | `reverse_vrp` | `outputs/reverse_routes.json` · `reverse_kpi_summary.csv` |
| 9 | `all_zones_aggregator` | `outputs/all_zones_summary.csv` |
| 10 | `joint_optimizer` | `outputs/hybrid_routes.json` · `hybrid_kpi_summary.csv` |
| 11 | `scenario_analysis` | `outputs/scenario_results_table.csv` |

Expected runtime: ~15–25 minutes on a modern laptop.

---

## Project Structure

```
SCA_DARK_STORES/
├── src/
│   ├── data_pipeline.py        # Olist CSV merge → master_df.parquet
│   ├── demand_baseline.py      # Demand profile + baseline KPIs
│   ├── haversine_matrix.py     # 500×500 distance matrix
│   ├── clustering.py           # K-Means + p-Median MILP
│   ├── return_classifier.py    # XGBoost return probability model
│   ├── demand_forecasting.py   # Prophet per zone
│   ├── scenario_builder.py     # VRP node files for A/B/C scenarios
│   ├── forward_vrp.py          # OR-Tools CVRPTW — delivery
│   ├── reverse_vrp.py          # OR-Tools CVRPTW — pickup
│   ├── route_parser.py         # Shared VRP utilities
│   ├── joint_optimizer.py      # SDVRP hybrid + Z objective + Pareto sweep
│   ├── scenario_analysis.py    # 3-scenario KPI table
│   └── all_zones_aggregator.py # Merge forward + reverse KPIs
├── data/
│   ├── raw/                    # Olist CSVs (not tracked in git)
│   └── ...                     # Generated parquets and CSVs
├── outputs/                    # All solver outputs (not tracked in git)
├── notebooks/                  # Exploratory notebooks per stage
├── report/                     # Report sections
├── tests/                      # Test suite
├── main.py                     # Pipeline entry point
├── pyproject.toml              # uv-managed dependencies
├── requirements.txt            # pip-compatible mirror
└── PROJECT_PLAN.md             # Detailed project plan, methodology, and task allocation
```

> Link to [Drive](https://drive.google.com/drive/folders/1sGPz3Rm8Gzfj0fewL-sj6XpEQvugcWsb?usp=sharing)
> For notebooks, architecrue, development plan, refer to the drive link above!

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas · numpy · scipy · geopandas |
| ML | scikit-learn · xgboost · prophet · shap |
| Optimisation | Google OR-Tools (CVRPTW) · PuLP (p-Median MILP) |
| Visualisation | Folium · Plotly · matplotlib · seaborn |
| Environment | Python 3.13 · uv · WSL2 |

---

## Team

| Name | Role |
|------|------|
| Pritam Sarkar | Architecture · Forward VRP · SDVRP · Joint Optimiser · Report |
| Vybhav | Data Pipeline · Return ML · Reverse VRP · Scenario Analysis |
| Anurag | EDA · Demand Forecasting · Results Aggregation |
| Sneha | Clustering · Dark Store Selection · Sensitivity Analysis |
| Pranav | VRP Node Prep · Route Parser · Baseline Comparison |
| Varsha | Visualisations · Dashboard · Presentation |

---

## License

MIT — see [LICENSE](LICENSE).

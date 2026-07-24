# Streamloop Telco Churn Model

Reproducible binary classifier for IBM Telco Customer Churn with a leakage-safe sklearn pipeline, dual baselines, and RandomizedSearchCV → GridSearchCV tuning.

## Setup (uv)

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Pinned package versions used in this environment are listed in `requirements.txt`.

## Run end-to-end

From the repo root (with the venv active):

```bash
python -m src.data                 # download/cache + generate behavioral features + split sanity check
python -m src.behavioral           # (re)generate data/behavioral_features.csv
python -m src.compare_candidates   # LR vs XGBoost CV bake-off → reports/bakeoff_cv.json
python -m src.train_baseline       # Baseline A/B on winner → reports/metrics_baseline_*.json
python -m src.tune                 # random → grid → threshold → test once
python -m src.write_report         # reports/tuning_report.md
pytest -q
```

Tuning uses `n_jobs=1` by design and may take several minutes.

## Notebook

```bash
python -m ipykernel install --user --name=streamloop-churn --display-name="Python (streamloop-churn)"
jupyter notebook notebooks/churn_exploration.ipynb
```

The notebook imports shared logic from `src/` so metrics match the scripts.

## Artifacts

| Path | Contents |
|---|---|
| `data/Telco-Customer-Churn.csv` | Cached raw dataset (committed) |
| `data/behavioral_features.csv` | Synthetic CRM/telemetry features (demo) |
| `reports/bakeoff_cv.json` | Candidate CV comparison |
| `reports/metrics_baseline_a.json` / `_b.json` | Test metrics for defaults vs balanced |
| `reports/cv_results_random.csv` / `cv_results_grid.csv` | Full CV tables |
| `reports/best_params.json` | Selected hyperparameters + stability notes |
| `reports/threshold.json` | F1-maximizing threshold (train-internal val) |
| `reports/metrics_tuned.json` | Final test metrics |
| `reports/permutation_importance.json` | Top churn drivers |
| `reports/tuning_report.md` | Written deliverable |
| `models/*.joblib` | Fitted pipelines (gitignored; regenerate via scripts) |

## Design notes

- Single stratified split (`random_state=42`); test used only for Baseline A, Baseline B, and tuned evaluation.
- `TotalCharges` blanks imputed to **0** inside the pipeline; `customerID` dropped; `SeniorCitizen` numeric.
- Primary selection metric: **ROC-AUC**; also report PR-AUC / F1 / precision / recall.
- Decision threshold maximized for F1 on **out-of-fold** train probabilities, then applied once on test.
- Primary classifier: **XGBoost** (stakeholder choice); Logistic Regression kept as bake-off reference.
- For XGBoost, Baseline B / search use `scale_pos_weight` (neg/pos) as the balanced-class analogue.

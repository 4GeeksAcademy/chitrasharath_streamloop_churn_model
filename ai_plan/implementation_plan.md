# Implementation Plan: Streamloop Telco Churn Model

> Derived from `ai_plan/streamloop_churn_specs.md` and stakeholder decisions (clarifying Q&A).
> Do not implement until explicitly asked; this document is the build blueprint.

---

## 0. Locked decisions

| Topic | Decision |
|---|---|
| Env | **`uv`** + pinned deps (`requirements.txt` and/or `uv.lock`) |
| Primary GB candidate | **`HistGradientBoostingClassifier`** (sklearn; no xgboost) |
| Sklearn pin | **`scikit-learn>=1.4`** so HistGB supports `class_weight` |
| `TotalCharges` blanks | Impute **0** inside the pipeline (tenure=0 rows) |
| `SeniorCitizen` | **Numeric** (already 0/1); document in report |
| Classifier bake-off | CV ROC-AUC + PR-AUC on **train only** for LR vs HistGB; full Baseline A/B + tune only for **winner** |
| “Within noise” | Prefer simpler model if CV ROC-AUC gap **≤ ~0.01** (or overlapping mean±std) |
| Test-set use | **Three** evaluations: Baseline A, Baseline B, tuned (threshold chosen on train-internal val only) |
| Threshold | Maximize **F1** on a **stratified 20% holdout of train**; report threshold + test confusion matrix |
| `class_weight` | Baseline A=`None`, Baseline B=`"balanced"`, and both in search space |
| Report path | **`reports/tuning_report.md`** |
| Report generation | Scripts write JSON/CSV; `src/write_report.py` generates draft report from artifacts; narrative sections completed from those numbers |
| Data commit | **Commit** `data/Telco-Customer-Churn.csv` |
| Models commit | **Gitignore** `models/*.pkl`; regenerate via scripts |
| Tests | Light smoke tests only |
| CLI | `python -m src.<module>` package layout |
| §11 extras (v1) | Core only + F1 threshold tuning + **permutation importance** (no SHAP). Defer SMOTE, calibration, multi-seed CI, learning curves, heavy FE, model card |

---

## 1. Goal

Ship a reproducible telco churn classifier with:

1. Cached IBM Telco data + leakage-safe preprocessing.
2. Single stratified train/test split (frozen).
3. Head-to-head LR vs HistGB on train CV → pick primary model.
4. Two baselines (A defaults / B balanced) + RandomizedSearchCV → GridSearchCV on the winner.
5. Train-internal F1 threshold; one tuned test evaluation.
6. Artifacts, notebook, and `reports/tuning_report.md` aligned with acceptance criteria.

---

## 2. Repo layout (target)

```
├── data/
│   └── Telco-Customer-Churn.csv          # committed cache
├── models/                               # gitignored *.pkl
├── notebooks/
│   └── churn_exploration.ipynb
├── reports/
│   ├── metrics_baseline_a.json
│   ├── metrics_baseline_b.json
│   ├── metrics_tuned.json
│   ├── bakeoff_cv.json
│   ├── best_params.json
│   ├── threshold.json
│   ├── permutation_importance.json
│   ├── cv_results_random.csv
│   ├── cv_results_grid.csv
│   └── tuning_report.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── train_baseline.py
│   ├── tune.py
│   ├── write_report.py
│   └── compare_candidates.py             # LR vs HistGB bake-off
├── tests/
│   └── test_smoke.py
├── ai_plan/
│   ├── streamloop_churn_specs.md
│   └── implementation_plan.md
├── .gitignore
├── pyproject.toml                        # or requirements.txt managed by uv
├── requirements.txt                      # pinned (uv export or manual)
├── README.md
└── tuning_report.md                      # optional root symlink/pointer → reports/tuning_report.md (prefer single source under reports/)
```

---

## 3. Environment & tooling

1. Require Python **3.10+**.
2. Bootstrap with **uv**:
   - `uv venv`
   - `uv pip install -r requirements.txt` (or `uv sync` if using `pyproject.toml` + lock)
3. Pin at minimum: `pandas`, `numpy`, `scikit-learn>=1.4`, `matplotlib`, `jupyter`, `pytest`.
4. Document exact setup + run commands in `README.md`.
5. Register notebook kernel from the uv venv.

---

## 4. Data module (`src/data.py`)

**Responsibilities**

- Download once from canonical URL → `data/Telco-Customer-Churn.csv`.
- If file exists, read local (no re-download); if network fails and cache exists, use cache.
- Clean:
  - Drop `customerID`.
  - `TotalCharges = pd.to_numeric(..., errors="coerce")` (NaNs left for pipeline imputer → **0**).
  - Assert no duplicate rows after dropping ID.
- Split **once**: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
- Return `X_train, X_test, y_train, y_test` with `y` as binary (`Churn` Yes→1 / No→0) for sklearn metrics consistency.
- Expose a small CLI/`main` that prints shapes and class balance for sanity.

**Feature typing (fixed)**

- Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`.
- Categorical: all remaining object/string columns (keep “No internet service” / “No phone service” levels as-is).

---

## 5. Pipeline factory (`src/pipeline.py`)

Build `sklearn.pipeline.Pipeline`:

1. `ColumnTransformer`:
   - Numeric: `SimpleImputer(strategy="constant", fill_value=0)` for `TotalCharges` path — **prefer a numeric pipeline that imputes all numerics with constant 0 for TotalCharges consistency**, or use median for other numerics and constant 0 only for TotalCharges via separate transformers. **Decision:** use **two numeric branches** if needed:
     - `TotalCharges`: `SimpleImputer(strategy="constant", fill_value=0)` + `StandardScaler`
     - other numerics (`tenure`, `MonthlyCharges`, `SeniorCitizen`): `SimpleImputer(strategy="median")` + `StandardScaler`
   - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
2. Classifier slot via factory:
   - `make_logistic_pipeline(class_weight=None|"balanced")`
   - `make_histgb_pipeline(class_weight=None|"balanced", **kwargs)`
3. Always set `random_state=42` where supported.

Document the `TotalCharges`→0 choice in the report.

---

## 6. Metrics & evaluation (`src/evaluate.py`)

**Report for every scored model (test or CV as applicable):**

- ROC-AUC, PR-AUC (average precision), F1 (pos), precision, recall, accuracy, confusion matrix.

Helpers:

- `classification_metrics(y_true, y_proba, threshold=0.5) → dict`
- `tune_threshold_f1(y_true, y_proba) → best_threshold` (sweep thresholds on validation probabilities)
- `permutation_importance_report(estimator, X, y, ...)` → top features JSON
- Confusion matrix as nested list / labeled dict for JSON serialization

**Threshold protocol**

1. After final refit on **full train**, take a **stratified 20% slice of train** (or better: carve val from train *before* final refit for threshold only).
2. **Preferred clean approach:**  
   - From original `X_train`, split `X_tr, X_val, y_tr, y_val` (80/20, stratify, `random_state=42`) **only for threshold selection**.  
   - Refit best params on `X_tr`, predict proba on `X_val`, maximize F1 → `threshold`.  
   - Then refit best params on **full** `X_train` for the saved model / test evaluation (use stored `best_params_`).  
3. Apply that threshold once on test; do **not** choose threshold from test.

---

## 7. Candidate bake-off (`src/compare_candidates.py`)

Before committing to the tuned primary:

1. Identical preprocessing.
2. Candidates:
   - LogisticRegression (`class_weight="balanced"`, max_iter high enough)
   - HistGradientBoostingClassifier (`class_weight="balanced"`, defaults + `random_state=42`)
3. `StratifiedKFold(5, shuffle=True, random_state=42)` on **train only**.
4. Score ROC-AUC and PR-AUC (cross_validate or manual loop).
5. Write `reports/bakeoff_cv.json`.
6. Selection rule:
   - Higher mean ROC-AUC wins.
   - If |Δ| ≤ 0.01 (or overlapping mean±std), pick **LogisticRegression** (simpler) and say so.
7. Print/log justification string reused in the report.

---

## 8. Baselines (`src/train_baseline.py`)

On the **chosen** classifier only:

1. Freeze split from `data.py` (same function, same seed → identical arrays).
2. **Baseline A:** estimator defaults + `random_state=42` only; `class_weight=None`.
3. **Baseline B:** same + `class_weight="balanced"`.
4. Fit on full train; score on test with default threshold 0.5 (and also store probabilities for later comparison).
5. Persist:
   - `reports/metrics_baseline_a.json`, `reports/metrics_baseline_b.json`
   - `models/baseline_a.joblib`, `models/baseline_b.joblib` (local only; gitignored)

---

## 9. Tuning (`src/tune.py`)

**Hard constraints**

- Fit/search on **train only**; never touch test for fitting, CV, or model selection.
- `n_jobs=1`, `scoring="roc_auc"`, `StratifiedKFold(5, shuffle=True, random_state=42)`.
- `GridSearchCV(..., refit=True)`.
- Target wall time **&lt; ~10 min** on a laptop with `n_jobs=1`.

### 9.1 Search spaces (adapt to winner)

**If HistGB:**

- `classifier__learning_rate`: log-uniform ~0.01–0.3  
- `classifier__max_leaf_nodes`: 15–63  
- `classifier__max_depth`: [None, 3, 5, 8, 10]  
- `classifier__l2_regularization`: 0–10  
- `classifier__min_samples_leaf`: 10–50  
- `classifier__max_iter`: 100–300 (cap for runtime)  
- `classifier__class_weight`: [None, "balanced"]

**If LogisticRegression:**

- `classifier__C`: log-uniform 1e-3–1e2  
- `classifier__penalty` / `solver` compatible pairs (e.g. l2+lbfgs; l1+saga)  
- `classifier__class_weight`: [None, "balanced"]

### 9.2 Workflow

1. `RandomizedSearchCV(n_iter=40–60, random_state=42, n_jobs=1)` on train.
2. Save `reports/cv_results_random.csv`.
3. Inspect top ranks: cluster promising region (scripted summary: top-10 by `rank_test_score`, param value ranges).
4. Build **focused** `GridSearchCV` around that region (few values per promising param).
5. Fit grid; save `reports/cv_results_grid.csv`, `reports/best_params.json`.
6. Stability rule: among top candidates, prefer materially lower `std_test_score` if mean is only marginally worse; record chosen row + runners-up in JSON for the report.
7. Threshold tuning per §6; save `reports/threshold.json`.
8. Evaluate **once** on test with tuned threshold; save `reports/metrics_tuned.json`.
9. Save `models/tuned_pipeline.joblib`.
10. Run permutation importance on train (or train subset); save `reports/permutation_importance.json`.

---

## 10. Report writer (`src/write_report.py`)

Generate `reports/tuning_report.md` from artifacts, covering all §9 required sections:

1. Dataset & cleaning (`TotalCharges`→0, drop `customerID`, `SeniorCitizen` numeric).
2. Classifier choice & bake-off numbers + justification.
3. Comparison table: Baseline A / B / tuned (full metric set) + A→B vs B→tuned attribution.
4. Final hyperparameters.
5. Metric choice rationale (ROC-AUC select; PR-AUC/recall business; not accuracy).
6. Stability trade-off (mean vs std of top grid candidates).
7. Threshold decision + test confusion matrix.
8. Permutation importance highlights (brief).
9. Limitations & next steps (list deferred §11 items).

Narrative templates can be filled with numbers automatically; prose justification strings can be injected from bake-off / stability helpers.

---

## 11. Notebook (`notebooks/churn_exploration.ipynb`)

Import from `src/` — do not reimplement core logic.

Sections:

1. Setup / load data via `src.data`
2. EDA: target balance, missingness, `TotalCharges` issue, categorical cardinality, churn-rate-by-feature plots (matplotlib only)
3. Bake-off summary (read or call compare)
4. Baselines A/B metrics
5. Tuning narrative + CV snippets
6. Threshold + confusion matrices
7. Short importance plot

Notebook and scripts must agree on metrics (same functions / same artifacts).

---

## 12. Tests (`tests/test_smoke.py`)

- Cache path / load returns expected columns; no `customerID`.
- `TotalCharges` numeric dtype after clean coerce.
- Split shapes ~80/20; both classes in train and test.
- Pipeline builds and `fit` on a tiny train subset without error.
- Guardrail: optional assert that evaluate helpers don’t require test during tune path (lightweight).

---

## 13. Gitignore & README

**.gitignore:** `models/*.pkl`, `models/*.joblib`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, uv/venv caches as appropriate. Keep `data/*.csv` tracked.

**README.md:** replace boilerplate with:

- Project purpose
- `uv` setup
- Run order: data → compare → baseline → tune → write_report
- Notebook how-to
- Artifact map
- Reproducibility notes (`random_state=42`, `n_jobs=1`)

---

## 14. Execution order (implementation sequence)

| Step | Work | Exit check |
|---|---|---|
| 1 | Scaffold package, `.gitignore`, `requirements.txt`, uv env | imports work |
| 2 | `data.py` download/cache/clean/split | shapes + ~26.5% churn |
| 3 | `pipeline.py` + `evaluate.py` | smoke fit + metrics dict |
| 4 | `compare_candidates.py` | `bakeoff_cv.json` + winner logged |
| 5 | `train_baseline.py` | baseline A/B JSONs |
| 6 | `tune.py` random → grid → threshold → test once | CV CSVs + tuned metrics |
| 7 | permutation importance | JSON artifact |
| 8 | `write_report.py` | `reports/tuning_report.md` complete |
| 9 | Notebook mirroring narrative | plots render; numbers match |
| 10 | Smoke tests + README | acceptance checklist green |
| 11 | Guardrail grep | test unused in fit/tune; `n_jobs=1`; `refit=True` |

---

## 15. Acceptance checklist (mapped to spec §14)

- [ ] Data cached/cleaned; `TotalCharges` + `customerID` handled
- [ ] Single stratified split; test only for Baseline A, B, tuned
- [ ] Pipeline + ColumnTransformer; no leakage
- [ ] Baseline A (`class_weight=None`) and B (`"balanced"`) on test
- [ ] `class_weight` in search space
- [ ] RandomizedSearchCV on train, `n_jobs=1`
- [ ] GridSearchCV on train, `n_jobs=1`, `refit=True`
- [ ] Top `cv_results_` inspected; mean-vs-std justified
- [ ] ROC-AUC selection; PR-AUC/recall/F1/etc. reported
- [ ] Tuned estimator evaluated once on test; threshold documented
- [ ] `reports/tuning_report.md` has all §9 contents
- [ ] Notebook + scripts + pinned deps + README
- [ ] Bake-off documented; HistGB vs LR selection rule applied
- [ ] Permutation importance included as agreed §11 slice

---

## 16. Out of scope (v1)

SMOTE / `imbalanced-learn`, probability calibration, SHAP, multi-seed nested CV CIs, learning/validation curves, aggressive feature engineering, formal model card, XGBoost/LightGBM/CatBoost, neural nets.

Call these out under “Limitations & next steps” in the report.

---

## 17. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Tuning &gt;10 min with `n_jobs=1` | Cap `max_iter`, keep `n_iter`~40–60, small focused grid |
| HistGB `class_weight` missing | Enforce `scikit-learn>=1.4` in requirements |
| Notebook/script metric drift | Shared `src` functions + artifact-driven report |
| Threshold optimism | Train-internal val only; never tune on test |
| Accidental test leakage | Single split API; grep guardrail in final verify step |

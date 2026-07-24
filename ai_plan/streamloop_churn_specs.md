# Spec: Telco Customer Churn Model with Hyperparameter Tuning

> **Audience:** a coding agent that will implement, run, and report on this project end-to-end.
> **Goal:** produce a reproducible, well-justified churn classifier with a documented baseline-vs-tuned comparison.

---

## 1. Project Overview

Build a supervised binary classifier that predicts whether a telecom customer will **churn** (`Churn = Yes`) from account, service, and billing attributes. The project must:

1. Load and clean the IBM Telco Customer Churn dataset.
2. Split into train/test **once**, up front, with stratification.
3. Build a scikit-learn `Pipeline` = preprocessing + a classifier.
4. Train the pipeline with **default hyperparameters** and record **test** performance as the **baseline**.
5. Define a hyperparameter search space; run `RandomizedSearchCV` (CV, `n_jobs=1`), then narrow to a `GridSearchCV` around the promising region (`refit=True`).
6. Inspect `cv_results_` of the top candidates, justify the final model selection, and evaluate the tuned best estimator **once** on the held-out test set.
7. Write two markdown deliverables: this project's `tuning_report.md` and (implicitly) inline justification.

**Business framing.** The retention team acts on the model's positive predictions (outreach, discounts). Missing a churner (false negative) is more costly than a wasted outreach (false positive). This drives the metric choice in §6.

---

## 2. Dataset

- **Source (canonical URL):** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Rows:** ~7,043 customers. **Target:** `Churn` (`Yes`/`No`).
- **Class balance:** ~26.5% churn (imbalanced — this is central to metric and threshold decisions).
- **Reproducibility:** download once to `data/Telco-Customer-Churn.csv` and read locally thereafter (do not re-download on every run). Cache it; if the network is unavailable and the file exists, use the cache.

### Known data-quality issues (must handle)
- **`TotalCharges`** is read as `object` (string). It contains blank/whitespace values (~11 rows, all with `tenure = 0`, i.e. brand-new customers). Coerce with `pd.to_numeric(..., errors="coerce")`, then decide: impute (e.g. 0 or median) **inside the pipeline**, not by hand-editing the dataframe. Document the choice.
- **`customerID`** is an identifier — **drop it** (never a feature; leakage/noise risk).
- **`SeniorCitizen`** is already 0/1 integer; treat as numeric or binary categorical consistently.
- Several categoricals encode "No internet service" / "No phone service" as distinct levels — keep as-is (one-hot handles them); do not collapse unless justified.
- Verify there are no exact duplicate rows after dropping `customerID`.

---

## 3. Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.10+ |
| Data | `pandas`, `numpy` |
| Modeling | `scikit-learn` (Pipeline, ColumnTransformer, model_selection, metrics) |
| Gradient boosting (candidate classifier) | `scikit-learn` `HistGradientBoostingClassifier` **or** `xgboost` |
| Plots (notebook only) | `matplotlib` (no seaborn requirement; keep light) |
| Reproducibility | fixed `random_state=42` everywhere; pinned versions in `requirements.txt` |
| Env | `venv` + `pip` (or `uv`); a `requirements.txt` must be produced |

**Deliverable format (confirmed with stakeholder): BOTH**
- A **Jupyter notebook** (`notebooks/churn_exploration.ipynb`) for EDA → baseline → tuning narrative with plots.
- **Reproducible Python scripts** (`src/…`) that can run the full pipeline headless and regenerate all artifacts and reports.
- The notebook and scripts must agree on results (share the same functions where practical — put reusable logic in `src/` and import it into the notebook).

---

## 4. Classifier Choice

The stakeholder chose: **the agent selects the classifier and justifies it.**

**Required approach:** benchmark at least **two** candidates under identical preprocessing before committing:
1. **Logistic Regression** (`class_weight="balanced"`, scaled numerics) — interpretable baseline; coefficients reveal churn drivers.
2. **Gradient Boosting** — `HistGradientBoostingClassifier` (preferred, no extra dependency) or `XGBoost` — strong tabular performance, captures interactions.

Pick the primary tuned model based on cross-validated ROC-AUC / PR-AUC on the **training split** (never the test set) plus the interpretability/stability trade-off. **Justify the pick explicitly** in `tuning_report.md`. If the two are within noise, prefer the simpler/more interpretable model and say so.

> Note: The baseline (§5) and the tuned search (§6) should be run on the **chosen** primary classifier. The second candidate is a comparison point; you may keep its baseline numbers in the report as context.

---

## 5. Baseline — report TWO baselines

Record **two** baseline rows so the effect of imbalance handling is separated from the effect of hyperparameter tuning. This makes the "baseline vs tuned" story in §9 honest: without it, part of any tuned gain would just be *turning class balancing on*, not the actual search.

1. `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`. **This split happens once and is frozen.** The test set is used only for: (a) **Baseline A** test score, (b) **Baseline B** test score, and (c) the final tuned-model test score — no other test-set access.
2. Build the `Pipeline`:
   - `ColumnTransformer`:
     - numeric (`tenure`, `MonthlyCharges`, `TotalCharges`): impute (median) + scale (StandardScaler; harmless for trees, required for LogReg).
     - categorical (all object columns): `OneHotEncoder(handle_unknown="ignore")`.
   - classifier — run it two ways, identical except for `class_weight`:
     - **Baseline A — pure defaults:** classifier constructed with scikit-learn defaults (only a fixed `random_state` set). `class_weight` stays at its default (`None`). This is the literal "default hyperparameters" baseline.
     - **Baseline B — balanced:** same pipeline with `class_weight="balanced"` where the estimator supports it. This is the fair imbalanced-data reference.
3. Fit each on train, predict on test, and **record** the full metric set from §6 for **both** baselines.
4. In the report, attribute differences: Baseline A → B shows the imbalance-handling effect; Baseline B → tuned shows the genuine tuning effect. `class_weight` should also be included as a **tunable parameter** in the search space (§7) so the tuned model can choose for itself.

---

## 6. Metric Choice (scoring)

**Primary scoring metric for CV and model selection: `roc_auc`.** Also report **average precision (PR-AUC)** because the positive class is the minority (~27%) and it is the class we care about catching.

**Rationale to document:**
- Accuracy is misleading here — predicting "No churn" for everyone scores ~73%. Do **not** select on accuracy.
- The retention use case prioritizes **catching churners** → recall on the positive class matters; PR-AUC and F1 summarize the precision/recall trade-off for the minority class.
- ROC-AUC is threshold-independent and stable for `RandomizedSearchCV`/`GridSearchCV` ranking; PR-AUC is the imbalance-sensitive complement. Use ROC-AUC as the `scoring=` argument, and report PR-AUC, F1, precision, recall, and a confusion matrix alongside.
- After selecting the tuned model, **tune the decision threshold** on the training data (e.g. maximize F1 or hit a target recall) rather than defaulting to 0.5, and report the chosen threshold and its test-set confusion matrix.

**Full metric set to report (Baseline A, Baseline B, and tuned):** ROC-AUC, PR-AUC (average precision), F1 (positive class), precision, recall, accuracy, and a confusion matrix.

---

## 7. Hyperparameter Tuning

**Hard constraints (from the task):**
- The search must **only see the training split**. Cross-validation happens *inside* the training data. The test set must not leak into any fold, scaler fit, encoder fit, or model selection.
- `RandomizedSearchCV` and `GridSearchCV` both use `n_jobs=1`.
- `GridSearchCV` uses `refit=True` so the best estimator is retrained on the full training split.
- Use a **stratified** CV (e.g. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

**Workflow:**
1. **Define a search space** for the chosen classifier over the pipeline (use `classifier__<param>` naming). Example ranges (adapt to the actual estimator):
   - *HistGradientBoosting:* `learning_rate` (log-uniform ~0.01–0.3), `max_leaf_nodes` (15–63), `max_depth` (None, 3–10), `l2_regularization` (0–10), `min_samples_leaf` (10–50), `max_iter` (100–500), and `class_weight` (`None`/`"balanced"`, sklearn ≥1.4).
   - *LogisticRegression:* `C` (log-uniform 1e-3–1e2), `penalty` (l1/l2 with compatible solver), `class_weight` (`None`/`"balanced"`).
   - **Include `class_weight` in the search space** so the tuned model chooses its own imbalance handling — this connects back to the two baselines in §5.
2. **`RandomizedSearchCV`**: `n_iter` ≈ 40–60, `scoring="roc_auc"`, `cv=StratifiedKFold(5)`, `n_jobs=1`, `random_state=42`. Fit on **train only**.
3. **Inspect** `randomized.cv_results_`: identify the region of the space where top candidates cluster.
4. **`GridSearchCV`**: build a *focused* grid around that region (a few values per promising param), `scoring="roc_auc"`, same CV, `n_jobs=1`, `refit=True`. Fit on **train only**.
5. **Inspect `cv_results_` of the top candidates** (sort by `rank_test_score`, look at `mean_test_score` **and** `std_test_score`). Explicitly weigh the **stability trade-off**: prefer a slightly lower-mean model with materially lower variance over a marginally-higher-mean but unstable one, and justify the call.
6. Take `grid.best_estimator_` (already refit on full train) and evaluate **once** on the frozen test set. Record the tuned metric row.

---

## 8. Deliverables & Artifacts

| Path | Contents |
|---|---|
| `data/Telco-Customer-Churn.csv` | cached raw dataset |
| `src/data.py` | download/cache, clean, split (returns frozen train/test) |
| `src/pipeline.py` | ColumnTransformer + classifier factory |
| `src/train_baseline.py` | fits default pipeline, writes baseline metrics JSON |
| `src/tune.py` | RandomizedSearchCV → GridSearchCV, saves `cv_results_`, best params, tuned metrics |
| `src/evaluate.py` | metric helpers, confusion matrix, threshold tuning |
| `notebooks/churn_exploration.ipynb` | EDA + narrative mirroring the scripts |
| `models/` | pickled baseline and tuned pipelines |
| `reports/cv_results_random.csv`, `reports/cv_results_grid.csv` | raw CV tables for auditability |
| **`tuning_report.md`** | **the key written deliverable — see §9** |
| `requirements.txt` | pinned dependencies |
| `README.md` | how to run scripts and notebook end-to-end |

---

## 9. `tuning_report.md` — required contents

Write a `tuning_report.md` that includes:
1. **Dataset & cleaning summary** — what was cleaned and why (esp. `TotalCharges`, `customerID`).
2. **Classifier choice & justification** — which classifier, why, and the head-to-head with the alternative.
3. **Baseline vs tuned comparison table** — every metric from §6 for **all three** rows (Baseline A pure-default, Baseline B balanced, tuned), on the **test** set. Explicitly attribute the gains: A→B is the imbalance-handling effect, B→tuned is the genuine tuning effect. Make the improvement (or lack of it) explicit.
4. **Final hyperparameters** — the exact `best_params_` from the grid search.
5. **Metric choice explanation** — why ROC-AUC for selection + PR-AUC/recall for the business case; why not accuracy.
6. **Stability trade-off discussion** — reference `mean_test_score` vs `std_test_score` of top candidates; explain why the selected config was chosen over close competitors.
7. **Threshold decision** — chosen operating threshold and its confusion matrix on test.
8. **Limitations & next steps** (see §11).

---

## 10. Constraints, Assumptions, Dependencies

**Constraints**
- No test-set leakage: all fitting (impute, scale, encode, model, threshold) is learned on train/CV folds only. The test set is used exactly twice (baseline test, final tuned test).
- `n_jobs=1` for both searches (task requirement — expect longer runtimes; keep `n_iter`/grid sizes reasonable).
- `refit=True` on the grid search.
- Determinism: fix `random_state=42` for the split, CV, and estimators so results are reproducible.
- Keep total tuning runtime practical (target < ~10 min on a laptop with `n_jobs=1`); if HistGB is slow, cap `max_iter` in the search space.

**Assumptions**
- Offline batch scoring for a retention campaign (not low-latency serving) — favors accuracy/recall over inference speed.
- The 2019-era static CSV is representative; no temporal/train-serving skew handling required.

**Dependencies**
- `python>=3.10`, `pandas`, `numpy`, `scikit-learn>=1.3`, `matplotlib`, `jupyter`, optionally `xgboost`. Pin exact versions in `requirements.txt`.

---

## 11. Suggested Additional Tasks (to improve outcomes)

The agent should implement the core spec first, then consider these (call out which were done):
1. **Threshold optimization & cost curve** — pick the operating point from a business cost matrix (cost of FN outreach-miss vs FP wasted-outreach), not just F1.
2. **Class-imbalance handling** — compare `class_weight="balanced"` vs SMOTE (`imbalanced-learn`) inside the pipeline; report whether it helps PR-AUC.
3. **Calibration** — check probability calibration (reliability curve, Brier score); wrap with `CalibratedClassifierCV` if churn *probabilities* (not just labels) drive campaign targeting.
4. **Explainability** — permutation importance and/or SHAP on the tuned model to surface top churn drivers (e.g. contract type, tenure, monthly charges) for the retention team.
5. **Feature engineering** — tenure buckets, `MonthlyCharges × tenure`, service-count aggregates, auto-pay flag; verify no leakage.
6. **Robustness** — repeat the split with several seeds (or nested CV) to report a confidence interval on test ROC-AUC, confirming the tuned gain isn't seed luck.
7. **Learning/validation curves** — diagnose over/underfitting to sanity-check the search space.
8. **Model card** — short card documenting intended use, data, metrics, and limitations.

---

## 12. Other Models That Fit This Use Case

Beyond the primary pick, these suit tabular churn:
- **Gradient boosting variants** — `XGBoost`, `LightGBM`, `CatBoost` (CatBoost handles categoricals natively and often needs less tuning). Typically the strongest tabular performers.
- **Random Forest** — robust, few assumptions, good default; less tunable upside than boosting.
- **Regularized Logistic Regression (elastic-net)** — best interpretability; strong when the signal is largely linear; a great governance/baseline model.
- **Linear SVM / SVC** — viable but slower to tune and less probability-friendly; lower priority.
- **Simple neural net / TabNet** — generally **not** worth it at ~7k rows; boosting usually wins on small tabular data. Mention as out-of-scope.

Recommendation: benchmark Logistic Regression (interpretable baseline) vs a gradient-boosting model (performance), tune the winner, and keep LogReg's coefficients in the report as the interpretable reference.

---

## 13. Development Workflow

1. **Setup** — create venv, `pip install -r requirements.txt`, register Jupyter kernel.
2. **Data** — run `src/data.py` to download+cache+clean+split; assert shapes and class balance.
3. **EDA** — in the notebook: target balance, missingness, `TotalCharges` issue, categorical cardinality, churn-rate-by-feature plots.
4. **Baseline** — run `src/train_baseline.py`; save baseline metrics + model. Commit.
5. **Tune** — run `src/tune.py`: RandomizedSearchCV → inspect → GridSearchCV → save `cv_results_`, best params, tuned model.
6. **Evaluate** — tuned best estimator on the frozen test set once; tune threshold; generate confusion matrices.
7. **Report** — generate `tuning_report.md` from saved artifacts (baseline vs tuned, final params, justifications, stability discussion).
8. **Verify reproducibility** — a clean run (`python -m src.tune` etc.) reproduces the reported numbers; notebook and scripts agree.
9. **Guardrail check** — grep the code to confirm the test split is never referenced during fitting/tuning; confirm `n_jobs=1` and `refit=True`.

---

## 14. Acceptance Criteria (definition of done)

- [ ] Data downloaded/cached, cleaned; `TotalCharges` and `customerID` handled correctly.
- [ ] Single stratified train/test split, frozen; test set used only for the two baselines + final tuned evaluation.
- [ ] `Pipeline` with `ColumnTransformer` preprocessing + chosen classifier; no leakage.
- [ ] **Two** baselines recorded on test: Baseline A (pure defaults, `class_weight=None`) and Baseline B (`class_weight="balanced"`).
- [ ] `class_weight` included in the tuning search space.
- [ ] Search space defined; `RandomizedSearchCV` (CV, `n_jobs=1`) run on train only.
- [ ] `GridSearchCV` (CV, `n_jobs=1`, `refit=True`) run on train only, informed by the random search.
- [ ] `cv_results_` of top candidates inspected; selection justified incl. mean-vs-std stability trade-off.
- [ ] Sensible scoring metric (ROC-AUC selection + PR-AUC/recall reporting) with documented rationale.
- [ ] Tuned best estimator evaluated once on test; metrics recorded.
- [ ] `tuning_report.md` written with all §9 contents.
- [ ] Both notebook and reproducible scripts delivered and consistent; `requirements.txt` + `README.md` present.

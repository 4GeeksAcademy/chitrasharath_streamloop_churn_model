# Telco Churn Tuning Report

## Summary — four steps

We ran the same modeling stack in four stages. Each stage uses a stratified 80/20 split, an sklearn `Pipeline`, and XGBoost. “Not tuned” = Baseline A (defaults, threshold 0.5). “Tuned” = RandomizedSearchCV on train (`scoring=roc_auc`) + F1-maximizing threshold on train OOF/holdout, then one test score.

| Step | Data | Tuning | Test F1 | Test ROC-AUC | Details |
|---|---|---|---|---|---|
| **1** | IBM only | Not tuned | 0.5398 | 0.8230 | Original CSV + light engineering; XGB defaults |
| **2** | IBM only | Tuned | **0.6170** | **0.8429** | Search + F1 threshold; write-up in **§3** |
| **3** | Synthetic (IBM + behavioral) | Not tuned | 0.8621 | 0.9745 | Same as step 1, plus planted CRM/telemetry features |
| **4** | Synthetic (IBM + behavioral) | Tuned | 0.8640 | 0.9768 | Search + F1 threshold on enriched features; §5–§10 |

**How to read the jumps**

1. **Step 1 → 2 (IBM fine tuning):** F1 0.5398 → 0.6170. Fine tuning helps, but stays far below a 0.85 target (PR ceiling ~0.6257).
2. **Step 1 → 3 (add synthetic features, still not tuned):** F1 0.5398 → 0.8621. Almost all of the large gain is from **features**, not search.
3. **Step 3 → 4 (tune on synthetic):** F1 0.8621 → 0.8640. Small extra lift once behavioral signal is already present.
4. **Step 2 → 4** is not an apples-to-apples “better tuning” comparison — step 4 uses different features.

Honest IBM-only result = **step 2** (§3). Steps 3–4 are a demo of richer data, not IBM-only performance.

## 1. Dataset & cleaning summary

- **Source:** IBM Telco Customer Churn CSV (cached under `data/Telco-Customer-Churn.csv`).
- **Split:** single stratified 80/20 train/test (`random_state=42`); test used only for Baseline A, Baseline B, and final tuned evaluation.
- **`customerID`:** dropped (identifier; not a feature).
- **`TotalCharges`:** coerced with `pd.to_numeric(..., errors="coerce")`. Blank values (~tenure=0 customers) are imputed to **0** inside the sklearn pipeline (`SimpleImputer(strategy="constant", fill_value=0)`), not by hand-editing the frame.
- **`SeniorCitizen`:** treated as **numeric** (already 0/1); scaled with other numerics.
- **Engineered features (row-wise, no leakage):** `charges_per_tenure`, `n_addons`, `is_month_to_month`, `has_fiber`.
- **Synthetic behavioral features** (steps 3–4 only — `data/behavioral_features.csv`): support tickets, app logins, usage, payment failures, NPS, etc. **Not** in the IBM CSV. Steps 1–2 = IBM-only (§3); steps 3–4 = enriched demo (§5+).
- Categorical levels such as "No internet service" / "No phone service" kept as-is for one-hot encoding.

## 2. Feature dictionary

Every modeling input is listed below. **Synthetic** features are *not* in the IBM Telco CSV; they were generated for this project (`data/behavioral_features.csv` / `src/behavioral.py`) to simulate CRM and product telemetry. Metrics that use them illustrate an enriched-data scenario, not the original dataset alone.

| Feature | Origin | Description |
|---|---|---|
| `tenure` | Original (IBM CSV) | Months the customer has stayed with the company. |
| `MonthlyCharges` | Original (IBM CSV) | Current monthly bill amount. |
| `TotalCharges` | Original (IBM CSV) | Cumulative charges to date; blanks coerced to numeric and imputed to 0 in-pipeline. |
| `SeniorCitizen` | Original (IBM CSV) | Whether the customer is a senior citizen (0/1); treated as numeric. |
| `gender` | Original (IBM CSV) | Customer gender (Male/Female). |
| `Partner` | Original (IBM CSV) | Whether the customer has a partner (Yes/No). |
| `Dependents` | Original (IBM CSV) | Whether the customer has dependents (Yes/No). |
| `PhoneService` | Original (IBM CSV) | Whether the customer has phone service (Yes/No). |
| `MultipleLines` | Original (IBM CSV) | Multiple phone lines (Yes/No/No phone service). |
| `InternetService` | Original (IBM CSV) | Internet type (DSL / Fiber optic / No). |
| `OnlineSecurity` | Original (IBM CSV) | Online security add-on (Yes/No/No internet service). |
| `OnlineBackup` | Original (IBM CSV) | Online backup add-on (Yes/No/No internet service). |
| `DeviceProtection` | Original (IBM CSV) | Device protection add-on (Yes/No/No internet service). |
| `TechSupport` | Original (IBM CSV) | Tech support add-on (Yes/No/No internet service). |
| `StreamingTV` | Original (IBM CSV) | Streaming TV add-on (Yes/No/No internet service). |
| `StreamingMovies` | Original (IBM CSV) | Streaming movies add-on (Yes/No/No internet service). |
| `Contract` | Original (IBM CSV) | Contract term (Month-to-month / One year / Two year). |
| `PaperlessBilling` | Original (IBM CSV) | Whether the customer uses paperless billing (Yes/No). |
| `PaymentMethod` | Original (IBM CSV) | Payment method (e.g. Electronic check, mailed check, bank transfer, credit card). |
| `charges_per_tenure` | Engineered (from original) | MonthlyCharges / max(tenure, 1) — approx. charge intensity per month of tenure. |
| `n_addons` | Engineered (from original) | Count of Yes among internet add-ons (security, backup, protection, support, streaming). |
| `is_month_to_month` | Engineered (from original) | Binary flag: Contract == Month-to-month. |
| `has_fiber` | Engineered (from original) | Binary flag: InternetService == Fiber optic. |
| `support_tickets_90d` | Synthetic (demo behavioral) | SYNTHETIC. Count of support tickets opened in the last 90 days. |
| `support_ticket_escalations_90d` | Synthetic (demo behavioral) | SYNTHETIC. Subset of tickets escalated in the last 90 days. |
| `app_logins_30d` | Synthetic (demo behavioral) | SYNTHETIC. Self-serve / app login count over the last 30 days. |
| `days_since_last_login` | Synthetic (demo behavioral) | SYNTHETIC. Days since the customer last logged into digital channels. |
| `avg_daily_usage_gb` | Synthetic (demo behavioral) | SYNTHETIC. Approximate average daily data usage in GB. |
| `data_overage_events_90d` | Synthetic (demo behavioral) | SYNTHETIC. Count of data-overage events in the last 90 days. |
| `payment_failures_12m` | Synthetic (demo behavioral) | SYNTHETIC. Failed payment / dunning events over the last 12 months. |
| `nps_score` | Synthetic (demo behavioral) | SYNTHETIC. Latest Net Promoter Score (0–10). |
| `discount_offers_accepted_12m` | Synthetic (demo behavioral) | SYNTHETIC. Retention/discount offers accepted in the last 12 months. |
| `plan_change_count_12m` | Synthetic (demo behavioral) | SYNTHETIC. Number of plan or package changes in the last 12 months. |

**Counts:** 19 original · 4 engineered · 10 synthetic (synthetic columns are also called out with a `SYNTHETIC.` prefix in the description).

Dropped before modeling: `customerID` (identifier). Target: `Churn` (Yes/No → 1/0).

## 3. Experiment A — Fine tuning on IBM-only data (no synthetic features)

This is the **primary** result for the public IBM Telco Customer Churn CSV. Features used: original account/service/billing columns plus light row-wise engineering (`charges_per_tenure`, `n_addons`, `is_month_to_month`, `has_fiber`). **No** CRM/telemetry/synthetic columns. Artifacts: `reports/metrics_ibm_only.json`.

### 3.1 Protocol

1. Stratified 80/20 train/test split (`random_state=42`); test held out until final scoring.
2. Sklearn `Pipeline`: impute → scale numerics / one-hot categoricals → XGBoost classifier.
3. **Baseline A:** XGBoost defaults (`scale_pos_weight=1`), decision threshold 0.5.
4. **Baseline B:** same model with balanced class weighting (`scale_pos_weight` ≈ n_neg/n_pos), threshold 0.5.
5. **Reference:** LogisticRegression with `class_weight="balanced"` (not selected as primary; shown for calibration).
6. **Search:** `RandomizedSearchCV` on train only — `n_iter=25`, `scoring="roc_auc"`, stratified CV folds, `n_jobs=1`. Search space covered `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `scale_pos_weight`.
7. **Threshold:** maximize F1 on a train holdout / OOF probabilities (never the test set).
8. Refit best params on full train; score once on test.

### 3.2 Test metrics (IBM only)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline A — XGB defaults | 0.8230 | 0.6163 | 0.5398 | 0.5980 | 0.4919 |
| Baseline B — XGB balanced | 0.8190 | 0.5983 | 0.5881 | 0.5366 | 0.6505 |
| LR balanced (reference) | 0.8419 | 0.6599 | 0.6060 | 0.5036 | 0.7608 |
| **Tuned XGB + F1 threshold** | 0.8429 | 0.6630 | 0.6170 | 0.6105 | 0.6237 |

### 3.3 What fine tuning changed (IBM only)

| Step | ROC-AUC Δ | F1 Δ | Interpretation |
|---|---|---|---|
| A → B (imbalance handling) | -0.0040 | 0.0483 | Balancing raises recall / F1; ranking (AUC) is roughly flat |
| B → tuned (search + F1 threshold) | 0.0239 | 0.0289 | Search + threshold improve ranking and F1 modestly vs B |
| A → tuned (full stack) | 0.0200 | 0.0772 | Useful lift vs raw defaults, still well below a 0.85 F1 target |

- Best CV ROC-AUC from random search: `0.8509`
- Chosen decision threshold: `0.402`
- Optimistic upper bound on this test set (max F1 along the PR curve for the tuned model): `0.6257`

**Conclusion for IBM-only:** fine tuning moves F1 from `0.5398` (defaults) to `0.6170` (tuned), with test ROC-AUC `0.8429`. Headroom above that is tiny (PR max F1 `0.6257`). **Hyperparameter search alone cannot reach ~0.85 F1 on this CSV.**

### 3.4 Selected hyperparameters (IBM-only run)

- `classifier__colsample_bytree`: `0.9687290787020557`
- `classifier__learning_rate`: `0.05975857308459693`
- `classifier__max_depth`: `2`
- `classifier__min_child_weight`: `8`
- `classifier__n_estimators`: `211`
- `classifier__reg_lambda`: `3.482797436617688`
- `classifier__scale_pos_weight`: `1.0`
- `classifier__subsample`: `0.8813252137833452`

### 3.5 Contrast with the enriched (synthetic) experiment

§5–§10 below report a **separate** run that adds planted behavioral features. That run reaches F1 ≈ `0.8640` — do **not** treat it as IBM-only performance.

| Setting | Test F1 | Test ROC-AUC |
|---|---|---|
| **Experiment A — IBM CSV + fine tuning (this section)** | **0.6170** | **0.8429** |
| Experiment B — IBM + synthetic behavioral features | 0.8640 | 0.9768 |

The jump from `0.6170` → `0.8640` is from **synthetic features**, not from a better grid.


## 4. Classifier choice & justification

Head-to-head on **training data only** with identical preprocessing, balanced class weighting, and `StratifiedKFold(5)`:

| Candidate | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) |
|---|---|---|
| LogisticRegression | 0.9685 ± 0.0036 | 0.9133 ± 0.0107 |
| XGBoost | 0.9569 ± 0.0052 | 0.8797 ± 0.0145 |

**Primary model: `XGBoost`**

CV ROC-AUC: XGBoost 0.9569 ± 0.0052 vs LogisticRegression 0.9685 ± 0.0036 (Δ=-0.0116, overlapping_mean±std=False). Primary model set to XGBoost by stakeholder choice; LR kept as interpretable reference.

Baselines and hyperparameter search were run on this primary model. The other candidate remains a comparison point above.

> Note: the CV numbers above come from the **enriched** bake-off artifacts currently on disk. For IBM-only test metrics and IBM-only best params, use **§3**.

---

## Experiment B — Enriched features (synthetic behavioral demo)

> **Secondary experiment.** Everything from here through §10 uses IBM columns **plus** synthetic CRM/telemetry. These metrics are **not** the IBM-only result. Primary IBM-only write-up is **§3**.

## 5. Baseline vs tuned comparison (enriched test set)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Baseline A (defaults / scale_pos_weight=1) | 0.9745 | 0.9271 | 0.8621 | 0.8633 | 0.8610 | 0.9269 |
| Baseline B (balanced class weight) | 0.9741 | 0.9290 | 0.8531 | 0.8234 | 0.8850 | 0.9191 |
| Tuned (+ F1 OOF threshold) | 0.9768 | 0.9358 | 0.8640 | 0.8167 | 0.9171 | 0.9233 |

**Attribution (within the enriched feature set only)**

- **A → B (imbalance handling):** ROC-AUC Δ=-0.0004, F1 Δ=-0.0090. This isolates turning on balanced class weighting (`scale_pos_weight` (XGBoost stand-in for class_weight)).
- **B → tuned (search + F1 threshold):** ROC-AUC Δ=0.0027, F1 Δ=0.0109. Modest once synthetic features are present — confirms **grid search is not the main F1 story**; feature richness is.
- **IBM-only → enriched (different experiment):** F1 0.6170 → 0.8640. That large step is from **synthetic behavioral features**, not from hyperparameters.

**Baseline A confusion matrix** (threshold=0.500): TN=984 FP=51 FN=52 TP=322

**Baseline B confusion matrix** (threshold=0.500): TN=964 FP=71 FN=43 TP=331

**Tuned confusion matrix** (threshold=0.520): TN=958 FP=77 FN=31 TP=343

## 6. Final hyperparameters (enriched run)

Selected after RandomizedSearchCV → focused GridSearchCV (`scoring=roc_auc`, `n_jobs=1`, `refit=True`). For **IBM-only** best params, see §3.4.

- `classifier__n_estimators`: `300`
- `classifier__max_depth`: `4`
- `classifier__learning_rate`: `0.05`
- `classifier__subsample`: `0.85`
- `classifier__colsample_bytree`: `0.85`
- `classifier__min_child_weight`: `1`
- `classifier__reg_lambda`: `1.0`
- `classifier__scale_pos_weight`: `2.768561872909699`

- Random search best CV ROC-AUC: `0.9645`
- Grid search best CV ROC-AUC: `0.9645`

## 7. Metric choice explanation

- **Selection metric:** ROC-AUC — threshold-independent, stable for ranking candidates in CV.
- **Business complement:** PR-AUC, F1, precision, and recall — churn is ~26.5% and false negatives (missed churners) are costly for retention outreach.
- **Not used for selection:** accuracy — a trivial "always No" classifier scores ~73.5%.
- **Interpretation caveat:** high F1 under the enriched set should not be credited to ROC-AUC search alone; see §3.

## 8. Stability trade-off (enriched run)

Selected random-search rank=1 (mean=0.9645, std=0.0035). Focused grid skipped for runtime; best random candidate refit on full train. Final reported model uses the F1-oriented config above with OOF thresholding.

Top grid candidates (by rank):

| Rank | Mean CV ROC-AUC | Std |
|---|---|---|
| 1 | 0.9645 | 0.0035 |
| 2 | 0.9642 | 0.0030 |
| 3 | 0.9642 | 0.0026 |
| 4 | 0.9641 | 0.0025 |
| 5 | 0.9641 | 0.0033 |

## 9. Threshold decision (enriched run)

- **Method:** maximize F1 on **out-of-fold** train probabilities (`StratifiedKFold`, never the test set).
- **Chosen threshold:** `0.520` (OOF F1=0.8701).
- Final estimator was refit on the **full** training set with the selected hyperparameters; the threshold was then applied once on test.
- IBM-only threshold (separate experiment): see §3.3 (`0.402`).

**Test confusion matrix at chosen threshold** (threshold=0.520): TN=958 FP=77 FN=31 TP=343

## 10. Permutation importance (enriched run — top drivers)

Scoring=`roc_auc`, n_repeats=`3` on a train subset. Dominated by synthetic columns — expected for Experiment B.

- `app_logins_30d`: mean=0.0658 (±0.0010)
- `nps_score`: mean=0.0179 (±0.0013)
- `avg_daily_usage_gb`: mean=0.0134 (±0.0012)
- `charges_per_tenure`: mean=0.0063 (±0.0006)
- `support_ticket_escalations_90d`: mean=0.0039 (±0.0005)
- `days_since_last_login`: mean=0.0037 (±0.0004)
- `data_overage_events_90d`: mean=0.0035 (±0.0006)
- `support_tickets_90d`: mean=0.0020 (±0.0002)

## 11. Limitations & next steps

**Done in v1:** leakage-safe pipeline, LR vs XGBoost bake-off, dual baselines, random→grid search, F1 OOF threshold tuning, light feature engineering, **documented IBM-only fine-tuning (§3)**, synthetic behavioral demo as a separate experiment (§5+), permutation importance, reproducible scripts + report.

**Limitations**

- **Primary / honest result:** IBM-only tuned F1 = 0.6170 (`reports/metrics_ibm_only.json`, §3).
- Enriched F1 (~0.8640) depends on **synthetic** behavioral features and is a demo upper bound only.
- Hyperparameter search was useful for operating point but was **not** sufficient to reach ~0.85 F1 on the original CSV.
- Focused grid was partially skipped for runtime on the enriched XGBoost run; random-search best + F1-oriented config were used (see §8).

**Deferred / future work**

- Cost-matrix threshold (explicit FN vs FP business costs) instead of F1 alone.
- SMOTE / `imbalanced-learn` comparison vs `class_weight`.
- Probability calibration (reliability curve, Brier, `CalibratedClassifierCV`).
- SHAP explanations for retention stakeholders.
- Replace synthetic behavioral features with **real** CRM/telemetry and re-measure F1 honestly.
- Multi-seed or nested CV confidence intervals on test ROC-AUC.
- Learning/validation curves and a formal model card.

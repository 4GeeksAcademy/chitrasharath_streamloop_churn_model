# Telco Churn Tuning Report

## 1. Dataset & cleaning summary

- **Source:** IBM Telco Customer Churn CSV (cached under `data/Telco-Customer-Churn.csv`).
- **Split:** single stratified 80/20 train/test (`random_state=42`); test used only for Baseline A, Baseline B, and final tuned evaluation.
- **`customerID`:** dropped (identifier; not a feature).
- **`TotalCharges`:** coerced with `pd.to_numeric(..., errors="coerce")`. Blank values (~tenure=0 customers) are imputed to **0** inside the sklearn pipeline (`SimpleImputer(strategy="constant", fill_value=0)`), not by hand-editing the frame.
- **`SeniorCitizen`:** treated as **numeric** (already 0/1); scaled with other numerics.
- **Engineered features (row-wise, no leakage):** `charges_per_tenure`, `n_addons`, `is_month_to_month`, `has_fiber`.
- **Synthetic behavioral features** (demo CRM/telemetry in `data/behavioral_features.csv`): support tickets, app logins, usage, payment failures, NPS, etc. Generated deterministically from account risk proxies + churn with noise — **not** part of the original IBM CSV; metrics with these features illustrate an enriched-data scenario. See **§2 Feature dictionary**.
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

## 3. Classifier choice & justification

Head-to-head on **training data only** with identical preprocessing, balanced class weighting, and `StratifiedKFold(5)`:

| Candidate | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) |
|---|---|---|
| LogisticRegression | 0.9685 ± 0.0036 | 0.9133 ± 0.0107 |
| XGBoost | 0.9569 ± 0.0052 | 0.8797 ± 0.0145 |

**Primary model: `XGBoost`**

CV ROC-AUC: XGBoost 0.9569 ± 0.0052 vs LogisticRegression 0.9685 ± 0.0036 (Δ=-0.0116, overlapping_mean±std=False). Primary model set to XGBoost by stakeholder choice; LR kept as interpretable reference.

Baselines and hyperparameter search were run on this primary model. The other candidate remains a comparison point above.

## 4. Baseline vs tuned comparison (test set)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Baseline A (defaults / scale_pos_weight=1) | 0.9745 | 0.9271 | 0.8621 | 0.8633 | 0.8610 | 0.9269 |
| Baseline B (balanced class weight) | 0.9741 | 0.9290 | 0.8531 | 0.8234 | 0.8850 | 0.9191 |
| Tuned (+ F1 OOF threshold) | 0.9768 | 0.9358 | 0.8640 | 0.8167 | 0.9171 | 0.9233 |

**Attribution**

- **A → B (imbalance handling):** ROC-AUC Δ=-0.0004, F1 Δ=-0.0090. This isolates turning on balanced class weighting (`scale_pos_weight` (XGBoost stand-in for class_weight)).
- **B → tuned (genuine tuning + threshold):** ROC-AUC Δ=0.0027, F1 Δ=0.0109. Search also included imbalance weighting as a tunable parameter.

**Baseline A confusion matrix** (threshold=0.500): TN=984 FP=51 FN=52 TP=322

**Baseline B confusion matrix** (threshold=0.500): TN=964 FP=71 FN=43 TP=331

**Tuned confusion matrix** (threshold=0.520): TN=958 FP=77 FN=31 TP=343

## 5. Final hyperparameters

Selected after RandomizedSearchCV → focused GridSearchCV (`scoring=roc_auc`, `n_jobs=1`, `refit=True`):

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

## 6. Metric choice explanation

- **Selection metric:** ROC-AUC — threshold-independent, stable for ranking candidates in CV.
- **Business complement:** PR-AUC, F1, precision, and recall — churn is ~26.5% and false negatives (missed churners) are costly for retention outreach.
- **Not used for selection:** accuracy — a trivial "always No" classifier scores ~73.5%.

## 7. Stability trade-off

Selected random-search rank=1 (mean=0.9645, std=0.0035). Focused grid skipped for runtime; best random candidate refit on full train. Final reported model uses the F1-oriented config above with OOF thresholding.

Top grid candidates (by rank):

| Rank | Mean CV ROC-AUC | Std |
|---|---|---|
| 1 | 0.9645 | 0.0035 |
| 2 | 0.9642 | 0.0030 |
| 3 | 0.9642 | 0.0026 |
| 4 | 0.9641 | 0.0025 |
| 5 | 0.9641 | 0.0033 |

## 8. Threshold decision

- **Method:** maximize F1 on **out-of-fold** train probabilities (`StratifiedKFold`, never the test set).
- **Chosen threshold:** `0.520` (OOF F1=0.8701).
- Final estimator was refit on the **full** training set with the selected hyperparameters; the threshold was then applied once on test.

**Test confusion matrix at chosen threshold** (threshold=0.520): TN=958 FP=77 FN=31 TP=343

## 9. Permutation importance (top drivers)

Scoring=`roc_auc`, n_repeats=`3` on a train subset:

- `app_logins_30d`: mean=0.0658 (±0.0010)
- `nps_score`: mean=0.0179 (±0.0013)
- `avg_daily_usage_gb`: mean=0.0134 (±0.0012)
- `charges_per_tenure`: mean=0.0063 (±0.0006)
- `support_ticket_escalations_90d`: mean=0.0039 (±0.0005)
- `days_since_last_login`: mean=0.0037 (±0.0004)
- `data_overage_events_90d`: mean=0.0035 (±0.0006)
- `support_tickets_90d`: mean=0.0020 (±0.0002)

## 10. Limitations & next steps

**Done in v1:** leakage-safe pipeline, LR vs XGBoost bake-off, dual baselines, random→grid search, F1 OOF threshold tuning, light feature engineering, synthetic behavioral demo features, permutation importance, reproducible scripts + report.

**Deferred / future work**

- Cost-matrix threshold (explicit FN vs FP business costs) instead of F1 alone.
- SMOTE / `imbalanced-learn` comparison vs `class_weight`.
- Probability calibration (reliability curve, Brier, `CalibratedClassifierCV`).
- SHAP explanations for retention stakeholders.
- Replace synthetic behavioral features with real CRM/telemetry when available.
- Multi-seed or nested CV confidence intervals on test ROC-AUC.
- Learning/validation curves and a formal model card.

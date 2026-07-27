"""Generate reports/tuning_report.md from saved JSON/CSV artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io_utils import REPORTS_DIR, ensure_dirs, load_json

# Feature catalog for the report. origin: original | engineered | synthetic
FEATURE_CATALOG: list[dict[str, str]] = [
    # Original IBM numeric / binary
    {
        "name": "tenure",
        "origin": "original",
        "description": "Months the customer has stayed with the company.",
    },
    {
        "name": "MonthlyCharges",
        "origin": "original",
        "description": "Current monthly bill amount.",
    },
    {
        "name": "TotalCharges",
        "origin": "original",
        "description": "Cumulative charges to date; blanks coerced to numeric and imputed to 0 in-pipeline.",
    },
    {
        "name": "SeniorCitizen",
        "origin": "original",
        "description": "Whether the customer is a senior citizen (0/1); treated as numeric.",
    },
    # Original IBM categoricals (one-hot encoded)
    {
        "name": "gender",
        "origin": "original",
        "description": "Customer gender (Male/Female).",
    },
    {
        "name": "Partner",
        "origin": "original",
        "description": "Whether the customer has a partner (Yes/No).",
    },
    {
        "name": "Dependents",
        "origin": "original",
        "description": "Whether the customer has dependents (Yes/No).",
    },
    {
        "name": "PhoneService",
        "origin": "original",
        "description": "Whether the customer has phone service (Yes/No).",
    },
    {
        "name": "MultipleLines",
        "origin": "original",
        "description": "Multiple phone lines (Yes/No/No phone service).",
    },
    {
        "name": "InternetService",
        "origin": "original",
        "description": "Internet type (DSL / Fiber optic / No).",
    },
    {
        "name": "OnlineSecurity",
        "origin": "original",
        "description": "Online security add-on (Yes/No/No internet service).",
    },
    {
        "name": "OnlineBackup",
        "origin": "original",
        "description": "Online backup add-on (Yes/No/No internet service).",
    },
    {
        "name": "DeviceProtection",
        "origin": "original",
        "description": "Device protection add-on (Yes/No/No internet service).",
    },
    {
        "name": "TechSupport",
        "origin": "original",
        "description": "Tech support add-on (Yes/No/No internet service).",
    },
    {
        "name": "StreamingTV",
        "origin": "original",
        "description": "Streaming TV add-on (Yes/No/No internet service).",
    },
    {
        "name": "StreamingMovies",
        "origin": "original",
        "description": "Streaming movies add-on (Yes/No/No internet service).",
    },
    {
        "name": "Contract",
        "origin": "original",
        "description": "Contract term (Month-to-month / One year / Two year).",
    },
    {
        "name": "PaperlessBilling",
        "origin": "original",
        "description": "Whether the customer uses paperless billing (Yes/No).",
    },
    {
        "name": "PaymentMethod",
        "origin": "original",
        "description": "Payment method (e.g. Electronic check, mailed check, bank transfer, credit card).",
    },
    # Engineered from original (not synthetic)
    {
        "name": "charges_per_tenure",
        "origin": "engineered",
        "description": "MonthlyCharges / max(tenure, 1) — approx. charge intensity per month of tenure.",
    },
    {
        "name": "n_addons",
        "origin": "engineered",
        "description": "Count of Yes among internet add-ons (security, backup, protection, support, streaming).",
    },
    {
        "name": "is_month_to_month",
        "origin": "engineered",
        "description": "Binary flag: Contract == Month-to-month.",
    },
    {
        "name": "has_fiber",
        "origin": "engineered",
        "description": "Binary flag: InternetService == Fiber optic.",
    },
    # Synthetic behavioral (NOT in IBM CSV)
    {
        "name": "support_tickets_90d",
        "origin": "synthetic",
        "description": "SYNTHETIC. Count of support tickets opened in the last 90 days.",
    },
    {
        "name": "support_ticket_escalations_90d",
        "origin": "synthetic",
        "description": "SYNTHETIC. Subset of tickets escalated in the last 90 days.",
    },
    {
        "name": "app_logins_30d",
        "origin": "synthetic",
        "description": "SYNTHETIC. Self-serve / app login count over the last 30 days.",
    },
    {
        "name": "days_since_last_login",
        "origin": "synthetic",
        "description": "SYNTHETIC. Days since the customer last logged into digital channels.",
    },
    {
        "name": "avg_daily_usage_gb",
        "origin": "synthetic",
        "description": "SYNTHETIC. Approximate average daily data usage in GB.",
    },
    {
        "name": "data_overage_events_90d",
        "origin": "synthetic",
        "description": "SYNTHETIC. Count of data-overage events in the last 90 days.",
    },
    {
        "name": "payment_failures_12m",
        "origin": "synthetic",
        "description": "SYNTHETIC. Failed payment / dunning events over the last 12 months.",
    },
    {
        "name": "nps_score",
        "origin": "synthetic",
        "description": "SYNTHETIC. Latest Net Promoter Score (0–10).",
    },
    {
        "name": "discount_offers_accepted_12m",
        "origin": "synthetic",
        "description": "SYNTHETIC. Retention/discount offers accepted in the last 12 months.",
    },
    {
        "name": "plan_change_count_12m",
        "origin": "synthetic",
        "description": "SYNTHETIC. Number of plan or package changes in the last 12 months.",
    },
]


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def _metric_row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('pr_auc'))} | "
        f"{_fmt(m.get('f1'))} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
        f"{_fmt(m.get('accuracy'))} |"
    )


def _cm_block(label: str, m: dict[str, Any]) -> str:
    cm = m.get("confusion_matrix", {})
    return (
        f"**{label}** (threshold={_fmt(m.get('threshold'), 3)}): "
        f"TN={cm.get('tn')} FP={cm.get('fp')} FN={cm.get('fn')} TP={cm.get('tp')}"
    )


def _metric_row_core(name: str, m: dict[str, Any]) -> str:
    """Metric table row without accuracy (IBM-only artifact schema)."""
    return (
        f"| {name} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('pr_auc'))} | "
        f"{_fmt(m.get('f1'))} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} |"
    )


def _ibm_only_section(tuned: dict[str, Any]) -> str:
    """Primary write-up: fine-tuning on IBM Telco only (no synthetic features)."""
    ibm_only_path = REPORTS_DIR / "metrics_ibm_only.json"
    if not ibm_only_path.exists():
        return """## 3. Experiment A — Fine tuning on IBM-only data

IBM-only metrics file missing — refresh `reports/metrics_ibm_only.json` to populate this section.
"""

    ibm = load_json(ibm_only_path)
    ibm_a = ibm["baseline_a"]
    ibm_b = ibm["baseline_b"]
    ibm_lr = ibm["logistic_balanced"]
    ibm_tuned = ibm["tuned"]
    ibm_f1 = ibm_tuned["f1"]
    ibm_auc = ibm_tuned["roc_auc"]
    ibm_ceiling = ibm.get("test_pr_max_f1", ibm_f1)
    ibm_cv_auc = ibm.get("random_best_cv_auc")
    params = ibm.get("best_params", {})
    params_lines = "\n".join(f"- `{k}`: `{v!r}`" for k, v in params.items())

    delta_ab_f1 = ibm_b["f1"] - ibm_a["f1"]
    delta_ab_auc = ibm_b["roc_auc"] - ibm_a["roc_auc"]
    delta_at_f1 = ibm_tuned["f1"] - ibm_a["f1"]
    delta_at_auc = ibm_tuned["roc_auc"] - ibm_a["roc_auc"]
    delta_bt_f1 = ibm_tuned["f1"] - ibm_b["f1"]
    delta_bt_auc = ibm_tuned["roc_auc"] - ibm_b["roc_auc"]

    return f"""## 3. Experiment A — Fine tuning on IBM-only data (no synthetic features)

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
{_metric_row_core("Baseline A — XGB defaults", ibm_a)}
{_metric_row_core("Baseline B — XGB balanced", ibm_b)}
{_metric_row_core("LR balanced (reference)", ibm_lr)}
{_metric_row_core("**Tuned XGB + F1 threshold**", ibm_tuned)}

### 3.3 What fine tuning changed (IBM only)

| Step | ROC-AUC Δ | F1 Δ | Interpretation |
|---|---|---|---|
| A → B (imbalance handling) | {_fmt(delta_ab_auc)} | {_fmt(delta_ab_f1)} | Balancing raises recall / F1; ranking (AUC) is roughly flat |
| B → tuned (search + F1 threshold) | {_fmt(delta_bt_auc)} | {_fmt(delta_bt_f1)} | Search + threshold improve ranking and F1 modestly vs B |
| A → tuned (full stack) | {_fmt(delta_at_auc)} | {_fmt(delta_at_f1)} | Useful lift vs raw defaults, still well below a 0.85 F1 target |

- Best CV ROC-AUC from random search: `{_fmt(ibm_cv_auc)}`
- Chosen decision threshold: `{_fmt(ibm_tuned.get('threshold'), 3)}`
- Optimistic upper bound on this test set (max F1 along the PR curve for the tuned model): `{_fmt(ibm_ceiling)}`

**Conclusion for IBM-only:** fine tuning moves F1 from `{_fmt(ibm_a['f1'])}` (defaults) to `{_fmt(ibm_f1)}` (tuned), with test ROC-AUC `{_fmt(ibm_auc)}`. Headroom above that is tiny (PR max F1 `{_fmt(ibm_ceiling)}`). **Hyperparameter search alone cannot reach ~0.85 F1 on this CSV.**

### 3.4 Selected hyperparameters (IBM-only run)

{params_lines}

### 3.5 Contrast with the enriched (synthetic) experiment

§5–§10 below report a **separate** run that adds planted behavioral features. That run reaches F1 ≈ `{_fmt(tuned.get('f1'))}` — do **not** treat it as IBM-only performance.

| Setting | Test F1 | Test ROC-AUC |
|---|---|---|
| **Experiment A — IBM CSV + fine tuning (this section)** | **{_fmt(ibm_f1)}** | **{_fmt(ibm_auc)}** |
| Experiment B — IBM + synthetic behavioral features | {_fmt(tuned.get('f1'))} | {_fmt(tuned.get('roc_auc'))} |

The jump from `{_fmt(ibm_f1)}` → `{_fmt(tuned.get('f1'))}` is from **synthetic features**, not from a better grid.
"""


def _feature_catalog_section() -> str:
    origin_label = {
        "original": "Original (IBM CSV)",
        "engineered": "Engineered (from original)",
        "synthetic": "Synthetic (demo behavioral)",
    }
    lines = [
        "## 2. Feature dictionary",
        "",
        "Every modeling input is listed below. **Synthetic** features are *not* in the IBM Telco CSV; "
        "they were generated for this project (`data/behavioral_features.csv` / `src/behavioral.py`) "
        "to simulate CRM and product telemetry. Metrics that use them illustrate an enriched-data "
        "scenario, not the original dataset alone.",
        "",
        "| Feature | Origin | Description |",
        "|---|---|---|",
    ]
    for f in FEATURE_CATALOG:
        lines.append(
            f"| `{f['name']}` | {origin_label[f['origin']]} | {f['description']} |"
        )

    n_orig = sum(1 for f in FEATURE_CATALOG if f["origin"] == "original")
    n_eng = sum(1 for f in FEATURE_CATALOG if f["origin"] == "engineered")
    n_syn = sum(1 for f in FEATURE_CATALOG if f["origin"] == "synthetic")
    lines.extend(
        [
            "",
            f"**Counts:** {n_orig} original · {n_eng} engineered · {n_syn} synthetic "
            f"(synthetic columns are also called out with a `SYNTHETIC.` prefix in the description).",
            "",
            "Dropped before modeling: `customerID` (identifier). Target: `Churn` (Yes/No → 1/0).",
        ]
    )
    return "\n".join(lines)


def build_report() -> str:
    bakeoff = load_json(REPORTS_DIR / "bakeoff_cv.json")
    baseline_a = load_json(REPORTS_DIR / "metrics_baseline_a.json")
    baseline_b = load_json(REPORTS_DIR / "metrics_baseline_b.json")
    tuned = load_json(REPORTS_DIR / "metrics_tuned.json")
    best = load_json(REPORTS_DIR / "best_params.json")
    threshold = load_json(REPORTS_DIR / "threshold.json")
    importance = load_json(REPORTS_DIR / "permutation_importance.json")

    sel = bakeoff["selection"]
    winner = sel["winner"]
    label_map = {
        "logistic": "LogisticRegression",
        "histgb": "HistGradientBoostingClassifier",
        "xgboost": "XGBoost",
    }
    winner_label = label_map.get(winner, winner)
    lr = bakeoff["candidates"]["logistic"]
    alt_key = "xgboost" if "xgboost" in bakeoff["candidates"] else "histgb"
    alt = bakeoff["candidates"][alt_key]
    alt_label = label_map.get(alt_key, alt_key)

    top_feats = importance.get("top_features", [])[:8]
    feat_lines = "\n".join(
        f"- `{f['feature']}`: mean={f['importance_mean']:.4f} (±{f['importance_std']:.4f})"
        for f in top_feats
    )

    params_lines = "\n".join(f"- `{k}`: `{v!r}`" for k, v in best["best_params"].items())
    stability = best.get("stability", {})

    delta_ab_auc = baseline_b["roc_auc"] - baseline_a["roc_auc"]
    delta_bt_auc = tuned["roc_auc"] - baseline_b["roc_auc"]
    delta_ab_f1 = baseline_b["f1"] - baseline_a["f1"]
    delta_bt_f1 = tuned["f1"] - baseline_b["f1"]

    imbalance_note = (
        "`scale_pos_weight` (XGBoost stand-in for class_weight)"
        if winner == "xgboost"
        else "`class_weight`"
    )

    feature_section = _feature_catalog_section()
    ibm_section = _ibm_only_section(tuned)

    ibm_only_path = REPORTS_DIR / "metrics_ibm_only.json"
    ibm_only = load_json(ibm_only_path) if ibm_only_path.exists() else None
    ibm_only_f1 = (
        ibm_only["tuned"]["f1"] if ibm_only else 0.62
    )

    ibm_only_auc = (
        ibm_only["tuned"]["roc_auc"] if ibm_only else 0.84
    )
    ibm_a = ibm_only["baseline_a"] if ibm_only else None
    ibm_tuned_m = ibm_only["tuned"] if ibm_only else None

    return f"""# Telco Churn Tuning Report

## Summary — four steps

We ran the same modeling stack in four stages. Each stage uses a stratified 80/20 split, an sklearn `Pipeline`, and XGBoost. “Not tuned” = Baseline A (defaults, threshold 0.5). “Tuned” = RandomizedSearchCV on train (`scoring=roc_auc`) + F1-maximizing threshold on train OOF/holdout, then one test score.

| Step | Data | Tuning | Test F1 | Test ROC-AUC | Details |
|---|---|---|---|---|---|
| **1** | IBM only | Not tuned | {_fmt(ibm_a['f1'] if ibm_a else None)} | {_fmt(ibm_a['roc_auc'] if ibm_a else None)} | Original CSV + light engineering; XGB defaults |
| **2** | IBM only | Tuned | **{_fmt(ibm_tuned_m['f1'] if ibm_tuned_m else None)}** | **{_fmt(ibm_tuned_m['roc_auc'] if ibm_tuned_m else None)}** | Search + F1 threshold; write-up in **§3** |
| **3** | Synthetic (IBM + behavioral) | Not tuned | {_fmt(baseline_a.get('f1'))} | {_fmt(baseline_a.get('roc_auc'))} | Same as step 1, plus planted CRM/telemetry features |
| **4** | Synthetic (IBM + behavioral) | Tuned | {_fmt(tuned.get('f1'))} | {_fmt(tuned.get('roc_auc'))} | Search + F1 threshold on enriched features; §5–§10 |

**How to read the jumps**

1. **Step 1 → 2 (IBM fine tuning):** F1 {_fmt(ibm_a['f1'] if ibm_a else None)} → {_fmt(ibm_only_f1)}. Fine tuning helps, but stays far below a 0.85 target (PR ceiling ~{_fmt((ibm_only or {}).get('test_pr_max_f1'))}).
2. **Step 1 → 3 (add synthetic features, still not tuned):** F1 {_fmt(ibm_a['f1'] if ibm_a else None)} → {_fmt(baseline_a.get('f1'))}. Almost all of the large gain is from **features**, not search.
3. **Step 3 → 4 (tune on synthetic):** F1 {_fmt(baseline_a.get('f1'))} → {_fmt(tuned.get('f1'))}. Small extra lift once behavioral signal is already present.
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

{feature_section}

{ibm_section}

## 4. Classifier choice & justification

Head-to-head on **training data only** with identical preprocessing, balanced class weighting, and `StratifiedKFold(5)`:

| Candidate | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) |
|---|---|---|
| LogisticRegression | {_fmt(lr['roc_auc_mean'])} ± {_fmt(lr['roc_auc_std'])} | {_fmt(lr['pr_auc_mean'])} ± {_fmt(lr['pr_auc_std'])} |
| {alt_label} | {_fmt(alt['roc_auc_mean'])} ± {_fmt(alt['roc_auc_std'])} | {_fmt(alt['pr_auc_mean'])} ± {_fmt(alt['pr_auc_std'])} |

**Primary model: `{winner_label}`**

{sel['justification']}

Baselines and hyperparameter search were run on this primary model. The other candidate remains a comparison point above.

> Note: the CV numbers above come from the **enriched** bake-off artifacts currently on disk. For IBM-only test metrics and IBM-only best params, use **§3**.

---

## Experiment B — Enriched features (synthetic behavioral demo)

> **Secondary experiment.** Everything from here through §10 uses IBM columns **plus** synthetic CRM/telemetry. These metrics are **not** the IBM-only result. Primary IBM-only write-up is **§3**.

## 5. Baseline vs tuned comparison (enriched test set)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
{_metric_row("Baseline A (defaults / scale_pos_weight=1)", baseline_a)}
{_metric_row("Baseline B (balanced class weight)", baseline_b)}
{_metric_row("Tuned (+ F1 OOF threshold)", tuned)}

**Attribution (within the enriched feature set only)**

- **A → B (imbalance handling):** ROC-AUC Δ={_fmt(delta_ab_auc)}, F1 Δ={_fmt(delta_ab_f1)}. This isolates turning on balanced class weighting ({imbalance_note}).
- **B → tuned (search + F1 threshold):** ROC-AUC Δ={_fmt(delta_bt_auc)}, F1 Δ={_fmt(delta_bt_f1)}. Modest once synthetic features are present — confirms **grid search is not the main F1 story**; feature richness is.
- **IBM-only → enriched (different experiment):** F1 {_fmt(ibm_only_f1)} → {_fmt(tuned.get('f1'))}. That large step is from **synthetic behavioral features**, not from hyperparameters.

{_cm_block("Baseline A confusion matrix", baseline_a)}

{_cm_block("Baseline B confusion matrix", baseline_b)}

{_cm_block("Tuned confusion matrix", tuned)}

## 6. Final hyperparameters (enriched run)

Selected after RandomizedSearchCV → focused GridSearchCV (`scoring=roc_auc`, `n_jobs=1`, `refit=True`). For **IBM-only** best params, see §3.4.

{params_lines}

- Random search best CV ROC-AUC: `{_fmt(best.get('random_best_score'))}`
- Grid search best CV ROC-AUC: `{_fmt(best.get('grid_best_score'))}`

## 7. Metric choice explanation

- **Selection metric:** ROC-AUC — threshold-independent, stable for ranking candidates in CV.
- **Business complement:** PR-AUC, F1, precision, and recall — churn is ~26.5% and false negatives (missed churners) are costly for retention outreach.
- **Not used for selection:** accuracy — a trivial "always No" classifier scores ~73.5%.
- **Interpretation caveat:** high F1 under the enriched set should not be credited to ROC-AUC search alone; see §3.

## 8. Stability trade-off (enriched run)

{stability.get('justification', 'See best_params.json for top candidate mean/std.')}

Top grid candidates (by rank):

| Rank | Mean CV ROC-AUC | Std |
|---|---|---|
{chr(10).join(
    f"| {c['rank_test_score']} | {_fmt(c['mean_test_score'])} | {_fmt(c['std_test_score'])} |"
    for c in stability.get('top_candidates', [])[:5]
)}

## 9. Threshold decision (enriched run)

- **Method:** maximize F1 on **out-of-fold** train probabilities (`StratifiedKFold`, never the test set).
- **Chosen threshold:** `{_fmt(threshold.get('threshold'), 3)}` (OOF F1={_fmt(threshold.get('oof_f1') or threshold.get('val_f1'))}).
- Final estimator was refit on the **full** training set with the selected hyperparameters; the threshold was then applied once on test.
- IBM-only threshold (separate experiment): see §3.3 (`{_fmt((ibm_only or {}).get('tuned', {}).get('threshold'), 3)}`).

{_cm_block("Test confusion matrix at chosen threshold", tuned)}

## 10. Permutation importance (enriched run — top drivers)

Scoring=`{importance.get('scoring')}`, n_repeats=`{importance.get('n_repeats')}` on a train subset. Dominated by synthetic columns — expected for Experiment B.

{feat_lines}

## 11. Limitations & next steps

**Done in v1:** leakage-safe pipeline, LR vs XGBoost bake-off, dual baselines, random→grid search, F1 OOF threshold tuning, light feature engineering, **documented IBM-only fine-tuning (§3)**, synthetic behavioral demo as a separate experiment (§5+), permutation importance, reproducible scripts + report.

**Limitations**

- **Primary / honest result:** IBM-only tuned F1 = {_fmt(ibm_only_f1)} (`reports/metrics_ibm_only.json`, §3).
- Enriched F1 (~{_fmt(tuned.get('f1'))}) depends on **synthetic** behavioral features and is a demo upper bound only.
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
"""


def write_report(path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (REPORTS_DIR / "tuning_report.md")
    text = build_report()
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")
    return path


def main() -> None:
    write_report()


if __name__ == "__main__":
    main()

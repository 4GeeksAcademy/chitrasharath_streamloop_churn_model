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

    return f"""# Telco Churn Tuning Report

## 1. Dataset & cleaning summary

- **Source:** IBM Telco Customer Churn CSV (cached under `data/Telco-Customer-Churn.csv`).
- **Split:** single stratified 80/20 train/test (`random_state=42`); test used only for Baseline A, Baseline B, and final tuned evaluation.
- **`customerID`:** dropped (identifier; not a feature).
- **`TotalCharges`:** coerced with `pd.to_numeric(..., errors="coerce")`. Blank values (~tenure=0 customers) are imputed to **0** inside the sklearn pipeline (`SimpleImputer(strategy="constant", fill_value=0)`), not by hand-editing the frame.
- **`SeniorCitizen`:** treated as **numeric** (already 0/1); scaled with other numerics.
- **Engineered features (row-wise, no leakage):** `charges_per_tenure`, `n_addons`, `is_month_to_month`, `has_fiber`.
- **Synthetic behavioral features** (demo CRM/telemetry in `data/behavioral_features.csv`): support tickets, app logins, usage, payment failures, NPS, etc. Generated deterministically from account risk proxies + churn with noise — **not** part of the original IBM CSV; metrics with these features illustrate an enriched-data scenario. See **§2 Feature dictionary**.
- Categorical levels such as "No internet service" / "No phone service" kept as-is for one-hot encoding.

{feature_section}

## 3. Classifier choice & justification

Head-to-head on **training data only** with identical preprocessing, balanced class weighting, and `StratifiedKFold(5)`:

| Candidate | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) |
|---|---|---|
| LogisticRegression | {_fmt(lr['roc_auc_mean'])} ± {_fmt(lr['roc_auc_std'])} | {_fmt(lr['pr_auc_mean'])} ± {_fmt(lr['pr_auc_std'])} |
| {alt_label} | {_fmt(alt['roc_auc_mean'])} ± {_fmt(alt['roc_auc_std'])} | {_fmt(alt['pr_auc_mean'])} ± {_fmt(alt['pr_auc_std'])} |

**Primary model: `{winner_label}`**

{sel['justification']}

Baselines and hyperparameter search were run on this primary model. The other candidate remains a comparison point above.

## 4. Baseline vs tuned comparison (test set)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
{_metric_row("Baseline A (defaults / scale_pos_weight=1)", baseline_a)}
{_metric_row("Baseline B (balanced class weight)", baseline_b)}
{_metric_row("Tuned (+ F1 OOF threshold)", tuned)}

**Attribution**

- **A → B (imbalance handling):** ROC-AUC Δ={_fmt(delta_ab_auc)}, F1 Δ={_fmt(delta_ab_f1)}. This isolates turning on balanced class weighting ({imbalance_note}).
- **B → tuned (genuine tuning + threshold):** ROC-AUC Δ={_fmt(delta_bt_auc)}, F1 Δ={_fmt(delta_bt_f1)}. Search also included imbalance weighting as a tunable parameter.

{_cm_block("Baseline A confusion matrix", baseline_a)}

{_cm_block("Baseline B confusion matrix", baseline_b)}

{_cm_block("Tuned confusion matrix", tuned)}

## 5. Final hyperparameters

Selected after RandomizedSearchCV → focused GridSearchCV (`scoring=roc_auc`, `n_jobs=1`, `refit=True`):

{params_lines}

- Random search best CV ROC-AUC: `{_fmt(best.get('random_best_score'))}`
- Grid search best CV ROC-AUC: `{_fmt(best.get('grid_best_score'))}`

## 6. Metric choice explanation

- **Selection metric:** ROC-AUC — threshold-independent, stable for ranking candidates in CV.
- **Business complement:** PR-AUC, F1, precision, and recall — churn is ~26.5% and false negatives (missed churners) are costly for retention outreach.
- **Not used for selection:** accuracy — a trivial "always No" classifier scores ~73.5%.

## 7. Stability trade-off

{stability.get('justification', 'See best_params.json for top candidate mean/std.')}

Top grid candidates (by rank):

| Rank | Mean CV ROC-AUC | Std |
|---|---|---|
{chr(10).join(
    f"| {c['rank_test_score']} | {_fmt(c['mean_test_score'])} | {_fmt(c['std_test_score'])} |"
    for c in stability.get('top_candidates', [])[:5]
)}

## 8. Threshold decision

- **Method:** maximize F1 on **out-of-fold** train probabilities (`StratifiedKFold`, never the test set).
- **Chosen threshold:** `{_fmt(threshold.get('threshold'), 3)}` (OOF F1={_fmt(threshold.get('oof_f1') or threshold.get('val_f1'))}).
- Final estimator was refit on the **full** training set with the selected hyperparameters; the threshold was then applied once on test.

{_cm_block("Test confusion matrix at chosen threshold", tuned)}

## 9. Permutation importance (top drivers)

Scoring=`{importance.get('scoring')}`, n_repeats=`{importance.get('n_repeats')}` on a train subset:

{feat_lines}

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

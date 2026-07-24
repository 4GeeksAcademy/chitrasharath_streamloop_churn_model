"""CV bake-off: Logistic Regression vs XGBoost on train only.

Stakeholder selected XGBoost as the primary non-linear model; LR is retained as
the interpretable comparison. Baselines + tuning always proceed on XGBoost.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE, load_train_test
from src.io_utils import REPORTS_DIR, ensure_dirs, save_json
from src.pipeline import make_logistic_pipeline, make_xgboost_pipeline

NOISE_THRESHOLD = 0.01
CV_SPLITS = 5
PRIMARY_MODEL = "xgboost"


def _cv_scores(pipeline, X, y, cv) -> dict[str, Any]:
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
        "roc_auc_std": float(np.std(results["test_roc_auc"], ddof=1)),
        "pr_auc_mean": float(np.mean(results["test_pr_auc"])),
        "pr_auc_std": float(np.std(results["test_pr_auc"], ddof=1)),
        "roc_auc_folds": [float(x) for x in results["test_roc_auc"]],
        "pr_auc_folds": [float(x) for x in results["test_pr_auc"]],
    }


def select_primary(logistic: dict[str, Any], xgboost: dict[str, Any]) -> dict[str, Any]:
    """Report head-to-head; primary is XGBoost by stakeholder choice."""
    lr_auc = logistic["roc_auc_mean"]
    xgb_auc = xgboost["roc_auc_mean"]
    delta = xgb_auc - lr_auc

    lr_lo = lr_auc - logistic["roc_auc_std"]
    lr_hi = lr_auc + logistic["roc_auc_std"]
    xgb_lo = xgb_auc - xgboost["roc_auc_std"]
    xgb_hi = xgb_auc + xgboost["roc_auc_std"]
    overlapping = not (xgb_lo > lr_hi or lr_lo > xgb_hi)

    comparison = (
        f"CV ROC-AUC: XGBoost {xgb_auc:.4f} ± {xgboost['roc_auc_std']:.4f} vs "
        f"LogisticRegression {lr_auc:.4f} ± {logistic['roc_auc_std']:.4f} "
        f"(Δ={delta:.4f}, overlapping_mean±std={overlapping}). "
        "Primary model set to XGBoost by stakeholder choice; LR kept as interpretable reference."
    )
    return {
        "winner": PRIMARY_MODEL,
        "delta_roc_auc_xgb_minus_lr": float(delta),
        "noise_threshold": NOISE_THRESHOLD,
        "overlapping_mean_pm_std": overlapping,
        "selection_rule": "stakeholder_xgboost",
        "justification": comparison,
    }


def run_bakeoff() -> dict[str, Any]:
    ensure_dirs()
    X_train, _X_test, y_train, _y_test = load_train_test()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    logistic_pipe = make_logistic_pipeline(X_train, class_weight="balanced")
    xgb_pipe = make_xgboost_pipeline(X_train, class_weight="balanced", y=y_train)

    logistic_scores = _cv_scores(logistic_pipe, X_train, y_train, cv)
    xgb_scores = _cv_scores(xgb_pipe, X_train, y_train, cv)
    selection = select_primary(logistic_scores, xgb_scores)

    payload = {
        "cv": {"n_splits": CV_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "class_weight": "balanced",
        "candidates": {
            "logistic": logistic_scores,
            "xgboost": xgb_scores,
        },
        "selection": selection,
    }
    out = REPORTS_DIR / "bakeoff_cv.json"
    save_json(payload, out)
    print(json.dumps(selection, indent=2))
    print(f"Wrote {out}")
    return payload


def main() -> None:
    run_bakeoff()


if __name__ == "__main__":
    main()

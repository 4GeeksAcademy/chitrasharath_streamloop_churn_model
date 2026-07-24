"""Metric helpers, threshold tuning, and permutation importance."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def predict_proba_positive(estimator, X) -> np.ndarray:
    proba = estimator.predict_proba(X)
    return proba[:, 1]


def classification_metrics(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "labels": ["actual_0", "actual_1"],
            "matrix": cm.tolist(),
        },
    }


def tune_threshold_f1(y_true, y_proba, n_grid: int = 101) -> dict[str, Any]:
    """Pick threshold that maximizes F1 on provided labels/probabilities."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.01, 0.99, n_grid)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return {
        "threshold": best_t,
        "val_f1": best_f1,
        "method": "maximize_f1",
        "n_grid": n_grid,
    }


def train_val_split_for_threshold(
    X_train,
    y_train,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Stratified holdout carved from train for threshold selection only."""
    return train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )


def select_threshold_on_train(
    estimator_factory,
    X_train,
    y_train,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    **_ignored,
) -> dict[str, Any]:
    """
    Choose an F1-maximizing threshold from out-of-fold train probabilities.

    Uses StratifiedKFold so every training row contributes an honest score;
    more stable than a single 20% holdout. Caller should refit on full train
    for the saved model (threshold does not require refitting).
    """
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X_train), dtype=float)
    y_arr = np.asarray(y_train)
    for tr_idx, va_idx in cv.split(X_train, y_train):
        model = estimator_factory()
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else y_arr[tr_idx]
        X_va = X_train.iloc[va_idx]
        model.fit(X_tr, y_tr)
        oof[va_idx] = predict_proba_positive(model, X_va)

    result = tune_threshold_f1(y_arr, oof)
    result["method"] = "maximize_f1_oof"
    result["n_splits"] = n_splits
    result["random_state"] = random_state
    result["oof_f1"] = result.pop("val_f1")
    return result


def permutation_importance_report(
    estimator,
    X,
    y,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    scoring: str = "roc_auc",
    top_n: int = 15,
) -> dict[str, Any]:
    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )
    # Feature names after ColumnTransformer are not trivial; use raw input columns.
    names = list(X.columns)
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    items = [
        {
            "feature": names[i],
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in order
    ]
    return {
        "scoring": scoring,
        "n_repeats": n_repeats,
        "top_features": items,
    }

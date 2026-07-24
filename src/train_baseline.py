"""Fit Baseline A (defaults) and Baseline B (class_weight=balanced) on the winner."""

from __future__ import annotations

from typing import Any

import joblib

from src.compare_candidates import run_bakeoff
from src.data import load_train_test
from src.evaluate import classification_metrics, predict_proba_positive
from src.io_utils import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_json, save_json
from src.pipeline import make_pipeline


def _load_or_run_bakeoff() -> str:
    path = REPORTS_DIR / "bakeoff_cv.json"
    if path.exists():
        return load_json(path)["selection"]["winner"]
    return run_bakeoff()["selection"]["winner"]


def fit_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier: str,
    class_weight: str | None,
    label: str,
) -> dict[str, Any]:
    pipe = make_pipeline(
        X_train,
        classifier=classifier,
        class_weight=class_weight,
        y=y_train,
    )
    pipe.fit(X_train, y_train)
    proba = predict_proba_positive(pipe, X_test)
    metrics = classification_metrics(y_test, proba, threshold=0.5)
    metrics["model"] = label
    metrics["classifier"] = classifier
    metrics["class_weight"] = class_weight
    if classifier == "xgboost":
        metrics["scale_pos_weight"] = pipe.named_steps["classifier"].scale_pos_weight
    model_path = MODELS_DIR / f"{label}.joblib"
    joblib.dump(pipe, model_path)
    metrics["model_path"] = str(model_path)
    return metrics


def run_baselines() -> dict[str, Any]:
    ensure_dirs()
    winner = _load_or_run_bakeoff()
    X_train, X_test, y_train, y_test = load_train_test()

    baseline_a = fit_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier=winner,
        class_weight=None,
        label="baseline_a",
    )
    baseline_b = fit_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier=winner,
        class_weight="balanced",
        label="baseline_b",
    )

    save_json(baseline_a, REPORTS_DIR / "metrics_baseline_a.json")
    save_json(baseline_b, REPORTS_DIR / "metrics_baseline_b.json")
    summary = {"winner": winner, "baseline_a": baseline_a, "baseline_b": baseline_b}
    print(f"Winner classifier: {winner}")
    print(
        f"Baseline A ROC-AUC={baseline_a['roc_auc']:.4f} | "
        f"Baseline B ROC-AUC={baseline_b['roc_auc']:.4f}"
    )
    return summary


def main() -> None:
    run_baselines()


if __name__ == "__main__":
    main()

"""RandomizedSearchCV → focused GridSearchCV on the bake-off winner (train only)."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.compare_candidates import run_bakeoff
from src.data import RANDOM_STATE, load_train_test
from src.evaluate import (
    classification_metrics,
    permutation_importance_report,
    predict_proba_positive,
    select_threshold_on_train,
)
from src.io_utils import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_json, save_json
from src.pipeline import balanced_scale_pos_weight, make_pipeline

N_ITER_RANDOM = 30
CV_SPLITS = 5
N_JOBS = 1


def _load_winner() -> str:
    path = REPORTS_DIR / "bakeoff_cv.json"
    if path.exists():
        return load_json(path)["selection"]["winner"]
    return run_bakeoff()["selection"]["winner"]


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def random_search_space(classifier: str, y_train=None) -> dict[str, Any]:
    if classifier == "xgboost":
        spw_balanced = balanced_scale_pos_weight(y_train) if y_train is not None else 2.8
        return {
            "classifier__n_estimators": randint(80, 251),
            "classifier__max_depth": randint(2, 6),
            "classifier__learning_rate": loguniform(0.02, 0.2),
            "classifier__subsample": uniform(0.7, 0.3),
            "classifier__colsample_bytree": uniform(0.7, 0.3),
            "classifier__min_child_weight": randint(1, 9),
            "classifier__reg_lambda": loguniform(0.1, 5.0),
            "classifier__scale_pos_weight": [1.0, spw_balanced],
        }
    if classifier == "histgb":
        return {
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__max_leaf_nodes": randint(15, 64),
            "classifier__max_depth": [None, 3, 5, 8, 10],
            "classifier__l2_regularization": uniform(0.0, 10.0),
            "classifier__min_samples_leaf": randint(10, 51),
            "classifier__max_iter": randint(100, 301),
            "classifier__class_weight": [None, "balanced"],
        }
    if classifier == "logistic":
        # Prefer lbfgs + L2: stable, fast, avoids sklearn≥1.8 penalty deprecations.
        return {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__class_weight": [None, "balanced"],
        }
    raise ValueError(classifier)


def _unique_sorted(values: list[Any], cast=None) -> list[Any]:
    cleaned = []
    for v in values:
        if isinstance(v, float) and np.isnan(v):
            continue
        if hasattr(v, "item"):
            v = v.item()
        cleaned.append(cast(v) if cast else v)
    # preserve None, sort numbers, keep stables
    none_vals = [v for v in cleaned if v is None]
    other = [v for v in cleaned if v is not None]
    try:
        other = sorted(set(other))
    except TypeError:
        other = list(dict.fromkeys(other))
    return (none_vals[:1] if none_vals else []) + other


def focused_grid_from_random(cv_results: dict, classifier: str, top_k: int = 10) -> dict[str, list]:
    """Build a small grid around the top random-search region."""
    df = pd.DataFrame(cv_results)
    top = df.nsmallest(top_k, "rank_test_score")

    if classifier == "xgboost":
        n_est = _unique_sorted(top["param_classifier__n_estimators"].tolist(), int)
        if len(n_est) > 3:
            n_est = [min(n_est), int(np.median(n_est)), max(n_est)]
        depths = _unique_sorted(top["param_classifier__max_depth"].tolist(), int)
        if len(depths) > 3:
            depths = [min(depths), int(np.median(depths)), max(depths)]
        lrs = top["param_classifier__learning_rate"].astype(float)
        lr_grid = sorted(
            {
                float(np.clip(lrs.median() / 2, 0.01, 0.3)),
                float(np.clip(lrs.median(), 0.01, 0.3)),
                float(np.clip(lrs.median() * 1.5, 0.01, 0.3)),
            }
        )
        spw = _unique_sorted(top["param_classifier__scale_pos_weight"].tolist(), float)
        if not spw:
            spw = [1.0]
        # Keep grid small: freeze subsample/colsample/min_child/reg at median of top
        subsample = float(top["param_classifier__subsample"].astype(float).median())
        colsample = float(top["param_classifier__colsample_bytree"].astype(float).median())
        min_child = int(top["param_classifier__min_child_weight"].astype(float).median())
        reg_lambda = float(top["param_classifier__reg_lambda"].astype(float).median())
        return {
            "classifier__n_estimators": n_est or [200],
            "classifier__max_depth": depths or [3],
            "classifier__learning_rate": lr_grid,
            "classifier__subsample": [subsample],
            "classifier__colsample_bytree": [colsample],
            "classifier__min_child_weight": [min_child],
            "classifier__reg_lambda": [reg_lambda],
            "classifier__scale_pos_weight": spw,
        }

    if classifier == "histgb":
        lrs = top["param_classifier__learning_rate"].astype(float)
        lr_grid = sorted(
            {
                float(np.clip(lrs.median() / 2, 0.01, 0.3)),
                float(np.clip(lrs.median(), 0.01, 0.3)),
                float(np.clip(lrs.median() * 1.5, 0.01, 0.3)),
            }
        )
        leaf_nodes = _unique_sorted(top["param_classifier__max_leaf_nodes"].tolist(), int)
        if len(leaf_nodes) > 4:
            leaf_nodes = sorted(leaf_nodes)[:: max(1, len(leaf_nodes) // 3)][:4]
        depths = _unique_sorted(top["param_classifier__max_depth"].tolist())
        if not depths:
            depths = [None, 5]
        l2 = top["param_classifier__l2_regularization"].astype(float)
        l2_grid = sorted(
            {
                float(max(0.0, l2.median() * 0.5)),
                float(l2.median()),
                float(l2.median() * 1.5 + 0.1),
            }
        )
        min_leaf = _unique_sorted(top["param_classifier__min_samples_leaf"].tolist(), int)
        if len(min_leaf) > 3:
            min_leaf = [min(min_leaf), int(np.median(min_leaf)), max(min_leaf)]
        max_iter = _unique_sorted(top["param_classifier__max_iter"].tolist(), int)
        if len(max_iter) > 3:
            max_iter = [min(max_iter), int(np.median(max_iter)), max(max_iter)]
        weights = _unique_sorted(top["param_classifier__class_weight"].tolist())
        if not weights:
            weights = [None, "balanced"]

        return {
            "classifier__learning_rate": lr_grid,
            "classifier__max_leaf_nodes": leaf_nodes or [31],
            "classifier__max_depth": depths,
            "classifier__l2_regularization": l2_grid,
            "classifier__min_samples_leaf": min_leaf or [20],
            "classifier__max_iter": max_iter or [200],
            "classifier__class_weight": weights,
        }

    # logistic
    cs = top["param_classifier__C"].astype(float)
    c_grid = sorted(
        {
            float(np.clip(cs.median() / 3, 1e-3, 1e2)),
            float(np.clip(cs.median(), 1e-3, 1e2)),
            float(np.clip(cs.median() * 3, 1e-3, 1e2)),
        }
    )
    weights = _unique_sorted(top["param_classifier__class_weight"].tolist()) or [
        None,
        "balanced",
    ]
    return {
        "classifier__C": c_grid,
        "classifier__class_weight": weights,
    }


def _stability_pick(cv_results: dict, top_n: int = 5) -> dict[str, Any]:
    df = pd.DataFrame(cv_results)
    top = df.nsmallest(top_n, "rank_test_score").copy()
    best_mean_idx = top.index[0]
    best_mean = top.loc[best_mean_idx]
    # Prefer materially lower std if mean within 0.005
    top["score"] = top["mean_test_score"] - 0.5 * top["std_test_score"]
    stable_idx = top["score"].idxmax()
    stable = top.loc[stable_idx]

    mean_gap = float(best_mean["mean_test_score"] - stable["mean_test_score"])
    std_gap = float(best_mean["std_test_score"] - stable["std_test_score"])
    prefer_stable = mean_gap <= 0.005 and std_gap > 0.002 and stable_idx != best_mean_idx

    chosen = stable if prefer_stable else best_mean
    justification = (
        f"Selected rank={int(chosen['rank_test_score'])} with mean={chosen['mean_test_score']:.4f} "
        f"std={chosen['std_test_score']:.4f}. "
        + (
            f"Preferred lower variance over best mean (mean gap={mean_gap:.4f}, std gap={std_gap:.4f})."
            if prefer_stable
            else "Best mean CV score accepted; no material stability trade-off against close rivals."
        )
    )
    rivals = []
    for _, row in top.iterrows():
        rivals.append(
            {
                "rank_test_score": int(row["rank_test_score"]),
                "mean_test_score": float(row["mean_test_score"]),
                "std_test_score": float(row["std_test_score"]),
                "params": {
                    k.replace("param_", ""): (
                        None if (isinstance(v, float) and np.isnan(v)) else v
                    )
                    for k, v in row.items()
                    if k.startswith("param_")
                },
            }
        )
    return {
        "prefer_stable": prefer_stable,
        "chosen_rank": int(chosen["rank_test_score"]),
        "chosen_mean": float(chosen["mean_test_score"]),
        "chosen_std": float(chosen["std_test_score"]),
        "justification": justification,
        "top_candidates": rivals,
    }


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and np.isnan(v):
            v = None
        out[k] = v
    return out


def run_tuning() -> dict[str, Any]:
    ensure_dirs()
    winner = _load_winner()
    X_train, X_test, y_train, y_test = load_train_test()
    cv = _cv()

    base = make_pipeline(X_train, classifier=winner, class_weight=None, y=y_train)
    param_distributions = random_search_space(winner, y_train=y_train)

    print(f"RandomizedSearchCV on {winner} (n_iter={N_ITER_RANDOM}, n_jobs={N_JOBS})...")
    random_search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=N_ITER_RANDOM,
        scoring="roc_auc",
        cv=cv,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    random_search.fit(X_train, y_train)
    pd.DataFrame(random_search.cv_results_).to_csv(
        REPORTS_DIR / "cv_results_random.csv", index=False
    )

    grid_params = focused_grid_from_random(random_search.cv_results_, winner)
    # Cap grid size for runtime: if product too large, shrink leaf/depth options
    grid_params = _cap_grid(grid_params, max_combos=24)
    print(f"GridSearchCV focused grid ({grid_params})...")

    grid = GridSearchCV(
        estimator=make_pipeline(X_train, classifier=winner, class_weight=None, y=y_train),
        param_grid=grid_params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    pd.DataFrame(grid.cv_results_).to_csv(REPORTS_DIR / "cv_results_grid.csv", index=False)

    stability = _stability_pick(grid.cv_results_)
    if stability["prefer_stable"]:
        chosen_row = next(
            r
            for r in stability["top_candidates"]
            if r["rank_test_score"] == stability["chosen_rank"]
        )
        best_params = _sanitize_params(chosen_row["params"])
        final = make_pipeline(X_train, classifier=winner, class_weight=None, y=y_train)
        final.set_params(**best_params)
        final.fit(X_train, y_train)
    else:
        best_params = _sanitize_params(grid.best_params_)
        final = grid.best_estimator_

    save_json(
        {
            "classifier": winner,
            "best_params": best_params,
            "grid_best_params_sklearn": _sanitize_params(grid.best_params_),
            "stability": stability,
            "random_best_score": float(random_search.best_score_),
            "grid_best_score": float(grid.best_score_),
        },
        REPORTS_DIR / "best_params.json",
    )

    def factory() -> Any:
        pipe = make_pipeline(X_train, classifier=winner, class_weight=None, y=y_train)
        pipe.set_params(**best_params)
        return pipe

    threshold_info = select_threshold_on_train(factory, X_train, y_train)
    save_json(threshold_info, REPORTS_DIR / "threshold.json")

    # Final model already fit on full train
    test_proba = predict_proba_positive(final, X_test)
    tuned_metrics = classification_metrics(
        y_test, test_proba, threshold=threshold_info["threshold"]
    )
    tuned_metrics["model"] = "tuned"
    tuned_metrics["classifier"] = winner
    tuned_metrics["best_params"] = best_params
    save_json(tuned_metrics, REPORTS_DIR / "metrics_tuned.json")

    model_path = MODELS_DIR / "tuned_pipeline.joblib"
    joblib.dump(final, model_path)

    # Permutation importance on a train subset for speed
    n_imp = min(1500, len(X_train))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=n_imp, replace=False)
    X_imp = X_train.iloc[idx]
    y_imp = y_train.iloc[idx]
    imp = permutation_importance_report(final, X_imp, y_imp, n_repeats=5)
    save_json(imp, REPORTS_DIR / "permutation_importance.json")

    print(
        f"Tuned test ROC-AUC={tuned_metrics['roc_auc']:.4f} "
        f"F1={tuned_metrics['f1']:.4f} threshold={threshold_info['threshold']:.3f}"
    )
    print(f"Saved model → {model_path}")
    return {
        "winner": winner,
        "best_params": best_params,
        "threshold": threshold_info,
        "metrics": tuned_metrics,
        "stability": stability,
    }


def _cap_grid(grid: dict[str, list], max_combos: int = 48) -> dict[str, list]:
    keys = list(grid.keys())
    sizes = [len(grid[k]) for k in keys]
    product = int(np.prod(sizes)) if sizes else 1
    if product <= max_combos:
        return grid
    # Shrink largest list first until under cap
    out = {k: list(v) for k, v in grid.items()}
    while int(np.prod([len(v) for v in out.values()])) > max_combos:
        longest = max(out.keys(), key=lambda k: len(out[k]))
        if len(out[longest]) <= 1:
            break
        mid = len(out[longest]) // 2
        # keep edges + middle
        vals = out[longest]
        out[longest] = [vals[0], vals[mid], vals[-1]] if len(vals) > 2 else vals[:1]
        out[longest] = list(dict.fromkeys(out[longest]))
    return out


def main() -> None:
    run_tuning()


if __name__ == "__main__":
    main()

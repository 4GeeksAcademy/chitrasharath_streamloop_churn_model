"""Light smoke tests for data, split, and pipeline fit."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import ID_COL, TARGET_COL, clean, ensure_dataset, load_train_test, prepare_xy
from src.evaluate import classification_metrics, tune_threshold_f1
from src.pipeline import make_histgb_pipeline, make_logistic_pipeline


def test_dataset_cached_and_no_customer_id():
    path = ensure_dataset()
    assert path.exists()
    raw = pd.read_csv(path)
    assert ID_COL in raw.columns
    assert TARGET_COL in raw.columns
    X, y = prepare_xy(raw)
    assert ID_COL not in X.columns
    assert y.isin([0, 1]).all()


def test_total_charges_numeric_after_clean():
    raw = pd.read_csv(ensure_dataset())
    cleaned = clean(raw)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_stratified_split_shapes_and_classes():
    X_train, X_test, y_train, y_test = load_train_test()
    n = len(X_train) + len(X_test)
    assert len(X_train) == pytest.approx(0.8 * n, rel=0.02)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_pipelines_fit_tiny_subset():
    X_train, _, y_train, _ = load_train_test()
    X_tiny = X_train.iloc[:200]
    y_tiny = y_train.iloc[:200]
    from src.pipeline import make_xgboost_pipeline

    for factory_args in (
        (make_logistic_pipeline, {"class_weight": "balanced"}),
        (make_histgb_pipeline, {"class_weight": "balanced"}),
        (make_xgboost_pipeline, {"class_weight": "balanced", "y": y_tiny}),
    ):
        factory, kwargs = factory_args
        pipe = factory(X_tiny, **kwargs)
        pipe.fit(X_tiny, y_tiny)
        proba = pipe.predict_proba(X_tiny.iloc[:10])
        assert proba.shape == (10, 2)


def test_metrics_and_threshold_helpers():
    y_true = [0, 0, 1, 1, 1, 0]
    y_proba = [0.1, 0.4, 0.6, 0.8, 0.55, 0.2]
    metrics = classification_metrics(y_true, y_proba, threshold=0.5)
    assert "roc_auc" in metrics and "confusion_matrix" in metrics
    thr = tune_threshold_f1(y_true, y_proba)
    assert 0.01 <= thr["threshold"] <= 0.99


def test_engineered_features_present():
    X_train, _, _, _ = load_train_test()
    for col in ("charges_per_tenure", "n_addons", "is_month_to_month", "has_fiber"):
        assert col in X_train.columns


def test_behavioral_features_present():
    from src.behavioral import BEHAVIORAL_FEATURES

    X_train, _, _, _ = load_train_test()
    for col in BEHAVIORAL_FEATURES:
        assert col in X_train.columns

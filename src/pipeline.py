"""ColumnTransformer + classifier pipeline factories."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import feature_columns

RANDOM_STATE = 42
ClassifierName = Literal["logistic", "histgb", "xgboost"]


def balanced_scale_pos_weight(y) -> float:
    """XGBoost equivalent of class_weight='balanced' for binary labels."""
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos)


def build_preprocessor(X) -> ColumnTransformer:
    """Leakage-safe preprocessing; TotalCharges imputed to 0 (tenure=0 rows)."""
    numeric, categorical = feature_columns(X)
    total_charges = [c for c in numeric if c == "TotalCharges"]
    other_numeric = [c for c in numeric if c != "TotalCharges"]

    transformers: list[tuple[str, Any, list[str]]] = []
    if total_charges:
        transformers.append(
            (
                "total_charges",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                total_charges,
            )
        )
    if other_numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                other_numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier(
    name: ClassifierName,
    class_weight: str | dict | None = None,
    scale_pos_weight: float | None = None,
    **kwargs: Any,
):
    if name == "logistic":
        params: dict[str, Any] = {
            "random_state": RANDOM_STATE,
            "max_iter": 1000,
            "class_weight": class_weight,
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    if name == "histgb":
        params = {
            "random_state": RANDOM_STATE,
            "class_weight": class_weight,
        }
        params.update(kwargs)
        return HistGradientBoostingClassifier(**params)

    if name == "xgboost":
        # Map class_weight → scale_pos_weight when caller did not pass it explicitly.
        spw = scale_pos_weight
        if spw is None:
            if class_weight == "balanced":
                raise ValueError(
                    "xgboost with class_weight='balanced' requires scale_pos_weight; "
                    "use make_pipeline(..., y=y_train) or pass scale_pos_weight."
                )
            spw = 1.0
        params = {
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "scale_pos_weight": spw,
        }
        params.update(kwargs)
        return XGBClassifier(**params)

    raise ValueError(f"Unknown classifier: {name}")


def make_pipeline(
    X,
    classifier: ClassifierName = "xgboost",
    class_weight: str | dict | None = None,
    y=None,
    **classifier_kwargs: Any,
) -> Pipeline:
    kwargs = dict(classifier_kwargs)
    if classifier == "xgboost":
        if "scale_pos_weight" not in kwargs:
            if class_weight == "balanced":
                if y is None:
                    raise ValueError("y is required to set balanced scale_pos_weight for xgboost")
                kwargs["scale_pos_weight"] = balanced_scale_pos_weight(y)
            else:
                kwargs["scale_pos_weight"] = 1.0
        # class_weight is not an XGBClassifier arg
        return Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "classifier",
                    make_classifier(
                        classifier,
                        class_weight=None,
                        **kwargs,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("classifier", make_classifier(classifier, class_weight, **kwargs)),
        ]
    )


def make_logistic_pipeline(X, class_weight: str | None = None, **kwargs: Any) -> Pipeline:
    return make_pipeline(X, "logistic", class_weight, **kwargs)


def make_histgb_pipeline(X, class_weight: str | None = None, **kwargs: Any) -> Pipeline:
    return make_pipeline(X, "histgb", class_weight, **kwargs)


def make_xgboost_pipeline(
    X,
    class_weight: str | None = None,
    y=None,
    **kwargs: Any,
) -> Pipeline:
    return make_pipeline(X, "xgboost", class_weight, y=y, **kwargs)

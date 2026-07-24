"""Download, cache, clean, and split the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

TARGET_COL = "Churn"
ID_COL = "customerID"

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_NUMERIC = [
    "charges_per_tenure",
    "n_addons",
    "is_month_to_month",
    "has_fiber",
]

# Synthetic CRM / telemetry columns (see src/behavioral.py).
BEHAVIORAL_NUMERIC = [
    "support_tickets_90d",
    "support_ticket_escalations_90d",
    "app_logins_30d",
    "days_since_last_login",
    "avg_daily_usage_gb",
    "data_overage_events_90d",
    "payment_failures_12m",
    "nps_score",
    "discount_offers_accepted_12m",
    "plan_change_count_12m",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    *ENGINEERED_NUMERIC,
    *BEHAVIORAL_NUMERIC,
]


def ensure_dataset(path: Path = DATA_PATH, url: str = DATA_URL) -> Path:
    """Download the CSV once; reuse the local cache thereafter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    try:
        df = pd.read_csv(url)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Dataset not found at {path} and download from {url} failed: {exc}"
        ) from exc
    df.to_csv(path, index=False)
    return path


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    ensure_dataset(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier, coerce TotalCharges; leave NaNs for pipeline imputation."""
    out = df.copy()
    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    n_dupes = out.duplicated().sum()
    if n_dupes:
        out = out.drop_duplicates().reset_index(drop=True)
    return out


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe row-wise features (no train statistics)."""
    out = X.copy()
    tenure = out["tenure"].clip(lower=1) if "tenure" in out.columns else 1
    out["charges_per_tenure"] = out["MonthlyCharges"] / tenure
    present_addons = [c for c in ADDON_COLS if c in out.columns]
    out["n_addons"] = (
        (out[present_addons] == "Yes").sum(axis=1).astype(float)
        if present_addons
        else 0.0
    )
    out["is_month_to_month"] = (
        (out["Contract"] == "Month-to-month").astype(float)
        if "Contract" in out.columns
        else 0.0
    )
    out["has_fiber"] = (
        (out["InternetService"] == "Fiber optic").astype(float)
        if "InternetService" in out.columns
        else 0.0
    )
    # Replace any accidental inf from division
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for modeling."""
    numeric = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Late import avoids circular dependency with src.behavioral.
    from src.behavioral import attach_behavioral

    enriched = attach_behavioral(df)
    cleaned = clean(enriched)
    y = (cleaned[TARGET_COL] == "Yes").astype(int)
    X = engineer_features(cleaned.drop(columns=[TARGET_COL]))
    return X, y


def load_train_test(
    path: Path = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Single stratified split — call this everywhere for a frozen test set."""
    raw = load_raw(path)
    X, y = prepare_xy(raw)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def main() -> None:
    path = ensure_dataset()
    from src.behavioral import ensure_behavioral, BEHAVIORAL_PATH

    ensure_behavioral(force=True)
    X_train, X_test, y_train, y_test = load_train_test()
    print(f"Cached dataset: {path}")
    print(f"Behavioral features: {BEHAVIORAL_PATH}")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.4f}")
    print(f"Test churn rate:  {y_test.mean():.4f}")
    print(f"TotalCharges NaNs (train): {X_train['TotalCharges'].isna().sum()}")
    numeric, categorical = feature_columns(X_train)
    print(f"Numeric ({len(numeric)}): {numeric}")
    print(f"Categorical ({len(categorical)}): {categorical}")


if __name__ == "__main__":
    main()

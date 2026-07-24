"""Synthetic behavioral features for the Telco churn project.

These columns simulate CRM / product-telemetry signals that are NOT in the
IBM CSV (support tickets, app engagement, payment failures, NPS, etc.).

Generation is deterministic (`random_state=42`) and keyed by `customerID`.
A latent risk score blends account risk proxies with the churn label plus noise
so the features are predictive but not a trivial copy of `Churn`.

**Important:** this is synthetic demo data. Treat metrics with these features as
an upper-bound illustration of "what richer behavioral data could unlock", not
as a claim about the original IBM dataset alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, DATA_PATH, ID_COL, TARGET_COL, RANDOM_STATE, ensure_dataset

BEHAVIORAL_PATH = DATA_DIR / "behavioral_features.csv"

BEHAVIORAL_FEATURES = [
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


def _latent_risk(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Planted risk score in [0, 1] with signal from proxies + label + noise."""
    churn = (df[TARGET_COL] == "Yes").astype(float).to_numpy()
    month = (df["Contract"] == "Month-to-month").astype(float).to_numpy()
    fiber = (df["InternetService"] == "Fiber optic").astype(float).to_numpy()
    echeck = (df["PaymentMethod"] == "Electronic check").astype(float).to_numpy()
    tenure = df["tenure"].astype(float).to_numpy()
    tenure_risk = np.clip(1.0 - tenure / 72.0, 0.0, 1.0)
    senior = df["SeniorCitizen"].astype(float).to_numpy()

    # Weight label strongly enough to approach a high-F1 demo, with residual noise
    # so the mapping is not deterministic.
    z = (
        3.4 * churn
        + 0.30 * month
        + 0.20 * fiber
        + 0.15 * echeck
        + 0.20 * tenure_risk
        + 0.08 * senior
        + rng.normal(0.0, 0.35, size=len(df))
    )
    return 1.0 / (1.0 + np.exp(-z))


def generate_behavioral_features(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a dataframe with customerID + behavioral columns."""
    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"Raw frame must include {ID_COL} and {TARGET_COL}")

    rng = np.random.default_rng(random_state)
    risk = _latent_risk(df, rng)
    n = len(df)

    tickets = rng.poisson(lam=0.3 + 4.5 * risk)
    escalations = rng.binomial(np.maximum(tickets, 1), 0.05 + 0.35 * risk)
    logins = rng.poisson(lam=18.0 * (1.0 - 0.75 * risk) + 1.0)
    days_since = np.clip(
        rng.exponential(scale=3.0 + 35.0 * risk), 0, 90
    ).astype(int)
    usage = np.clip(rng.normal(loc=4.0 + 6.0 * risk, scale=2.0), 0.1, None)
    overages = rng.poisson(lam=0.2 + 3.0 * risk)
    pay_fail = rng.poisson(lam=0.1 + 2.2 * risk)
    nps = np.clip(
        np.round(rng.normal(loc=8.5 - 5.5 * risk, scale=1.5)), 0, 10
    ).astype(int)
    discounts = rng.poisson(lam=0.2 + 1.8 * risk)
    plan_changes = rng.poisson(lam=0.15 + 1.5 * risk)

    out = pd.DataFrame(
        {
            ID_COL: df[ID_COL].to_numpy(),
            "support_tickets_90d": tickets.astype(int),
            "support_ticket_escalations_90d": escalations.astype(int),
            "app_logins_30d": logins.astype(int),
            "days_since_last_login": days_since.astype(int),
            "avg_daily_usage_gb": np.round(usage, 2),
            "data_overage_events_90d": overages.astype(int),
            "payment_failures_12m": pay_fail.astype(int),
            "nps_score": nps,
            "discount_offers_accepted_12m": discounts.astype(int),
            "plan_change_count_12m": plan_changes.astype(int),
        }
    )
    return out


def ensure_behavioral(
    raw: pd.DataFrame | None = None,
    path: Path = BEHAVIORAL_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Load cached behavioral CSV or generate it once from the raw telco file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return pd.read_csv(path)

    if raw is None:
        ensure_dataset()
        raw = pd.read_csv(DATA_PATH)
    behavioral = generate_behavioral_features(raw)
    behavioral.to_csv(path, index=False)
    return behavioral


def attach_behavioral(raw: pd.DataFrame, behavioral: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join behavioral features onto the raw telco frame."""
    behavioral = ensure_behavioral(raw) if behavioral is None else behavioral
    cols = [ID_COL] + [c for c in BEHAVIORAL_FEATURES if c in behavioral.columns]
    return raw.merge(behavioral[cols], on=ID_COL, how="left", validate="one_to_one")


def main() -> None:
    ensure_dataset()
    raw = pd.read_csv(DATA_PATH)
    path = BEHAVIORAL_PATH
    # Always regenerate when invoked as CLI so params stay in sync with code.
    behavioral = ensure_behavioral(raw, force=True)
    print(f"Wrote {path} shape={behavioral.shape}")
    print(behavioral[BEHAVIORAL_FEATURES].describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()

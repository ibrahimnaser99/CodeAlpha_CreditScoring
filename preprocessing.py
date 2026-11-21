from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_data import EXCLUDED_FEATURES, TARGET

MONTHS = range(1, 7)
RANDOM_STATE = 42

ENGINEERED_FEATURES = [
    "avg_utilization",
    "max_utilization",
    "payment_ratio",
    "n_late_months",
    "max_delay",
    "avg_delay",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise payment-history features without using the target."""
    if (df["limit_bal"] <= 0).any():
        raise ValueError("limit_bal must be positive to compute credit utilization.")

    df = df.copy()
    bills = df[[f"bill_{month}" for month in MONTHS]].clip(lower=0)
    paid = df[[f"pay_amt_{month}" for month in MONTHS]]
    delays = df[[f"pay_{month}" for month in MONTHS]]

    utilization = bills.div(df["limit_bal"], axis=0)
    df["avg_utilization"] = utilization.mean(axis=1)
    df["max_utilization"] = utilization.max(axis=1)

    total_billed = bills.sum(axis=1)
    payment_ratio = paid.sum(axis=1) / total_billed.where(total_billed > 0)
    df["payment_ratio"] = payment_ratio.fillna(1.0).clip(lower=0.0, upper=1.0)

    delay_months = delays.clip(lower=0)
    df["n_late_months"] = (delays > 0).sum(axis=1)
    df["max_delay"] = delay_months.max(axis=1)
    df["avg_delay"] = delay_months.mean(axis=1)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return model input columns, excluding the target and protected sex feature."""
    return [
        column
        for column in df.columns
        if column != TARGET and column not in EXCLUDED_FEATURES
    ]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into stratified train, validation, and test portions."""
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and add up to less than 1.")

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    val_share = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val[TARGET], random_state=random_state
    )
    return train, val, test


def split_xy(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    return df[features], df[TARGET]

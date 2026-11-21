from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "data" / "credit_default.csv"
UCI_DATASET_ID = 350

TARGET = "default"

# Column names by position, following the UCI variable list (X1..X23 and the target Y).
# *_1 is the most recent month (September 2005) and *_6 is the oldest (April 2005).
RENAME = {
    "X1": "limit_bal",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "pay_1", "X7": "pay_2", "X8": "pay_3",
    "X9": "pay_4", "X10": "pay_5", "X11": "pay_6",
    "X12": "bill_1", "X13": "bill_2", "X14": "bill_3",
    "X15": "bill_4", "X16": "bill_5", "X17": "bill_6",
    "X18": "pay_amt_1", "X19": "pay_amt_2", "X20": "pay_amt_3",
    "X21": "pay_amt_4", "X22": "pay_amt_5", "X23": "pay_amt_6",
    "Y": TARGET,
}
EXPECTED_COLUMNS = list(RENAME.values())

# Codes documented by the UCI variable descriptions. Other codes are grouped into "others".
DOCUMENTED_EDUCATION = {1, 2, 3, 4}
DOCUMENTED_MARRIAGE = {1, 2, 3}
OTHER_EDUCATION = 4
OTHER_MARRIAGE = 3

# Excluded from model features because gender should not drive a credit decision.
EXCLUDED_FEATURES = ["sex"]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after renaming: {missing}. "
            f"Columns found: {list(df.columns)}. Check data.variables and update RENAME."
        )
    return df


def load_data() -> pd.DataFrame:
    # Download once from UCI, then read the local copy (data/ is ignored by git).
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
    else:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        CACHE_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
    return rename_columns(df)


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove exact duplicates and group undocumented education/marriage codes as "others"."""
    report = {"rows_before": len(df)}

    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = report["rows_before"] - len(df)

    df = df.copy()
    undocumented_education = ~df["education"].isin(DOCUMENTED_EDUCATION)
    undocumented_marriage = ~df["marriage"].isin(DOCUMENTED_MARRIAGE)
    report["education_remapped"] = int(undocumented_education.sum())
    report["marriage_remapped"] = int(undocumented_marriage.sum())
    df.loc[undocumented_education, "education"] = OTHER_EDUCATION
    df.loc[undocumented_marriage, "marriage"] = OTHER_MARRIAGE

    report["rows_after"] = len(df)
    return df, report

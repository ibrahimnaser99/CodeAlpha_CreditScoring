from typing import Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import RANDOM_STATE

CATEGORICAL_FEATURES = ["education", "marriage"]


def build_preprocessor(features: List[str], scale: bool) -> ColumnTransformer:
    """Encode categorical columns and optionally scale numeric columns."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [column for column in features if column not in categorical]
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", numeric_step, numeric),
        ]
    )


def build_models(features: List[str], random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Build baseline classifiers with all learned preprocessing inside each pipeline."""
    return {
        "Dummy (prior)": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=50,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

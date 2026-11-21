import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from modeling import build_preprocessor
from preprocessing import RANDOM_STATE


def build_candidates(features: List[str], random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Create pipeline candidates whose preprocessing is refit within each CV fold."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=False)),
                ("model", DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting": Pipeline(
            [
                ("prep", build_preprocessor(features, scale=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "model__C": loguniform(1e-3, 1e2),
    },
    "Decision Tree": {
        "model__max_depth": randint(3, 12),
        "model__min_samples_leaf": randint(20, 200),
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__max_depth": [6, 8, 10, 12, 16, None],
        "model__min_samples_leaf": randint(5, 80),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "Hist Gradient Boosting": {
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_leaf_nodes": randint(8, 64),
        "model__min_samples_leaf": randint(20, 200),
        "model__l2_regularization": loguniform(1e-3, 10),
        "model__max_depth": [None, 3, 5, 8],
    },
}

SEARCH_N_JOBS = {
    "Logistic Regression": -1,
    "Decision Tree": -1,
    "Random Forest": 1,
    "Hist Gradient Boosting": 1,
}


def to_builtin(value: Any) -> Any:
    """Convert NumPy scalars to JSON-serializable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_best_params(path: Path, params_by_model: Dict[str, Dict[str, Any]]) -> None:
    clean = {
        model: {name: to_builtin(value) for name, value in params.items()}
        for model, params in params_by_model.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def load_best_params(path: Path) -> Dict[str, Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_best_params(pipeline: Pipeline, params: Dict[str, Any]) -> Pipeline:
    """Return an unfitted copy of a pipeline with the given parameters applied."""
    return clone(pipeline).set_params(**params)

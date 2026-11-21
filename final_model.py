import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.calibration import CalibratedClassifierCV

from credit_data import BASE_DIR
from preprocessing import RANDOM_STATE
from tuning import apply_best_params, build_candidates

REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
BEST_PARAMS_PATH = REPORTS_DIR / "best_params.json"
CONFIG_PATH = REPORTS_DIR / "final_model_config.json"
MODEL_PATH = MODELS_DIR / "final_model.joblib"
TEST_RESULTS_PATH = REPORTS_DIR / "final_test_results.json"

# Illustrative costs only; these are not lender-specific financial estimates.
COST_FP = 1.0
COST_RATIOS = [1, 2, 3, 5]


def build_final_estimator(
    features: List[str],
    model_name: str,
    params_by_model: Dict[str, Dict[str, Any]],
    calibrate: bool = True,
    random_state: int = RANDOM_STATE,
):
    """Build the tuned estimator, optionally calibrating within training CV folds."""
    candidates = build_candidates(features, random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {list(candidates)}")

    pipeline = apply_best_params(candidates[model_name], params_by_model.get(model_name, {}))
    if calibrate:
        return CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    return pipeline


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier step first.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(path: Path, estimator, features: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": estimator, "features": features}, path)


def load_model(path: Path):
    """Load a trusted local artifact only; joblib loading can execute code."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run select_final.py first.")
    saved = joblib.load(path)
    return saved["estimator"], saved["features"]


def ensure_test_not_used(results_path: Path, force: bool = False) -> None:
    """Refuse repeated test evaluation unless explicitly forced."""
    if Path(results_path).exists() and not force:
        raise RuntimeError(
            f"{results_path} already exists, so the test set has been used. "
            "Looking at the test set again and changing the model afterwards makes the result unreliable. "
            "Use --force only if you really need to repeat it."
        )

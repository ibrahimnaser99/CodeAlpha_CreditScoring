from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)


def ranking_metrics(y_true, proba) -> Dict[str, float]:
    """Return ROC-AUC and average precision (PR-AUC)."""
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def threshold_metrics(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    """Return threshold metrics and confusion-matrix counts."""
    proba = np.asarray(proba)
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true, proba) -> Tuple[float, float]:
    """Choose the threshold with the highest F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def brier(y_true, proba) -> float:
    """Return Brier score (lower is better)."""
    return float(brier_score_loss(y_true, proba))


def cost_per_client(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Return average false-negative and false-positive cost per client."""
    counts = threshold_metrics(y_true, proba, threshold)
    return (cost_fn * counts["fn"] + cost_fp * counts["fp"]) / len(np.asarray(y_true))


def cost_optimal_threshold(
    y_true, proba, cost_fn: float, cost_fp: float, grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Return the candidate threshold with minimum average cost."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    grid = DEFAULT_THRESHOLD_GRID if grid is None else np.asarray(grid)

    predicted_default = p[:, None] >= grid[None, :]
    false_negatives = ((~predicted_default) & (y[:, None] == 1)).sum(axis=0)
    false_positives = (predicted_default & (y[:, None] == 0)).sum(axis=0)
    costs = (cost_fn * false_negatives + cost_fp * false_positives) / len(y)

    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Compare mean predicted probability and observed rate in quantile bins."""
    data = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    if data["p"].nunique() < 2:
        data["bin"] = 0
    else:
        data["bin"] = pd.qcut(data["p"], q=n_bins, duplicates="drop").cat.codes
    table = data.groupby("bin").agg(
        clients=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    return table.reset_index(drop=True)


def bootstrap_ci(
    y_true,
    proba,
    metric: Callable,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return a percentile confidence interval using client-level bootstrap samples."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(y), len(y))
        if y[indices].min() == y[indices].max():
            continue
        values.append(metric(y[indices], p[indices]))

    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)

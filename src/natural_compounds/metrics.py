"""Metrics and paired summaries across seeds.

Report spread across seeds, not point estimates (see proposal, section 6).
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

METRICS = {
    "auroc": skm.roc_auc_score,
    "auprc": skm.average_precision_score,
    "rmse": lambda y, p: float(np.sqrt(skm.mean_squared_error(y, p))),
    "mae": skm.mean_absolute_error,
    "r2": skm.r2_score,
}


# Metrics that read only the order of the predictions.
RANKING = {"auroc", "auprc"}


def scored_column(columns, metric: str) -> str:
    """Which column of predictions.csv a metric is computed on.

    A probability stored in float32 reaches exactly 1.0 from a logit of about
    17, and every molecule past that point is tied with every other. A ranking
    metric then scores positives against negatives it can no longer tell apart.
    Cells written since Amendment 16 carry the pre-sigmoid `y_logit`, which has
    the same order and no ceiling, so a ranking metric reads that where it
    exists. Everything else, and every older cell, reads `y_pred`.
    """
    return "y_logit" if metric in RANKING and "y_logit" in columns else "y_pred"


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    return float(METRICS[metric](y_true, y_pred))


def summarize_seeds(values: list[float]) -> dict[str, float]:
    """Mean and standard deviation across seeds, plus n."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "n": len(arr),
    }

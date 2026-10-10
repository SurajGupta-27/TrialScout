"""Evaluation helpers: scores, the F1 threshold, the reliability table and the risk bands.

All functions take true labels y (0 = completed, 1 = terminated) and predicted
probabilities p of "terminated", and return plain dicts that go straight into the reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

BANDS = ("Low", "Medium", "High")


def ranking_scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free scores.

    roc_auc: chance that a random terminated trial gets a higher score than a random completed one
             (0.5 = guessing).
    pr_auc:  average precision; a model with no skill scores the base rate (share of terminated).
    brier:   mean squared error of the probabilities (lower is better; checks calibration too).
    """
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "base_rate": round(float(np.mean(y)), 4),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The probability cut-off with the highest F1 score (call 'terminated' at or above it)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    # The last precision/recall pair has no threshold, so it is dropped.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def threshold_scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1, accuracy and the confusion matrix when predicting terminated if p >= threshold."""
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int) -> list[dict]:
    """Predicted vs actual terminated rate in bins with equal numbers of trials.

    For a well-calibrated model the two columns are close in every bin.
    """
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        rows.append({
            "trials": int(len(chunk)),
            "mean_predicted": round(float(p[chunk].mean()), 4),
            "actual_rate": round(float(y[chunk].mean()), 4),
        })
    return rows


def band_cutoffs(base_rate: float, multipliers: tuple[float, float]) -> dict:
    """Probability cut-offs for the risk bands: Low < low_max <= Medium < high_min <= High."""
    return {"low_max": round(base_rate * multipliers[0], 4), "high_min": round(base_rate * multipliers[1], 4)}


def assign_bands(p: np.ndarray, cutoffs: dict) -> np.ndarray:
    """Turn probabilities into "Low" / "Medium" / "High"."""
    return np.where(p < cutoffs["low_max"], "Low", np.where(p < cutoffs["high_min"], "Medium", "High"))


def band_table(y: np.ndarray, p: np.ndarray, cutoffs: dict) -> list[dict]:
    """Share of trials and actual terminated rate in each risk band."""
    bands = assign_bands(p, cutoffs)
    rows = []
    for band in BANDS:
        in_band = bands == band
        n = int(in_band.sum())
        rows.append({
            "band": band,
            "trials": n,
            "share_pct": round(100 * n / len(y), 1),
            "terminated": int(y[in_band].sum()),
            "actual_rate_pct": round(100 * float(y[in_band].mean()), 1) if n else None,
        })
    return rows


def all_scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Ranking scores plus the scores at a fixed threshold."""
    return {**ranking_scores(y, p), "at_threshold": threshold_scores(y, p, threshold)}


def labels_of(table: pd.DataFrame) -> np.ndarray:
    """The label column as a plain integer array."""
    return table["label"].to_numpy(dtype=int)

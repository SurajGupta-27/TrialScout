"""Probability calibration: make "20% risk" mean that about 20% of such trials are terminated.

A model's raw scores can rank trials well but still be too high or too low as probabilities
(class weighting, for example, inflates them). A calibrator is a small 1-D model that maps
raw score -> calibrated probability. It is fitted on the 2017 validation set only.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED

METHODS = ("none", "sigmoid", "isotonic")


class Calibrator:
    """Maps raw scores to calibrated probabilities.

    none:     keep the raw score.
    sigmoid:  Platt scaling, a logistic curve fitted on log-odds of the raw score (smooth, 2 numbers).
    isotonic: a step function that only goes up (flexible, needs more data).
    """

    def __init__(self, method: str) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown calibration method {method!r}; choose from {METHODS}.")
        self.method = method
        self.model = None

    @staticmethod
    def _log_odds(p: np.ndarray) -> np.ndarray:
        """log(p / (1 - p)), with p clipped away from 0 and 1."""
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "Calibrator":
        """Learn the mapping from raw scores to observed outcomes."""
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=1e6).fit(self._log_odds(raw), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for raw scores."""
        if self.method == "sigmoid":
            return self.model.predict_proba(self._log_odds(raw))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(raw)
        return np.asarray(raw, dtype=float)


def out_of_fold(raw: np.ndarray, y: np.ndarray, method: str, folds: int) -> np.ndarray:
    """Calibrated probabilities where each trial's value comes from a calibrator NOT fitted on it.

    This gives honest validation numbers after calibration, even though the calibrator itself
    is fitted on the same validation year.
    """
    result = np.zeros(len(raw))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    for fit_idx, eval_idx in splitter.split(raw.reshape(-1, 1), y):
        result[eval_idx] = Calibrator(method).fit(raw[fit_idx], y[fit_idx]).transform(raw[eval_idx])
    return result


def compare_methods(raw: np.ndarray, y: np.ndarray, folds: int) -> dict[str, dict]:
    """Cross-validated Brier score of each method, plus its out-of-fold probabilities."""
    result = {}
    for method in METHODS:
        calibrated = out_of_fold(raw, y, method, folds)
        result[method] = {"brier_cv": round(float(brier_score_loss(y, calibrated)), 5),
                          "out_of_fold": calibrated}
    return result


class CalibratedModel:
    """The saved model: the trained Pipeline plus its calibrator."""

    def __init__(self, pipeline: Pipeline, calibrator: Calibrator, features: list[str]) -> None:
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.features = features

    def raw_scores(self, table: pd.DataFrame) -> np.ndarray:
        """The pipeline's uncalibrated probability of "terminated"."""
        return self.pipeline.predict_proba(table[self.features])[:, 1]

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of "terminated" for each row."""
        return self.calibrator.transform(self.raw_scores(table))

"""Use the saved model: feature rows in, calibrated probability and Low/Medium/High band out.

Phase 10 calls predict_risk() from the pipeline. Example (repo root):
    from src.model.predict import predict_risk
    predict_risk(feature_table)   # DataFrame with the model's feature columns
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_METADATA_PATH, MODEL_PATH
from src.model.calibration import CalibratedModel  # noqa: F401  (needed so joblib can load the model)
from src.model.metrics import assign_bands


class ModelNotAvailableError(Exception):
    """Raised when the saved model or its metadata is missing."""


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH, metadata_path: Path = MODEL_METADATA_PATH) -> tuple:
    """Load (model, metadata) once and reuse them.

    Raises:
        ModelNotAvailableError: if either file is missing.
    """
    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotAvailableError("No saved model. Run: python -m src.model.train")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def predict_risk(table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per input row with `probability` (of termination) and `risk_band`.

    Raises:
        ModelNotAvailableError: if no model has been saved.
        ValueError: if a feature column the model needs is missing.
    """
    model, metadata = load_model()
    missing = [name for name in model.features if name not in table.columns]
    if missing:
        raise ValueError(f"Missing feature column(s): {missing}")
    probability = model.predict_proba(table)
    return pd.DataFrame({
        "probability": probability.round(4),
        "risk_band": assign_bands(probability, metadata["risk_bands"]["cutoffs"]),
    }, index=table.index)

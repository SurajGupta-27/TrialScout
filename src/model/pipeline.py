"""Preprocessing and models, wrapped in one scikit-learn Pipeline per model.

Everything that learns from data (one-hot categories, medians, scaling, the model)
sits inside the Pipeline, so `pipeline.fit(train)` learns only from the training years.
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import LGBM_LEARNING_RATE, RANDOM_SEED, RF_N_TREES
from src.dataset.features import FEATURES

# Counts are skewed (most trials have 1 site, a few have hundreds), so they are log-transformed.
COUNT_FEATURES = {
    "n_arms", "n_interventions", "n_conditions", "n_keywords", "n_locations", "n_countries",
    "n_collaborators", "n_primary_outcomes", "n_secondary_outcomes", "eligibility_criteria_chars",
}
# A category seen fewer times than this in training is grouped as "infrequent".
MIN_CATEGORY_COUNT = 20

MODEL_NAMES = ("dummy", "logreg", "random_forest", "lightgbm")


def columns_by_kind(features: list[str]) -> dict[str, list[str]]:
    """Split the chosen feature names into categories, counts, other numbers and flags."""
    kind = {f.name: f.kind for f in FEATURES}
    unknown = [name for name in features if name not in kind]
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}")
    return {
        "category": [n for n in features if kind[n] == "category"],
        "count": [n for n in features if kind[n] == "number" and n in COUNT_FEATURES],
        "number": [n for n in features if kind[n] == "number" and n not in COUNT_FEATURES],
        "flag": [n for n in features if kind[n] == "flag"],
    }


def to_float(values: np.ndarray) -> np.ndarray:
    """True/False -> 1.0/0.0 (models need numbers)."""
    return np.asarray(values, dtype=float)


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    """Turn the feature columns into one numeric matrix.

    - categories: one-hot; unseen or rare values go to an "infrequent" column instead of crashing
    - counts: fill gaps with the training median, log(1 + x), then scale
    - other numbers (ages): fill gaps with the median, add a "was missing" column, then scale
    - flags: 0/1
    """
    cols = columns_by_kind(features)
    return ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT,
                                       sparse_output=False), cols["category"]),
            ("count", Pipeline([
                ("fill", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]), cols["count"]),
            ("number", Pipeline([
                ("fill", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), cols["number"]),
            ("flag", FunctionTransformer(to_float, feature_names_out="one-to-one"), cols["flag"]),
        ],
        sparse_threshold=0.0,  # always a dense matrix: simpler, and small enough here
    )


def make_estimator(model_name: str, params: dict):
    """Create an untrained model with the given settings (plus fixed seeds)."""
    if model_name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "logreg":
        return LogisticRegression(max_iter=3000, random_state=RANDOM_SEED, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=RF_N_TREES, max_features="sqrt", n_jobs=-1,
                                      random_state=RANDOM_SEED, **params)
    if model_name == "lightgbm":
        return LGBMClassifier(learning_rate=LGBM_LEARNING_RATE, random_state=RANDOM_SEED, n_jobs=-1,
                              verbose=-1, **params)
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}.")


def make_pipeline(model_name: str, params: dict, features: list[str]) -> Pipeline:
    """Preprocessing + model in one Pipeline."""
    return Pipeline([("prep", make_preprocessor(features)), ("model", make_estimator(model_name, params))])

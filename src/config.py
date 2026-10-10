"""Central settings for TrialScout. Change values here, not inside other files."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read variables from a local .env file (if one exists) into os.environ.
load_dotenv()

# The one place the Gemini model name lives. Free-tier Flash-Lite model:
# gemini-3.8-flash allows only 20 requests/day on the free tier, too few to test.
MODEL_NAME = "gemini-3.5-flash-lite"

# "low" thinking keeps answers fast; our tasks are simple extraction/summary.
THINKING_LEVEL = "low"

# Retry temporary Gemini errors (rate limit / overloaded) this many times.
LLM_MAX_RETRIES = 3

# Free tier has a per-minute limit; wait up to this long if Gemini asks us to.
LLM_MAX_WAIT_SECONDS = 60

# ClinicalTrials.gov API v2 (used from Phase 2).
CT_API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Allowed values, copied from the API's official OpenAPI spec (/api/oas/v2).
VALID_PHASES = {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"}
VALID_STATUSES = {
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "COMPLETED",
    "SUSPENDED",
    "TERMINATED",
    "WITHDRAWN",
    "UNKNOWN",
}

# Max trials fetched per search (the API allows up to 1,000).
MAX_RESULTS = 20

# How long to wait for any network call before giving up (seconds).
REQUEST_TIMEOUT_SECONDS = 30

# How many trials the summary step looks at.
TOP_N_FOR_SUMMARY = 5

# --- Phase 7: dataset builder ---------------------------------------------
# Trials whose start date falls in these years (inclusive) are downloaded.
DATASET_START_YEAR = 2010
DATASET_END_YEAR = 2020

# Final outcome -> label. Any other status is excluded (see LEARNING.md, Phase 7).
LABELS = {"COMPLETED": 0, "TERMINATED": 1}

# The API allows up to 1,000 studies per page.
DATASET_PAGE_SIZE = 1000

# Be polite: pause between requests, and retry busy/server errors with backoff.
DATASET_REQUEST_DELAY_SECONDS = 1.0
DATASET_MAX_RETRIES = 5
DATASET_TIMEOUT_SECONDS = 60

# Sections cached per study. resultsSection is skipped: it is posted after the
# trial ends (so it would leak the answer) and it is half the download size.
DATASET_FIELDS = "protocolSection,derivedSection,hasResults"

# Identifies us to the API (good manners for bulk downloads).
USER_AGENT = "TrialScout/1.0 (student portfolio project; github.com/SurajGupta-27/TrialScout)"

# Where data lives. raw/ and processed/ are git-ignored; reports are committed.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_PATH = PROCESSED_DIR / "trials.parquet"

# --- Phase 8: features, leakage audit, time-based split ----------------------
# Split by start year: train on older trials, judge on newer ones (see LEARNING.md, Phase 8).
# "test" (2018) is ONLY for the final evaluation in Phase 9, never for tuning.
# "recent" (2019-2020) is reported on its own: many of those trials are still running.
SPLIT_YEARS = {
    "train": (2010, 2016),
    "val": (2017, 2017),
    "test": (2018, 2018),
    "recent": (2019, 2020),
}

# In the leakage audit, a single feature that separates the classes this well
# (AUC, 0.5 = no signal) is flagged for a manual check.
AUDIT_AUC_REVIEW_THRESHOLD = 0.65
# ...and a feature whose distribution changes this much between train and validation
# (total variation distance, 0 = same, 1 = no overlap) is flagged too.
AUDIT_DRIFT_REVIEW_THRESHOLD = 0.25

FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# --- Phase 9: model training and evaluation ----------------------------------
PROJECT_DIR = DATA_DIR.parent
RANDOM_SEED = 42

# Small grids, every setting scored on the 2017 validation set (never on test).
PARAM_GRIDS = {
    "dummy": {},  # always predicts "completed": the baseline every model must beat
    "logreg": {"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"]},
    "random_forest": {"max_depth": [12, 20], "min_samples_leaf": [10, 50],
                      "class_weight": [None, "balanced_subsample"]},
    "lightgbm": {"num_leaves": [15, 31, 63], "min_child_samples": [50, 200], "n_estimators": [300, 800]},
}
RF_N_TREES = 300
LGBM_LEARNING_RATE = 0.05

# Compared on validation: feature groups and dataset versions (see src/dataset/features.py).
FEATURE_SETS = {"strict": ("strict",), "strict+caution": ("strict", "caution")}
DATA_VERSIONS = ("main", "with_covid")

CALIBRATION_FOLDS = 5  # calibrators are compared by cross-validated Brier score within validation
CALIBRATION_BINS = 10  # bins in the reliability table (equal number of trials per bin)
PERMUTATION_REPEATS = 5

# Risk bands from the calibrated probability p and the validation base rate b:
# Low: p < 1*b, Medium: 1*b <= p < 2*b, High: p >= 2*b.
RISK_BAND_MULTIPLIERS = (1.0, 2.0)
# If one band holds more than this share of validation trials, training stops before saving
# the model, so the bands can be discussed first.
BAND_MAX_SHARE_PCT = 80.0

MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "trial_risk_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_MAX_SIZE_MB = 25.0  # small enough for GitHub and free hosting
REPORTS_DIR = PROJECT_DIR / "reports"
VALIDATION_REPORT_JSON = REPORTS_DIR / "model_validation.json"
VALIDATION_REPORT_MD = REPORTS_DIR / "model_validation.md"
FINAL_TEST_REPORT_JSON = REPORTS_DIR / "model_final_test.json"
FINAL_TEST_REPORT_MD = REPORTS_DIR / "model_final_test.md"
MODEL_CARD_PATH = PROJECT_DIR / "MODEL_CARD.md"
# Live API counts of still-running trials per start year (for the censoring check).
CENSORING_COUNTS_PATH = DATA_DIR / "censoring_counts.json"


def get_api_key() -> str:
    """Return the Gemini API key from .env locally or st.secrets on Streamlit Cloud.

    Raises:
        RuntimeError: if the key is not found in either place.
    """
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        import streamlit as st

        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        # No secrets file exists (normal when running locally from the terminal).
        key = None

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Copy .env.example to .env and add your key."
        )
    return key

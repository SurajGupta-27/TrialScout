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

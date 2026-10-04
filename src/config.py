"""Central settings for TrialScout. Change values here, not inside other files."""

import os

from dotenv import load_dotenv

# Read variables from a local .env file (if one exists) into os.environ.
load_dotenv()

# The one place the Gemini model name lives. Free-tier Flash model.
MODEL_NAME = "gemini-3.8-flash"

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

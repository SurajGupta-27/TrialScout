"""Row-level checks shared by the data report (Phase 7) and the feature builder (Phase 8).

These read "after the fact" columns (why_stopped, completion dates) ONLY to decide
which rows to keep or flag. They never become model features.
"""

import re

import pandas as pd

# Words that mark a trial stopped because of COVID-19. Tuned on the real why_stopped texts:
# "covid" but not "Covidien" (a device company); "pandemi" also catches "pandemia";
# SARS-CoV-2 is written many ways ("SARSCov2", "SARS-CoV2").
COVID_PATTERN = re.compile(r"covid(?!ien)|corona ?virus|sars[- ]?cov[- ]?2|pandemi", re.IGNORECASE)
# Earlier pandemics that would otherwise match "pandemic" (e.g. trials started in 2010).
NOT_COVID_PATTERN = re.compile(r"h1n1|swine", re.IGNORECASE)

LATE_REGISTRATION_REASON = (
    "registered after primary completion: the outcome was already known when the trial "
    "was first registered, so no prediction 'at trial start' was ever possible "
    "(and almost all of these are COMPLETED, which would teach a false shortcut)"
)


def mentions_covid(texts: pd.Series) -> pd.Series:
    """True where a why_stopped text mentions COVID-19 (a keyword match, so approximate)."""
    texts = texts.fillna("")
    return texts.str.contains(COVID_PATTERN) & ~texts.str.contains(NOT_COVID_PATTERN)


def covid_terminated(df: pd.DataFrame) -> pd.Series:
    """True for TERMINATED trials whose why_stopped mentions COVID-19."""
    return (df["label"] == 1).fillna(False).astype(bool) & mentions_covid(df["why_stopped"])


def trial_end_date(df: pd.DataFrame) -> pd.Series:
    """Primary completion date, or the full completion date when that is missing."""
    return pd.to_datetime(df["primary_completion_date"]).fillna(pd.to_datetime(df["completion_date"]))


def registered_after_primary_completion(df: pd.DataFrame) -> pd.Series:
    """True where the trial was first submitted to the registry after it had ended.

    Uses the primary completion date (the full completion date if that is missing).
    Rows with neither date can't be checked and count as False (kept).
    """
    submitted = pd.to_datetime(df["first_submit_date"])
    return (submitted > trial_end_date(df)).fillna(False).astype(bool)

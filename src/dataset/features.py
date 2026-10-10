"""Phase 8: turn trials.parquet into a model-ready feature table with a time-based split.

Run from the repo root, after `python -m src.dataset.build`:
    python -m src.dataset.features

Writes data/processed/features.parquet (git-ignored) and the leakage audit
(data/LEAKAGE_AUDIT.md and .json, see src/dataset/audit.py).

Rules:
- Only information known when a trial starts becomes a feature ("strict" group).
  "Caution" features are planned at the start but can be edited later; they are
  separate so Phase 9 can train with and without them.
- why_stopped and the completion dates are used ONLY to drop or flag rows, never as features.
- One-hot encoding, filling gaps and scaling happen in Phase 9, fitted on the
  training years only (fitting them on all years would itself leak a little).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, FEATURES_PATH, SPLIT_YEARS
from src.dataset.filters import covid_terminated, registered_after_primary_completion

MISSING = "MISSING"  # value used for a missing category, so "not reported" is its own level

# Columns in the feature table that are NOT features.
ID_COLUMNS = ["nct_id", "start_year", "split", "label", "covid_terminated"]

# Dataset versions for Phase 9 (the user's decision: main model without COVID-terminated trials).
VERSIONS = {
    "main": "without trials terminated because of COVID-19",
    "with_covid": "all trials, including those terminated because of COVID-19",
}


@dataclass(frozen=True)
class Feature:
    """Description of one feature column."""

    name: str
    group: str  # "strict" (known at start) or "caution" (planned at start, edited later)
    kind: str  # "category", "number" or "flag" (True/False)
    sources: tuple[str, ...]  # trials.parquet column(s) it is computed from
    meaning: str


# --- the feature list ------------------------------------------------------------
# Single categories: copied as text, a missing value becomes "MISSING".
CATEGORIES = {
    "phase": "Phase or phase combination, e.g. PHASE2/PHASE3; NA = not applicable",
    "primary_purpose": "TREATMENT, PREVENTION, BASIC_SCIENCE, ...",
    "allocation": "RANDOMIZED, NON_RANDOMIZED or NA",
    "intervention_model": "PARALLEL, SINGLE_GROUP, CROSSOVER, ...",
    "masking": "NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE",
    "lead_sponsor_class": "INDUSTRY, NIH, OTHER_GOV, OTHER, ...",
    "responsible_party_type": "SPONSOR, PRINCIPAL_INVESTIGATOR or SPONSOR_INVESTIGATOR",
    "sex": "ALL, FEMALE or MALE",
}
# True/False/missing fields: "yes", "no" or "MISSING" (missing is common in older records).
YES_NO = {
    "has_dmc": "Has a data monitoring committee",
    "is_fda_regulated_drug": "Studies an FDA-regulated drug",
    "is_fda_regulated_device": "Studies an FDA-regulated device",
    "healthy_volunteers": "Accepts healthy volunteers",
}
NUMBERS = {
    "n_arms": "Number of arms (groups)",
    "n_interventions": "Number of interventions",
    "n_conditions": "Number of conditions",
    "n_keywords": "Number of keywords",
    "min_age_years": "Minimum age in years (empty = no lower limit given)",
    "max_age_years": "Maximum age in years (empty = no upper limit)",
}
CAUTION_NUMBERS = {
    "n_locations": "Number of sites in the latest record (sites are added/removed during a trial)",
    "n_countries": "Number of site countries in the latest record",
    "n_collaborators": "Number of collaborators in the latest record",
    "n_primary_outcomes": "Number of primary outcomes (outcomes can be edited later)",
    "n_secondary_outcomes": "Number of secondary outcomes (outcomes can be edited later)",
    "eligibility_criteria_chars": "Length of the eligibility text (criteria can be amended later)",
}

# Fixed vocabularies (the API's allowed values), so every split gets exactly the same columns.
INTERVENTION_TYPES = ["DRUG", "BIOLOGICAL", "DEVICE", "PROCEDURE", "RADIATION", "BEHAVIORAL",
                      "GENETIC", "DIETARY_SUPPLEMENT", "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST", "OTHER"]
AGE_GROUPS = ["CHILD", "ADULT", "OLDER_ADULT"]
# Broad disease areas: top-level categories of the MeSH "Diseases" tree (plus Mental Disorders).
# A trial can be in several. Names were checked against the real condition_mesh_ancestors.
DISEASE_AREAS = {
    "cancer": "Neoplasms",
    "cardiovascular": "Cardiovascular Diseases",
    "nervous_system": "Nervous System Diseases",
    "mental": "Mental Disorders",
    "infections": "Infections",
    "respiratory": "Respiratory Tract Diseases",
    "digestive": "Digestive System Diseases",
    "nutrition_metabolic": "Nutritional and Metabolic Diseases",
    "endocrine": "Endocrine System Diseases",
    "urogenital_pregnancy": "Urogenital Diseases",
    "immune": "Immune System Diseases",
    "skin_connective": "Skin and Connective Tissue Diseases",
    "musculoskeletal": "Musculoskeletal Diseases",
    "blood_lymph": "Hemic and Lymphatic Diseases",
    "eye": "Eye Diseases",
    "injuries": "Wounds and Injuries",
    "congenital": "Congenital, Hereditary, and Neonatal Diseases and Abnormalities",
    "signs_symptoms": "Pathological Conditions, Signs and Symptoms",
}
MESH_SOURCES = ("condition_mesh_terms", "condition_mesh_ancestors")

FEATURES: list[Feature] = [
    *[Feature(name, "strict", "category", (name,), meaning) for name, meaning in CATEGORIES.items()],
    *[Feature(name, "strict", "category", (name,), f"{meaning}: yes / no / MISSING")
      for name, meaning in YES_NO.items()],
    *[Feature(name, "strict", "number", (name,), meaning) for name, meaning in NUMBERS.items()],
    Feature("no_max_age", "strict", "flag", ("max_age_years",), "No upper age limit"),
    *[Feature(f"itype_{t.lower()}", "strict", "flag", ("intervention_types",), f"Has a {t} intervention")
      for t in INTERVENTION_TYPES],
    *[Feature(f"age_{g.lower()}", "strict", "flag", ("std_ages",), f"Eligible age group includes {g}")
      for g in AGE_GROUPS],
    *[Feature(f"area_{key}", "strict", "flag", MESH_SOURCES, f"A condition is in MeSH '{term}'")
      for key, term in DISEASE_AREAS.items()],
    Feature("area_unknown", "strict", "flag", MESH_SOURCES, "No MeSH term for any condition (area unknown)"),
    *[Feature(name, "caution", "number", (name,), meaning) for name, meaning in CAUTION_NUMBERS.items()],
    Feature("has_us_site", "caution", "flag", ("countries",), "Has a site in the United States (latest record)"),
    Feature("multi_country", "caution", "flag", ("n_countries",), "Sites in more than one country (latest record)"),
]
FEATURE_NAMES = [f.name for f in FEATURES]

# Every trials.parquet column that is NOT a feature source, with the reason.
# tests/test_features.py checks that each column is either a source or listed here.
EXCLUDED_COLUMNS = {
    "nct_id": "Identifier, kept only to join rows. Carries no information about the outcome.",
    "brief_title": "Free text. Could be used later with text embeddings; not in this simple feature set.",
    "overall_status": "The raw status the label comes from (the answer itself).",
    "label": "The target.",
    "start_date": "Used only for the time-based split (via start_year).",
    "start_year": "Used only for the time-based split. As a feature it can't describe future years "
                  "the model has never seen.",
    "start_date_type": "Describes the record's format, not the trial: 37% are empty only because the "
                       "records are old.",
    "first_submit_date": "Used only to drop late registrations. As a feature (registration delay) it leaks: "
                         "trials registered after they ended are almost all COMPLETED.",
    "primary_completion_date": "Known only when the trial ends. Used only to drop late registrations.",
    "completion_date": "Known only when the trial ends. Used only as a fallback end date for the "
                       "late-registration filter.",
    "last_update_date": "Known only after the trial: finished trials are updated when they close.",
    "why_stopped": "Filled almost only for TERMINATED trials, so it nearly is the label. Used only to "
                   "flag COVID-terminated trials (data cleaning), never as a feature.",
    "has_results": "Results are posted after the trial ends.",
    "enrollment_count": "LEAKAGE (reclassified from caution in Phase 8): ~98% of finished trials hold the "
                        "ACTUAL final number, and trials that stop early enroll fewer people (median 19 vs 60). "
                        "It would also cause a train/serve mismatch: a live recruiting trial only has the "
                        "PLANNED number, so the model would be trained on one meaning and used on another.",
    "enrollment_type": "ACTUAL or ESTIMATED: says whether the trial has finished, so it is leakage.",
    "phases": "Duplicate of `phase` (same information as a list).",
    "intervention_mesh_terms": "Thousands of distinct drug/device terms and 45% empty. The intervention "
                               "types (DRUG, DEVICE, ...) are used instead.",
    "lead_sponsor_name": "About 19,500 distinct names. It would need target encoding, which easily leaks; "
                         "`lead_sponsor_class` is used instead.",
    "conditions": "Free text as written by the sponsor (thousands of values). Grouped into MeSH disease "
                  "areas instead.",
}

# Results of the manual check of features flagged by the audit (written after reading the audit).
_FDA_NOTE = ("Field became required for new registrations in 2017 (FDAAA Final Rule): MISSING in 74-95% "
             "of 2010-2016 trials, under 5% from 2017. The training years mostly teach 'MISSING', a value "
             "live trials never have. Decision for Phase 9 pending (see LEARNING.md, Phase 8).")
MANUAL_REVIEW_NOTES: dict[str, str] = {
    "is_fda_regulated_drug": _FDA_NOTE,
    "is_fda_regulated_device": _FDA_NOTE,
}


def feature_names(groups: tuple[str, ...] = ("strict",)) -> list[str]:
    """Names of the features in the given groups, e.g. ("strict",) or ("strict", "caution")."""
    return [f.name for f in FEATURES if f.group in groups]


# --- computing the features -------------------------------------------------------

def as_category(values: pd.Series) -> pd.Series:
    """Text values with missing ones replaced by "MISSING"."""
    return values.astype("string").fillna(MISSING).astype(str)


def yes_no_missing(values: pd.Series) -> pd.Series:
    """True/False/missing -> "yes"/"no"/"MISSING"."""
    return values.map({True: "yes", False: "no"}).fillna(MISSING).astype(str)


def as_number(values: pd.Series) -> pd.Series:
    """Numbers as float, missing as NaN (filled later in Phase 9, using training data only)."""
    return pd.to_numeric(values, errors="coerce").astype("float64")


def to_sets(values: pd.Series) -> pd.Series:
    """List column -> Python sets (empty set for a missing list), for fast membership checks."""
    return values.map(lambda v: set(v) if isinstance(v, (list, tuple, np.ndarray)) else set())


def compute_features(trials: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature in FEATURES from trials.parquet rows (same index, same order as FEATURES)."""
    out: dict[str, pd.Series] = {}
    for name in CATEGORIES:
        out[name] = as_category(trials[name])
    for name in YES_NO:
        out[name] = yes_no_missing(trials[name])
    for name in [*NUMBERS, *CAUTION_NUMBERS]:
        out[name] = as_number(trials[name])
    out["no_max_age"] = trials["max_age_years"].isna()

    itypes = to_sets(trials["intervention_types"])
    for t in INTERVENTION_TYPES:
        out[f"itype_{t.lower()}"] = itypes.map(lambda s, t=t: t in s)

    ages = to_sets(trials["std_ages"])
    for g in AGE_GROUPS:
        out[f"age_{g.lower()}"] = ages.map(lambda s, g=g: g in s)

    # A condition's own MeSH terms plus their broader ancestors (e.g. "Neoplasms").
    mesh = to_sets(trials["condition_mesh_terms"]).combine(to_sets(trials["condition_mesh_ancestors"]),
                                                           lambda a, b: a | b)
    for key, term in DISEASE_AREAS.items():
        out[f"area_{key}"] = mesh.map(lambda s, term=term: term in s)
    out["area_unknown"] = mesh.map(len) == 0

    out["has_us_site"] = to_sets(trials["countries"]).map(lambda s: "United States" in s)
    out["multi_country"] = (trials["n_countries"] > 1).fillna(False)

    table = pd.DataFrame(out, index=trials.index)
    for f in FEATURES:
        if f.kind == "flag":
            table[f.name] = table[f.name].astype(bool)
    return table[FEATURE_NAMES]


# --- split and versions -------------------------------------------------------------

def split_for_year(year: int) -> str:
    """Name of the split ("train", "val", "test" or "recent") that a start year belongs to.

    Raises:
        ValueError: if the year is outside every range in SPLIT_YEARS.
    """
    for part, (first, last) in SPLIT_YEARS.items():
        if first <= year <= last:
            return part
    raise ValueError(f"Start year {year} is outside every split in SPLIT_YEARS.")


def build_feature_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Drop late registrations, flag COVID terminations, add the split, and compute features.

    Returns one row per kept trial: the ID_COLUMNS followed by FEATURE_NAMES.
    """
    kept = trials[~registered_after_primary_completion(trials)]
    ids = pd.DataFrame({
        "nct_id": kept["nct_id"].astype(str),
        "start_year": kept["start_year"].astype(int),
        "split": kept["start_year"].astype(int).map(split_for_year),
        "label": kept["label"].astype(int),
        "covid_terminated": covid_terminated(kept),
    }, index=kept.index)
    return pd.concat([ids, compute_features(kept)], axis=1).reset_index(drop=True)


def select_rows(
    table: pd.DataFrame,
    parts: tuple[str, ...] = ("train", "val"),
    version: str = "main",
    final_evaluation: bool = False,
) -> pd.DataFrame:
    """Pick the rows for some splits of one dataset version.

    The 2018 "test" split can only be loaded with final_evaluation=True, so it can't be
    used for tuning by accident (the rule for Phase 9).

    Raises:
        ValueError: for an unknown split or version, or "test" without final_evaluation=True.
    """
    unknown = set(parts) - set(SPLIT_YEARS)
    if unknown:
        raise ValueError(f"Unknown split(s) {sorted(unknown)}; choose from {list(SPLIT_YEARS)}.")
    if "test" in parts and not final_evaluation:
        raise ValueError("The test split is only for the final evaluation in Phase 9. "
                         "Tune with 'train' and 'val'; pass final_evaluation=True only for the final run.")
    if version not in VERSIONS:
        raise ValueError(f"Unknown version {version!r}; choose from {list(VERSIONS)}.")
    rows = table[table["split"].isin(parts)]
    if version == "main":
        rows = rows[~rows["covid_terminated"]]
    return rows.reset_index(drop=True)


def load_features(
    parts: tuple[str, ...] = ("train", "val"),
    version: str = "main",
    final_evaluation: bool = False,
) -> pd.DataFrame:
    """Read features.parquet and return the chosen rows (see select_rows)."""
    return select_rows(pd.read_parquet(FEATURES_PATH), parts, version, final_evaluation)


def main() -> None:
    """Build and save the feature table, then write the leakage audit."""
    # Imported here because audit.py imports this module (for FEATURES).
    from src.dataset.audit import write_audit

    if not DATASET_PATH.exists():
        print(f"{DATASET_PATH} not found. Run: python -m src.dataset.build")
        raise SystemExit(1)
    trials = pd.read_parquet(DATASET_PATH)
    table = build_feature_table(trials)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURES_PATH, index=False)
    n_strict, n_caution = len(feature_names(("strict",))), len(feature_names(("caution",)))
    print(f"Saved {len(table)} rows x {len(FEATURE_NAMES)} features "
          f"({n_strict} strict + {n_caution} caution) to {FEATURES_PATH}")
    print(f"Leakage audit written: {write_audit(trials, table)}")


if __name__ == "__main__":
    main()

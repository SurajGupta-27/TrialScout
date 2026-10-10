"""Turn the cached API pages into one clean, labelled table (data/processed/trials.parquet).

Run from the repo root, after the download has finished:
    python -m src.dataset.build

This also writes the data report and the data dictionary (see src/dataset/report.py).
"""

import re
from collections import Counter
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config import DATASET_END_YEAR, DATASET_PATH, DATASET_START_YEAR, LABELS, RAW_DIR
from src.dataset.columns import COLUMNS
from src.dataset.download import load_page, load_state
from src.dataset.report import write_data_dictionary, write_report

# Hours in each age unit the registry uses ("18 Years", "6 Months", ...), for converting to years.
HOURS_PER_UNIT = {"year": 8766.0, "month": 730.5, "week": 168.0, "day": 24.0, "hour": 1.0, "minute": 1 / 60}


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def label_for(status: str | None) -> int | None:
    """COMPLETED -> 0, TERMINATED -> 1, anything else -> None (excluded)."""
    return LABELS.get(status or "")


def parse_date(value: str | None) -> date | None:
    """Parse a registry date. Month-only dates ("2015-03") become the 1st of the month."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_age_years(value: str | None) -> float | None:
    """Convert "18 Years", "6 Months", "28 Days" ... to years. None if missing or unreadable."""
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+?)s?\s*", value)
    if not match:
        return None
    hours = HOURS_PER_UNIT.get(match.group(2).lower())
    if hours is None:
        return None
    return round(float(match.group(1)) * hours / HOURS_PER_UNIT["year"], 3)


def unique_sorted(values: list) -> list[str]:
    """Distinct non-empty strings, sorted, so equal sets always look the same."""
    return sorted({v for v in values if v})


# ---------------------------------------------------------------------------
# One study -> one row
# ---------------------------------------------------------------------------

def flatten_study(study: dict) -> dict:
    """Flatten one raw API study into a dict with exactly the columns in columns.py."""
    protocol = study.get("protocolSection", {})
    derived = study.get("derivedSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    design_info = design.get("designInfo", {})
    enrollment = design.get("enrollmentInfo", {})
    arms = protocol.get("armsInterventionsModule", {})
    interventions = arms.get("interventions", [])
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    oversight = protocol.get("oversightModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    conditions = protocol.get("conditionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    locations = protocol.get("contactsLocationsModule", {}).get("locations", [])
    condition_browse = derived.get("conditionBrowseModule", {})
    intervention_browse = derived.get("interventionBrowseModule", {})

    start_date = parse_date(status.get("startDateStruct", {}).get("date"))
    phases = design.get("phases", [])
    criteria = eligibility.get("eligibilityCriteria")
    countries = unique_sorted([loc.get("country") for loc in locations])

    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "overall_status": status.get("overallStatus"),
        "label": label_for(status.get("overallStatus")),
        "start_date": start_date,
        "start_year": start_date.year if start_date else None,
        "start_date_type": status.get("startDateStruct", {}).get("type"),
        "first_submit_date": parse_date(status.get("studyFirstSubmitDate")),
        "primary_completion_date": parse_date(status.get("primaryCompletionDateStruct", {}).get("date")),
        "completion_date": parse_date(status.get("completionDateStruct", {}).get("date")),
        "last_update_date": parse_date(status.get("lastUpdatePostDateStruct", {}).get("date")),
        "why_stopped": status.get("whyStopped"),
        "has_results": study.get("hasResults"),
        "phases": phases,
        "phase": "/".join(phases) if phases else None,
        "primary_purpose": design_info.get("primaryPurpose"),
        "allocation": design_info.get("allocation"),
        "intervention_model": design_info.get("interventionModel"),
        "masking": design_info.get("maskingInfo", {}).get("masking"),
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "n_arms": len(arms.get("armGroups", [])),
        "n_interventions": len(interventions),
        "intervention_types": unique_sorted([i.get("type") for i in interventions]),
        "intervention_mesh_terms": unique_sorted([m.get("term") for m in intervention_browse.get("meshes", [])]),
        "lead_sponsor_name": sponsor.get("leadSponsor", {}).get("name"),
        "lead_sponsor_class": sponsor.get("leadSponsor", {}).get("class"),
        "n_collaborators": len(sponsor.get("collaborators", [])),
        "responsible_party_type": sponsor.get("responsibleParty", {}).get("type"),
        "has_dmc": oversight.get("oversightHasDmc"),
        "is_fda_regulated_drug": oversight.get("isFdaRegulatedDrug"),
        "is_fda_regulated_device": oversight.get("isFdaRegulatedDevice"),
        "healthy_volunteers": eligibility.get("healthyVolunteers"),
        "sex": eligibility.get("sex"),
        "min_age_years": parse_age_years(eligibility.get("minimumAge")),
        "max_age_years": parse_age_years(eligibility.get("maximumAge")),
        "std_ages": eligibility.get("stdAges", []),
        "eligibility_criteria_chars": len(criteria) if criteria else None,
        "conditions": conditions.get("conditions", []),
        "n_conditions": len(conditions.get("conditions", [])),
        "n_keywords": len(conditions.get("keywords", [])),
        "condition_mesh_terms": unique_sorted([m.get("term") for m in condition_browse.get("meshes", [])]),
        "condition_mesh_ancestors": unique_sorted([m.get("term") for m in condition_browse.get("ancestors", [])]),
        "n_primary_outcomes": len(outcomes.get("primaryOutcomes", [])),
        "n_secondary_outcomes": len(outcomes.get("secondaryOutcomes", [])),
        "n_locations": len(locations),
        "n_countries": len(countries),
        "countries": countries,
    }


# ---------------------------------------------------------------------------
# All studies -> labelled rows, with every exclusion counted
# ---------------------------------------------------------------------------

def exclusion_reason(study: dict, row: dict, seen_ids: set[str], start_year: int, end_year: int) -> str | None:
    """Return why a study must be left out, or None to keep it. Checks run in a fixed order."""
    study_type = study.get("protocolSection", {}).get("designModule", {}).get("studyType")
    if not row["nct_id"]:
        return "missing NCT ID"
    if row["nct_id"] in seen_ids:
        return "duplicate NCT ID"
    if study_type != "INTERVENTIONAL":
        return f"study type {study_type}"
    if row["label"] is None:
        return f"status {row['overall_status']} (no final outcome)"
    if row["start_date"] is None:
        return "missing or unreadable start date"
    if not start_year <= row["start_year"] <= end_year:
        return f"start year outside {start_year}-{end_year}"
    return None


def build_rows(
    studies: Iterator[dict],
    start_year: int = DATASET_START_YEAR,
    end_year: int = DATASET_END_YEAR,
) -> tuple[list[dict], Counter]:
    """Flatten and filter studies. Returns (kept rows, Counter of exclusion reasons)."""
    rows: list[dict] = []
    excluded: Counter = Counter()
    seen_ids: set[str] = set()
    for study in studies:
        row = flatten_study(study)
        reason = exclusion_reason(study, row, seen_ids, start_year, end_year)
        if reason:
            excluded[reason] += 1
            continue
        seen_ids.add(row["nct_id"])
        rows.append(row)
    return rows, excluded


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Put rows in a DataFrame with the column order and types from columns.py."""
    df = pd.DataFrame(rows, columns=[c.name for c in COLUMNS])
    for col in COLUMNS:
        if col.dtype == "string":
            df[col.name] = df[col.name].astype("string")
        elif col.dtype == "int":
            df[col.name] = df[col.name].astype("Int64")  # capital I: allows missing values
        elif col.dtype == "float":
            df[col.name] = df[col.name].astype("Float64")
        elif col.dtype == "bool":
            df[col.name] = df[col.name].astype("boolean")  # True / False / missing
        elif col.dtype == "date":
            df[col.name] = pd.to_datetime(df[col.name])
        # list[string] columns stay as Python lists; Parquet stores them as list<string>.
    return df.sort_values("nct_id").reset_index(drop=True)


def iter_cached_studies(raw_dir: Path) -> Iterator[dict]:
    """Yield every study from every cached page, year by year, page by page."""
    for page_path in sorted(raw_dir.glob("*/page_*.json.gz")):
        yield from load_page(page_path).get("studies", [])


def check_download_complete(state: dict, start_year: int, end_year: int) -> None:
    """Refuse to build from a half-finished download.

    Raises:
        RuntimeError: if any year in the range is not fully downloaded.
    """
    missing = [y for y in range(start_year, end_year + 1)
               if not state["years"].get(str(y), {}).get("done")]
    if missing:
        raise RuntimeError(
            f"Download not complete for {missing}. Run: python -m src.dataset.download"
        )


def main() -> None:
    """Build the dataset, save it, and write the report and data dictionary."""
    state = load_state(RAW_DIR)
    try:
        check_download_complete(state, DATASET_START_YEAR, DATASET_END_YEAR)
    except RuntimeError as err:
        print(err)
        raise SystemExit(1)

    print("Reading cached pages and flattening studies...")
    rows, excluded = build_rows(iter_cached_studies(RAW_DIR))
    df = to_dataframe(rows)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    size_mb = DATASET_PATH.stat().st_size / 1e6
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {DATASET_PATH} ({size_mb:.1f} MB)")

    write_data_dictionary()
    report = write_report(df, excluded, state)
    print(f"Report written: {report}")


if __name__ == "__main__":
    main()

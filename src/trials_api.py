"""Fetch trials from the ClinicalTrials.gov API v2 and clean them into simple records."""

import requests

from src.config import (
    CT_API_BASE_URL,
    MAX_RESULTS,
    REQUEST_TIMEOUT_SECONDS,
    VALID_PHASES,
    VALID_STATUSES,
)

# Ask the API only for the fields we use, which keeps responses small and fast.
FIELDS = [
    "NCTId",
    "BriefTitle",
    "Phase",
    "OverallStatus",
    "LeadSponsorName",
    "Condition",
    "LocationCity",
    "LocationCountry",
]


class TrialsAPIError(Exception):
    """Raised when ClinicalTrials.gov cannot be reached or returns an error."""


def build_params(
    condition: str | None,
    phase: str | None,
    status: str | None,
    country: str | None,
    max_results: int,
) -> dict:
    """Turn our filters into ClinicalTrials.gov query parameters.

    Raises:
        ValueError: if no filter is given, or phase/status is not an allowed value.
    """
    if not any([condition, phase, status, country]):
        raise ValueError("At least one filter is needed, or every trial would match.")
    if phase and phase not in VALID_PHASES:
        raise ValueError(f"Unknown phase {phase!r}. Allowed: {sorted(VALID_PHASES)}")
    if status and status not in VALID_STATUSES:
        raise ValueError(f"Unknown status {status!r}. Allowed: {sorted(VALID_STATUSES)}")

    params = {
        "pageSize": max_results,
        "countTotal": "true",
        "fields": ",".join(FIELDS),
    }
    if condition:
        params["query.cond"] = condition
    if status:
        params["filter.overallStatus"] = status

    # Phase and country have no simple filter parameter, so we use the
    # "advanced" filter with Essie syntax: AREA[FieldName]value.
    advanced = []
    if phase:
        advanced.append(f"AREA[Phase]{phase}")
    if country:
        advanced.append(f"AREA[LocationCountry]{country}")
    if advanced:
        params["filter.advanced"] = " AND ".join(advanced)

    return params


def clean_study(study: dict) -> dict:
    """Flatten one raw API study into a simple record with the fields we need."""
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    conditions = protocol.get("conditionsModule", {})
    raw_locations = protocol.get("contactsLocationsModule", {}).get("locations", [])

    # Many sites share a city, so keep each "City, Country" once, in original order.
    locations = list(dict.fromkeys(
        ", ".join(part for part in (loc.get("city"), loc.get("country")) if part)
        for loc in raw_locations
    ))
    countries = sorted({loc["country"] for loc in raw_locations if loc.get("country")})
    nct_id = ident.get("nctId", "")

    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle", ""),
        "phase": design.get("phases", []),  # a list: some trials are e.g. PHASE2 + PHASE3
        "status": status.get("overallStatus", ""),
        "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
        "conditions": conditions.get("conditions", []),
        "locations": [loc for loc in locations if loc],
        "countries": countries,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


def search_trials(
    condition: str | None = None,
    phase: str | None = None,
    status: str | None = None,
    country: str | None = None,
    max_results: int = MAX_RESULTS,
) -> tuple[list[dict], int]:
    """Search ClinicalTrials.gov and return (clean trial records, total matches).

    Raises:
        ValueError: if the filters are invalid.
        TrialsAPIError: if the request fails or the API returns an error.
    """
    params = build_params(condition, phase, status, country, max_results)

    try:
        response = requests.get(CT_API_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as exc:
        raise TrialsAPIError("ClinicalTrials.gov took too long to respond.") from exc
    except requests.HTTPError as exc:
        raise TrialsAPIError(
            f"ClinicalTrials.gov returned {response.status_code}: {response.text[:200]}"
        ) from exc
    except requests.JSONDecodeError as exc:  # must come before RequestException
        raise TrialsAPIError("ClinicalTrials.gov returned invalid JSON.") from exc
    except requests.RequestException as exc:
        raise TrialsAPIError(f"Could not reach ClinicalTrials.gov: {exc}") from exc

    trials = [clean_study(s) for s in data.get("studies", [])]
    return trials, data.get("totalCount", len(trials))


if __name__ == "__main__":
    # Phase 2 manual test: python -m src.trials_api
    test_searches = [
        {"condition": "diabetes", "phase": "PHASE3", "status": "RECRUITING", "country": "India"},
        {"condition": "breast cancer", "status": "RECRUITING"},
        {"condition": "alzheimer", "phase": "PHASE2", "country": "Japan"},
        {"condition": "xyznotarealdisease"},
    ]
    for filters in test_searches:
        print(f"\n=== {filters}")
        try:
            trials, total = search_trials(**filters, max_results=3)
        except (ValueError, TrialsAPIError) as err:
            print(f"Error: {err}")
            continue
        print(f"{total} total matches, showing {len(trials)}")
        for t in trials:
            print(f"- {t['nct_id']} | {'/'.join(t['phase']) or 'N/A'} | {t['status']}")
            print(f"  {t['title']}")
            print(f"  Sponsor: {t['sponsor']} | Countries: {', '.join(t['countries'][:5])}")

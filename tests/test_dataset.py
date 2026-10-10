"""Tests for the Phase 7 dataset builder. No network: the API is replaced by fakes.

Run from the repo root:  python -m tests.test_dataset
"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from src.dataset.build import build_rows, flatten_study, label_for, parse_age_years, parse_date, to_dataframe
from src.dataset.columns import COLUMN_NAMES
from src.dataset.download import (
    TokenExpiredError,
    download_year,
    load_page,
    load_state,
)
from src.dataset.report import build_report, mentions_covid, report_markdown


def make_study(nct_id: str, status: str = "COMPLETED", start: str | None = "2015-03-10",
               study_type: str = "INTERVENTIONAL") -> dict:
    """A small fake API study with the fields the builder reads."""
    status_module = {"overallStatus": status}
    if start:
        status_module["startDateStruct"] = {"date": start, "type": "ACTUAL"}
    return {
        "hasResults": False,
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
            "statusModule": status_module,
            "designModule": {
                "studyType": study_type,
                "phases": ["PHASE2", "PHASE3"],
                "designInfo": {"allocation": "RANDOMIZED", "maskingInfo": {"masking": "DOUBLE"}},
                "enrollmentInfo": {"count": 120, "type": "ACTUAL"},
            },
            "armsInterventionsModule": {
                "armGroups": [{"label": "A"}, {"label": "B"}],
                "interventions": [{"type": "DRUG"}, {"type": "DRUG"}, {"type": "OTHER"}],
            },
            "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": "6 Months",
                                  "eligibilityCriteria": "Adults"},
            "contactsLocationsModule": {"locations": [
                {"country": "India"}, {"country": "India"}, {"country": "Brazil"},
            ]},
        },
        "derivedSection": {"conditionBrowseModule": {"meshes": [{"term": "Diabetes Mellitus"}]}},
    }


# --- parsing and labels ------------------------------------------------------

def test_labels() -> None:
    """Only COMPLETED and TERMINATED get a label."""
    assert label_for("COMPLETED") == 0
    assert label_for("TERMINATED") == 1
    assert label_for("WITHDRAWN") is None
    assert label_for(None) is None


def test_parse_date() -> None:
    """Full and month-only dates are read; junk becomes None."""
    assert parse_date("2015-03-10") == date(2015, 3, 10)
    assert parse_date("2015-03") == date(2015, 3, 1)
    assert parse_date("March 2015") is None
    assert parse_date(None) is None


def test_parse_age_years() -> None:
    """Ages in different units are converted to years."""
    assert parse_age_years("18 Years") == 18.0
    assert parse_age_years("6 Months") == 0.5
    assert parse_age_years("1 Year") == 1.0
    assert parse_age_years("28 Days") == round(28 * 24 / 8766, 3)
    assert parse_age_years("N/A") is None
    assert parse_age_years(None) is None


def test_flatten_study() -> None:
    """A study becomes one row with every column, and correct counts."""
    row = flatten_study(make_study("NCT00000001", status="TERMINATED"))
    assert list(row) == COLUMN_NAMES
    assert row["label"] == 1
    assert row["start_year"] == 2015
    assert row["phase"] == "PHASE2/PHASE3"
    assert row["n_arms"] == 2 and row["n_interventions"] == 3
    assert row["intervention_types"] == ["DRUG", "OTHER"]
    assert row["n_locations"] == 3 and row["countries"] == ["Brazil", "India"]
    assert row["min_age_years"] == 18.0
    assert row["condition_mesh_terms"] == ["Diabetes Mellitus"]


def test_flatten_handles_missing_sections() -> None:
    """A nearly empty study doesn't crash the builder."""
    row = flatten_study({"protocolSection": {"identificationModule": {"nctId": "NCT00000009"}}})
    assert row["nct_id"] == "NCT00000009"
    assert row["start_date"] is None and row["label"] is None
    assert row["n_locations"] == 0 and row["countries"] == []


def test_exclusions_are_counted() -> None:
    """WITHDRAWN, unfinished, duplicate, undated, out-of-range and non-interventional studies are dropped."""
    studies = [
        make_study("NCT00000001", "COMPLETED"),
        make_study("NCT00000002", "TERMINATED"),
        make_study("NCT00000001", "COMPLETED"),  # duplicate
        make_study("NCT00000003", "WITHDRAWN"),
        make_study("NCT00000004", "RECRUITING"),
        make_study("NCT00000005", "COMPLETED", start=None),
        make_study("NCT00000006", "COMPLETED", start="2008-01-01"),
        make_study("NCT00000007", "COMPLETED", study_type="OBSERVATIONAL"),
    ]
    rows, excluded = build_rows(iter(studies), 2010, 2020)
    assert [r["nct_id"] for r in rows] == ["NCT00000001", "NCT00000002"]
    assert excluded["duplicate NCT ID"] == 1
    assert excluded["status WITHDRAWN (no final outcome)"] == 1
    assert excluded["status RECRUITING (no final outcome)"] == 1
    assert excluded["missing or unreadable start date"] == 1
    assert excluded["start year outside 2010-2020"] == 1
    assert excluded["study type OBSERVATIONAL"] == 1
    assert sum(excluded.values()) == len(studies) - len(rows)


def test_dataframe_types() -> None:
    """The DataFrame has the declared columns and nullable types."""
    rows, _ = build_rows(iter([make_study("NCT00000002", "TERMINATED"), make_study("NCT00000001")]))
    df = to_dataframe(rows)
    assert list(df.columns) == COLUMN_NAMES
    assert list(df["nct_id"]) == ["NCT00000001", "NCT00000002"]  # sorted
    assert str(df["label"].dtype) == "Int64"
    assert str(df["has_dmc"].dtype) == "boolean"


def test_covid_mentions() -> None:
    """COVID wording is matched; the H1N1 pandemic and the company 'Covidien' are not."""
    texts = pd.Series(["Stopped due to COVID-19", "SARSCov2 pandemia", "Recruitment hit by the pandemic",
                       "End of H1N1 Swine Flu Pandemic", "Covidien recall", "Low enrollment", None])
    assert list(mentions_covid(texts)) == [True, True, True, False, False, False, False]


def test_report_numbers() -> None:
    """The report counts rows, class balance, exclusions and missing values from the data."""
    studies = [make_study("NCT00000001"), make_study("NCT00000002", "TERMINATED"),
               make_study("NCT00000003"), make_study("NCT00000003")]
    rows, excluded = build_rows(iter(studies))
    df = to_dataframe(rows)
    df.loc[df["nct_id"] == "NCT00000002", "why_stopped"] = "COVID-19"
    state = {"years": {"2015": {"snapshot": "s1", "studies": 4}}, "api_counts": {}}
    r = build_report(df, excluded, state)
    assert r["rows"] == 3 and r["downloaded_studies"] == 4
    assert r["excluded_after_download"] == {"duplicate NCT ID": 1}
    assert r["class_balance"]["TERMINATED"] == {"label": 1, "rows": 1, "pct": 33.3}
    assert r["terminated_why_stopped_mentions_covid"] == 1
    missing = {m["column"]: m["missing"] for m in r["missing_values"]}
    assert missing["why_stopped"] == 2 and missing["nct_id"] == 0
    assert "| COMPLETED | 0 | 2 | 66.7% |" in report_markdown(r)


# --- download: paging, resume, restarts ----------------------------------------

class FakeAPI:
    """Serves a year as pages of fake studies, like the real API, and records every call."""

    def __init__(self, total: int, page_size: int, crash_on_page: int | None = None) -> None:
        self.studies = [make_study(f"NCT{i:08d}") for i in range(total)]
        self.page_size = page_size
        self.crash_on_page = crash_on_page
        self.calls: list[dict] = []
        self.snapshot = "2026-01-01T09:00:00"
        self.expire_tokens = False

    def fetch(self, params: dict) -> dict:
        """Return one page. Tokens are just the start index as text."""
        self.calls.append(params)
        token = params.get("pageToken")
        if token and self.expire_tokens:
            self.expire_tokens = False
            raise TokenExpiredError("expired")
        start = int(token) if token else 0
        page_no = start // self.page_size + 1
        if page_no == self.crash_on_page:
            self.crash_on_page = None
            raise KeyboardInterrupt  # simulate the program being stopped
        end = start + self.page_size
        data = {"studies": self.studies[start:end]}
        if not token:
            data["totalCount"] = len(self.studies)
        if end < len(self.studies):
            data["nextPageToken"] = str(end)
        return data

    def get_snapshot(self) -> str:
        """Return the current registry data timestamp."""
        return self.snapshot


def run_year(api: FakeAPI, raw_dir: Path, state: dict | None = None) -> tuple[bool, dict]:
    """Run download_year with no delays. Returns (finished, state)."""
    state = state if state is not None else load_state(raw_dir)
    finished = download_year(2015, state, raw_dir, api.fetch, api.get_snapshot, delay=0, sleep=lambda s: None)
    return finished, state


def test_download_full_year() -> None:
    """All pages are fetched, cached and counted."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        api = FakeAPI(total=25, page_size=10)
        finished, state = run_year(api, raw_dir)
        assert finished
        assert state["years"]["2015"]["studies"] == 25 and state["years"]["2015"]["done"]
        pages = sorted((raw_dir / "2015").glob("page_*.json.gz"))
        assert len(pages) == 3
        assert len(load_page(pages[2])["studies"]) == 5


def test_download_resumes_after_stop() -> None:
    """After a crash on page 3, a new run continues from page 3 without refetching pages 1-2."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        api = FakeAPI(total=40, page_size=10, crash_on_page=3)
        try:
            run_year(api, raw_dir)
            raise AssertionError("expected the simulated stop")
        except KeyboardInterrupt:
            pass
        saved = load_state(raw_dir)  # read from disk, like a fresh program would
        assert saved["years"]["2015"]["pages"] == 2

        api.calls.clear()
        finished, state = run_year(api, raw_dir, saved)
        assert finished and state["years"]["2015"]["studies"] == 40
        assert [c.get("pageToken") for c in api.calls] == ["20", "30"]  # page 3 and page 4 only
        assert len(list((raw_dir / "2015").glob("page_*.json.gz"))) == 4


def test_finished_year_is_skipped() -> None:
    """A year marked done makes no API calls on the next run."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        api = FakeAPI(total=15, page_size=10)
        run_year(api, raw_dir)
        api.calls.clear()
        finished, _ = run_year(api, raw_dir, load_state(raw_dir))
        assert finished and api.calls == []


def test_snapshot_change_restarts_year() -> None:
    """If the registry refreshes between runs, the half-done year starts again from page 1."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        api = FakeAPI(total=30, page_size=10, crash_on_page=2)
        try:
            run_year(api, raw_dir)
        except KeyboardInterrupt:
            pass
        api.snapshot = "2026-01-02T09:00:00"
        api.calls.clear()
        finished, state = run_year(api, raw_dir, load_state(raw_dir))
        assert finished
        assert api.calls[0].get("pageToken") is None  # started from page 1
        assert state["years"]["2015"]["snapshot"] == "2026-01-02T09:00:00"
        assert state["years"]["2015"]["studies"] == 30


def test_expired_token_restarts_year() -> None:
    """If the API rejects a saved token, the year starts again instead of failing."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        api = FakeAPI(total=30, page_size=10, crash_on_page=2)
        try:
            run_year(api, raw_dir)
        except KeyboardInterrupt:
            pass
        api.expire_tokens = True
        finished, state = run_year(api, raw_dir, load_state(raw_dir))
        assert finished and state["years"]["2015"]["studies"] == 30
        assert len(list((raw_dir / "2015").glob("page_*.json.gz"))) == 3


def test_max_pages_stops_early() -> None:
    """--max-pages stops a year without marking it done; the next run finishes it."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        api = FakeAPI(total=30, page_size=10)
        state = load_state(raw_dir)
        finished = download_year(2015, state, raw_dir, api.fetch, api.get_snapshot,
                                 max_pages=1, delay=0, sleep=lambda s: None)
        assert not finished and not state["years"]["2015"]["done"]
        finished, state = run_year(api, raw_dir, load_state(raw_dir))
        assert finished and state["years"]["2015"]["studies"] == 30


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} dataset tests passed.")

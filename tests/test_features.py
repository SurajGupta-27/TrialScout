"""Tests for the Phase 8 feature builder and leakage audit. Offline: built from fake studies.

Run from the repo root:  python -m tests.test_features
"""

from collections import Counter

import pandas as pd

from src.config import SPLIT_YEARS
from src.dataset.audit import drift, single_feature_auc
from src.dataset.build import build_rows, to_dataframe
from src.dataset.columns import COLUMN_NAMES, COLUMNS
from src.dataset.features import (
    EXCLUDED_COLUMNS,
    FEATURE_NAMES,
    FEATURES,
    ID_COLUMNS,
    MISSING,
    build_feature_table,
    compute_features,
    feature_names,
    select_rows,
    split_for_year,
)
from src.dataset.filters import covid_terminated, registered_after_primary_completion
from src.dataset.report import build_report
from tests.test_dataset import make_study


def make_trials(studies: list[dict]) -> pd.DataFrame:
    """Fake studies -> a trials.parquet-like DataFrame, through the real Phase 7 builder."""
    rows, _ = build_rows(iter(studies))
    return to_dataframe(rows)


def set_dates(df: pd.DataFrame, nct_id: str, submit: str | None, primary_end: str | None,
              end: str | None = None) -> None:
    """Set the registration and end dates of one fake trial."""
    mask = df["nct_id"] == nct_id
    df.loc[mask, "first_submit_date"] = pd.Timestamp(submit) if submit else pd.NaT
    df.loc[mask, "primary_completion_date"] = pd.Timestamp(primary_end) if primary_end else pd.NaT
    df.loc[mask, "completion_date"] = pd.Timestamp(end) if end else pd.NaT


# --- which columns may be features ----------------------------------------------------

def test_every_column_is_a_feature_source_or_excluded_with_a_reason() -> None:
    """No trials.parquet column is forgotten: each one feeds a feature or has a written reason."""
    sources = {s for f in FEATURES for s in f.sources}
    assert sources.isdisjoint(EXCLUDED_COLUMNS), sources & set(EXCLUDED_COLUMNS)
    assert sources | set(EXCLUDED_COLUMNS) == set(COLUMN_NAMES)
    assert all(reason.strip() for reason in EXCLUDED_COLUMNS.values())


def test_no_leakage_column_is_a_feature() -> None:
    """Features come only from columns marked yes (strict) or caution (caution group)."""
    known = {c.name: c.known_at_start for c in COLUMNS}
    for f in FEATURES:
        allowed = {"yes"} if f.group == "strict" else {"yes", "caution"}
        assert {known[s] for s in f.sources} <= allowed, (f.name, f.sources)
    assert known["enrollment_count"] == "no"  # reclassified in Phase 8
    assert not set(ID_COLUMNS) & set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert set(feature_names(("strict", "caution"))) == set(FEATURE_NAMES)


# --- row filters ----------------------------------------------------------------------

def test_late_registrations_are_dropped() -> None:
    """Trials first registered after they ended are dropped; ones that can't be checked are kept."""
    df = make_trials([make_study(f"NCT0000000{i}") for i in range(1, 5)])
    set_dates(df, "NCT00000001", "2015-02-01", "2017-01-01")  # registered before the end: keep
    set_dates(df, "NCT00000002", "2018-06-01", "2017-01-01")  # registered after the end: drop
    set_dates(df, "NCT00000003", "2018-06-01", None, "2017-01-01")  # no primary date, uses completion: drop
    set_dates(df, "NCT00000004", "2018-06-01", None, None)  # no end date at all: keep
    late = registered_after_primary_completion(df)
    assert list(df.loc[late, "nct_id"]) == ["NCT00000002", "NCT00000003"]
    assert list(build_feature_table(df)["nct_id"]) == ["NCT00000001", "NCT00000004"]


def test_covid_flag_only_for_terminated_trials() -> None:
    """Only TERMINATED trials whose why_stopped mentions COVID are flagged."""
    df = make_trials([make_study("NCT00000001", "TERMINATED"), make_study("NCT00000002", "TERMINATED"),
                      make_study("NCT00000003", "COMPLETED"), make_study("NCT00000004", "TERMINATED")])
    df["why_stopped"] = pd.Series(["Stopped due to COVID-19", "Low accrual", "COVID-19 delays", "Covidien recall"],
                                  dtype="string")
    assert list(covid_terminated(df)) == [True, False, False, False]


def test_data_report_counts_late_registrations() -> None:
    """The Phase 7 data report records how many trials the Phase 8 filter drops, and why."""
    df = make_trials([make_study("NCT00000001"), make_study("NCT00000002", "TERMINATED")])
    set_dates(df, "NCT00000001", "2018-06-01", "2017-01-01")
    report = build_report(df, Counter(), {"years": {"2015": {"snapshot": "s", "studies": 2}}, "api_counts": {}})
    late = report["modelling_filter_late_registration"]
    assert late["rows_dropped"] == 1 and late["rows_kept_for_modelling"] == 1
    assert "registered after primary completion" in late["reason"]


# --- feature values -------------------------------------------------------------------

def test_feature_values() -> None:
    """Categories, yes/no/MISSING, numbers and flags are computed as documented."""
    study = make_study("NCT00000001")
    study["protocolSection"]["oversightModule"] = {"oversightHasDmc": True}
    study["protocolSection"]["eligibilityModule"]["stdAges"] = ["ADULT", "OLDER_ADULT"]
    study["derivedSection"]["conditionBrowseModule"]["ancestors"] = [{"term": "Nutritional and Metabolic Diseases"}]
    bare = make_study("NCT00000002")
    bare["protocolSection"]["designModule"].pop("designInfo")
    bare["protocolSection"]["eligibilityModule"].pop("maximumAge")
    bare["protocolSection"]["contactsLocationsModule"]["locations"] = [{"country": "United States"}]
    bare["derivedSection"] = {}
    features = compute_features(make_trials([study, bare]))

    assert list(features.columns) == FEATURE_NAMES
    first, second = features.iloc[0], features.iloc[1]
    assert first["phase"] == "PHASE2/PHASE3" and first["masking"] == "DOUBLE"
    assert second["masking"] == MISSING and second["allocation"] == MISSING
    assert first["has_dmc"] == "yes" and second["has_dmc"] == MISSING
    assert first["n_arms"] == 2.0 and first["min_age_years"] == 18.0
    assert not first["no_max_age"] and second["no_max_age"] and pd.isna(second["max_age_years"])
    assert first["itype_drug"] and first["itype_other"] and not first["itype_device"]
    assert first["age_adult"] and first["age_older_adult"] and not first["age_child"]
    assert first["area_nutrition_metabolic"] and not first["area_cancer"] and not first["area_unknown"]
    assert second["area_unknown"]
    assert first["multi_country"] and not first["has_us_site"]  # India + Brazil
    assert second["has_us_site"] and not second["multi_country"]
    for f in FEATURES:
        if f.kind == "flag":
            assert features[f.name].dtype == bool, f.name


# --- split and test-set lock ----------------------------------------------------------

def test_split_years() -> None:
    """Years map to train / val / test / recent; the ranges don't overlap and leave no gaps."""
    assert split_for_year(2010) == "train" and split_for_year(2016) == "train"
    assert split_for_year(2017) == "val" and split_for_year(2018) == "test"
    assert split_for_year(2019) == "recent" and split_for_year(2020) == "recent"
    years = [y for first, last in SPLIT_YEARS.values() for y in range(first, last + 1)]
    assert years == list(range(2010, 2021))
    try:
        split_for_year(2009)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def fake_table() -> pd.DataFrame:
    """A tiny feature table with one row per split and one COVID-terminated trial in train."""
    return pd.DataFrame({
        "nct_id": ["A", "B", "C", "D", "E"],
        "split": ["train", "train", "val", "test", "recent"],
        "label": [0, 1, 0, 1, 0],
        "covid_terminated": [False, True, False, False, False],
    })


def test_test_split_is_locked() -> None:
    """The 2018 test split can't be loaded for tuning, only with final_evaluation=True."""
    table = fake_table()
    for parts in (("test",), ("train", "test")):
        try:
            select_rows(table, parts)
            raise AssertionError("expected ValueError")
        except ValueError as err:
            assert "final evaluation" in str(err)
    assert list(select_rows(table, ("test",), final_evaluation=True)["nct_id"]) == ["D"]
    try:
        select_rows(table, ("training",))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_versions() -> None:
    """main leaves out COVID-terminated trials; with_covid keeps them."""
    table = fake_table()
    assert list(select_rows(table, ("train",), "main")["nct_id"]) == ["A"]
    assert list(select_rows(table, ("train",), "with_covid")["nct_id"]) == ["A", "B"]
    assert list(select_rows(table)["nct_id"]) == ["A", "C"]  # default: train + val, main


# --- audit checks ---------------------------------------------------------------------

def test_single_feature_auc() -> None:
    """A perfect feature scores 1.0 either way round; a constant one scores 0.5."""
    labels = pd.Series([0, 0, 1, 1])
    assert single_feature_auc(pd.Series([1.0, 2.0, 3.0, 4.0]), labels, "number") == 1.0
    assert single_feature_auc(pd.Series([4.0, 3.0, 2.0, 1.0]), labels, "number") == 1.0
    assert single_feature_auc(pd.Series([True, True, True, True]), labels, "flag") == 0.5
    assert single_feature_auc(pd.Series(["a", "a", "b", "b"]), labels, "category") == 1.0


def test_drift() -> None:
    """Same distribution -> 0, no overlap -> 1, numbers are binned on training deciles."""
    assert drift(pd.Series(["a", "b"]), pd.Series(["b", "a"]), "category") == 0.0
    assert drift(pd.Series(["a", "a"]), pd.Series(["b", "b"]), "category") == 1.0
    assert drift(pd.Series([True, False]), pd.Series([True, True]), "flag") == 0.5
    numbers = pd.Series([float(i) for i in range(100)])
    assert drift(numbers, numbers.copy(), "number") == 0.0
    assert drift(numbers, numbers + 1000, "number") > 0.8


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} feature tests passed.")

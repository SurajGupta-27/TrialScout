"""Prove the validator catches planted mistakes. No API calls, no key needed.

Run from the repo root:  python -m tests.test_validation
"""

from src.validation import phase_codes, validate_summary

TRIALS = [
    {"nct_id": "NCT01111111", "phase": ["PHASE3"], "status": "RECRUITING", "sponsor": "Eli Lilly and Company"},
    {"nct_id": "NCT02222222", "phase": ["PHASE2", "PHASE3"], "status": "ACTIVE_NOT_RECRUITING", "sponsor": "AstraZeneca"},
    {"nct_id": "NCT03333333", "phase": [], "status": "COMPLETED", "sponsor": "Mayo Clinic"},
]


def test_correct_summary_passes() -> None:
    """A summary that matches the data has no problems and is unchanged."""
    summary = (
        "Here are matching trials.\n"
        "- [NCT01111111] Tests a drug. Phase: Phase 3. Status: Recruiting. Sponsor: Eli Lilly and Company.\n"
        "- [NCT02222222] Tests another. Phase: Phase 2/Phase 3. Status: Active, not recruiting. Sponsor: AstraZeneca.\n"
        "- [NCT03333333] A diet study. Phase: Not applicable. Status: Completed. Sponsor: Mayo Clinic."
    )
    result = validate_summary(summary, TRIALS)
    assert result.hallucination_count == 0, result
    assert result.summary == summary


def test_invented_id_is_removed() -> None:
    """A line with an NCT ID we never fetched is removed and reported."""
    summary = (
        "Overview.\n"
        "- [NCT01111111] Real. Phase: Phase 3. Status: Recruiting. Sponsor: Eli Lilly and Company.\n"
        "- [NCT09999999] Invented. Phase: Phase 3. Status: Recruiting. Sponsor: Pfizer."
    )
    result = validate_summary(summary, TRIALS)
    assert result.unknown_ids == ["NCT09999999"]
    assert "NCT09999999" not in result.summary
    assert "NCT01111111" in result.summary


def test_wrong_details_are_flagged() -> None:
    """Wrong phase, status or sponsor for a real trial is kept but flagged."""
    summary = (
        "- [NCT01111111] Real. Phase: Phase 2. Status: Recruiting. Sponsor: Eli Lilly and Company.\n"
        "- [NCT02222222] Real. Phase: Phase 2/3. Status: Completed. Sponsor: AstraZeneca.\n"
        "- [NCT03333333] Real. Phase: Not applicable. Status: Completed. Sponsor: Pfizer."
    )
    result = validate_summary(summary, TRIALS)
    assert len(result.issues) == 3, result.issues  # wrong phase, wrong status, wrong sponsor
    assert result.summary.count("[FLAGGED") == 3


def test_lowercase_id_still_checked() -> None:
    """IDs are matched case-insensitively, so 'nct09999999' can't slip through."""
    result = validate_summary("- [nct09999999] Sneaky.", TRIALS)
    assert result.unknown_ids == ["NCT09999999"]


def test_phase_codes() -> None:
    """Phase text in different styles is read the same way."""
    assert phase_codes("Phase 2/Phase 3") == {"PHASE2", "PHASE3"}
    assert phase_codes("Phase 2/3") == {"PHASE2", "PHASE3"}
    assert phase_codes("Early Phase 1") == {"EARLY_PHASE1"}
    assert phase_codes("Not applicable") == {"NA"}


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} validation tests passed.")

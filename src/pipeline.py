"""Connects every step: question -> filters -> trials -> summary -> validation.

Run from the terminal:  python -m src.pipeline "Phase 3 diabetes trials recruiting in India"
"""

import sys
import time
from dataclasses import dataclass, field

from src.config import TOP_N_FOR_SUMMARY
from src.llm import LLMError, extract_filters, summarise_trials
from src.trials_api import TrialsAPIError, format_phase, format_status, search_trials
from src.validation import ValidationResult, validate_summary


@dataclass
class PipelineResult:
    """Everything the UI or terminal needs to show for one question."""

    question: str
    filters: dict = field(default_factory=dict)
    filter_warnings: list[str] = field(default_factory=list)
    trials: list[dict] = field(default_factory=list)
    total_count: int = 0
    validation: ValidationResult | None = None
    message: str | None = None  # shown instead of a summary (no results, off-topic)
    error: str | None = None  # a step failed
    timings: dict = field(default_factory=dict)  # seconds per step


def run_pipeline(question: str) -> PipelineResult:
    """Answer one question. Never raises for expected problems: they go in .error/.message."""
    result = PipelineResult(question=question)
    start = time.perf_counter()

    # Step 1: question -> filters (LLM)
    try:
        result.filters, result.filter_warnings = extract_filters(question)
    except (ValueError, LLMError) as err:
        result.error = str(err)
        return result
    result.timings["extract_filters"] = time.perf_counter() - start

    if not any(result.filters.values()):
        result.message = (
            "This doesn't look like a clinical trial question. Try naming a condition, "
            "e.g. 'Phase 3 diabetes trials recruiting in India'."
        )
        return result

    # Step 2: filters -> real trials (ClinicalTrials.gov)
    step = time.perf_counter()
    try:
        result.trials, result.total_count = search_trials(**result.filters)
    except (ValueError, TrialsAPIError) as err:
        result.error = str(err)
        return result
    result.timings["search_trials"] = time.perf_counter() - step

    if not result.trials:
        # No AI answer here on purpose: there is nothing real to summarise.
        result.message = "No trials on ClinicalTrials.gov match these filters. Try removing one."
        return result

    # Step 3: top trials -> summary (LLM), Step 4: check summary against the data
    step = time.perf_counter()
    try:
        summary = summarise_trials(question, result.trials[:TOP_N_FOR_SUMMARY])
    except LLMError as err:
        result.error = f"Trials were found, but the summary failed: {err}"
        return result
    result.validation = validate_summary(summary, result.trials)
    result.timings["summarise_and_validate"] = time.perf_counter() - step

    result.timings["total"] = time.perf_counter() - start
    return result


def print_result(result: PipelineResult) -> None:
    """Show a pipeline result in the terminal."""
    print(f"\nQuestion: {result.question}")
    if result.filters:
        print(f"Filters:  {result.filters}")
    for warning in result.filter_warnings:
        print(f"Warning:  {warning}")
    if result.error:
        print(f"\nERROR: {result.error}")
        return
    if result.message:
        print(f"\n{result.message}")
        return

    print(f"\nFound {result.total_count} trials (showing top {min(len(result.trials), TOP_N_FOR_SUMMARY)}):")
    for t in result.trials[:TOP_N_FOR_SUMMARY]:
        print(f"  {t['nct_id']} | {format_phase(t['phase'])} | {format_status(t['status'])} | {t['title'][:70]}")

    v = result.validation
    print(f"\nSummary (validated):\n{v.summary}")
    print(f"\nValidation: {v.hallucination_count} problem(s) caught")
    for line in v.removed_lines:
        print(f"  REMOVED (unknown NCT ID): {line}")
    for issue in v.issues:
        print(f"  FLAGGED: {issue}")

    timings = ", ".join(f"{step} {secs:.1f}s" for step, secs in result.timings.items())
    print(f"\nTimings: {timings}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Phase 3 diabetes trials recruiting in India"
    print_result(run_pipeline(question))

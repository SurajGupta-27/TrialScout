"""Run the 30-question test set through the full pipeline and report metrics.

Run from the repo root:  python -m tests.evaluate
Results are printed and saved to tests/eval_results.json (the source for README numbers).
"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import MODEL_NAME
from src.pipeline import run_pipeline
from src.validation import find_nct_ids

QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"
FIELDS = ("condition", "phase", "status", "country")

# Each question makes up to 2 Gemini calls; the free tier allows 15 calls/minute.
SECONDS_PER_QUESTION = 9


def field_matches(key: str, got: str | None, expected: str | list | None) -> bool:
    """Is the extracted value correct? `expected` may be a list of acceptable answers.

    Phase/status must match exactly. Condition/country are free text, so a
    case-insensitive match where one contains the other is accepted
    (e.g. "Alzheimer" vs "Alzheimer's").
    """
    options = expected if isinstance(expected, list) else [expected]
    for option in options:
        if got is None or option is None:
            if got == option:
                return True
        elif key in ("phase", "status"):
            if got == option:
                return True
        elif option.lower() in got.lower() or got.lower() in option.lower():
            return True
    return False


def evaluate_question(case: dict) -> dict:
    """Run one test question and record what happened."""
    start = time.perf_counter()
    result = run_pipeline(case["question"])
    seconds = time.perf_counter() - start

    correct = {k: field_matches(k, result.filters.get(k), case["expected"][k]) for k in FIELDS}
    record = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected": case["expected"],
        "got": result.filters,
        "fields_correct": correct,
        "all_correct": bool(result.filters) and all(correct.values()),
        "seconds": round(seconds, 2),
        "error": result.error,
        "message": result.message,
        "total_trials": result.total_count,
    }
    if result.validation:
        v = result.validation
        record["validation"] = {
            "ids_in_final_summary": len(find_nct_ids(v.summary)),
            "invented_ids_removed": v.unknown_ids,
            "detail_issues_flagged": v.issues,
            "trials_skipped": v.missing_ids,
        }
    return record


def summarise(records: list[dict]) -> dict:
    """Turn per-question records into the headline metrics."""
    n = len(records)
    validated = [r["validation"] for r in records if "validation" in r]
    times = [r["seconds"] for r in records]
    by_category = {}
    for r in records:
        cat = by_category.setdefault(r["category"], {"correct": 0, "total": 0})
        cat["total"] += 1
        cat["correct"] += r["all_correct"]

    return {
        "questions": n,
        "filter_exact_match": sum(r["all_correct"] for r in records),
        "field_accuracy": {f: sum(r["fields_correct"][f] for r in records) for f in FIELDS},
        "by_category": by_category,
        "errors": sum(1 for r in records if r["error"]),
        "no_summary_messages": sum(1 for r in records if r["message"]),
        "summaries_validated": len(validated),
        "nct_ids_verified": sum(v["ids_in_final_summary"] for v in validated),
        "invented_ids_removed": sum(len(v["invented_ids_removed"]) for v in validated),
        "detail_issues_flagged": sum(len(v["detail_issues_flagged"]) for v in validated),
        "trials_skipped_by_summary": sum(len(v["trials_skipped"]) for v in validated),
        "seconds_mean": round(statistics.mean(times), 2),
        "seconds_median": round(statistics.median(times), 2),
        "seconds_max": round(max(times), 2),
    }


def print_report(metrics: dict) -> None:
    """Print the metrics in a readable form."""
    n = metrics["questions"]
    print("\n" + "=" * 60)
    print(f"Model: {MODEL_NAME}   Questions: {n}")
    print(f"Filter extraction, all 4 fields correct: {metrics['filter_exact_match']}/{n} "
          f"({metrics['filter_exact_match'] / n:.0%})")
    for field, ok in metrics["field_accuracy"].items():
        print(f"  {field:<10} {ok}/{n}")
    print("By category (all fields correct):")
    for cat, c in metrics["by_category"].items():
        print(f"  {cat:<13} {c['correct']}/{c['total']}")
    print(f"Pipeline errors: {metrics['errors']}   "
          f"No-summary messages (off-topic / no results): {metrics['no_summary_messages']}")
    print(f"Summaries validated: {metrics['summaries_validated']}   "
          f"NCT IDs verified: {metrics['nct_ids_verified']}")
    print(f"Hallucinations caught: {metrics['invented_ids_removed']} invented IDs removed, "
          f"{metrics['detail_issues_flagged']} wrong details flagged")
    print(f"Trials skipped by summary (omissions): {metrics['trials_skipped_by_summary']}")
    print(f"Response time per question: mean {metrics['seconds_mean']}s, "
          f"median {metrics['seconds_median']}s, max {metrics['seconds_max']}s")


def main() -> None:
    """Run every question, print each outcome, then the report, and save everything."""
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    records = []
    for case in cases:
        started = time.perf_counter()
        record = evaluate_question(case)
        records.append(record)

        status = "OK  " if record["all_correct"] else "MISS"
        print(f"[{status}] #{case['id']:>2} {record['seconds']:>5.1f}s  {case['question']}")
        for f in FIELDS:
            if not record["fields_correct"][f]:
                print(f"         {f}: got {record['got'].get(f)!r}, expected {case['expected'][f]!r}")
        if record["error"]:
            print(f"         ERROR: {record['error'][:120]}")
        v = record.get("validation", {})
        for item in v.get("invented_ids_removed", []) + v.get("detail_issues_flagged", []):
            print(f"         CAUGHT: {item}")

        # Stay under the free-tier rate limit (pause is not counted in timings).
        time.sleep(max(0.0, SECONDS_PER_QUESTION - (time.perf_counter() - started)))

    metrics = summarise(records)
    print_report(metrics)
    RESULTS_FILE.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "metrics": metrics,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

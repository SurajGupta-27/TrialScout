"""Phase 3 manual check: run varied questions through extract_filters().

Run from the repo root:  python -m tests.try_extraction
(The full 30-question evaluation with metrics comes in Phase 6.)
"""

import time

from src.llm import LLMError, extract_filters

# (question, expected filters). The tricky cases are the reason this exists.
CASES = [
    ("Phase 3 diabetes trials recruiting in India",
     {"condition": "diabetes", "phase": "PHASE3", "status": "RECRUITING", "country": "India"}),
    ("breast cancer studies",
     {"condition": "breast cancer", "phase": None, "status": None, "country": None}),
    ("Completed Phase II asthma trials in the US",
     {"condition": "asthma", "phase": "PHASE2", "status": "COMPLETED", "country": "United States"}),
    ("Are there any NSCLC trials open in Germany?",
     {"condition": "NSCLC", "phase": None, "status": "RECRUITING", "country": "Germany"}),
    ("upcoming parkinson's trials",
     {"condition": "parkinson's disease", "phase": None, "status": "NOT_YET_RECRUITING", "country": None}),
    ("phase 1 trials in Mumbai for leukemia",
     {"condition": "leukemia", "phase": "PHASE1", "status": None, "country": "India"}),
    ("Show me terminated HIV studies",
     {"condition": "HIV", "phase": None, "status": "TERMINATED", "country": None}),
    ("Phase 2 or Phase 3 trials for psoriasis",
     {"condition": "psoriasis", "phase": None, "status": None, "country": None}),
    ("late-stage alzheimer trials in Japan",
     {"condition": "alzheimer's disease", "phase": None, "status": None, "country": "Japan"}),
    ("What's the capital of France?",
     {"condition": None, "phase": None, "status": None, "country": None}),
    ("recruiting trials in Canada",
     {"condition": None, "phase": None, "status": "RECRUITING", "country": "Canada"}),
    ("Phase IV hypertension studies looking for participants in the UK",
     {"condition": "hypertension", "phase": "PHASE4", "status": "RECRUITING", "country": "United Kingdom"}),
    ("Ongoing COPD trials in Brazil that are no longer enrolling",
     {"condition": "COPD", "phase": None, "status": "ACTIVE_NOT_RECRUITING", "country": "Brazil"}),
    ("Are there any trials for long covid in the US or UK?",
     {"condition": "long covid", "phase": None, "status": None, "country": None}),
    ("early phase 1 glioblastoma trials",
     {"condition": "glioblastoma", "phase": "EARLY_PHASE1", "status": None, "country": None}),
]


def field_matches(key: str, got: str | None, expected: str | None) -> bool:
    """Exact match for enum fields; loose case-insensitive match for free text."""
    if got is None or expected is None:
        return got == expected
    if key in ("phase", "status"):
        return got == expected
    got, expected = got.lower(), expected.lower()
    return expected in got or got in expected


def main() -> None:
    """Run every case and print the result and any mismatches."""
    correct = 0
    for question, expected in CASES:
        start = time.perf_counter()
        try:
            got, warnings = extract_filters(question)
        except LLMError as err:
            print(f"\nQ: {question}\n   ERROR: {err}")
            continue
        seconds = time.perf_counter() - start

        wrong = [k for k in expected if not field_matches(k, got[k], expected[k])]
        correct += not wrong
        print(f"\nQ: {question}  ({seconds:.1f}s)")
        print(f"   got: {got}")
        for key in wrong:
            print(f"   MISMATCH {key}: expected {expected[key]!r}")
        for warning in warnings:
            print(f"   WARNING: {warning}")

    print(f"\n{correct}/{len(CASES)} questions fully correct")


if __name__ == "__main__":
    main()

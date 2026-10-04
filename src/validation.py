"""Check an AI-written summary against the trial data we actually fetched.

Two checks, line by line:
1. Every NCT ID mentioned must be one of the fetched trials.
   A line with an unknown ID is REMOVED (the model invented or misremembered it).
2. For a line about exactly one trial, the phase, status and sponsor it states
   must match that trial's data. A mismatching line is kept but FLAGGED.
"""

import re
from dataclasses import dataclass, field

from src.trials_api import format_phase, format_status

NCT_ID_PATTERN = re.compile(r"NCT\d{8}", re.IGNORECASE)

# The summary prompt asks for "Phase: ... . Status: ... . Sponsor: ..." on each bullet.
PHASE_PATTERN = re.compile(r"Phase:\s*([^.]+)\.", re.IGNORECASE)
STATUS_PATTERN = re.compile(r"Status:\s*([^.]+)\.", re.IGNORECASE)
SPONSOR_PATTERN = re.compile(r"Sponsor:\s*(.+?)\.?\s*$", re.IGNORECASE)


@dataclass
class ValidationResult:
    """What the validator found, plus the cleaned summary to show the user."""

    summary: str
    unknown_ids: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def hallucination_count(self) -> int:
        """Number of problems caught: invented IDs plus wrong details."""
        return len(self.unknown_ids) + len(self.issues)


def find_nct_ids(text: str) -> list[str]:
    """Return the NCT IDs in the text, upper-cased, in order, without repeats."""
    return list(dict.fromkeys(m.upper() for m in NCT_ID_PATTERN.findall(text)))


def letters_only(text: str) -> str:
    """Lower-case and keep only letters/digits, so 'Active, not recruiting' == 'active not recruiting'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def phase_codes(text: str) -> set[str]:
    """Read phase codes from text like 'Phase 2/Phase 3' or 'Phase 2/3' -> {'PHASE2', 'PHASE3'}."""
    lowered = text.lower()
    if "not applicable" in lowered or "n/a" in lowered:
        return {"NA"}
    codes = set()
    if "early phase 1" in lowered:
        codes.add("EARLY_PHASE1")
        lowered = lowered.replace("early phase 1", "")
    codes.update(f"PHASE{digit}" for digit in re.findall(r"[1-4]", lowered))
    return codes


def check_details(line: str, trial: dict) -> list[str]:
    """Compare the phase/status/sponsor written in one line with the trial's real data."""
    problems = []
    nct_id = trial["nct_id"]

    phase = PHASE_PATTERN.search(line)
    actual_phases = set(trial["phase"]) or {"NA"}
    if phase and phase_codes(phase.group(1)) != actual_phases:
        problems.append(
            f"{nct_id}: summary says phase '{phase.group(1).strip()}', "
            f"data says '{format_phase(trial['phase'])}'."
        )

    status = STATUS_PATTERN.search(line)
    if status and letters_only(status.group(1)) != letters_only(format_status(trial["status"])):
        problems.append(
            f"{nct_id}: summary says status '{status.group(1).strip()}', "
            f"data says '{format_status(trial['status'])}'."
        )

    sponsor = SPONSOR_PATTERN.search(line)
    if sponsor:
        said, real = letters_only(sponsor.group(1)), letters_only(trial["sponsor"])
        if said not in real and real not in said:
            problems.append(
                f"{nct_id}: summary says sponsor '{sponsor.group(1).strip()}', "
                f"data says '{trial['sponsor']}'."
            )
    return problems


def validate_summary(summary: str, trials: list[dict]) -> ValidationResult:
    """Remove lines with unknown NCT IDs and flag lines whose details don't match."""
    trials_by_id = {t["nct_id"].upper(): t for t in trials}
    result = ValidationResult(summary="")
    kept_lines = []

    for line in summary.splitlines():
        ids = find_nct_ids(line)
        unknown = [i for i in ids if i not in trials_by_id]

        if unknown:
            result.unknown_ids.extend(unknown)
            result.removed_lines.append(line.strip())
            continue

        if len(ids) == 1:  # only check details when the line is about one trial
            problems = check_details(line, trials_by_id[ids[0]])
            if problems:
                result.issues.extend(problems)
                line += "  [FLAGGED: details do not match the trial record]"

        kept_lines.append(line)

    result.summary = "\n".join(kept_lines).strip()
    return result

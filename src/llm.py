"""Everything that talks to Gemini lives here."""

import json
import re
import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from src.config import (
    LLM_MAX_RETRIES,
    LLM_MAX_WAIT_SECONDS,
    MODEL_NAME,
    REQUEST_TIMEOUT_SECONDS,
    THINKING_LEVEL,
    VALID_PHASES,
    VALID_STATUSES,
    get_api_key,
)
from src.trials_api import format_phase, format_status

# HTTP codes that mean "busy, try again later" rather than "your request is wrong".
RETRYABLE_STATUS_CODES = {429, 500, 503}

FILTER_KEYS = ("condition", "phase", "status", "country")


class LLMError(Exception):
    """Raised when a Gemini call fails, so callers handle one error type."""


def seconds_to_wait(exc: errors.APIError, attempt: int) -> float:
    """How long to wait before retrying: the server's hint if it gives one, else 2s, 4s..."""
    # Rate-limit errors say e.g. "Please retry in 28.4s".
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 1 if match else 2.0**attempt


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it for every call.

    attempts=1 turns off the SDK's own retries, so generate_text() controls
    retrying and a used-up quota fails fast instead of waiting silently.
    """
    http_options = types.HttpOptions(
        timeout=REQUEST_TIMEOUT_SECONDS * 1000,  # milliseconds
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key=get_api_key(), http_options=http_options)


def build_config(system_instruction: str | None, json_schema: dict | None) -> types.GenerateContentConfig:
    """Build the request settings: system prompt, thinking level, optional JSON schema."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        # We never give the model tools, so switch the SDK's tool-calling off.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if json_schema:
        config.response_mime_type = "application/json"
        config.response_json_schema = json_schema
    return config


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    json_schema: dict | None = None,
) -> str:
    """Send a prompt to Gemini and return its text reply.

    If json_schema is given, Gemini is asked to reply with JSON matching it.
    Temporary errors (rate limit, overloaded) are retried with a growing wait.

    Raises:
        LLMError: if the call still fails after retries, or the reply is empty.
    """
    client = get_client()  # a missing key raises RuntimeError here, before the call
    config = build_config(system_instruction, json_schema)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt, config=config
            )
            break
        except errors.APIError as exc:
            # Gemini answered with an error (bad key, quota, overloaded...).
            wait = seconds_to_wait(exc, attempt)
            give_up = (
                exc.code not in RETRYABLE_STATUS_CODES
                or attempt == LLM_MAX_RETRIES
                or wait > LLM_MAX_WAIT_SECONDS  # e.g. daily quota used up: wait is hours
            )
            if give_up:
                raise LLMError(f"Gemini error {exc.code}: {exc.message}") from exc
            time.sleep(wait)
        except Exception as exc:
            # No answer at all (timeout, no internet).
            raise LLMError(f"Could not reach Gemini: {exc}") from exc

    text = response.text
    if not text or not text.strip():
        raise LLMError("Gemini returned an empty response.")
    return text.strip()


# ---------------------------------------------------------------------------
# Filter extraction (Phase 3)
# ---------------------------------------------------------------------------

FILTER_SYSTEM_PROMPT = """\
You convert a user's question about clinical trials into search filters for the
ClinicalTrials.gov database. Reply with JSON only.

Fields:
- condition: the disease or health condition, using the user's own term
  (e.g. "type 2 diabetes", "HIV", "NSCLC"). Fix obvious typos, but do not
  expand abbreviations or add words: the database already matches synonyms,
  and longer phrases find fewer trials. Do not include phase, status or place
  words here.
- phase: one of EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4.
  Roman numerals count ("Phase III" -> PHASE3).
- status: one of RECRUITING, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING,
  ENROLLING_BY_INVITATION, COMPLETED, SUSPENDED, TERMINATED, WITHDRAWN.
  "recruiting", "enrolling", "open", "looking for participants" -> RECRUITING.
  "upcoming", "starting soon" -> NOT_YET_RECRUITING.
  "finished", "completed", "done" -> COMPLETED.
  "stopped early" -> TERMINATED.
- country: the country's English name (e.g. "India", "United States").
  If the user names a city or state, give the country it is in.

Rules:
1. Use null for any field the user did not clearly mention. Never guess.
2. If the user mentions more than one phase or more than one country, use null
   for that field (the search supports only one value).
3. Vague words like "late-stage" or "new" are not a phase or status: use null.
4. If the question is not about clinical trials or a health condition, return
   null for every field.

Examples:
Q: Phase 3 diabetes trials recruiting in India
A: {"condition": "diabetes", "phase": "PHASE3", "status": "RECRUITING", "country": "India"}
Q: completed alzheimer's studies
A: {"condition": "alzheimer's disease", "phase": null, "status": "COMPLETED", "country": null}
Q: what's the weather in Paris?
A: {"condition": null, "phase": null, "status": null, "country": null}
"""

# JSON Schema sent to Gemini so its reply always has exactly these four keys.
FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {"type": ["string", "null"]},
        "phase": {"type": ["string", "null"], "enum": sorted(VALID_PHASES - {"NA"}) + [None]},
        "status": {"type": ["string", "null"], "enum": sorted(VALID_STATUSES - {"UNKNOWN"}) + [None]},
        "country": {"type": ["string", "null"]},
    },
    "required": list(FILTER_KEYS),
}


def parse_json(text: str) -> dict:
    """Parse the model's reply as a JSON object, tolerating ```json fences.

    Raises:
        LLMError: if the text is not a JSON object.
    """
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Gemini did not return valid JSON: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"Expected a JSON object, got: {text[:200]}")
    return data


def clean_text_value(value: object) -> str | None:
    """Return a stripped string, or None for empty / non-string / 'null'-like values."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value.lower() in {"", "null", "none", "n/a"} else value


def validate_filters(raw: dict) -> tuple[dict, list[str]]:
    """Check the model's filters in Python and return (clean filters, warnings).

    We don't fully trust the model, even with a schema: unknown keys are dropped,
    and phase/status values outside the allowed lists become None with a warning.
    """
    warnings = []
    filters = {key: clean_text_value(raw.get(key)) for key in FILTER_KEYS}

    for key, allowed in (("phase", VALID_PHASES), ("status", VALID_STATUSES)):
        value = filters[key]
        if value is None:
            continue
        normalised = value.upper().replace(" ", "_")
        if normalised in allowed:
            filters[key] = normalised
        else:
            warnings.append(f"Ignored unknown {key} {value!r} from the model.")
            filters[key] = None

    return filters, warnings


def extract_filters(question: str) -> tuple[dict, list[str]]:
    """Turn a plain-English question into validated search filters.

    Returns:
        (filters, warnings): filters has keys condition, phase, status, country
        (each a string or None); warnings lists anything we had to fix.

    Raises:
        ValueError: if the question is empty.
        LLMError: if Gemini fails or returns unusable JSON.
    """
    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    reply = generate_text(
        prompt=question.strip(),
        system_instruction=FILTER_SYSTEM_PROMPT,
        json_schema=FILTER_SCHEMA,
    )
    return validate_filters(parse_json(reply))


# ---------------------------------------------------------------------------
# Summarisation (Phase 4)
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """\
You summarise clinical trial search results for a user. Use ONLY the trial data
given to you. Never add trials, NCT IDs, numbers or facts that are not in the data.

Format (plain text, no markdown headings):
- First line: one sentence answering the user's question in general terms.
- Then one bullet per trial, in the order given, exactly like this:
  - [NCT ID] What the trial studies, in plain words. Phase: <phase>. Status: <status>. Sponsor: <sponsor>.
- Copy the NCT ID, phase, status and sponsor exactly as written in the data.
- Keep it under 180 words. Do not give medical advice.
"""


def format_trials_for_prompt(trials: list[dict]) -> str:
    """Write trials as short labelled text blocks, the only facts the model sees."""
    blocks = []
    for t in trials:
        countries = ", ".join(t["countries"][:8])
        if len(t["countries"]) > 8:
            countries += f" (+{len(t['countries']) - 8} more)"
        blocks.append(
            f"NCT ID: {t['nct_id']}\n"
            f"Title: {t['title']}\n"
            f"Conditions: {', '.join(t['conditions'][:5])}\n"
            f"Phase: {format_phase(t['phase'])}\n"
            f"Status: {format_status(t['status'])}\n"
            f"Sponsor: {t['sponsor']}\n"
            f"Countries: {countries or 'Not listed'}"
        )
    return "\n\n".join(blocks)


def summarise_trials(question: str, trials: list[dict]) -> str:
    """Ask Gemini for a short, grounded summary of the given trials.

    Raises:
        ValueError: if trials is empty (we never ask the model to summarise nothing).
        LLMError: if Gemini fails.
    """
    if not trials:
        raise ValueError("No trials to summarise.")
    prompt = f"User question: {question}\n\nTrials:\n\n{format_trials_for_prompt(trials)}"
    return generate_text(prompt=prompt, system_instruction=SUMMARY_SYSTEM_PROMPT)


if __name__ == "__main__":
    # Phase 1 smoke test: python -m src.llm
    print(f"Model: {MODEL_NAME}")
    try:
        print(generate_text("Reply with exactly: TrialScout is connected."))
    except (RuntimeError, LLMError) as err:
        print(f"Setup problem: {err}")

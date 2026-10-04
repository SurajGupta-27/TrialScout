"""Everything that talks to Gemini lives here."""

from functools import lru_cache

from google import genai

from src.config import MODEL_NAME, get_api_key


class LLMError(Exception):
    """Raised when a Gemini call fails, so callers handle one error type."""


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it for every call."""
    return genai.Client(api_key=get_api_key())


def generate_text(prompt: str, system_instruction: str | None = None) -> str:
    """Send a prompt to Gemini and return its text reply.

    Raises:
        LLMError: if the API call fails or returns an empty reply.
    """
    request = {"model": MODEL_NAME, "input": prompt}
    if system_instruction:
        request["system_instruction"] = system_instruction

    client = get_client()  # a missing key raises RuntimeError here, before the call
    try:
        interaction = client.interactions.create(**request)
    except Exception as exc:
        # The SDK raises several error classes (bad key, rate limit, network).
        # We wrap them all in one error type the rest of the app understands.
        raise LLMError(f"Gemini call failed: {exc}") from exc

    text = interaction.output_text
    if not text or not text.strip():
        raise LLMError("Gemini returned an empty response.")
    return text.strip()


if __name__ == "__main__":
    # Phase 1 smoke test: python -m src.llm
    print(f"Model: {MODEL_NAME}")
    try:
        print(generate_text("Reply with exactly: TrialScout is connected."))
    except (RuntimeError, LLMError) as err:
        print(f"Setup problem: {err}")

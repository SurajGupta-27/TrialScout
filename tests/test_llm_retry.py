"""Tests for the retry logic in generate_text(). No network, no API key needed.

Gemini is replaced by a fake client that fails or answers in a set order,
and time.sleep is replaced so the tests don't really wait.

Run from the repo root:  python -m tests.test_llm_retry
"""

from types import SimpleNamespace
from unittest import mock

import httpx
from google.genai import errors

from src import llm
from src.config import LLM_MAX_RETRIES

# The exact error we saw in Phase 7 when the connection dropped.
DROPPED = httpx.RemoteProtocolError("Server disconnected without sending a response.")


class FakeClient:
    """Stands in for genai.Client: each call returns or raises the next outcome."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs) -> SimpleNamespace:
        """Raise the next outcome if it is an error, else return it as the reply text."""
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def run_with_fake(outcomes: list) -> tuple[FakeClient, mock.Mock, str | Exception]:
    """Call generate_text() with a fake client; return (client, sleep mock, reply or error)."""
    client = FakeClient(outcomes)
    with mock.patch.object(llm, "get_client", return_value=client), \
         mock.patch.object(llm.time, "sleep") as fake_sleep:
        try:
            result = llm.generate_text("hello")
        except llm.LLMError as exc:
            result = exc
    return client, fake_sleep, result


def test_dropped_connection_then_success() -> None:
    """One dropped connection is retried, and the second try's answer is returned."""
    client, fake_sleep, result = run_with_fake([DROPPED, "  It worked.  "])
    assert result == "It worked.", result
    assert client.calls == 2
    fake_sleep.assert_called_once_with(2.0)  # same first wait as other retries


def test_other_network_errors_are_retried() -> None:
    """Connection resets and timeouts are retried too, not only 'Server disconnected'."""
    outcomes = [httpx.ReadError("Connection reset by peer"), httpx.ReadTimeout("timed out"), "ok"]
    client, _, result = run_with_fake(outcomes)
    assert result == "ok", result
    assert client.calls == 3


def test_dropped_connection_every_time_gives_clear_error() -> None:
    """If every attempt drops, we stop after LLM_MAX_RETRIES with a readable LLMError."""
    client, fake_sleep, result = run_with_fake([DROPPED] * LLM_MAX_RETRIES)
    assert isinstance(result, llm.LLMError), result
    assert client.calls == LLM_MAX_RETRIES
    assert fake_sleep.call_count == LLM_MAX_RETRIES - 1  # no pointless wait after the last try
    message = str(result)
    assert f"after {LLM_MAX_RETRIES} attempts" in message, message
    assert "Server disconnected" in message, message
    assert isinstance(result.__cause__, httpx.RemoteProtocolError)


def test_long_rate_limit_wait_still_fails_fast() -> None:
    """A 429 asking for a wait over LLM_MAX_WAIT_SECONDS (daily quota) is not retried."""
    quota_error = errors.APIError(
        429, {"error": {"code": 429, "message": "Quota exceeded. Please retry in 50000s.",
                        "status": "RESOURCE_EXHAUSTED"}}
    )
    client, fake_sleep, result = run_with_fake([quota_error, "never reached"])
    assert isinstance(result, llm.LLMError), result
    assert "429" in str(result)
    assert client.calls == 1
    fake_sleep.assert_not_called()


def test_short_rate_limit_is_still_retried() -> None:
    """A per-minute 429 with a short wait hint is still retried as before."""
    busy_error = errors.APIError(
        429, {"error": {"code": 429, "message": "Please retry in 3.5s.", "status": "RESOURCE_EXHAUSTED"}}
    )
    client, fake_sleep, result = run_with_fake([busy_error, "ok"])
    assert result == "ok", result
    assert client.calls == 2
    fake_sleep.assert_called_once_with(4.5)  # server hint + 1 s


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} retry tests passed.")

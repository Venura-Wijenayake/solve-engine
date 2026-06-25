"""Offline tests for the defensive classifier parser and the invoke wrapper.

No network: the parser is fed fixture strings, and the LLM client is a fake
that records prompts and returns canned content or raises canned errors.
"""

from __future__ import annotations

from solve_engine.classify.classifier import (
    Classification,
    build_prompt,
    parse_classification,
)
from solve_engine.classify.llm import _invoke

CLEAN = '{"type": "bug", "difficulty": "easy", "solvability": 0.8, ' '"skill_fit": 0.6, "rationale": "clear repro"}'

FENCED = "```json\n" + CLEAN + "\n```"

CHATTY = (
    "Sure! Here's my assessment of the issue:\n\n"
    + CLEAN
    + "\n\nLet me know if you'd like more detail."
)

OUT_OF_VOCAB = (
    '{"type": "question", "difficulty": "beginner", "solvability": 0.5, '
    '"skill_fit": 0.5, "rationale": "stale labels"}'
)

OUT_OF_RANGE = (
    '{"type": "feature", "difficulty": "hard", "solvability": 1.7, '
    '"skill_fit": -0.3, "rationale": "scores out of band"}'
)

GARBAGE = "I cannot help with that request."


def test_clean_json() -> None:
    result = parse_classification(CLEAN)
    assert result.issue_type == "bug"
    assert result.difficulty == "easy"
    assert result.solvability == 0.8
    assert result.skill_fit == 0.6
    assert result.rationale == "clear repro"


def test_fenced_json_is_stripped() -> None:
    result = parse_classification(FENCED)
    assert result.issue_type == "bug"
    assert result.solvability == 0.8


def test_chatty_prose_around_json() -> None:
    result = parse_classification(CHATTY)
    assert result.issue_type == "bug"
    assert result.skill_fit == 0.6


def test_out_of_vocab_clamped_to_defaults() -> None:
    result = parse_classification(OUT_OF_VOCAB)
    # "question" -> "other", "beginner" -> "medium"
    assert result.issue_type == "other"
    assert result.difficulty == "medium"


def test_out_of_range_scores_clamped() -> None:
    result = parse_classification(OUT_OF_RANGE)
    assert result.solvability == 1.0
    assert result.skill_fit == 0.0


def test_total_garbage_falls_back() -> None:
    result = parse_classification(GARBAGE)
    assert result == Classification(
        issue_type="other",
        difficulty="medium",
        solvability=0.0,
        skill_fit=0.0,
        rationale="classification failed",
    )


def test_empty_and_none_fall_back() -> None:
    assert parse_classification("").rationale == "classification failed"
    assert parse_classification(None).rationale == "classification failed"


def test_missing_keys_default_safely() -> None:
    result = parse_classification('{"type": "docs"}')
    assert result.issue_type == "docs"
    assert result.difficulty == "medium"  # missing -> default
    assert result.solvability == 0.0  # missing -> 0.0
    assert result.rationale == "no rationale provided"


def test_build_prompt_truncates_body_and_handles_empties() -> None:
    prompt = build_prompt("A title", "x" * 5000, [])
    assert "A title" in prompt
    assert "(none)" in prompt  # no labels
    # body truncated to 1500 chars: a 1500-run survives, a 1501-run does not.
    assert "x" * 1500 in prompt
    assert "x" * 1501 not in prompt

    no_body = build_prompt("T", None, ["bug"])
    assert "(no body)" in no_body
    assert "bug" in no_body


class _FakeChat:
    """A chat client double: returns canned content or raises a canned error."""

    def __init__(self, *, content: str = "", error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls = 0

    def invoke(self, prompt: str) -> "_FakeResponse":
        self.calls += 1
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


def test_invoke_returns_content() -> None:
    chat = _FakeChat(content=CLEAN)
    assert _invoke(chat, "prompt") == CLEAN
    assert chat.calls == 1


def test_invoke_daily_quota_fails_fast() -> None:
    chat = _FakeChat(error=RuntimeError("429 RequestsPerDay quota exceeded"))
    sleeps: list[float] = []
    assert _invoke(chat, "prompt", sleep=sleeps.append) == None  # noqa: E711
    assert chat.calls == 1  # no retries on a per-day quota
    assert sleeps == []


def test_invoke_rate_limit_retries_then_gives_up() -> None:
    chat = _FakeChat(error=RuntimeError("429 RESOURCE_EXHAUSTED per-minute"))
    sleeps: list[float] = []
    assert _invoke(chat, "prompt", max_retries=3, sleep=sleeps.append) is None
    assert chat.calls == 4  # initial try + 3 retries
    assert sleeps == [2.0, 4.0, 8.0]  # exponential backoff


def test_invoke_other_error_returns_none_without_retry() -> None:
    chat = _FakeChat(error=ValueError("malformed request"))
    sleeps: list[float] = []
    assert _invoke(chat, "prompt", sleep=sleeps.append) is None
    assert chat.calls == 1
    assert sleeps == []

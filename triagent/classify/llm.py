"""Model-agnostic chat client, plus a defensive invoke wrapper.

The LLM is treated as an unreliable network dependency, not a trusted function:

* ``_chat()`` returns a chat client chosen by the ``LLM_BACKEND`` env switch.
  The default ``"gemini"`` builds a ``ChatGoogleGenerativeAI``; an ``"ollama"``
  branch is stubbed for running locally later. Both expose
  ``.invoke(prompt).content``, so nothing downstream changes when the backend
  swaps.
* ``invoke()`` does ONE attempt and classifies the outcome into an
  ``InvokeOutcome`` (ok / rate_limit / daily_quota / error). It deliberately
  owns no retry or sleep policy: the caller (the scoring pass) decides whether
  to pace, cool down and resume (a transient per-minute limit), or exit (a
  per-day quota that won't clear soon). Keeping the classification here means
  the caller never has to parse provider error strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

# The Gemini model we score with; also stamped into Score.model_version.
GEMINI_MODEL = "gemini-2.5-flash-lite"


def model_version() -> str:
    """The model string to stamp onto each Score row, per active backend."""
    backend = os.environ.get("LLM_BACKEND", "gemini").lower()
    if backend == "ollama":
        return os.environ.get("OLLAMA_MODEL", "llama3.1")
    return GEMINI_MODEL


def _chat() -> Any:
    """Build a chat client for the configured backend.

    Returns an object exposing ``.invoke(prompt).content``. Returns ``Any``
    because the two backend client types share only that duck-typed surface.
    """
    backend = os.environ.get("LLM_BACKEND", "gemini").lower()

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env to run the scoring pass."
            )
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            google_api_key=api_key,
        )

    if backend == "ollama":
        # --- ollama branch (stub): wire this up when running models locally. ---
        from langchain_ollama import ChatOllama  # type: ignore[import-not-found]

        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            temperature=0,
        )

    raise RuntimeError(f"Unknown LLM_BACKEND: {backend!r} (expected 'gemini' or 'ollama')")


# Why an attempt did not yield text. The caller maps these to policy:
# ok -> use it; rate_limit -> cool down and resume; daily_quota -> exit; error -> stop.
InvokeReason = Literal["ok", "rate_limit", "daily_quota", "error"]


@dataclass(frozen=True)
class InvokeOutcome:
    """One attempt's result: ``text`` on success, plus a classified ``reason``
    so the caller can choose to pace, cool down and resume, or exit."""

    text: str | None
    reason: InvokeReason


def _is_daily_quota(message: str) -> bool:
    """A per-day quota is exhausted — it won't clear in a short cooldown."""
    return "PerDay" in message or "RequestsPerDay" in message


def _is_rate_limited(message: str) -> bool:
    """A transient per-minute limit — clears after a short cooldown."""
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def invoke(chat: Any, prompt: str) -> InvokeOutcome:
    """Invoke the chat client once and classify the outcome.

    Single attempt by design: pacing, cooldown, and retry policy live in the
    caller (the scoring pass), which can tell a transient per-minute limit
    (cool down and resume) from a per-day quota (exit).
    """
    try:
        return InvokeOutcome(str(chat.invoke(prompt).content), "ok")
    except Exception as exc:  # noqa: BLE001 — LLM client raises many error types
        message = str(exc)
        if _is_daily_quota(message):
            return InvokeOutcome(None, "daily_quota")
        if _is_rate_limited(message):
            return InvokeOutcome(None, "rate_limit")
        return InvokeOutcome(None, "error")

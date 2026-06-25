"""Model-agnostic chat client, plus a defensive invoke wrapper.

The LLM is treated as an unreliable network dependency, not a trusted function:

* ``_chat()`` returns a chat client chosen by the ``LLM_BACKEND`` env switch.
  The default ``"gemini"`` builds a ``ChatGoogleGenerativeAI``; an ``"ollama"``
  branch is stubbed for running locally later. Both expose
  ``.invoke(prompt).content``, so nothing downstream changes when the backend
  swaps.
* ``_invoke()`` never lets a rate limit hammer the API: it fails fast on a
  per-day quota error, backs off and retries on a transient per-minute limit,
  and swallows everything else into ``None`` so callers can fall back.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

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


def _is_daily_quota(message: str) -> bool:
    """A per-day quota is exhausted — retrying tonight is pointless."""
    return "PerDay" in message or "RequestsPerDay" in message


def _is_rate_limited(message: str) -> bool:
    """A transient per-minute limit — safe to back off and retry."""
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _invoke(
    chat: Any,
    prompt: str,
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Invoke the chat client, returning its text content or ``None``.

    * Per-day quota error -> ``None`` immediately (fail fast; never retry).
    * Per-minute limit (429 / RESOURCE_EXHAUSTED) -> exponential backoff and
      retry, capped at ``max_retries`` tries, then ``None``.
    * Any other error -> ``None`` (the caller falls back).
    """
    for attempt in range(max_retries + 1):
        try:
            return str(chat.invoke(prompt).content)
        except Exception as exc:  # noqa: BLE001 — LLM client raises many error types
            message = str(exc)
            if _is_daily_quota(message):
                return None
            if _is_rate_limited(message) and attempt < max_retries:
                sleep(base_delay * (2.0**attempt))
                continue
            return None
    return None

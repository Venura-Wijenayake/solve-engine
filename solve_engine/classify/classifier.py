"""The classification prompt and a parser that survives chatty LLM output.

The model is asked for a strict JSON object with four scored dimensions plus a
one-line rationale. Real models ignore "ONLY JSON" instructions often enough
that the parser must be defensive: strip code fences, pull out the first
balanced ``{...}`` object, ``json.loads`` it, then **coerce** every field to the
exact vocabulary the ``score`` table's CHECK constraints allow. Anything the
parser can't make sense of degrades to a FALLBACK row rather than crashing the
run or violating a constraint.
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel

from solve_engine.models import Difficulty, IssueType

# Bumping this invalidates prior scores: the unscored anti-join keys on it, so a
# new prompt version re-opens every issue for re-scoring.
PROMPT_VERSION = "v1"

BODY_TRUNCATE = 1500

_ALLOWED_TYPES: frozenset[str] = frozenset({"bug", "feature", "docs", "other"})
_ALLOWED_DIFFICULTY: frozenset[str] = frozenset({"easy", "medium", "hard"})

_PROMPT_TEMPLATE = """\
You are triaging a GitHub issue for an open-source contributor.

Return ONLY a JSON object — no prose, no markdown, no code fences — with exactly these keys:
"type": one of [bug, feature, docs, other]
"difficulty": one of [easy, medium, hard]
"solvability": number 0..1
"skill_fit": number 0..1
"rationale": one short sentence (max 20 words)

SOLVABILITY (0-1): how realistically a competent outside contributor could take this issue to a \
merged PR. High = clearly described, reproducible or well-specified, scoped to a concrete change, \
enough context to start, actionable now. Low = vague, huge/architectural, blocked or contentious, \
needs insider knowledge, or looks abandoned/stale.
SKILL_FIT (0-1): fit for a contributor strong in Python and the data/SQL ecosystem (pandas, dbt, \
duckdb, SQL) at a junior-to-mid level. High = Python/data domain at a difficulty they could \
realistically handle. Low = unfamiliar stack or beyond that level.

ISSUE TITLE: {title}
LABELS: {labels}
BODY:
{body}
"""


class Classification(BaseModel):
    """The four scored dimensions plus a rationale, already coerced to the
    exact vocabulary the ``score`` table accepts."""

    issue_type: IssueType
    difficulty: Difficulty
    solvability: float
    skill_fit: float
    rationale: str


def build_prompt(title: str, body: str | None, labels: list[str]) -> str:
    """Render the classification prompt for one issue (body truncated)."""
    truncated = (body or "").strip()[:BODY_TRUNCATE]
    label_text = ", ".join(labels) if labels else "(none)"
    return _PROMPT_TEMPLATE.format(title=title, labels=label_text, body=truncated or "(no body)")


def fallback() -> Classification:
    """The result used when the model reply can't be parsed at all."""
    return Classification(
        issue_type="other",
        difficulty="medium",
        solvability=0.0,
        skill_fit=0.0,
        rationale="classification failed",
    )


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, ignoring braces in strings.

    Robust to leading/trailing code fences and surrounding prose, because it
    just scans from the first ``{`` to its matching ``}``.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_type(value: object) -> IssueType:
    """Clamp the model's ``type`` to the allowed vocab; default ``other``.

    Catches the reference repo's stale values (e.g. "question") too."""
    if isinstance(value, str) and value.strip().lower() in _ALLOWED_TYPES:
        return cast(IssueType, value.strip().lower())
    return "other"


def _coerce_difficulty(value: object) -> Difficulty:
    """Clamp the model's ``difficulty`` to the allowed vocab; default ``medium``.

    Catches stale values like "beginner"/"intermediate" too."""
    if isinstance(value, str) and value.strip().lower() in _ALLOWED_DIFFICULTY:
        return cast(Difficulty, value.strip().lower())
    return "medium"


def _coerce_score(value: object) -> float:
    """Clamp a 0..1 score; default 0.0 on missing/non-numeric input."""
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def parse_classification(raw: str | None) -> Classification:
    """Parse and coerce a raw model reply into a Classification.

    Returns the FALLBACK on empty input, no JSON object, invalid JSON, or a
    non-object payload. Otherwise every field is coerced into range/vocab.
    """
    if not raw:
        return fallback()
    blob = _extract_json_object(raw)
    if blob is None:
        return fallback()
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return fallback()
    if not isinstance(data, dict):
        return fallback()

    rationale = str(data.get("rationale", "")).strip() or "no rationale provided"
    return Classification(
        issue_type=_coerce_type(data.get("type")),
        difficulty=_coerce_difficulty(data.get("difficulty")),
        solvability=_coerce_score(data.get("solvability")),
        skill_fit=_coerce_score(data.get("skill_fit")),
        rationale=rationale,
    )

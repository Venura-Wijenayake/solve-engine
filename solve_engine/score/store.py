"""Persistence for the scoring pass: select the unscored set, write Score rows.

Run bookkeeping (``start_run`` / ``finish_run``) is reused from the ingest
store — the ``run`` table is shared across pipeline stages.
"""

from __future__ import annotations

from typing import Any

import psycopg

from solve_engine.models import Score

# Issues with no score row at the given prompt_version. Mirrors the anti-join
# the score_issue_key_idx supports; ORDER BY keeps re-runs deterministic.
_UNSCORED_SQL = """
SELECT i.key, i.title, i.body, i.labels
FROM issue i
WHERE NOT EXISTS (
    SELECT 1 FROM score s
    WHERE s.issue_key = i.key AND s.prompt_version = %(prompt_version)s
)
ORDER BY i.last_seen DESC
LIMIT %(limit)s
"""

_INSERT_SQL = """
INSERT INTO score (
    issue_key, solvability, skill_fit, difficulty, issue_type,
    model_version, prompt_version, rationale, scored_at
) VALUES (
    %(issue_key)s, %(solvability)s, %(skill_fit)s, %(difficulty)s, %(issue_type)s,
    %(model_version)s, %(prompt_version)s, %(rationale)s, %(scored_at)s
)
"""


def select_unscored(
    conn: psycopg.Connection[tuple[Any, ...]],
    *,
    limit: int,
    prompt_version: str,
) -> list[tuple[str, str, str | None, list[str]]]:
    """Return up to ``limit`` issues with no score at ``prompt_version``.

    Each tuple is ``(key, title, body, labels)``.
    """
    rows = conn.execute(
        _UNSCORED_SQL, {"prompt_version": prompt_version, "limit": limit}
    ).fetchall()
    return [(row[0], row[1], row[2], row[3]) for row in rows]


def insert_score(conn: psycopg.Connection[tuple[Any, ...]], score: Score) -> None:
    """Write one Score row. Called per issue so a mid-run failure loses nothing."""
    conn.execute(
        _INSERT_SQL,
        {
            "issue_key": score.issue_key,
            "solvability": score.solvability,
            "skill_fit": score.skill_fit,
            "difficulty": score.difficulty,
            "issue_type": score.issue_type,
            "model_version": score.model_version,
            "prompt_version": score.prompt_version,
            "rationale": score.rationale,
            "scored_at": score.scored_at,
        },
    )

"""Persist issues and record ingest runs via psycopg.

Upsert is keyed on the natural ``key`` (``repo#number``) using
``INSERT ... ON CONFLICT (key) DO UPDATE``, so re-running never duplicates:
existing rows have their mutable fields refreshed and ``last_seen`` bumped,
while ``first_seen`` (and original provenance) are preserved from the insert.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from solve_engine.models import Issue

_UPSERT_SQL = """
INSERT INTO issue (
    key, repo, number, title, body, html_url, state, labels, language,
    created_at, updated_at, source, first_seen, last_seen, raw
) VALUES (
    %(key)s, %(repo)s, %(number)s, %(title)s, %(body)s, %(html_url)s, %(state)s,
    %(labels)s, %(language)s, %(created_at)s, %(updated_at)s, %(source)s,
    %(first_seen)s, %(last_seen)s, %(raw)s
)
ON CONFLICT (key) DO UPDATE SET
    title      = EXCLUDED.title,
    body       = EXCLUDED.body,
    state      = EXCLUDED.state,
    labels     = EXCLUDED.labels,
    language   = EXCLUDED.language,
    updated_at = EXCLUDED.updated_at,
    raw        = EXCLUDED.raw,
    last_seen  = EXCLUDED.last_seen
RETURNING (xmax = 0) AS inserted
"""


def _params(issue: Issue) -> dict[str, Any]:
    return {
        "key": issue.key,
        "repo": issue.repo,
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "html_url": issue.html_url,
        "state": issue.state,
        "labels": issue.labels,
        "language": issue.language,
        "created_at": issue.created_at,
        "updated_at": issue.updated_at,
        "source": issue.source,
        "first_seen": issue.first_seen,
        "last_seen": issue.last_seen,
        "raw": Jsonb(issue.raw),
    }


def upsert_issues(
    conn: psycopg.Connection[tuple[Any, ...]], issues: list[Issue]
) -> tuple[int, int]:
    """Insert new issues and update changed ones. Returns (new, updated)."""
    new = 0
    updated = 0
    with conn.transaction():
        for issue in issues:
            row = conn.execute(_UPSERT_SQL, _params(issue)).fetchone()
            inserted = bool(row[0]) if row is not None else False
            if inserted:
                new += 1
            else:
                updated += 1
    return new, updated


def start_run(conn: psycopg.Connection[tuple[Any, ...]], kind: str) -> int:
    """Record the start of a run; returns its id."""
    row = conn.execute(
        "INSERT INTO run (kind, started_at, status) "
        "VALUES (%s, now(), 'running') RETURNING id",
        (kind,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def finish_run(
    conn: psycopg.Connection[tuple[Any, ...]],
    run_id: int,
    *,
    status: str,
    seen: int,
    new: int,
    updated: int,
) -> None:
    """Mark a run finished with its final counters."""
    conn.execute(
        "UPDATE run SET finished_at = now(), status = %s, seen = %s, new = %s, "
        "updated = %s WHERE id = %s",
        (status, seen, new, updated, run_id),
    )


def issue_count(conn: psycopg.Connection[tuple[Any, ...]]) -> int:
    row = conn.execute("SELECT count(*) FROM issue").fetchone()
    assert row is not None
    return int(row[0])

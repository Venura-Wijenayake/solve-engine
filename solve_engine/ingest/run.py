"""End-to-end ingest: fetch (live) -> normalize -> upsert, recorded in `run`.

    python -m solve_engine.ingest.run

Pulls a small live batch, normalizes it onto the Issue model, upserts into
Postgres, writes one row to the `run` table, and prints a summary.
"""

from __future__ import annotations

import io
import sys

from solve_engine.config import get_settings
from solve_engine.db.connection import get_connection
from solve_engine.ingest.github_client import GitHubClient
from solve_engine.ingest.normalize import normalize_issues
from solve_engine.ingest.store import finish_run, issue_count, start_run, upsert_issues

# Keep the live pull modest.
SEARCH_MAX_RESULTS = 60
SEARCH_MAX_PAGES = 1
WATCH_REPOS = 3
WATCH_MAX_PAGES = 1


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    language = settings.language_focus[0] if settings.language_focus else None

    with GitHubClient.from_settings(settings) as client:
        raws = client.fetch_issues(
            settings,
            search_max_results=SEARCH_MAX_RESULTS,
            search_max_pages=SEARCH_MAX_PAGES,
            watch_repos=settings.watchlist[:WATCH_REPOS],
            watch_max_pages=WATCH_MAX_PAGES,
        )

    issues = normalize_issues(raws, language=language)

    with get_connection() as conn:
        conn.autocommit = True
        run_id = start_run(conn, "ingest")
        try:
            new, updated = upsert_issues(conn, issues)
        except Exception:
            finish_run(conn, run_id, status="error", seen=len(issues), new=0, updated=0)
            raise
        finish_run(
            conn, run_id, status="success", seen=len(issues), new=new, updated=updated
        )
        total = issue_count(conn)

    print("=== solve-engine ingest run ===")
    print(f"pulled (raw)   : {len(raws)}")
    print(f"normalized     : {len(issues)}")
    print(f"new            : {new}")
    print(f"updated        : {updated}")
    print(f"total in DB    : {total}")
    print(f"run id         : {run_id}")


if __name__ == "__main__":
    main()

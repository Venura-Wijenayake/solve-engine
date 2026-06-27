"""Migration runner: applies pending numbered .sql files in order.

Each ``NNNN_*.sql`` file is applied inside its own transaction and recorded in a
``schema_migrations`` table. Already-applied files are skipped, so the runner is
idempotent and re-runnable:

    python -m triagent.db.migrate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

from triagent.db.connection import get_connection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _ensure_migrations_table(conn: psycopg.Connection[tuple[Any, ...]]) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version text PRIMARY KEY, "
        "applied_at timestamptz NOT NULL DEFAULT now())"
    )


def _applied_versions(conn: psycopg.Connection[tuple[Any, ...]]) -> set[str]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {str(row[0]) for row in rows}


def main() -> None:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    applied_now = 0

    with get_connection() as conn:
        # Autocommit so each DDL file is its own explicit transaction block.
        conn.autocommit = True
        _ensure_migrations_table(conn)
        already = _applied_versions(conn)

        for path in files:
            version = path.stem
            if version in already:
                print(f"skip   {version} (already applied)")
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(sql)  # no params -> multi-statement files are allowed
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
            applied_now += 1
            print(f"apply  {version}")

    print(f"done: {applied_now} migration(s) applied, {len(files)} total on disk")


if __name__ == "__main__":
    main()

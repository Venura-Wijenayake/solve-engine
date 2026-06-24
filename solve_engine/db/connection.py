"""Connection helper: yields a psycopg connection using the configured URL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg

from solve_engine.config import get_settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[tuple[Any, ...]]]:
    """Open a psycopg connection to ``DATABASE_URL`` and close it on exit."""
    settings = get_settings()
    conn = psycopg.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()

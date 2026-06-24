"""Minimal Streamlit board: browse the stored issue queue.

    streamlit run solve_engine/board/app.py

The DB read is factored into ``load_issues()`` so it can be checked without
launching the server.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from solve_engine.db.connection import get_connection

_COLUMNS = ["repo", "number", "title", "html_url", "state", "labels", "source", "last_seen"]


def load_issues() -> list[dict[str, Any]]:
    """Read all stored issues, most-recently-seen first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT repo, number, title, html_url, state, labels, source, last_seen "
            "FROM issue ORDER BY last_seen DESC"
        ).fetchall()
    return [dict(zip(_COLUMNS, row, strict=True)) for row in rows]


def _render() -> None:
    st.set_page_config(page_title="Solve Engine — Board", layout="wide")
    st.title("Solve Engine — Ticket Queue")

    issues = load_issues()
    if not issues:
        st.info("No issues stored yet. Run `python -m solve_engine.ingest.run` first.")
        return

    repos = sorted({row["repo"] for row in issues})
    sources = sorted({row["source"] for row in issues})

    col1, col2, col3 = st.columns(3)
    repo_filter = col1.selectbox("Repo", ["(all)", *repos])
    source_filter = col2.selectbox("Source", ["(all)", *sources])
    query = col3.text_input("Search title")

    rows = issues
    if repo_filter != "(all)":
        rows = [r for r in rows if r["repo"] == repo_filter]
    if source_filter != "(all)":
        rows = [r for r in rows if r["source"] == source_filter]
    if query:
        needle = query.lower()
        rows = [r for r in rows if needle in r["title"].lower()]

    st.caption(f"{len(rows)} of {len(issues)} issues")
    table = [
        {
            "repo": r["repo"],
            "number": r["number"],
            "title": r["title"],
            "link": r["html_url"],
            "state": r["state"],
            "labels": ", ".join(r["labels"]),
            "source": r["source"],
            "last_seen": r["last_seen"],
        }
        for r in rows
    ]
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={"link": st.column_config.LinkColumn("link", display_text="open")},
    )


if __name__ == "__main__":
    _render()

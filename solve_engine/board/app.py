"""Minimal Streamlit board: browse the scored issue queue.

    streamlit run solve_engine/board/app.py

Each issue is LEFT JOINed to its latest score, so the queue sorts "most
solvable first" instead of a flat dump. Unscored issues still appear (with
blank score cells). The DB read is factored into ``load_issues()`` so it can be
checked without launching the server.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from solve_engine.db.connection import get_connection

_COLUMNS = [
    "repo",
    "number",
    "title",
    "html_url",
    "state",
    "labels",
    "source",
    "last_seen",
    "solvability",
    "skill_fit",
    "difficulty",
    "issue_type",
]

# Each issue joined to its single latest score (by scored_at). Issues with no
# score yield NULL score columns. Highest solvability first; unscored last.
_LOAD_SQL = """
SELECT i.repo, i.number, i.title, i.html_url, i.state, i.labels, i.source, i.last_seen,
       s.solvability, s.skill_fit, s.difficulty, s.issue_type
FROM issue i
LEFT JOIN LATERAL (
    SELECT solvability, skill_fit, difficulty, issue_type
    FROM score
    WHERE issue_key = i.key
    ORDER BY scored_at DESC
    LIMIT 1
) s ON true
ORDER BY s.solvability DESC NULLS LAST, i.last_seen DESC
"""


def load_issues() -> list[dict[str, Any]]:
    """Read all stored issues joined to their latest score, most solvable first."""
    with get_connection() as conn:
        rows = conn.execute(_LOAD_SQL).fetchall()
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
    difficulties = sorted({row["difficulty"] for row in issues if row["difficulty"]})

    col1, col2, col3 = st.columns(3)
    repo_filter = col1.selectbox("Repo", ["(all)", *repos])
    source_filter = col2.selectbox("Source", ["(all)", *sources])
    query = col3.text_input("Search title")

    col4, col5 = st.columns(2)
    difficulty_filter = col4.selectbox("Difficulty", ["(all)", *difficulties])
    min_solvability = col5.slider("Minimum solvability", 0.0, 1.0, 0.0, 0.05)

    rows = issues
    if repo_filter != "(all)":
        rows = [r for r in rows if r["repo"] == repo_filter]
    if source_filter != "(all)":
        rows = [r for r in rows if r["source"] == source_filter]
    if query:
        needle = query.lower()
        rows = [r for r in rows if needle in r["title"].lower()]
    if difficulty_filter != "(all)":
        rows = [r for r in rows if r["difficulty"] == difficulty_filter]
    if min_solvability > 0.0:
        rows = [r for r in rows if (r["solvability"] or 0.0) >= min_solvability]

    st.caption(f"{len(rows)} of {len(issues)} issues")
    table = [
        {
            "repo": r["repo"],
            "number": r["number"],
            "title": r["title"],
            "link": r["html_url"],
            "type": r["issue_type"],
            "difficulty": r["difficulty"],
            "solvability": r["solvability"],
            "skill_fit": r["skill_fit"],
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

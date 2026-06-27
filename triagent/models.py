"""Domain models — the typed contract everything downstream reads and writes.

Four entities mirror the Postgres tables in ``triagent/db/migrations``:

* ``Issue``    — a GitHub issue we've ingested (identity = ``repo#number``).
* ``Score``    — one scoring of an issue (solvability / skill-fit / difficulty).
* ``Run``      — one ingest/score/agent execution, with counters and observability.
* ``SolveLog`` — tracks a real PR opened to solve an issue, through to merge.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field

IssueState = Literal["open", "closed"]
IssueSource = Literal["search", "watchlist"]
Difficulty = Literal["easy", "medium", "hard"]
IssueType = Literal["bug", "feature", "docs", "other"]
RunKind = Literal["ingest", "score", "agent"]
RunStatus = Literal["running", "success", "error"]
SolveStatus = Literal["open", "review", "merged", "closed"]


class Issue(BaseModel):
    """A GitHub issue. Pull requests are filtered out at ingest, never stored,
    so there is no ``is_pull_request`` field here by design."""

    repo: str  # "owner/name", e.g. "dbt-labs/dbt-core"
    number: int  # issue number within the repo
    title: str
    body: str | None = None
    html_url: str
    state: IssueState
    labels: list[str] = []
    language: str | None = None
    created_at: datetime  # when the issue was opened on GitHub
    updated_at: datetime  # last GitHub-side update
    source: IssueSource  # how we found it: targeted search vs. a watchlist repo
    first_seen: datetime  # first time our ingest saw it
    last_seen: datetime  # most recent time our ingest saw it
    raw: dict[str, Any]  # full GitHub payload, retained for re-derivation

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> str:
        """Stable natural key used as the primary key and for foreign keys."""
        return f"{self.repo}#{self.number}"


class Score(BaseModel):
    """A single scoring of an issue. Many scores may exist per issue over time
    (different model/prompt versions); the latest is the one that counts."""

    # Pydantic protects the ``model_`` prefix; we use it deliberately here.
    model_config = ConfigDict(protected_namespaces=())

    issue_key: str  # references Issue.key ("repo#number")
    solvability: float  # 0.0-1.0: how tractable the issue looks
    skill_fit: float  # 0.0-1.0: match to the user's skills
    difficulty: Difficulty
    issue_type: IssueType
    model_version: str  # model that produced the score, e.g. "claude-opus-4-8"
    prompt_version: str  # scoring-prompt version, for reproducibility
    rationale: str  # human-readable justification
    scored_at: datetime


class Run(BaseModel):
    """One execution of a pipeline stage (ingest / score / agent), with counters
    and room for cost + latency so runs are observable later."""

    id: int | None = None  # assigned by the database
    kind: RunKind
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = "running"
    seen: int = 0  # issues encountered
    new: int = 0  # issues inserted
    updated: int = 0  # issues updated
    cost_usd: float | None = None
    latency_ms: int | None = None


class SolveLog(BaseModel):
    """Tracks a real PR opened to solve an issue, from open through merge."""

    id: int | None = None  # assigned by the database
    issue_key: str  # references Issue.key
    pr_url: str
    status: SolveStatus = "open"
    opened_at: datetime
    merged_at: datetime | None = None
    notes: str | None = None

from datetime import datetime, timezone

from triagent.models import Issue, Run, Score, SolveLog

_NOW = datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc)


def test_issue_builds_and_derives_key() -> None:
    issue = Issue(
        repo="dbt-labs/dbt-core",
        number=123,
        title="Fix off-by-one in parser",
        body="Steps to reproduce ...",
        html_url="https://github.com/dbt-labs/dbt-core/issues/123",
        state="open",
        labels=["bug", "good first issue"],
        language="python",
        created_at=_NOW,
        updated_at=_NOW,
        source="watchlist",
        first_seen=_NOW,
        last_seen=_NOW,
        raw={"id": 1, "number": 123},
    )
    assert issue.key == "dbt-labs/dbt-core#123"
    assert issue.state == "open"


def test_score_builds() -> None:
    score = Score(
        issue_key="dbt-labs/dbt-core#123",
        solvability=0.82,
        skill_fit=0.7,
        difficulty="medium",
        issue_type="bug",
        model_version="claude-opus-4-8",
        prompt_version="score-v1",
        rationale="Clear repro, isolated module, tests exist.",
        scored_at=_NOW,
    )
    assert score.difficulty == "medium"
    assert score.model_version == "claude-opus-4-8"


def test_run_builds_with_defaults() -> None:
    run = Run(kind="ingest", started_at=_NOW)
    assert run.status == "running"
    assert run.seen == 0 and run.new == 0 and run.updated == 0
    assert run.id is None


def test_solve_log_builds() -> None:
    log = SolveLog(
        issue_key="dbt-labs/dbt-core#123",
        pr_url="https://github.com/dbt-labs/dbt-core/pull/456",
        opened_at=_NOW,
    )
    assert log.status == "open"
    assert log.merged_at is None

"""Offline tests for the paced auto-continue scoring loop.

No network, no DB, no real sleeping: ``run_scoring_pass`` is driven with a
fake issue source, a fake invoker scripted with InvokeOutcomes, and a fake
sleep that records every wait. These assert the pacing and cooldown policy
without exercising the LLM, the parser, or Postgres.
"""

from __future__ import annotations

from triagent.classify.classifier import Classification
from triagent.classify.llm import InvokeOutcome
from triagent.score.run import IssueRow, PassResult, run_scoring_pass

# A minimal valid classifier reply; parse_classification turns it into a Classification.
_OK_JSON = (
    '{"type": "bug", "difficulty": "easy", "solvability": 0.5, '
    '"skill_fit": 0.5, "rationale": "ok"}'
)


def _issues(n: int) -> list[IssueRow]:
    return [(f"repo#{i}", f"title {i}", "body", ["bug"]) for i in range(n)]


class _FakeSource:
    """Serves unscored issues in pages, dropping served ones (mimics checkpointing)."""

    def __init__(self, issues: list[IssueRow]) -> None:
        self._pending = list(issues)

    def fetch(self, n: int) -> list[IssueRow]:
        page = self._pending[:n]
        self._pending = self._pending[n:]
        return page


class _Recorder:
    """Records persisted scores."""

    def __init__(self) -> None:
        self.persisted: list[tuple[str, Classification]] = []

    def persist(self, key: str, c: Classification) -> None:
        self.persisted.append((key, c))


def _ok() -> InvokeOutcome:
    return InvokeOutcome(_OK_JSON, "ok")


class _ScriptedInvoker:
    """Returns a scripted sequence of outcomes; the last entry repeats."""

    def __init__(self, outcomes: list[InvokeOutcome]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def __call__(self, prompt: str) -> InvokeOutcome:
        idx = min(self.calls, len(self._outcomes) - 1)
        self.calls += 1
        return self._outcomes[idx]


def _run(
    *,
    source: _FakeSource,
    invoker: _ScriptedInvoker,
    recorder: _Recorder,
    sleeps: list[float],
    target: int | None = None,
    batch_size: int = 25,
    sleep_s: float = 4.0,
    cooldown_s: float = 65.0,
    continue_on_limit: bool = True,
) -> PassResult:
    return run_scoring_pass(
        fetch_batch=source.fetch,
        persist=recorder.persist,
        invoke_fn=invoker,
        target=target,
        batch_size=batch_size,
        sleep_s=sleep_s,
        cooldown_s=cooldown_s,
        continue_on_limit=continue_on_limit,
        sleep=sleeps.append,
        log=lambda _msg: None,
    )


def test_paces_between_successful_calls() -> None:
    source = _FakeSource(_issues(3))
    invoker = _ScriptedInvoker([_ok()])
    recorder = _Recorder()
    sleeps: list[float] = []

    result = _run(source=source, invoker=invoker, recorder=recorder, sleeps=sleeps, sleep_s=4.0)

    assert result.new == 3
    assert result.stop_reason is None
    assert len(recorder.persisted) == 3
    # A pacing sleep of sleep_s after each successful score (no cooldown waits).
    assert sleeps == [4.0, 4.0, 4.0]


def test_per_minute_limit_cools_down_and_continues() -> None:
    source = _FakeSource(_issues(1))
    # First attempt hits the per-minute limit; the retry of the SAME issue succeeds.
    invoker = _ScriptedInvoker([InvokeOutcome(None, "rate_limit"), _ok()])
    recorder = _Recorder()
    sleeps: list[float] = []

    result = _run(
        source=source,
        invoker=invoker,
        recorder=recorder,
        sleeps=sleeps,
        sleep_s=4.0,
        cooldown_s=65.0,
        continue_on_limit=True,
    )

    assert result.new == 1  # the issue was scored after cooling down
    assert result.stop_reason is None
    assert len(recorder.persisted) == 1
    assert invoker.calls == 2  # rate-limited, then retried
    assert 65.0 in sleeps  # a cooldown wait happened
    assert sleeps.index(65.0) < sleeps.index(4.0)  # cooldown before the pacing sleep


def test_per_minute_limit_exits_when_continue_disabled() -> None:
    source = _FakeSource(_issues(3))
    invoker = _ScriptedInvoker([InvokeOutcome(None, "rate_limit")])
    recorder = _Recorder()
    sleeps: list[float] = []

    result = _run(
        source=source,
        invoker=invoker,
        recorder=recorder,
        sleeps=sleeps,
        continue_on_limit=False,
    )

    assert result.stop_reason == "rate_limit"
    assert result.new == 0
    assert sleeps == []  # no cooldown when continue-on-limit is off


def test_per_day_quota_exits_without_cooldown() -> None:
    source = _FakeSource(_issues(3))
    # Score one, then hit the per-day quota on the second.
    invoker = _ScriptedInvoker([_ok(), InvokeOutcome(None, "daily_quota")])
    recorder = _Recorder()
    sleeps: list[float] = []

    result = _run(source=source, invoker=invoker, recorder=recorder, sleeps=sleeps)

    assert result.stop_reason == "daily_quota"
    assert result.new == 1  # the first score is kept (checkpointed)
    assert len(recorder.persisted) == 1
    assert 65.0 not in sleeps  # a per-day quota never triggers a cooldown


def test_max_caps_total_across_pages() -> None:
    source = _FakeSource(_issues(50))
    invoker = _ScriptedInvoker([_ok()])
    recorder = _Recorder()
    sleeps: list[float] = []

    result = _run(
        source=source,
        invoker=invoker,
        recorder=recorder,
        sleeps=sleeps,
        target=2,
        batch_size=25,
    )

    assert result.new == 2  # stopped at --max even though 50 were available
    assert len(recorder.persisted) == 2


def test_cooldown_safety_cap_stops_runaway() -> None:
    source = _FakeSource(_issues(1))
    invoker = _ScriptedInvoker([InvokeOutcome(None, "rate_limit")])  # never clears
    recorder = _Recorder()
    sleeps: list[float] = []

    result = run_scoring_pass(
        fetch_batch=source.fetch,
        persist=recorder.persist,
        invoke_fn=invoker,
        target=None,
        batch_size=25,
        sleep_s=4.0,
        cooldown_s=65.0,
        continue_on_limit=True,
        sleep=sleeps.append,
        log=lambda _msg: None,
        max_cooldowns=3,
    )

    assert result.stop_reason == "rate_limit"
    assert result.new == 0
    assert sleeps == [65.0, 65.0, 65.0]  # bailed after the cap, not forever

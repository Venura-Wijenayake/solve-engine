"""Offline tests for the triage agent graph (mocked model, in-memory checkpointer).

No live LLM and no Postgres: the model is a scripted fake returning InvokeOutcomes,
and the graph is compiled with MemorySaver. Pure decision logic (should_verify,
merge_verify, parse_verify) is tested directly; the wired graph is exercised
end-to-end against the mock to prove the conditional verify branch.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from solve_engine.agent.graph import (
    VERIFY_THRESHOLD,
    AgentState,
    VerifyResult,
    build_graph,
    merge_verify,
    parse_verify,
    should_verify,
)
from solve_engine.classify.llm import InvokeOutcome


def _triage_json(*, solvability: float, issue_type: str = "bug", difficulty: str = "medium") -> str:
    return (
        f'{{"type": "{issue_type}", "difficulty": "{difficulty}", '
        f'"solvability": {solvability}, "skill_fit": 0.5, "rationale": "t"}}'
    )


def _verify_json(*, agree: bool, ctype: str = "bug", cdiff: str = "medium") -> str:
    return (
        f'{{"agree": {str(agree).lower()}, "corrected_type": "{ctype}", '
        f'"corrected_difficulty": "{cdiff}", "reason": "v"}}'
    )


class _ScriptedInvoker:
    """Returns scripted InvokeOutcomes in order; records how many times called."""

    def __init__(self, outcomes: list[InvokeOutcome]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def __call__(self, prompt: str) -> InvokeOutcome:
        idx = min(self.calls, len(self._outcomes) - 1)
        self.calls += 1
        return self._outcomes[idx]


def _run(invoker: _ScriptedInvoker, *, key: str = "o/r#1") -> AgentState:
    graph = build_graph(invoker, checkpointer=MemorySaver())
    state_in: AgentState = {"key": key, "title": "t", "body": "b", "labels": ["bug"]}
    result = graph.invoke(state_in, config={"configurable": {"thread_id": key}})
    return result  # type: ignore[no-any-return]


# --- pure decision logic ---------------------------------------------------


def test_should_verify_threshold() -> None:
    assert should_verify({"solvability": VERIFY_THRESHOLD - 0.01}) is True
    assert should_verify({"solvability": VERIFY_THRESHOLD}) is False  # boundary: not uncertain
    assert should_verify({"solvability": 0.9}) is False
    assert should_verify({}) is True  # missing -> 0.0 -> uncertain


def test_merge_verify_agree_keeps_triage() -> None:
    verdict = VerifyResult(agree=True, corrected_type="feature", corrected_difficulty="hard", reason="r")
    merged = merge_verify(triage_type="bug", triage_difficulty="easy", verdict=verdict)
    assert merged.needs_review is False
    assert merged.issue_type == "bug"  # corrected values ignored when agree
    assert merged.difficulty == "easy"


def test_merge_verify_disagree_applies_correction() -> None:
    verdict = VerifyResult(agree=False, corrected_type="feature", corrected_difficulty="hard", reason="r")
    merged = merge_verify(triage_type="bug", triage_difficulty="easy", verdict=verdict)
    assert merged.needs_review is True
    assert merged.issue_type == "feature"
    assert merged.difficulty == "hard"


def test_parse_verify_fails_open_on_garbage() -> None:
    verdict = parse_verify("not json at all", proposed_type="docs", proposed_difficulty="medium")
    assert verdict.agree is True  # never invent a disagreement
    assert verdict.corrected_type == "docs"
    assert verdict.corrected_difficulty == "medium"


def test_parse_verify_fails_open_on_none() -> None:
    verdict = parse_verify(None, proposed_type="bug", proposed_difficulty="hard")
    assert verdict.agree is True
    assert verdict.corrected_type == "bug"


def test_parse_verify_clamps_out_of_vocab() -> None:
    raw = '{"agree": false, "corrected_type": "question", "corrected_difficulty": "trivial", "reason": "x"}'
    verdict = parse_verify(raw, proposed_type="bug", proposed_difficulty="medium")
    assert verdict.agree is False
    assert verdict.corrected_type == "other"  # "question" -> other
    assert verdict.corrected_difficulty == "medium"  # "trivial" -> medium


# --- wired graph against the mock ------------------------------------------


def test_confident_triage_skips_verify() -> None:
    invoker = _ScriptedInvoker([InvokeOutcome(_triage_json(solvability=0.9), "ok")])
    final = _run(invoker)
    assert invoker.calls == 1  # only triage ran; verify was NOT invoked
    assert final["verified"] is False
    assert final["needs_review"] is False
    assert final["issue_type"] == "bug"


def test_low_confidence_triage_runs_verify_agree() -> None:
    invoker = _ScriptedInvoker(
        [
            InvokeOutcome(_triage_json(solvability=0.2), "ok"),
            InvokeOutcome(_verify_json(agree=True), "ok"),
        ]
    )
    final = _run(invoker)
    assert invoker.calls == 2  # triage + verify
    assert final["verified"] is True
    assert final["needs_review"] is False


def test_verify_disagreement_sets_needs_review_and_corrects() -> None:
    invoker = _ScriptedInvoker(
        [
            InvokeOutcome(_triage_json(solvability=0.2, issue_type="bug", difficulty="easy"), "ok"),
            InvokeOutcome(_verify_json(agree=False, ctype="feature", cdiff="hard"), "ok"),
        ]
    )
    final = _run(invoker)
    assert invoker.calls == 2
    assert final["verified"] is True
    assert final["needs_review"] is True
    assert final["issue_type"] == "feature"  # corrected
    assert final["difficulty"] == "hard"


def test_verify_fails_open_when_model_unavailable() -> None:
    invoker = _ScriptedInvoker(
        [
            InvokeOutcome(_triage_json(solvability=0.2, issue_type="docs", difficulty="medium"), "ok"),
            InvokeOutcome(None, "error"),  # verify call fails
        ]
    )
    final = _run(invoker)
    assert invoker.calls == 2
    assert final["verified"] is True
    assert final["needs_review"] is False  # failed open: no invented disagreement
    assert final["issue_type"] == "docs"  # triage labels kept


def test_triage_quota_surfaces_reason_and_falls_back() -> None:
    invoker = _ScriptedInvoker([InvokeOutcome(None, "daily_quota")])
    final = _run(invoker)
    assert final["triage_reason"] == "daily_quota"
    # fallback() -> solvability 0.0 -> would route to verify, which also fails open.
    assert final["issue_type"] == "other"

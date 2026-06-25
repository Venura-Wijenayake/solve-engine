"""The triage agent as a LangGraph state graph: Triage -> (conditional) Verify.

Why a graph and not the old single call: scoring used to be one Gemini call.
Here it's a checkpointed state machine. ``triage`` reuses the EXISTING classifier
(build_prompt -> invoke -> parse_classification) as its brain — no new scoring
logic. ``verify`` is a second, focused LLM call that only runs for *uncertain*
triage results (low solvability), re-checks type/difficulty, and on disagreement
flags the issue ``needs_review`` and applies the verifier's correction. Confident
triage skips straight to END — we verify the uncertain ones, not a blind 3x.

The decision logic (``should_verify``, ``merge_verify``) and the verify parser are
pure functions so they're unit-testable without a model, a graph, or Postgres.
The graph is compiled WITH a checkpointer; ``run.py`` gives each issue its own
``thread_id`` so every issue's run is persisted as its own resumable thread.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from solve_engine.classify.classifier import (
    _coerce_difficulty,
    _coerce_type,
    _extract_json_object,
    build_prompt,
    fallback,
    parse_classification,
)
from solve_engine.classify.llm import InvokeOutcome
from solve_engine.models import Difficulty, IssueType

# Run verify only when triage solvability is below this — the "uncertain" band.
VERIFY_THRESHOLD = 0.5

# Distinguishes agent scores from the v1 single-call scores in the score table.
AGENT_PROMPT_VERSION = "agent-v1"

# How the chat client is reached: a function prompt -> InvokeOutcome. Injecting
# this (rather than a hard dependency on _chat) is what lets tests use a mock.
InvokeFn = Callable[[str], InvokeOutcome]


class AgentState(TypedDict, total=False):
    """State threaded through the graph. ``total=False`` so each node contributes
    a partial update and earlier-unset keys are simply absent until set."""

    # Input (issue identity + content).
    key: str
    title: str
    body: str | None
    labels: list[str]
    # Triage output (the reused classifier's result).
    issue_type: str
    difficulty: str
    solvability: float
    skill_fit: float
    rationale: str
    triage_reason: str  # the invoke outcome reason: ok / daily_quota / rate_limit / error
    # Verify output (defaults set by triage; overridden iff verify runs).
    verified: bool  # did the verify node run?
    needs_review: bool  # verifier disagreed with triage
    verify_reason: str


class VerifyResult(BaseModel):
    """The verifier's parsed verdict, already clamped to the allowed vocab."""

    agree: bool
    corrected_type: IssueType
    corrected_difficulty: Difficulty
    reason: str


_VERIFY_PROMPT = """\
You are double-checking a triage decision on a GitHub issue.

Another model proposed: type={proposed_type}, difficulty={proposed_difficulty}.

Return ONLY a JSON object - no prose, no markdown, no code fences - with exactly these keys:
"agree": true or false
"corrected_type": one of [bug, feature, docs, other]
"corrected_difficulty": one of [easy, medium, hard]
"reason": one short sentence (max 20 words)

If the proposed type and difficulty look right, set agree=true and echo them back.
If either is wrong, set agree=false and give your corrected values.

ISSUE TITLE: {title}
LABELS: {labels}
BODY:
{body}
"""


def build_verify_prompt(
    title: str, body: str | None, labels: list[str], *, proposed_type: str, proposed_difficulty: str
) -> str:
    """Render the verify prompt, reusing the classifier's body truncation."""
    # Reuse build_prompt's truncation by extracting just the body line is overkill;
    # truncate inline to keep verify self-contained and cheap.
    truncated = (body or "").strip()[:1500] or "(no body)"
    label_text = ", ".join(labels) if labels else "(none)"
    return _VERIFY_PROMPT.format(
        title=title,
        labels=label_text,
        body=truncated,
        proposed_type=proposed_type,
        proposed_difficulty=proposed_difficulty,
    )


def parse_verify(
    raw: str | None, *, proposed_type: str, proposed_difficulty: str
) -> VerifyResult:
    """Parse the verifier reply defensively; fail OPEN on anything unusable.

    Fail-open means: if the reply is missing/garbage (e.g. the model was
    unavailable), return ``agree=True`` echoing the proposed labels — we never
    invent a disagreement out of a failed call.
    """
    open_default = VerifyResult(
        agree=True,
        corrected_type=_coerce_type(proposed_type),
        corrected_difficulty=_coerce_difficulty(proposed_difficulty),
        reason="verify unavailable; kept triage",
    )
    if not raw:
        return open_default
    blob = _extract_json_object(raw)
    if blob is None:
        return open_default
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return open_default
    if not isinstance(data, dict):
        return open_default

    return VerifyResult(
        agree=bool(data.get("agree", True)),
        corrected_type=_coerce_type(data.get("corrected_type", proposed_type)),
        corrected_difficulty=_coerce_difficulty(data.get("corrected_difficulty", proposed_difficulty)),
        reason=str(data.get("reason", "")).strip() or "no reason given",
    )


class VerifyOutcome(BaseModel):
    """The merged effect of a verify verdict on the triage labels."""

    needs_review: bool
    issue_type: str
    difficulty: str
    verify_reason: str


def merge_verify(
    *, triage_type: str, triage_difficulty: str, verdict: VerifyResult
) -> VerifyOutcome:
    """Apply a verify verdict: agree -> keep triage; disagree -> correct + flag."""
    if verdict.agree:
        return VerifyOutcome(
            needs_review=False,
            issue_type=triage_type,
            difficulty=triage_difficulty,
            verify_reason=verdict.reason,
        )
    return VerifyOutcome(
        needs_review=True,
        issue_type=verdict.corrected_type,
        difficulty=verdict.corrected_difficulty,
        verify_reason=verdict.reason,
    )


def should_verify(state: AgentState) -> bool:
    """Verify only uncertain triage: solvability below the threshold."""
    return state.get("solvability", 0.0) < VERIFY_THRESHOLD


def _make_triage_node(invoke_fn: InvokeFn) -> Callable[[AgentState], dict[str, Any]]:
    def triage(state: AgentState) -> dict[str, Any]:
        prompt = build_prompt(state["title"], state.get("body"), state.get("labels", []))
        outcome = invoke_fn(prompt)
        result = parse_classification(outcome.text) if outcome.reason == "ok" else fallback()
        return {
            "issue_type": result.issue_type,
            "difficulty": result.difficulty,
            "solvability": result.solvability,
            "skill_fit": result.skill_fit,
            "rationale": result.rationale,
            "triage_reason": outcome.reason,
            # Verify defaults; the verify node overrides these if it runs.
            "verified": False,
            "needs_review": False,
            "verify_reason": "",
        }

    return triage


def _make_verify_node(invoke_fn: InvokeFn) -> Callable[[AgentState], dict[str, Any]]:
    def verify(state: AgentState) -> dict[str, Any]:
        proposed_type = state["issue_type"]
        proposed_difficulty = state["difficulty"]
        prompt = build_verify_prompt(
            state["title"],
            state.get("body"),
            state.get("labels", []),
            proposed_type=proposed_type,
            proposed_difficulty=proposed_difficulty,
        )
        outcome = invoke_fn(prompt)
        # Non-ok -> raw is None -> parse_verify fails open (agree=True).
        verdict = parse_verify(
            outcome.text, proposed_type=proposed_type, proposed_difficulty=proposed_difficulty
        )
        merged = merge_verify(
            triage_type=proposed_type, triage_difficulty=proposed_difficulty, verdict=verdict
        )
        return {
            "verified": True,
            "needs_review": merged.needs_review,
            "issue_type": merged.issue_type,
            "difficulty": merged.difficulty,
            "verify_reason": merged.verify_reason,
        }

    return verify


def _route_after_triage(state: AgentState) -> Literal["verify", "skip"]:
    return "verify" if should_verify(state) else "skip"


def build_graph(invoke_fn: InvokeFn, *, checkpointer: Any) -> Any:
    """Wire START -> triage -> (conditional) verify -> END and compile with the
    given checkpointer (PostgresSaver in prod, MemorySaver in tests).

    ``builder`` is typed ``Any``: LangGraph's add_node overloads are intricate
    and the wiring is inherently dynamic, so we don't gain from fighting them.
    The compiled graph is exercised end-to-end in tests, which is the real check.
    """
    builder: Any = StateGraph(AgentState)
    builder.add_node("triage", _make_triage_node(invoke_fn))
    builder.add_node("verify", _make_verify_node(invoke_fn))
    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage", _route_after_triage, {"verify": "verify", "skip": END}
    )
    builder.add_edge("verify", END)
    return builder.compile(checkpointer=checkpointer)

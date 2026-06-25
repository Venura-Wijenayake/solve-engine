"""End-to-end agent pass: run the triage graph per issue, persist agent scores.

    python -m solve_engine.agent.run [--limit N] [--sleep S]

Selects issues with no score at prompt_version "agent-v1" (so agent scores are
distinguishable from the v1 single-call scores), runs the compiled Triage->Verify
graph for each (its own checkpointed thread keyed on the issue), and writes a
Score row from the final state. ``needs_review`` is encoded into the rationale
(no schema change). Paces with --sleep and stops gracefully on a per-day quota,
mirroring score/run.py. Each Score commits as a checkpoint.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, cast

from langgraph.checkpoint.postgres import PostgresSaver

from solve_engine.agent.graph import AGENT_PROMPT_VERSION, AgentState, InvokeFn, build_graph
from solve_engine.classify.classifier import _coerce_difficulty, _coerce_type
from solve_engine.classify.llm import _chat, invoke, model_version
from solve_engine.config import get_settings
from solve_engine.db.connection import get_connection
from solve_engine.ingest.store import finish_run, start_run
from solve_engine.models import Score
from solve_engine.score.store import count_unscored, insert_score, select_unscored

DEFAULT_LIMIT = 5
DEFAULT_SLEEP = 4.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the triage agent over unscored issues.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max issues this run")
    parser.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP, help="seconds between issues (pacing)"
    )
    return parser.parse_args(argv)


def _rationale_for(final: AgentState) -> str:
    """Encode the verify signal into the rationale (no extra columns)."""
    base = final.get("rationale", "")
    if not final.get("verified", False):
        return base  # confident triage; verify skipped
    if final.get("needs_review", False):
        return f"[needs_review] {base} | verify: {final.get('verify_reason', '')}"
    return f"[verified-ok] {base}"


def run_agent(
    invoke_fn: InvokeFn,
    *,
    limit: int,
    sleep_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Core agent pass: compile the checkpointed graph and score a batch.

    ``invoke_fn`` is injected so this exact persistence path (real PostgresSaver
    checkpointer + real score writes) can be driven by a mock model — the live
    Gemini path and a mocked proof share one code path.
    """
    settings = get_settings()
    model = model_version()

    seen = 0
    new = 0
    stop_reason: str | None = None
    samples: list[tuple[str, AgentState]] = []

    with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        checkpointer.setup()  # idempotent; ensures checkpointer tables exist
        graph = build_graph(invoke_fn, checkpointer=checkpointer)

        with get_connection() as conn:
            conn.autocommit = True  # each Score commits as a checkpoint
            backlog = count_unscored(conn, prompt_version=AGENT_PROMPT_VERSION)
            batch = select_unscored(conn, limit=limit, prompt_version=AGENT_PROMPT_VERSION)
            print(f"unscored-by-agent backlog: {backlog}; this run: up to {len(batch)}")

            run_id = start_run(conn, "agent")
            try:
                for key, title, body, labels in batch:
                    state_in: AgentState = {
                        "key": key,
                        "title": title,
                        "body": body,
                        "labels": labels,
                    }
                    final = _invoke_graph(graph, state_in, thread_id=key)

                    if final.get("triage_reason") != "ok":
                        stop_reason = final.get("triage_reason", "error")
                        break  # quota/network: stop before writing a garbage score

                    insert_score(
                        conn,
                        Score(
                            issue_key=key,
                            solvability=final["solvability"],
                            skill_fit=final["skill_fit"],
                            # Already in-vocab from triage/verify; coerce re-affirms the Literal type.
                            difficulty=_coerce_difficulty(final["difficulty"]),
                            issue_type=_coerce_type(final["issue_type"]),
                            model_version=model,
                            prompt_version=AGENT_PROMPT_VERSION,
                            rationale=_rationale_for(final),
                            scored_at=datetime.now(timezone.utc),
                        ),
                    )
                    seen += 1
                    new += 1
                    samples.append((title, final))
                    sleep(sleep_s)
            except Exception:
                finish_run(conn, run_id, status="error", seen=seen, new=new, updated=0)
                raise
            finish_run(conn, run_id, status="success", seen=seen, new=new, updated=0)
            remaining = count_unscored(conn, prompt_version=AGENT_PROMPT_VERSION)

    _print_summary(run_id, new, remaining, stop_reason, samples)


def main(argv: list[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args(argv)
    chat = _chat()
    run_agent(lambda prompt: invoke(chat, prompt), limit=args.limit, sleep_s=args.sleep)


def _invoke_graph(graph: Any, state_in: AgentState, *, thread_id: str) -> AgentState:
    """Invoke the compiled graph on its own checkpointed thread for this issue.

    The compiled graph returns the merged state dict; we treat it as AgentState.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return cast(AgentState, graph.invoke(state_in, config=config))


def _print_summary(
    run_id: int,
    new: int,
    remaining: int,
    stop_reason: str | None,
    samples: list[tuple[str, AgentState]],
) -> None:
    print("\n=== solve-engine agent run ===")
    print(f"scored (new)   : {new}")
    print(f"still unscored : {remaining}")
    print(f"run id         : {run_id}")
    if stop_reason is not None:
        print(f"stopped        : {stop_reason} (re-run to continue)")

    if samples:
        print("\n--- sample results ---")
        for title, final in samples[:8]:
            verified = "verify" if final.get("verified") else "skip"
            flag = "NEEDS_REVIEW" if final.get("needs_review") else "ok"
            head = title if len(title) <= 60 else title[:57] + "..."
            print(
                f"[{final.get('issue_type')}/{final.get('difficulty')}] "
                f"solv={final.get('solvability'):.2f} {verified}->{flag}  {head}"
            )


if __name__ == "__main__":
    main()

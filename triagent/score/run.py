"""End-to-end scoring pass: classify unscored issues, persist Scores, record a run.

    python -m triagent.score.run [--limit N] [--max N] [--sleep S] \
        [--cooldown S] [--no-continue-on-limit]

Selects issues with no score at prompt_version "v1", asks the LLM to classify
each (one call -> type / difficulty / solvability / skill_fit / rationale),
and writes a Score row per issue as it is produced (checkpointing, so a
rate-limit mid-run loses nothing). Re-runs skip already-scored issues.

Patient batch mode lets one command grind through the whole backlog under the
free-tier per-minute ceiling: ``--sleep`` paces calls to stay under the limit,
and on a per-MINUTE stop the pass cools down (``--cooldown``) and resumes the
SAME pass instead of exiting. A per-DAY quota still exits (it won't clear in a
short cooldown). The core loop is factored into ``run_scoring_pass`` so pacing
and cooldown behavior are unit-testable with injected fakes (no network/clock).
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from triagent.classify.classifier import (
    PROMPT_VERSION,
    Classification,
    build_prompt,
    parse_classification,
)
from triagent.classify.llm import InvokeOutcome, _chat, invoke, model_version
from triagent.config import get_settings
from triagent.db.connection import get_connection
from triagent.ingest.store import finish_run, start_run
from triagent.models import Score
from triagent.score.store import count_unscored, insert_score, select_unscored

DEFAULT_LIMIT = 25  # DB fetch page size (inner batch)
DEFAULT_SLEEP = 4.0  # seconds between successful calls (~15/min, under typical limits)
DEFAULT_COOLDOWN = 65.0  # seconds to wait out a per-minute limit before resuming
PROGRESS_EVERY = 10  # print a progress line every N scored issues
MAX_COOLDOWNS = 30  # safety cap on consecutive cooldowns with no progress

# What ended the pass; ``None`` means it completed (backlog drained or --max reached).
StopReason = str  # one of: "daily_quota" | "rate_limit" | "error"

# (key, title, body, labels) as returned by select_unscored.
IssueRow = tuple[str, str, str | None, list[str]]


@dataclass
class PassResult:
    seen: int = 0
    new: int = 0
    stop_reason: StopReason | None = None
    scored: list[tuple[str, Classification]] = field(default_factory=list)


def run_scoring_pass(
    *,
    fetch_batch: Callable[[int], list[IssueRow]],
    persist: Callable[[str, Classification], None],
    invoke_fn: Callable[[str], InvokeOutcome],
    target: int | None,
    batch_size: int,
    sleep_s: float,
    cooldown_s: float,
    continue_on_limit: bool,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    progress_every: int = PROGRESS_EVERY,
    max_cooldowns: int = MAX_COOLDOWNS,
) -> PassResult:
    """Score unscored issues page by page until the backlog drains, ``target``
    new scores are written, or a stop condition is hit.

    Policy:
    * After each successful score, ``sleep(sleep_s)`` to pace under the ceiling.
    * On a per-minute ``rate_limit`` with ``continue_on_limit``: ``sleep(cooldown_s)``
      and retry the SAME issue (bounded by ``max_cooldowns`` without progress).
    * On a per-day ``daily_quota`` (or ``error``, or rate_limit with continue off):
      stop and return the reason. Each score is persisted as produced, so a stop
      never loses scored work.
    """
    result = PassResult()
    cooldowns = 0

    while target is None or result.new < target:
        remaining = None if target is None else target - result.new
        fetch_n = batch_size if remaining is None else min(batch_size, remaining)
        batch = fetch_batch(fetch_n)
        if not batch:
            break  # backlog drained

        for key, title, body, labels in batch:
            prompt = build_prompt(title, body, labels)

            # Retry the SAME issue across per-minute cooldowns.
            while True:
                outcome = invoke_fn(prompt)
                if outcome.reason == "rate_limit" and continue_on_limit:
                    cooldowns += 1
                    if cooldowns > max_cooldowns:
                        result.stop_reason = "rate_limit"
                        return result
                    log(f"per-minute limit hit; cooling down {cooldown_s:.0f}s and resuming…")
                    sleep(cooldown_s)
                    continue
                break

            if outcome.reason != "ok":
                # daily_quota -> exit; error / rate_limit(no-continue) -> stop.
                result.stop_reason = outcome.reason
                return result

            classification = parse_classification(outcome.text)
            persist(key, classification)  # checkpoint: committed before we sleep
            result.seen += 1
            result.new += 1
            result.scored.append((title, classification))
            cooldowns = 0  # progress resets the safety counter

            if target is not None and result.new >= target:
                return result

            if result.new % progress_every == 0:
                shown = str(result.new) if target is None else f"{result.new}/{target}"
                log(f"scored {shown}, sleeping {sleep_s:.0f}s…")
            sleep(sleep_s)  # pace under the per-minute ceiling

    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patiently score unscored issues with the LLM.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"DB fetch page size / inner batch (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="overall cap on issues scored this invocation (default: all unscored)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"seconds between successful calls, to stay under the per-minute "
        f"ceiling (default {DEFAULT_SLEEP})",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN,
        help=f"seconds to wait out a per-minute limit before resuming "
        f"(default {DEFAULT_COOLDOWN})",
    )
    parser.add_argument(
        "--continue-on-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="on a per-minute limit, cool down and resume (default: on); "
        "use --no-continue-on-limit to exit instead",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args(argv)
    get_settings()  # loads .env so GEMINI_API_KEY is available to _chat()

    model = model_version()
    chat = _chat()

    counters = {"new": 0}  # mirrors result.new so a hard DB error still records counters

    with get_connection() as conn:
        conn.autocommit = True  # each Score commits as a checkpoint
        backlog = count_unscored(conn, prompt_version=PROMPT_VERSION)
        target = args.max if args.max is not None else None
        print(f"unscored backlog at {PROMPT_VERSION}: {backlog}")
        print(f"target this run     : {target if target is not None else 'all'} "
              f"(sleep {args.sleep:.0f}s, cooldown {args.cooldown:.0f}s, "
              f"continue-on-limit={args.continue_on_limit})")

        def fetch_batch(n: int) -> list[IssueRow]:
            return select_unscored(conn, limit=n, prompt_version=PROMPT_VERSION)

        def persist(key: str, c: Classification) -> None:
            insert_score(
                conn,
                Score(
                    issue_key=key,
                    solvability=c.solvability,
                    skill_fit=c.skill_fit,
                    difficulty=c.difficulty,
                    issue_type=c.issue_type,
                    model_version=model,
                    prompt_version=PROMPT_VERSION,
                    rationale=c.rationale,
                    scored_at=datetime.now(timezone.utc),
                ),
            )
            counters["new"] += 1

        run_id = start_run(conn, "score")
        try:
            result = run_scoring_pass(
                fetch_batch=fetch_batch,
                persist=persist,
                invoke_fn=lambda prompt: invoke(chat, prompt),
                target=target,
                batch_size=args.limit,
                sleep_s=args.sleep,
                cooldown_s=args.cooldown,
                continue_on_limit=args.continue_on_limit,
            )
        except Exception:
            finish_run(
                conn, run_id, status="error", seen=counters["new"], new=counters["new"], updated=0
            )
            raise
        finish_run(conn, run_id, status="success", seen=result.seen, new=result.new, updated=0)
        remaining = count_unscored(conn, prompt_version=PROMPT_VERSION)

    _print_summary(run_id, result, remaining)


def _print_summary(run_id: int, result: PassResult, remaining: int) -> None:
    print("\n=== triagent scoring run ===")
    print(f"scored (new)   : {result.new}")
    print(f"still unscored : {remaining}")
    print(f"run id         : {run_id}")
    print(f"outcome        : {_outcome_text(result.stop_reason)}")

    if result.scored:
        print("\n--- top by solvability ---")
        top = sorted(result.scored, key=lambda item: item[1].solvability, reverse=True)[:5]
        for title, c in top:
            head = title if len(title) <= 70 else title[:67] + "..."
            print(
                f"[{c.issue_type}/{c.difficulty}] "
                f"solv={c.solvability:.2f} fit={c.skill_fit:.2f}  {head}"
            )


def _outcome_text(stop_reason: StopReason | None) -> str:
    if stop_reason is None:
        return "completed (backlog drained or --max reached)"
    if stop_reason == "daily_quota":
        return "stopped: per-DAY quota hit — won't clear soon, re-run later"
    if stop_reason == "rate_limit":
        return "stopped: per-MINUTE limit (continue disabled or safety cap reached)"
    return "stopped: unexpected LLM error"


if __name__ == "__main__":
    main()

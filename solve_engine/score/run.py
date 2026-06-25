"""End-to-end scoring pass: classify unscored issues, persist Scores, record a run.

    python -m solve_engine.score.run [--limit N]

Selects issues with no score at prompt_version "v1", asks the LLM to classify
each (one call -> type / difficulty / solvability / skill_fit / rationale),
and writes a Score row per issue as it is produced (checkpointing, so a
rate-limit mid-run loses nothing). Re-runs skip already-scored issues, so
raising ``--limit`` over repeated runs scales coverage without duplicates.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timezone

from solve_engine.classify.classifier import (
    PROMPT_VERSION,
    Classification,
    build_prompt,
    parse_classification,
)
from solve_engine.classify.llm import _chat, _invoke, model_version
from solve_engine.config import get_settings
from solve_engine.db.connection import get_connection
from solve_engine.ingest.store import finish_run, start_run
from solve_engine.models import Score
from solve_engine.score.store import insert_score, select_unscored

DEFAULT_LIMIT = 25


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score unscored issues with the LLM.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"max issues to score this run (default {DEFAULT_LIMIT}); "
        "keeps the first run within the free-tier daily quota",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args(argv)
    get_settings()  # loads .env so GEMINI_API_KEY is available to _chat()

    model = model_version()
    chat = _chat()

    seen = 0
    new = 0
    quota_stopped = False
    scored: list[tuple[str, Classification]] = []

    with get_connection() as conn:
        conn.autocommit = True  # each Score commits as a checkpoint
        run_id = start_run(conn, "score")
        try:
            batch = select_unscored(conn, limit=args.limit, prompt_version=PROMPT_VERSION)
            for key, title, body, labels in batch:
                seen += 1
                raw = _invoke(chat, build_prompt(title, body, labels))
                if raw is None:
                    # Quota/network failure: stop rather than write garbage rows.
                    quota_stopped = True
                    seen -= 1
                    break
                result = parse_classification(raw)
                insert_score(
                    conn,
                    Score(
                        issue_key=key,
                        solvability=result.solvability,
                        skill_fit=result.skill_fit,
                        difficulty=result.difficulty,
                        issue_type=result.issue_type,
                        model_version=model,
                        prompt_version=PROMPT_VERSION,
                        rationale=result.rationale,
                        scored_at=datetime.now(timezone.utc),
                    ),
                )
                new += 1
                scored.append((title, result))
        except Exception:
            finish_run(conn, run_id, status="error", seen=seen, new=new, updated=0)
            raise
        finish_run(conn, run_id, status="success", seen=seen, new=new, updated=0)

    _print_summary(run_id, seen, new, quota_stopped, scored)


def _print_summary(
    run_id: int,
    seen: int,
    new: int,
    quota_stopped: bool,
    scored: list[tuple[str, Classification]],
) -> None:
    print("=== solve-engine scoring run ===")
    print(f"seen           : {seen}")
    print(f"scored (new)   : {new}")
    print(f"run id         : {run_id}")
    if quota_stopped:
        print("note           : stopped early — LLM quota/network limit hit")

    if scored:
        print("\n--- top by solvability ---")
        top = sorted(scored, key=lambda item: item[1].solvability, reverse=True)[:5]
        for title, c in top:
            head = title if len(title) <= 70 else title[:67] + "..."
            print(
                f"[{c.issue_type}/{c.difficulty}] "
                f"solv={c.solvability:.2f} fit={c.skill_fit:.2f}  {head}"
            )


if __name__ == "__main__":
    main()

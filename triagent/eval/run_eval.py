"""Evaluation harness: score the LLM's predictions against the human golden set.

    python -m triagent.eval.run_eval

Reads ``golden.jsonl`` (human ground truth), joins each filled row to that
issue's latest v1 score FROM THE DB (predictions never live in the golden file),
and reports agreement on three axes:

* TYPE       — precision / recall / F1 + confusion (human vs model issue_type).
* DIFFICULTY — same (human vs model difficulty).
* SOLVABILITY— the model's solvability float is thresholded to yes/no, then
  accuracy + a 2x2 confusion vs the human yes/no, plus the mean model solvability
  for human-yes vs human-no issues (it should be higher for yes if meaningful).

No live LLM call: this reads stored scores only. Safe to run before labelling —
if the golden set is still all TODO, it says so and exits cleanly.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from triagent.classify.classifier import PROMPT_VERSION
from triagent.db.connection import get_connection
from triagent.eval.golden import (
    SOLVABILITY_THRESHOLD,
    EvalPairs,
    GoldenRow,
    Prediction,
    pair_with_predictions,
    parse_golden_lines,
    solvability_label,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"


def load_predictions(prompt_version: str) -> dict[str, Prediction]:
    """Map issue_key -> latest prediction at ``prompt_version`` from the DB."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ON (issue_key) issue_key, issue_type, difficulty, solvability "
            "FROM score WHERE prompt_version = %s "
            "ORDER BY issue_key, scored_at DESC",
            (prompt_version,),
        ).fetchall()
    return {
        row[0]: Prediction(issue_type=row[1], difficulty=row[2], solvability=float(row[3]))
        for row in rows
    }


def _print_label_report(name: str, truth: list[str], pred: list[str]) -> None:
    labels = sorted(set(truth) | set(pred))
    print(f"\n=== {name} ===")
    print(classification_report(truth, pred, labels=labels, zero_division=0))
    print("confusion matrix (rows = human, cols = model):")
    print(f"labels: {labels}")
    print(confusion_matrix(truth, pred, labels=labels))


def _print_solvability_report(pairs: list[tuple[GoldenRow, Prediction]]) -> None:
    truth = [g.human_solvable for g, _ in pairs]
    pred = [solvability_label(p.solvability) for _, p in pairs]
    labels = ["yes", "no"]

    print("\n=== SOLVABILITY ===")
    print(f"model solvability thresholded at {SOLVABILITY_THRESHOLD:.2f} -> yes/no")
    print(f"accuracy: {accuracy_score(truth, pred):.3f}")
    print("confusion matrix (rows = human, cols = model):")
    print(f"labels: {labels}")
    print(confusion_matrix(truth, pred, labels=labels))

    yes_scores = [p.solvability for g, p in pairs if g.human_solvable == "yes"]
    no_scores = [p.solvability for g, p in pairs if g.human_solvable == "no"]
    print("\nmean model solvability by human label (should be higher for yes):")
    print(f"  human=yes: {_mean(yes_scores)}  (n={len(yes_scores)})")
    print(f"  human=no : {_mean(no_scores)}  (n={len(no_scores)})")


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.3f}" if values else "n/a"


def report(pairs: EvalPairs) -> None:
    """Print all three agreement reports over the joined (golden, prediction) pairs."""
    scored = pairs.paired
    _print_label_report(
        "TYPE",
        [g.human_type for g, _ in scored],
        [p.issue_type for _, p in scored],
    )
    _print_label_report(
        "DIFFICULTY",
        [g.human_difficulty for g, _ in scored],
        [p.difficulty for _, p in scored],
    )
    _print_solvability_report(scored)


def main(argv: Any = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = parse_golden_lines(GOLDEN_PATH.read_text(encoding="utf-8").splitlines())
    predictions = load_predictions(PROMPT_VERSION)
    pairs = pair_with_predictions(rows, predictions.get)

    print("=== triagent evaluation ===")
    print(f"golden file   : {GOLDEN_PATH}")
    print(f"golden rows   : {len(rows)} total")

    if not pairs.paired and not pairs.skipped_keys:
        print("\ngolden set is still a template - fill it by hand (all rows are TODO/invalid).")
        return

    print(f"scored        : {len(pairs.paired)} filled rows with a v1 prediction")
    print(f"skipped       : {len(pairs.skipped_keys)} filled rows with NO score in DB")
    for key in pairs.skipped_keys:
        print(f"  - no score for {key}")

    if not pairs.paired:
        print("\nno filled golden rows have a matching prediction yet - nothing to score.")
        return

    report(pairs)


if __name__ == "__main__":
    main()

"""Golden-set loading and the pure join/threshold logic the harness measures.

The golden set (``golden.jsonl``) is **human** ground truth: a person fills in
``human_type`` / ``human_difficulty`` / ``human_solvable`` by reading each issue,
never copying the model. The model's prediction lives only in the DB and is
joined in at eval time — so this file, and the human labelling, stay free of
model output. That is what keeps the evaluation honest instead of circular.

Everything here is pure (no DB, no sklearn, no network) so it is unit-testable:
loading + comment skipping, the "is this row filled?" check, the solvability
float -> yes/no threshold, and pairing golden rows with predictions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

# The placeholder a human replaces; rows still holding it are skipped.
TODO = "TODO"

# Allowed human-label vocab. Type/difficulty mirror the score table's CHECK
# constraints; solvability is asked of the human as a plain yes/no.
HUMAN_TYPES = frozenset({"bug", "feature", "docs", "other"})
HUMAN_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
HUMAN_SOLVABLE = frozenset({"yes", "no"})

# Model solvability is a float in [0,1]; >= this counts as a "yes" prediction.
SOLVABILITY_THRESHOLD = 0.5

_FIELDS = ("key", "title", "repo", "human_type", "human_difficulty", "human_solvable", "note")


@dataclass
class GoldenRow:
    """One human-labelled (or still-TODO) golden row. Holds NO model output."""

    key: str
    title: str
    repo: str
    human_type: str
    human_difficulty: str
    human_solvable: str
    note: str = ""


@dataclass(frozen=True)
class Prediction:
    """The model's latest scored prediction for an issue, joined from the DB."""

    issue_type: str
    difficulty: str
    solvability: float


@dataclass
class EvalPairs:
    """The result of joining filled golden rows to predictions."""

    paired: list[tuple[GoldenRow, Prediction]] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)  # filled, but no score in DB


def parse_golden_lines(lines: Iterable[str]) -> list[GoldenRow]:
    """Parse golden.jsonl text, skipping blank lines and ``#`` comments.

    Each non-comment line must be a JSON object; missing fields default to "".
    """
    rows: list[GoldenRow] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        rows.append(GoldenRow(**{f: str(data.get(f, "")) for f in _FIELDS}))
    return rows


def is_filled(row: GoldenRow) -> bool:
    """True only if all three human labels are present and in-vocab (not TODO)."""
    return (
        row.human_type in HUMAN_TYPES
        and row.human_difficulty in HUMAN_DIFFICULTIES
        and row.human_solvable in HUMAN_SOLVABLE
    )


def filled_rows(rows: Iterable[GoldenRow]) -> list[GoldenRow]:
    """Keep only fully, validly labelled rows; TODO/invalid rows are dropped."""
    return [row for row in rows if is_filled(row)]


def solvability_label(value: float, *, threshold: float = SOLVABILITY_THRESHOLD) -> str:
    """Map a model solvability float to a yes/no label at ``threshold``."""
    return "yes" if value >= threshold else "no"


def pair_with_predictions(
    rows: Iterable[GoldenRow], lookup: Callable[[str], Prediction | None]
) -> EvalPairs:
    """Join each filled golden row to its prediction via ``lookup``.

    Rows whose issue has no prediction are recorded in ``skipped_keys`` (and
    excluded from ``paired``) rather than silently dropped.
    """
    result = EvalPairs()
    for row in filled_rows(rows):
        prediction = lookup(row.key)
        if prediction is None:
            result.skipped_keys.append(row.key)
        else:
            result.paired.append((row, prediction))
    return result

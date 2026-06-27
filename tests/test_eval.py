"""Offline tests for the eval join/threshold logic (no DB, no sklearn, no network)."""

from __future__ import annotations

from triagent.eval.golden import (
    GoldenRow,
    Prediction,
    filled_rows,
    is_filled,
    pair_with_predictions,
    parse_golden_lines,
    solvability_label,
)

_GOLDEN_TEXT = """\
# a comment line, skipped
# fields: human_type | human_difficulty | human_solvable

{"key": "o/r#1", "title": "filled bug", "repo": "o/r", "human_type": "bug", "human_difficulty": "easy", "human_solvable": "yes", "note": "seed"}
{"key": "o/r#2", "title": "still todo", "repo": "o/r", "human_type": "TODO", "human_difficulty": "TODO", "human_solvable": "TODO", "note": ""}
{"key": "o/r#3", "title": "filled but no score", "repo": "o/r", "human_type": "docs", "human_difficulty": "medium", "human_solvable": "no", "note": ""}
"""


def test_parse_skips_comments_and_blanks() -> None:
    rows = parse_golden_lines(_GOLDEN_TEXT.splitlines())
    assert [r.key for r in rows] == ["o/r#1", "o/r#2", "o/r#3"]


def test_is_filled_rejects_todo_and_invalid() -> None:
    todo = GoldenRow("k", "t", "r", "TODO", "TODO", "TODO")
    bad_vocab = GoldenRow("k", "t", "r", "bug", "trivial", "yes")  # "trivial" not allowed
    good = GoldenRow("k", "t", "r", "feature", "hard", "no")
    assert not is_filled(todo)
    assert not is_filled(bad_vocab)
    assert is_filled(good)


def test_filled_rows_keeps_only_valid() -> None:
    rows = parse_golden_lines(_GOLDEN_TEXT.splitlines())
    keys = [r.key for r in filled_rows(rows)]
    assert keys == ["o/r#1", "o/r#3"]  # o/r#2 is TODO


def test_solvability_threshold_maps_floats() -> None:
    assert solvability_label(0.5) == "yes"  # boundary is inclusive
    assert solvability_label(0.51) == "yes"
    assert solvability_label(0.49) == "no"
    assert solvability_label(0.0) == "no"
    assert solvability_label(1.0) == "yes"
    # custom threshold
    assert solvability_label(0.6, threshold=0.7) == "no"
    assert solvability_label(0.8, threshold=0.7) == "yes"


def test_pairing_joins_predictions_and_counts_skips() -> None:
    rows = parse_golden_lines(_GOLDEN_TEXT.splitlines())
    # o/r#1 has a prediction; o/r#3 (filled) does NOT; o/r#2 is TODO and never paired.
    predictions = {
        "o/r#1": Prediction(issue_type="bug", difficulty="easy", solvability=0.9),
    }
    result = pair_with_predictions(rows, predictions.get)

    assert [g.key for g, _ in result.paired] == ["o/r#1"]
    assert result.paired[0][1].issue_type == "bug"
    assert result.skipped_keys == ["o/r#3"]  # filled but no score
    # The TODO row is neither paired nor counted as skipped.
    assert "o/r#2" not in result.skipped_keys


def test_empty_when_all_todo() -> None:
    text = '{"key": "o/r#9", "title": "t", "repo": "o/r", "human_type": "TODO", "human_difficulty": "TODO", "human_solvable": "TODO", "note": ""}'
    rows = parse_golden_lines([text])
    empty: dict[str, Prediction] = {}
    result = pair_with_predictions(rows, empty.get)
    assert result.paired == []
    assert result.skipped_keys == []

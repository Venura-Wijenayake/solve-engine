import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solve_engine.ingest.normalize import normalize_issue, normalize_issues

FIXTURES = Path(__file__).parent / "fixtures"
_NOW = datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc)


def _raws() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(
        (FIXTURES / "raw_normalize.json").read_text(encoding="utf-8")
    )
    return data


def test_search_item_maps_fully() -> None:
    issue = normalize_issue(_raws()[0], language="python", now=_NOW)
    assert issue is not None
    assert issue.repo == "pandas-dev/pandas"
    assert issue.number == 100
    assert issue.key == "pandas-dev/pandas#100"
    assert issue.title == "groupby regression on multiindex"  # trimmed
    assert issue.labels == ["good first issue", "bug"]  # objects -> names
    assert issue.source == "search"
    assert issue.language == "python"  # search items get configured language
    assert issue.first_seen == _NOW and issue.last_seen == _NOW
    assert issue.raw["number"] == 100  # full payload retained


def test_watchlist_item_has_no_language() -> None:
    issue = normalize_issue(_raws()[1], language="python", now=_NOW)
    assert issue is not None
    assert issue.source == "watchlist"
    assert issue.language is None  # watchlist items: language stays null
    # empty body but has a label -> still stored, body normalized to None
    assert issue.body is None
    assert issue.labels == ["help wanted"]


def test_missing_title_is_dropped() -> None:
    assert normalize_issue(_raws()[2], language="python", now=_NOW) is None


def test_empty_body_no_labels_is_dropped() -> None:
    assert normalize_issue(_raws()[3], language="python", now=_NOW) is None


def test_batch_drops_unusable_records() -> None:
    issues = normalize_issues(_raws(), language="python", now=_NOW)
    # 4 raw -> 2 usable (search#100, watchlist#200)
    keys = {i.key for i in issues}
    assert keys == {"pandas-dev/pandas#100", "duckdb/duckdb#200"}

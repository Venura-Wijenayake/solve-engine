import json
from pathlib import Path
from typing import Any

from triagent.ingest.query import SOURCE_KEY, is_pull_request, merge_and_dedupe

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


def _search() -> list[dict[str, Any]]:
    return _load("search_items.json")


def _watch() -> list[dict[str, Any]]:
    return _load("repo_items.json")


def test_prs_are_dropped() -> None:
    merged = merge_and_dedupe(_search(), _watch())
    # Fixtures contain two PRs (duckdb#999, pandas#888); none may survive.
    assert all(not is_pull_request(item) for item in merged)
    assert not any("pull_request" in item for item in merged)


def test_dedupe_collapses_duplicates() -> None:
    merged = merge_and_dedupe(_search(), _watch())
    keys = [f"{m['repository_url'].split('/repos/')[-1]}#{m['number']}" for m in merged]
    # 3 search + 4 watch (2 PRs, 1 dup) -> 4 unique issues.
    assert len(merged) == 4
    assert len(keys) == len(set(keys))
    assert "duckdb/duckdb#200" in keys


def test_provenance_tagging() -> None:
    merged = merge_and_dedupe(_search(), _watch())
    by_key = {f"{m['repository_url'].split('/repos/')[-1]}#{m['number']}": m for m in merged}
    # search-only issue keeps search provenance
    assert by_key["pandas-dev/pandas#100"][SOURCE_KEY] == "search"
    # issue in both sources records watchlist
    assert by_key["duckdb/duckdb#200"][SOURCE_KEY] == "watchlist"
    # watchlist-only issue is watchlist
    assert by_key["duckdb/duckdb#201"][SOURCE_KEY] == "watchlist"

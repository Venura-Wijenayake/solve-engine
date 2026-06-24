from datetime import date

from solve_engine.config import SearchConfig, Settings, Thresholds
from solve_engine.ingest.query import build_search_queries


def _settings(labels: list[str]) -> Settings:
    return Settings(
        github_token=None,
        database_url="postgresql://x/y",
        language_focus=["python"],
        data_tags=["sql"],
        watchlist=["duckdb/duckdb"],
        search=SearchConfig(labels=labels, recency_days=30, max_results=100, per_page=100),
        thresholds=Thresholds(solvability_min=0.6, skill_fit_min=0.5),
    )


def test_one_query_per_label_with_expected_qualifiers() -> None:
    settings = _settings(["good first issue", "help wanted", "bug"])
    queries = build_search_queries(settings, today=date(2026, 6, 24))

    assert len(queries) == 3
    for q in queries:
        assert "is:issue" in q
        assert "is:open" in q
        assert "archived:false" in q
        assert "language:python" in q
        assert "created:>=2026-05-25" in q  # 2026-06-24 minus 30 days

    assert 'label:"good first issue"' in queries[0]
    assert 'label:"help wanted"' in queries[1]
    assert 'label:"bug"' in queries[2]


def test_no_labels_emits_single_query() -> None:
    settings = _settings([])
    queries = build_search_queries(settings, today=date(2026, 6, 24))
    assert len(queries) == 1
    assert "label:" not in queries[0]

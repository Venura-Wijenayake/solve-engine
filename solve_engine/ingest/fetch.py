"""Live proof: a small GitHub pull that prints a summary and writes nothing.

    python -m solve_engine.ingest.fetch

Caps the pull low (a page or two) so it stays well inside the rate limit. Run it
twice to see the cache work: the second run serves 304s and consumes far less
budget. This never touches the database.
"""

from __future__ import annotations

import io
import sys

from solve_engine.config import get_settings
from solve_engine.ingest.github_client import GitHubClient
from solve_engine.ingest.query import (
    SOURCE_KEY,
    build_search_queries,
    is_pull_request,
    merge_and_dedupe,
)

# Keep the live pull tiny.
SEARCH_MAX_RESULTS = 30
SEARCH_MAX_PAGES = 1
WATCH_REPOS = 2
WATCH_MAX_PAGES = 1


def main() -> None:
    # GitHub issue titles contain non-cp1252 characters; force UTF-8 output.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()

    with GitHubClient.from_settings(settings) as client:
        before = client.rate_limit()

        queries = build_search_queries(settings)
        search_items = client.search_issues(
            queries,
            per_page=settings.search.per_page,
            max_results=SEARCH_MAX_RESULTS,
            max_pages=SEARCH_MAX_PAGES,
        )

        watch_items = []
        for repo in settings.watchlist[:WATCH_REPOS]:
            watch_items.extend(
                client.repo_issues(
                    repo, per_page=settings.search.per_page, max_pages=WATCH_MAX_PAGES
                )
            )

        raw_total = len(search_items) + len(watch_items)
        prs = sum(
            1 for it in (*search_items, *watch_items) if is_pull_request(it)
        )
        merged = merge_and_dedupe(search_items, watch_items)
        after = client.rate_limit()

    by_source = {"search": 0, "watchlist": 0}
    for item in merged:
        by_source[item[SOURCE_KEY]] += 1

    print("=== solve-engine live fetch (no DB writes) ===")
    print(f"raw items pulled : {raw_total} (search={len(search_items)}, "
          f"watchlist={len(watch_items)})")
    print(f"PRs filtered out : {prs}")
    print(f"after dedupe     : {len(merged)} issues "
          f"(search={by_source['search']}, watchlist={by_source['watchlist']})")
    print(f"http live calls  : {client.stats.live_calls}")
    print(f"http 304 cache   : {client.stats.cache_hits}")
    print(f"search budget    : {before.search.remaining} -> "
          f"{after.search.remaining} remaining (consumed "
          f"{before.search.remaining - after.search.remaining})")
    print(f"core budget      : {before.core.remaining} -> "
          f"{after.core.remaining} remaining (consumed "
          f"{before.core.remaining - after.core.remaining})")
    print("samples:")
    for item in merged[:5]:
        repo = item["repository_url"].split("/repos/", 1)[-1]
        print(f"  [{item[SOURCE_KEY]:9}] {repo}#{item['number']} :: "
              f"{item['title'][:60]}")


if __name__ == "__main__":
    main()

"""Pure helpers: build search queries, derive issue keys, filter and dedupe."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from triagent.config import Settings

# Provenance is recorded under this key on each returned raw dict. Namespaced to
# avoid colliding with any real GitHub field.
SOURCE_KEY = "_source"


def build_search_queries(settings: Settings, *, today: date | None = None) -> list[str]:
    """Compose one search query per configured label (OR'd by running each).

    Each query pins ``is:issue is:open archived:false``, a recency window, and
    the first configured language. Multiple labels are emitted as separate
    queries because a single query with several ``label:`` qualifiers ANDs them.
    """
    if today is None:
        today = date.today()
    cutoff = today - timedelta(days=settings.search.recency_days)

    base = ["is:issue", "is:open", "archived:false", f"created:>={cutoff.isoformat()}"]
    if settings.language_focus:
        base.append(f"language:{settings.language_focus[0]}")

    labels = settings.search.labels
    if not labels:
        return [" ".join(base)]
    return [" ".join([*base, f'label:"{label}"']) for label in labels]


def issue_key(item: dict[str, Any]) -> str:
    """Derive the ``owner/repo#number`` key from a raw issue dict.

    Both the search endpoint and the repo-issues endpoint include
    ``repository_url`` (".../repos/owner/name"), so one extractor serves both.
    """
    repo = item["repository_url"].split("/repos/", 1)[-1]
    return f"{repo}#{item['number']}"


def is_pull_request(item: dict[str, Any]) -> bool:
    """GitHub marks PRs returned by issue endpoints with a ``pull_request`` key."""
    return "pull_request" in item


def merge_and_dedupe(
    search_items: list[dict[str, Any]],
    watchlist_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter out PRs, dedupe by issue key, and tag provenance.

    When an issue appears in both sources, watchlist provenance wins.
    """
    merged: dict[str, dict[str, Any]] = {}
    tagged = [(item, "search") for item in search_items]
    tagged += [(item, "watchlist") for item in watchlist_items]

    for item, source in tagged:
        if is_pull_request(item):
            continue
        key = issue_key(item)
        if key in merged:
            if source == "watchlist":
                merged[key][SOURCE_KEY] = "watchlist"
            continue
        copy = dict(item)
        copy[SOURCE_KEY] = source
        merged[key] = copy

    return list(merged.values())

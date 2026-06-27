"""Authenticated, rate-limit-aware, caching GitHub client.

All GETs flow through :meth:`GitHubClient.get`, which layers the conditional-
request cache (``If-None-Match`` / 304) on top of the rate-limit guard. Search
calls (30/min) are paced and budget-tracked separately from core calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, cast

import httpx

from triagent.config import Settings, get_settings
from triagent.ingest.cache import ResponseCache, cache_key
from triagent.ingest.query import build_search_queries, merge_and_dedupe
from triagent.ingest.ratelimit import RateLimitBudget, RateLimitGuard, RateLimits

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "triagent"
# GitHub's Search API never returns more than 1000 results for one query.
SEARCH_RESULT_CAP = 1000


@dataclass
class FetchStats:
    """Counters for observability and the caching proof."""

    live_calls: int = 0  # 200 responses that consumed budget
    cache_hits: int = 0  # 304 responses served from disk (free)


class GitHubClient:
    """A thin GitHub REST client with caching and rate-limit handling."""

    def __init__(
        self,
        *,
        token: str,
        cache: ResponseCache,
        max_retries: int = 5,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._sleep = sleep
        self.max_retries = max_retries
        self.cache = cache
        self.stats = FetchStats()
        self.guard = RateLimitGuard(now=now, sleep=sleep)
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": USER_AGENT,
            },
            timeout=30.0,
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> GitHubClient:
        settings = settings or get_settings()
        if settings.github_token is None:
            raise RuntimeError(
                "GITHUB_TOKEN is not set. Add it to .env to fetch from GitHub."
            )
        if cache_dir is None:
            cache_dir = Path(".cache") / "github"
        return cls(token=settings.github_token, cache=ResponseCache(cache_dir))

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    # -- core GET: cache + rate-limit guard ------------------------------------

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        is_search: bool = False,
    ) -> Any:
        """GET with conditional caching and rate-limit handling."""
        key = cache_key(url, params)
        cached = self.cache.get(key)

        for attempt in range(self.max_retries + 1):
            if is_search:
                self.guard.throttle_search()

            headers = {}
            if cached is not None and cached.etag:
                headers["If-None-Match"] = cached.etag

            response = self.client.get(url, params=params, headers=headers)
            self.guard.note_headers(response.headers, is_search=is_search)

            if response.status_code == 304 and cached is not None:
                self.stats.cache_hits += 1
                return cached.body

            if response.status_code in (403, 429):
                wait = self.guard.compute_retry_wait(response.headers, attempt)
                self._sleep(wait)
                continue

            response.raise_for_status()
            body = response.json()
            self.cache.set(key, response.headers.get("ETag"), body)
            self.stats.live_calls += 1
            return body

        raise RuntimeError(f"Rate-limit retries exhausted for {url}")

    # -- endpoints -------------------------------------------------------------

    def rate_limit(self) -> RateLimits:
        """Read /rate_limit directly (it never counts against any budget)."""
        response = self.client.get(f"{API_ROOT}/rate_limit")
        response.raise_for_status()
        resources = response.json()["resources"]
        return RateLimits(
            core=RateLimitBudget(**resources["core"]),
            search=RateLimitBudget(**resources["search"]),
        )

    def search_issues(
        self,
        queries: list[str],
        *,
        per_page: int = 100,
        max_results: int = 300,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Paginate the Search API for each query; merge raw items."""
        out: list[dict[str, Any]] = []
        for query in queries:
            collected = 0
            page = 1
            while True:
                body = self.get(
                    f"{API_ROOT}/search/issues",
                    params={
                        "q": query,
                        "per_page": per_page,
                        "page": page,
                        "advanced_search": "true",
                    },
                    is_search=True,
                )
                items = cast(list[dict[str, Any]], body.get("items", []))
                total = int(body.get("total_count", 0))
                out.extend(items)
                collected += len(items)

                if len(items) < per_page:
                    break  # last page for this query
                if collected >= min(max_results, total, SEARCH_RESULT_CAP):
                    break  # hit our cap or all available results
                if page * per_page >= SEARCH_RESULT_CAP:
                    break  # GitHub's hard 1000-result ceiling
                if max_pages is not None and page >= max_pages:
                    break
                page += 1
        return out

    def repo_issues(
        self,
        repo: str,
        *,
        per_page: int = 100,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Paginate open issues for one watchlist repo via the core endpoint."""
        out: list[dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            body = self.get(
                f"{API_ROOT}/repos/{repo}/issues",
                params={"state": "open", "per_page": per_page, "page": page},
                is_search=False,
            )
            items = cast(list[dict[str, Any]], body)
            out.extend(items)
            if len(items) < per_page:
                break
            page += 1
        return out

    def fetch_issues(
        self,
        settings: Settings,
        *,
        search_max_results: int | None = None,
        search_max_pages: int | None = None,
        watch_repos: list[str] | None = None,
        watch_max_pages: int = 1,
    ) -> list[dict[str, Any]]:
        """Pull search + watchlist, filter PRs, dedupe, and tag provenance."""
        queries = build_search_queries(settings)
        search_items = self.search_issues(
            queries,
            per_page=settings.search.per_page,
            max_results=search_max_results or settings.search.max_results,
            max_pages=search_max_pages,
        )

        repos = settings.watchlist if watch_repos is None else watch_repos
        watch_items: list[dict[str, Any]] = []
        for repo in repos:
            watch_items.extend(
                self.repo_issues(
                    repo,
                    per_page=settings.search.per_page,
                    max_pages=watch_max_pages,
                )
            )

        return merge_and_dedupe(search_items, watch_items)

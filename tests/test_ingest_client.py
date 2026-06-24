from pathlib import Path

import httpx

from solve_engine.ingest.cache import ResponseCache
from solve_engine.ingest.github_client import GitHubClient


def _client(tmp_path: Path, handler: httpx.MockTransport) -> GitHubClient:
    return GitHubClient(
        token="test-token",
        cache=ResponseCache(tmp_path / "cache"),
        now=lambda: 0.0,
        sleep=lambda _seconds: None,
        transport=handler,
    )


def test_etag_304_serves_cache_without_consuming_budget(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.headers.get("If-None-Match") == '"abc"':
            return httpx.Response(304, headers={"ETag": '"abc"'})
        return httpx.Response(
            200, headers={"ETag": '"abc"'}, json={"items": [], "total_count": 0}
        )

    client = _client(tmp_path, httpx.MockTransport(handle))
    url = "https://api.github.com/search/issues"

    first = client.get(url, params={"q": "x"})
    second = client.get(url, params={"q": "x"})

    assert calls["n"] == 2  # both reached the server
    assert first == second  # 304 served the cached body
    assert client.stats.live_calls == 1
    assert client.stats.cache_hits == 1


def test_403_then_success_retries(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    client = _client(tmp_path, httpx.MockTransport(handle))
    body = client.get("https://api.github.com/repos/x/y/issues")

    assert body == {"ok": True}
    assert calls["n"] == 2


def test_search_pagination_stops_on_short_page(tmp_path: Path) -> None:
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": i} for i in range(100, 105)]

    def handle(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        items = page1 if page == "1" else page2
        return httpx.Response(200, json={"items": items, "total_count": 105})

    client = _client(tmp_path, httpx.MockTransport(handle))
    items = client.search_issues(["q"], per_page=100, max_results=300)

    assert len(items) == 105

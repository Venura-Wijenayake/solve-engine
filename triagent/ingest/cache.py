"""On-disk conditional-request cache for GitHub GETs.

For every request we store the response ETag alongside its JSON body in a file
named after a hash of the request (URL + params). On the next identical request
we send ``If-None-Match: <etag>``; GitHub answers ``304 Not Modified`` when the
resource is unchanged and we serve the stored body. A 304 does not count against
the rate limit, so re-runs are cheap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CachedResponse:
    """A previously stored response: its ETag and parsed JSON body."""

    etag: str | None
    body: Any


def cache_key(url: str, params: dict[str, Any] | None) -> str:
    """Stable hash of a request (URL + sorted params) used as the filename."""
    canonical = url
    if params:
        items = sorted((str(k), str(v)) for k, v in params.items())
        canonical += "?" + "&".join(f"{k}={v}" for k, v in items)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A simple file-per-request JSON cache under ``cache_dir``."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> CachedResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return CachedResponse(etag=data.get("etag"), body=data.get("body"))

    def set(self, key: str, etag: str | None, body: Any) -> None:
        path = self._path(key)
        payload = {"etag": etag, "body": body}
        path.write_text(json.dumps(payload), encoding="utf-8")

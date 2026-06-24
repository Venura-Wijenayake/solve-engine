"""Rate-limit budgets and the guard that keeps us inside GitHub's limits.

GitHub's Search API allows only 30 requests/minute, tracked separately from the
5000/hour core budget. The guard does two things:

* **Pacing:** enforce a minimum interval between *search* requests so we stay
  under 30/min, and if a response says search ``remaining == 0`` it waits until
  the reset time before the next search call.
* **Backoff:** on a 403/429 (primary exhaustion or a secondary/abuse limit) it
  computes how long to wait — honoring ``Retry-After`` first, then waiting until
  ``X-RateLimit-Reset`` when remaining is 0, otherwise exponential backoff.

The wait math is pure given an injected clock, so it is unit-testable with a
fake clock and no real sleeping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from pydantic import BaseModel


class RateLimitBudget(BaseModel):
    """One GitHub rate-limit resource (core or search)."""

    limit: int
    remaining: int
    reset: int  # epoch seconds


class RateLimits(BaseModel):
    """The two budgets we care about."""

    core: RateLimitBudget
    search: RateLimitBudget


class RateLimitGuard:
    """Paces search requests and computes backoff on rate-limit responses."""

    def __init__(
        self,
        *,
        max_search_per_min: int = 30,
        base_backoff: float = 1.0,
        now: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        # Leave a little headroom under the documented 30/min ceiling.
        self.min_interval = 60.0 / max_search_per_min
        self.base_backoff = base_backoff
        self._now = now
        self._sleep = sleep
        self._last_search: float | None = None
        self.search_remaining: int | None = None
        self.search_reset: int | None = None

    def throttle_search(self) -> float:
        """Block until it's safe to issue the next search request.

        Returns the total seconds waited (for observability/tests).
        """
        waited = 0.0
        if self._last_search is not None:
            gap = self.min_interval - (self._now() - self._last_search)
            if gap > 0:
                self._sleep(gap)
                waited += gap
        if self.search_remaining == 0 and self.search_reset is not None:
            until_reset = self.search_reset - self._now()
            if until_reset > 0:
                self._sleep(until_reset)
                waited += until_reset
        self._last_search = self._now()
        return waited

    def note_headers(self, headers: Mapping[str, str], *, is_search: bool) -> None:
        """Record X-RateLimit-Remaining/Reset from a search response."""
        if not is_search:
            return
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining is not None:
            self.search_remaining = int(remaining)
        if reset is not None:
            self.search_reset = int(reset)

    def compute_retry_wait(self, headers: Mapping[str, str], attempt: int) -> float:
        """Seconds to wait after a 403/429, by priority:

        Retry-After header > wait until reset when remaining is 0 > exp backoff.
        """
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            return float(retry_after)
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining == "0" and reset is not None:
            return max(0.0, float(reset) - self._now())
        return self.base_backoff * (2.0**attempt)

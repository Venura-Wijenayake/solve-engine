from solve_engine.ingest.ratelimit import RateLimitGuard


class FakeClock:
    """A controllable clock; sleeping advances time and is recorded."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _guard(clock: FakeClock) -> RateLimitGuard:
    return RateLimitGuard(max_search_per_min=30, now=clock.now, sleep=clock.sleep)


def test_first_search_does_not_wait() -> None:
    clock = FakeClock()
    guard = _guard(clock)
    assert guard.throttle_search() == 0.0
    assert clock.sleeps == []


def test_back_to_back_search_waits_min_interval() -> None:
    clock = FakeClock()
    guard = _guard(clock)
    guard.throttle_search()  # establishes last_search at t=1000
    waited = guard.throttle_search()  # no time passed -> wait full 2.0s
    assert waited == 2.0  # 60 / 30
    assert clock.sleeps == [2.0]


def test_partial_interval_waits_remainder() -> None:
    clock = FakeClock()
    guard = _guard(clock)
    guard.throttle_search()
    clock.t += 0.5
    waited = guard.throttle_search()
    assert waited == 1.5


def test_zero_remaining_waits_until_reset() -> None:
    clock = FakeClock(start=1000.0)
    guard = _guard(clock)
    guard.search_remaining = 0
    guard.search_reset = 1010
    waited = guard.throttle_search()
    assert waited == 10.0


def test_retry_after_header_wins() -> None:
    clock = FakeClock()
    guard = _guard(clock)
    wait = guard.compute_retry_wait({"retry-after": "7"}, attempt=3)
    assert wait == 7.0


def test_retry_waits_until_reset_when_exhausted() -> None:
    clock = FakeClock(start=1000.0)
    guard = _guard(clock)
    wait = guard.compute_retry_wait(
        {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"}, attempt=0
    )
    assert wait == 30.0


def test_exponential_backoff_fallback() -> None:
    clock = FakeClock()
    guard = _guard(clock)
    assert guard.compute_retry_wait({}, attempt=0) == 1.0
    assert guard.compute_retry_wait({}, attempt=1) == 2.0
    assert guard.compute_retry_wait({}, attempt=3) == 8.0

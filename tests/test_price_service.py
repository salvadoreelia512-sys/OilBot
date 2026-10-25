"""
Failover and staleness logic, tested with fake in-memory adapters so this
suite never touches the network - it tests the ORCHESTRATION logic, not
any particular vendor's API.
"""

from datetime import datetime, timedelta, timezone

import pytest

from adapters.base import PriceAdapter, PriceAdapterError, PriceQuote
from core.price_service import PriceService, PriceServiceConfig


class FakeAdapter(PriceAdapter):
    """A scripted adapter: fails `fail_times` calls, then returns a quote
    with a given age. Lets tests simulate 'source is down', 'source
    recovered', and 'source is up but returning stale data' precisely."""

    def __init__(self, name: str, fail_times: int = 0, age_seconds: float = 0.0, price: float = 80.0):
        self.name = name
        self._fail_remaining = fail_times
        self._age_seconds = age_seconds
        self._price = price
        self.calls = 0

    def get_quote(self, symbol: str) -> PriceQuote:
        self.calls += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise PriceAdapterError(f"{self.name} simulated failure")
        ts = datetime.now(timezone.utc) - timedelta(seconds=self._age_seconds)
        return PriceQuote(symbol=symbol, price=self._price, timestamp=ts, source=self.name)

    def get_bars(self, symbol, interval="1m", lookback="1d"):
        raise NotImplementedError("not needed for these tests")


def test_first_healthy_source_is_used():
    primary = FakeAdapter("primary")
    secondary = FakeAdapter("secondary")
    service = PriceService([primary, secondary])

    quote = service.get_quote("CL=F")

    assert quote.source == "primary"
    assert secondary.calls == 0  # never had to fall back


def test_fails_over_to_next_source():
    primary = FakeAdapter("primary", fail_times=99)
    secondary = FakeAdapter("secondary")
    service = PriceService([primary, secondary], PriceServiceConfig(max_retries_per_source=1))

    quote = service.get_quote("CL=F")

    assert quote.source == "secondary"


def test_falls_through_multiple_dead_sources():
    a = FakeAdapter("a", fail_times=99)
    b = FakeAdapter("b", fail_times=99)
    c = FakeAdapter("c")
    service = PriceService([a, b, c], PriceServiceConfig(max_retries_per_source=1))

    quote = service.get_quote("CL=F")

    assert quote.source == "c"


def test_raises_when_all_sources_fail_and_no_cache():
    a = FakeAdapter("a", fail_times=99)
    b = FakeAdapter("b", fail_times=99)
    service = PriceService([a, b], PriceServiceConfig(max_retries_per_source=1))

    with pytest.raises(PriceAdapterError):
        service.get_quote("CL=F")


def test_returns_last_good_quote_when_all_sources_later_fail():
    """Simulates: source works once, we cache it, then everything goes
    down. The service should hand back the cached quote instead of
    raising - but it is the CALLER's job (get_quote_if_fresh) to refuse to
    act on it once it's too old."""
    flaky = FakeAdapter("flaky", fail_times=0)
    service = PriceService([flaky], PriceServiceConfig(max_retries_per_source=1))

    first = service.get_quote("CL=F")
    assert first.source == "flaky"

    # now make it fail forever
    flaky._fail_remaining = 999
    second = service.get_quote("CL=F")
    assert second.source == "flaky"
    assert second.price == first.price  # it's the cached quote, not a new failure


def test_get_quote_if_fresh_returns_none_when_stale():
    stale = FakeAdapter("stale", age_seconds=120)
    service = PriceService([stale], PriceServiceConfig(stale_after_seconds=30))

    result = service.get_quote_if_fresh("CL=F")

    assert result is None


def test_get_quote_if_fresh_returns_quote_when_fresh():
    fresh = FakeAdapter("fresh", age_seconds=1)
    service = PriceService([fresh], PriceServiceConfig(stale_after_seconds=30))

    result = service.get_quote_if_fresh("CL=F")

    assert result is not None
    assert result.source == "fresh"


def test_health_tracks_per_source_status():
    a = FakeAdapter("a", fail_times=99)
    b = FakeAdapter("b")
    service = PriceService([a, b], PriceServiceConfig(max_retries_per_source=1))

    service.get_quote("CL=F")
    health = service.health()

    assert health["a"].status == "error"
    assert health["a"].consecutive_failures >= 1
    assert health["b"].status == "ok"
    assert health["b"].last_success_ts is not None


def test_empty_adapter_list_rejected():
    with pytest.raises(ValueError):
        PriceService([])


def test_quote_requires_timezone_aware_timestamp():
    with pytest.raises(ValueError):
        PriceQuote(symbol="CL=F", price=80.0, timestamp=datetime.now(), source="x")  # naive datetime

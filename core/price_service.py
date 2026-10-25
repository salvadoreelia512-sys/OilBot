"""
Phase 1: reliability layer over multiple price sources.

The core guarantee this module makes: the bot NEVER produces a probability
or a trade signal from a price it can't vouch for. `get_quote_if_fresh`
returns None - loudly - rather than silently handing back a stale number,
because a confident-looking probability built on a 10-minute-old price is
worse than no probability at all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from adapters.base import PriceAdapter, PriceAdapterError, PriceQuote

logger = logging.getLogger("oilbot.price_service")


@dataclass
class PriceServiceConfig:
    stale_after_seconds: float = 30.0
    max_retries_per_source: int = 1
    retry_backoff_seconds: float = 0.5


@dataclass
class SourceHealth:
    status: str = "unknown"          # "unknown" | "ok" | "error"
    last_error: Optional[str] = None
    last_success_ts: Optional[float] = None
    consecutive_failures: int = 0


class PriceService:
    """
    Tries each adapter in priority order until one returns a quote. Falls
    back through the whole list before giving up, and if every source is
    down, returns the last known-good quote rather than raising - but
    tagged with its real (likely large) age, so `get_quote_if_fresh` will
    correctly refuse to treat it as live.

    Usage:
        service = PriceService([YahooAdapter(), TwelveDataAdapter(key)])
        quote = service.get_quote_if_fresh("CL=F")
        if quote is None:
            # don't trade / don't show a probability - show a "data stale" state
            ...
    """

    def __init__(self, adapters: List[PriceAdapter], config: Optional[PriceServiceConfig] = None):
        if not adapters:
            raise ValueError("PriceService needs at least one adapter")
        self.adapters = adapters
        self.config = config or PriceServiceConfig()
        self._last_good: Dict[str, PriceQuote] = {}
        self._health: Dict[str, SourceHealth] = {a.name: SourceHealth() for a in adapters}

    def get_quote(self, symbol: str) -> PriceQuote:
        errors: List[str] = []
        for adapter in self.adapters:
            health = self._health[adapter.name]
            for attempt in range(self.config.max_retries_per_source):
                try:
                    quote = adapter.get_quote(symbol)
                    health.status = "ok"
                    health.last_error = None
                    health.last_success_ts = time.time()
                    health.consecutive_failures = 0
                    self._last_good[symbol] = quote
                    return quote
                except PriceAdapterError as e:
                    health.status = "error"
                    health.last_error = str(e)
                    health.consecutive_failures += 1
                    errors.append(str(e))
                    logger.warning("price source %s failed (attempt %d): %s", adapter.name, attempt + 1, e)
                    if attempt + 1 < self.config.max_retries_per_source:
                        time.sleep(self.config.retry_backoff_seconds)

        if symbol in self._last_good:
            logger.error(
                "all %d price sources failed for %s; returning last known quote (age=%.1fs)",
                len(self.adapters), symbol, self._last_good[symbol].age_seconds,
            )
            return self._last_good[symbol]

        raise PriceAdapterError(f"all price sources failed for {symbol} and no cached quote exists: {errors}")

    def get_quote_if_fresh(self, symbol: str) -> Optional[PriceQuote]:
        """The call the simulation / UI layer should actually use. Returns
        None instead of a stale quote so callers never silently act on
        old data."""
        quote = self.get_quote(symbol)
        if quote.age_seconds > self.config.stale_after_seconds:
            logger.warning(
                "quote for %s is stale (%.1fs old, source=%s, limit=%.1fs)",
                symbol, quote.age_seconds, quote.source, self.config.stale_after_seconds,
            )
            return None
        return quote

    def health(self) -> Dict[str, SourceHealth]:
        return dict(self._health)

"""
Shared interface every price source plugs into. The price service (see
core/price_service.py) only ever talks to this interface, so adding a new
source later (another broker, another data vendor) never touches the
failover logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


class PriceAdapterError(Exception):
    """Raised by an adapter when it cannot produce a quote or bars right now.
    The price service catches this and moves on to the next source - it is
    the expected, routine failure mode, not a bug."""


@dataclass
class PriceQuote:
    symbol: str
    price: float
    timestamp: datetime  # UTC - when the price was actually observed/reported
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("PriceQuote.timestamp must be timezone-aware (UTC)")

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


class PriceAdapter(ABC):
    name: str

    @abstractmethod
    def get_quote(self, symbol: str) -> PriceQuote:
        """Return the latest quote for `symbol`. Must raise PriceAdapterError
        (not let a raw exception escape) on any failure - network error,
        rate limit, bad response shape, missing symbol, etc."""

    @abstractmethod
    def get_bars(self, symbol: str, interval: str, lookback: str) -> pd.DataFrame:
        """Return recent OHLCV bars as a DataFrame indexed by UTC timestamp,
        with at least a 'close' column. Must raise PriceAdapterError on failure."""

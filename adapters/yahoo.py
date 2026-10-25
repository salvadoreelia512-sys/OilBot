"""
Yahoo Finance adapter, via the unofficial `yfinance` package. Free, no API
key, but unofficial - Yahoo can rate-limit or change response shape without
notice, which is exactly why this sits behind the same PriceAdapter
interface as everything else and is never the only source in production.

Useful symbols:
    "CL=F"   - WTI crude oil front-month futures (closest free proxy for
               TVC:USOIL, which has no public API of its own)
    "USO"    - United States Oil Fund ETF (tradeable on Alpaca for paper trading)
    "AAPL" etc. - any Nasdaq-listed equity
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from .base import PriceAdapter, PriceAdapterError, PriceQuote


class YahooAdapter(PriceAdapter):
    name = "yahoo"

    def get_quote(self, symbol: str) -> PriceQuote:
        try:
            fast = yf.Ticker(symbol).fast_info
            price = fast.get("lastPrice") if isinstance(fast, dict) else fast.last_price
            if price is None:
                raise ValueError("no lastPrice in fast_info response")
            return PriceQuote(
                symbol=symbol,
                price=float(price),
                timestamp=datetime.now(timezone.utc),
                source=self.name,
            )
        except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
            raise PriceAdapterError(f"yahoo quote failed for {symbol}: {e}") from e

    def get_bars(self, symbol: str, interval: str = "1m", lookback: str = "1d") -> pd.DataFrame:
        try:
            df = yf.Ticker(symbol).history(period=lookback, interval=interval)
            if df.empty:
                raise ValueError("empty dataframe returned")
            df.index = pd.to_datetime(df.index, utc=True)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:  # noqa: BLE001
            raise PriceAdapterError(f"yahoo bars failed for {symbol}: {e}") from e

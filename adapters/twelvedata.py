"""
Twelve Data adapter. Free tier covers WTI crude (symbol "WTI/USD" or
"CL1:COM" depending on plan) and US equities, with an 800 req/day,
8 req/min free quota - fine as a secondary/tertiary source behind Yahoo,
not as the only source. Needs a free API key from twelvedata.com.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from .base import PriceAdapter, PriceAdapterError, PriceQuote


class TwelveDataAdapter(PriceAdapter):
    name = "twelvedata"
    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str, timeout: float = 5.0):
        if not api_key:
            raise ValueError("TwelveDataAdapter requires an api_key")
        self.api_key = api_key
        self.timeout = timeout

    def get_quote(self, symbol: str) -> PriceQuote:
        try:
            resp = requests.get(
                f"{self.BASE_URL}/price",
                params={"symbol": symbol, "apikey": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "price" not in data:
                raise ValueError(data.get("message", f"unexpected response: {data}"))
            return PriceQuote(
                symbol=symbol,
                price=float(data["price"]),
                timestamp=datetime.now(timezone.utc),
                source=self.name,
            )
        except Exception as e:  # noqa: BLE001
            raise PriceAdapterError(f"twelvedata quote failed for {symbol}: {e}") from e

    def get_bars(self, symbol: str, interval: str = "1min", lookback: str = "1day") -> pd.DataFrame:
        outputsize = {"1day": 390, "5day": 1950}.get(lookback, 390)
        try:
            resp = requests.get(
                f"{self.BASE_URL}/time_series",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "outputsize": outputsize,
                    "apikey": self.api_key,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "values" not in data:
                raise ValueError(data.get("message", f"unexpected response: {data}"))
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("datetime").sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
        except Exception as e:  # noqa: BLE001
            raise PriceAdapterError(f"twelvedata bars failed for {symbol}: {e}") from e

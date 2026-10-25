"""
Alpaca Markets data adapter - also doubles as the paper-trading execution
venue in Phase 7. Covers US-listed equities and ETFs (USO, Nasdaq names)
on its free paper account. It does NOT quote oil futures (CL) or
TVC:USOIL directly, so in the price service this is typically the primary
source for USO/equities and a fallback confirmation source generally,
while Yahoo/Twelve Data cover the futures/CFD proxy.

Needs a free Alpaca paper-trading API key + secret from alpaca.markets.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from .base import PriceAdapter, PriceAdapterError, PriceQuote


class AlpacaAdapter(PriceAdapter):
    name = "alpaca"
    DATA_URL = "https://data.alpaca.markets/v2"

    def __init__(self, api_key: str, api_secret: str, timeout: float = 5.0):
        if not api_key or not api_secret:
            raise ValueError("AlpacaAdapter requires api_key and api_secret")
        self.headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        self.timeout = timeout

    def get_quote(self, symbol: str) -> PriceQuote:
        try:
            resp = requests.get(
                f"{self.DATA_URL}/stocks/{symbol}/trades/latest",
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            trade = data["trade"]
            ts = pd.to_datetime(trade["t"], utc=True).to_pydatetime()
            return PriceQuote(symbol=symbol, price=float(trade["p"]), timestamp=ts, source=self.name)
        except Exception as e:  # noqa: BLE001
            raise PriceAdapterError(f"alpaca quote failed for {symbol}: {e}") from e

    def get_bars(self, symbol: str, interval: str = "1Min", lookback: str = "1Day") -> pd.DataFrame:
        try:
            resp = requests.get(
                f"{self.DATA_URL}/stocks/{symbol}/bars",
                headers=self.headers,
                params={"timeframe": interval, "limit": 1000},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("bars", [])
            if not bars:
                raise ValueError("no bars returned")
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df = df.set_index("t").rename(
                columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            )
            return df
        except Exception as e:  # noqa: BLE001
            raise PriceAdapterError(f"alpaca bars failed for {symbol}: {e}") from e

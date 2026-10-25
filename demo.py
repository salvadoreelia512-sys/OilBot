"""
End-to-end demo of Phase 1 (price reliability) + Phase 2 (volatility +
Monte Carlo probability engine), wired together the way the real UI will
call them: target price in, long/short in, probability out.

IMPORTANT: this sandbox's shell cannot reach Yahoo Finance / Twelve Data /
Alpaca (outbound network here is restricted to package registries only),
so this demo uses a DemoAdapter with a clearly-labeled placeholder price
instead of live data. The adapters in adapters/yahoo.py, adapters/
twelvedata.py and adapters/alpaca.py are real, complete implementations -
run this same script with them swapped in on your own machine (where
normal internet access is available) to see it against live prices.

Run: python3 demo.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from adapters.base import PriceAdapter, PriceQuote
from core.config import EngineConfig
from core.price_service import PriceService, PriceServiceConfig
from core.simulation import simulate_jump_diffusion, target_for_required_return


class DemoAdapter(PriceAdapter):
    """Stands in for a real adapter so this script runs with zero network
    access and zero API keys. NOT a real price - swap in YahooAdapter (or
    TwelveDataAdapter / AlpacaAdapter) for actual data."""

    name = "demo-placeholder"

    def __init__(self, price: float):
        self._price = price

    def get_quote(self, symbol: str) -> PriceQuote:
        return PriceQuote(
            symbol=symbol,
            price=self._price,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
        )

    def get_bars(self, symbol, interval="1m", lookback="1d"):
        raise NotImplementedError("demo adapter does not implement historical bars")


def run(target_input: float | None, direction: str, leverage: float, required_return: float) -> None:
    config = EngineConfig()

    # --- Phase 1: get a price we can trust ---------------------------------
    # Swap DemoAdapter(68.40) for e.g. YahooAdapter() to use real data:
    #   service = PriceService([YahooAdapter(), TwelveDataAdapter(api_key=...)])
    service = PriceService(
        [DemoAdapter(price=68.40)],
        PriceServiceConfig(stale_after_seconds=config.stale_after_seconds),
    )
    quote = service.get_quote_if_fresh("CL=F")
    if quote is None:
        print("Data is stale - refusing to produce a probability. (This should never trigger in the demo.)")
        return

    print(f"Current price: {quote.price:.2f}  (source={quote.source}, age={quote.age_seconds:.2f}s)")
    print("*** DEMO PRICE - not live market data; see module docstring. ***\n")

    # --- Phase 2a: volatility ------------------------------------------------
    # In production this comes from EWMA + GARCH fit on real recent bars,
    # blended with OVX. Hardcoded here since the demo has no bar history.
    daily_vol = 0.022  # ~ a 35% annualized vol, a plausible oil figure
    hours_remaining = 4.5
    T = hours_remaining / 6.5  # fraction of a 6.5h session remaining
    from core.volatility import daily_vol_to_horizon

    sigma = daily_vol_to_horizon(daily_vol, T)
    print(f"Volatility used: {daily_vol:.4f}/day -> {sigma:.4f} over the remaining {hours_remaining}h")

    # --- Target: either user-specified, or solved from a required return ---
    if target_input is not None:
        target = target_input
        print(f"\nTarget price (user-specified): {target:.2f} ({direction})")
    else:
        target = target_for_required_return(quote.price, required_return, leverage, direction)
        print(
            f"\nNo target given - nearest level for {required_return:.0%} return "
            f"at {leverage:.0f}x leverage ({direction}): {target:.2f}"
        )

    # --- Phase 2b: simulate --------------------------------------------------
    result = simulate_jump_diffusion(
        S0=quote.price,
        target=target,
        mu=0.0,  # no directional view baked in - see conversation notes on why
        sigma=sigma,
        T=T,
        jump_intensity=0.0,  # Phase 4/5 will drive this from news/Hormuz signals
        n_paths=config.n_paths,
        n_steps=config.n_steps,
        seed=None,
    )

    print(f"\nProbability of touching {target:.2f} in the next {hours_remaining}h: {result.touch_probability:.1%}")
    print("\nExpected range for the rest of the session (terminal price percentiles):")
    for p in (5, 25, 50, 75, 95):
        print(f"  p{p:>2}: {result.terminal_percentiles[p]:.2f}")
    print("\nPath high/low bands (90% interval):")
    print(f"  high: {result.path_high_percentiles[5]:.2f} - {result.path_high_percentiles[95]:.2f}")
    print(f"  low:  {result.path_low_percentiles[5]:.2f} - {result.path_low_percentiles[95]:.2f}")


if __name__ == "__main__":
    # Example: "what's my probability of hitting $70 today, going long?"
    run(target_input=70.0, direction="long", leverage=1.0, required_return=0.0)

    print("\n" + "=" * 70 + "\n")

    # Example: "I want 20% return at 10x leverage, going long - what level
    # does that need, and how likely is it?"
    run(target_input=None, direction="long", leverage=10.0, required_return=0.20)

"""
Phase 2a: volatility estimation.

Three independent estimates feed the simulation engine:
  - EWMA: fast-reacting to recent moves (RiskMetrics-style decay)
  - GARCH(1,1): mean-reverting, captures volatility clustering
  - OVX (when available): the market's own forward-looking implied vol

`blended_volatility` combines them. This blend is a starting point, not a
tuned result - Phase 3 (backtesting) is where the weights get validated
against actual calibration, not assumed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def ewma_volatility(returns: pd.Series, lam: float = 0.94, annualization_factor: float = 252) -> float:
    """
    RiskMetrics-style EWMA volatility from a series of periodic simple
    returns. `annualization_factor` converts the periodic vol to an annual
    one - use 252 for daily bars, or (periods-per-day * 252) for intraday
    bars (e.g. 390 one-minute bars/day * 252 ~ 98280 for 1-minute data).

    Deliberately assumes a ZERO mean return, as RiskMetrics does, rather
    than demeaning by the sample mean. For short-horizon financial returns
    the true mean is negligible next to the noise in estimating it, and
    subtracting a realized sample mean actively hides volatility: a
    sustained directional move gets absorbed into "mean" instead of
    registering as the elevated variance it actually represents - exactly
    the kind of move this function exists to detect.
    """
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        raise ValueError("need at least 2 return observations for EWMA")
    weights = lam ** np.arange(len(r))[::-1]
    weights = weights / weights.sum()
    var = np.sum(weights * r**2)
    periodic_vol = np.sqrt(var)
    return float(periodic_vol * np.sqrt(annualization_factor))


def garch_volatility_forecast(returns_pct: pd.Series, horizon: int = 1) -> dict:
    """
    Fits GARCH(1,1) with a Student-t error distribution (fatter tails than
    normal - appropriate for commodity returns) and forecasts variance
    `horizon` steps ahead.

    `returns_pct` must be in PERCENT units (e.g. 0.5 for a 0.5% return) -
    this is what the `arch` package expects for numerically stable fitting;
    feeding it raw decimal returns (0.005) will produce a bad fit silently.

    Returns a dict with:
      periodic_vol   - forecast volatility, same periodicity as the input returns, decimal (not percent)
      params         - fitted model parameters, for logging/diagnostics
      fit_result     - the raw arch `ARCHModelResult`, in case callers need more (e.g. std resid for diagnostics)
    """
    from arch import arch_model  # imported lazily - optional heavy dependency

    r = returns_pct.dropna()
    if len(r) < 50:
        raise ValueError("GARCH needs at least ~50 observations to fit reliably")

    model = arch_model(r, vol="GARCH", p=1, q=1, dist="t")
    res = model.fit(disp="off")
    fcast = res.forecast(horizon=horizon, reindex=False)
    variance_pct2 = float(fcast.variance.values[-1, -1])  # percent^2
    periodic_vol = np.sqrt(variance_pct2) / 100.0

    return {
        "periodic_vol": periodic_vol,
        "params": res.params.to_dict(),
        "fit_result": res,
    }


def blended_volatility(
    ewma_vol: float,
    garch_vol: float,
    ovx_implied_vol: Optional[float] = None,
    weights: Tuple[float, float, float] = (0.3, 0.4, 0.3),
) -> float:
    """
    Weighted blend of EWMA, GARCH and (if available) OVX-implied volatility.
    If OVX isn't supplied, re-normalizes across the remaining two weights
    rather than silently zero-filling it.
    """
    w_ewma, w_garch, w_ovx = weights
    if ovx_implied_vol is None:
        total = w_ewma + w_garch
        if total <= 0:
            raise ValueError("ewma + garch weights must be positive")
        return (w_ewma / total) * ewma_vol + (w_garch / total) * garch_vol

    total = w_ewma + w_garch + w_ovx
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return (
        (w_ewma / total) * ewma_vol
        + (w_garch / total) * garch_vol
        + (w_ovx / total) * ovx_implied_vol
    )


def daily_vol_to_horizon(daily_vol: float, horizon_fraction_of_day: float) -> float:
    """
    Scales an annualization-free DAILY volatility down to a shorter horizon
    (e.g. '4.5 hours left in the session' = 4.5/6.5 of a trading day) under
    the standard square-root-of-time assumption. This is an approximation -
    intraday vol is not actually flat (it's typically higher near the open
    and close), which is exactly what the news/event layer in Phase 4-5
    is there to correct for.
    """
    if horizon_fraction_of_day <= 0:
        raise ValueError("horizon_fraction_of_day must be positive")
    return daily_vol * np.sqrt(horizon_fraction_of_day)

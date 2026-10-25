"""No network needed - uses synthetic return series with known properties."""

import numpy as np
import pandas as pd
import pytest

from core.volatility import (
    blended_volatility,
    daily_vol_to_horizon,
    ewma_volatility,
)


def test_ewma_recovers_known_volatility_roughly():
    """Generate i.i.d. normal returns with a KNOWN daily vol and check EWMA
    gets reasonably close. With lambda close to 1 (slow decay) and enough
    observations, EWMA approaches the sample std."""
    rng = np.random.default_rng(0)
    true_daily_vol = 0.02
    returns = pd.Series(rng.normal(0, true_daily_vol, size=2000))
    est = ewma_volatility(returns, lam=0.97, annualization_factor=1)  # periodic, not annualized, for direct comparison
    assert est == pytest.approx(true_daily_vol, rel=0.15)


def test_ewma_annualization_scales_correctly():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.02, size=500))
    periodic = ewma_volatility(returns, annualization_factor=1)
    annualized = ewma_volatility(returns, annualization_factor=252)
    assert annualized == pytest.approx(periodic * np.sqrt(252), rel=1e-9)


def test_ewma_rejects_too_few_observations():
    with pytest.raises(ValueError):
        ewma_volatility(pd.Series([0.01]))


def test_ewma_reacts_faster_than_plain_mean_to_recent_spike():
    """A volatility spike in the most recent observations should move EWMA
    (low lambda = fast decay) more than it moves a long-window simple
    estimate - this is the whole reason to use EWMA over a flat lookback."""
    calm = np.zeros(200) + 1e-6  # avoid exact-zero variance edge case
    spike = np.full(50, 0.05)
    returns = pd.Series(np.concatenate([calm, spike]))
    fast = ewma_volatility(returns, lam=0.80, annualization_factor=1)
    slow = ewma_volatility(returns, lam=0.99, annualization_factor=1)
    assert fast > slow


def test_blended_volatility_without_ovx_renormalizes():
    result = blended_volatility(ewma_vol=0.02, garch_vol=0.03, ovx_implied_vol=None, weights=(0.3, 0.4, 0.3))
    expected = (0.3 / 0.7) * 0.02 + (0.4 / 0.7) * 0.03
    assert result == pytest.approx(expected)


def test_blended_volatility_with_ovx():
    result = blended_volatility(ewma_vol=0.02, garch_vol=0.03, ovx_implied_vol=0.025, weights=(0.3, 0.4, 0.3))
    expected = 0.3 * 0.02 + 0.4 * 0.03 + 0.3 * 0.025
    assert result == pytest.approx(expected)


def test_daily_vol_to_horizon_sqrt_time():
    daily = 0.02
    half_day = daily_vol_to_horizon(daily, 0.5)
    assert half_day == pytest.approx(daily * np.sqrt(0.5))


def test_daily_vol_to_horizon_rejects_nonpositive():
    with pytest.raises(ValueError):
        daily_vol_to_horizon(0.02, 0)


def test_garch_fit_runs_on_synthetic_arch_data():
    """Smoke test: GARCH(1,1) should fit without error on a synthetic
    series with actual volatility clustering (simulated ARCH process), and
    produce a sane, finite, positive variance forecast. Not testing exact
    parameter recovery - that's `arch`'s job to guarantee, not ours."""
    from core.volatility import garch_volatility_forecast

    rng = np.random.default_rng(5)
    n = 500
    # omega scaled so returns_pct sits in the 1-1000 range arch expects for
    # stable optimizer convergence (these are already "percent" units, e.g.
    # a value of 1.2 means a 1.2% return) - avoids a DataScaleWarning that
    # would otherwise just be noise from this synthetic series, not a real
    # issue with garch_volatility_forecast itself.
    omega, alpha = 5.0, 0.3
    returns_pct = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns_pct[t - 1] ** 2
        returns_pct[t] = rng.normal(0, np.sqrt(sigma2[t]))
    series = pd.Series(returns_pct)

    result = garch_volatility_forecast(series, horizon=1)
    assert result["periodic_vol"] > 0
    assert np.isfinite(result["periodic_vol"])


def test_garch_rejects_too_short_series():
    from core.volatility import garch_volatility_forecast
    with pytest.raises(ValueError):
        garch_volatility_forecast(pd.Series(np.random.normal(0, 1, 10)))

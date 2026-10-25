"""
Phase 2b: the probability engine.

Two pieces that check each other:

1. `closed_form_barrier_touch_probability` - the textbook reflection-principle
   formula for a drifted Brownian motion hitting a level within time T.
   Exact, instant, but diffusion-only (no jumps).

2. `simulate_jump_diffusion` - Merton jump-diffusion Monte Carlo. Adds
   sudden jumps (for news/Hormuz shocks) on top of the same diffusion.
   With jump_intensity=0 it must converge to (1), which is exactly the
   cross-check `tests/test_simulation.py` runs - that's what catches a
   silent bug in the simulator rather than trusting it blindly.

Units convention used throughout this module: mu, sigma and jump_intensity
are all expressed PER THE SAME TIME UNIT as T. The natural choice is
"per trading day", with T as a fraction of a day (e.g. T=0.5 for half a
session remaining) - see core/volatility.daily_vol_to_horizon for getting
sigma into that unit from a daily volatility estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm


@dataclass
class SimulationResult:
    touch_probability: float
    terminal_percentiles: Dict[int, float]
    path_high_percentiles: Dict[int, float]
    path_low_percentiles: Dict[int, float]
    n_paths: int
    n_jumps_expected: float
    n_jumps_observed_mean: float


def closed_form_barrier_touch_probability(
    S0: float, barrier: float, mu: float, sigma: float, T: float
) -> float:
    """
    P(a Brownian motion on log-price, with drift mu and volatility sigma,
    touches log(barrier) at some point in [0, T]), via the reflection
    principle. No jumps - this is the ground-truth check for the diffusion
    component only.

    mu and sigma are the drift/vol of LOG returns, same time unit as T.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if S0 <= 0 or barrier <= 0:
        raise ValueError("S0 and barrier must be positive prices")

    x0 = np.log(S0)
    b = np.log(barrier)
    d = b - x0
    if d == 0:
        return 1.0

    sqrtT = np.sqrt(T)
    if d > 0:
        term1 = norm.cdf((mu * T - d) / (sigma * sqrtT))
        term2 = np.exp(2 * mu * d / sigma**2) * norm.cdf((-mu * T - d) / (sigma * sqrtT))
    else:
        d_abs = -d
        term1 = norm.cdf((-mu * T - d_abs) / (sigma * sqrtT))
        term2 = np.exp(-2 * mu * d_abs / sigma**2) * norm.cdf((mu * T - d_abs) / (sigma * sqrtT))

    return float(np.clip(term1 + term2, 0.0, 1.0))


# Broadie-Glasserman-Kou (1997) continuity correction constant:
# beta = -zeta(1/2) / sqrt(2*pi) ~ 0.5826. Discretely-monitored barrier
# checks systematically under-detect touches relative to continuous
# monitoring, because a path can cross and revert between two monitoring
# points without ever being observed past the barrier. Shifting the
# monitored barrier toward the starting price by beta*sigma*sqrt(dt) (in
# log-price space) corrects for this bias without needing an impractically
# large step count. This correction is applied ONLY to the touch-probability
# test - the percentile bands below reflect the literal simulated path,
# since those are meant to describe what the discrete (e.g. per-minute)
# price series itself looks like, not a continuous-time idealization of it.
_BGK_BETA = 0.5826


def simulate_jump_diffusion(
    S0: float,
    target: float,
    mu: float,
    sigma: float,
    T: float,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 390,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Merton jump-diffusion Monte Carlo on log-price:

        dX = (mu - 0.5*sigma^2 - jump_intensity*k) dt + sigma dW + dJ

    dJ is a compound Poisson process: jumps arrive at rate `jump_intensity`
    (per unit time, same unit as T), each jump sized ~ Normal(jump_mean,
    jump_std) in log-price space. `k = E[e^Y]-1` compensates the drift so
    adding jump risk doesn't silently bias the expected price - this is
    what makes jump_intensity>0 with jump_mean=jump_std=0 collapse exactly
    to the pure-diffusion case (see tests/test_simulation.py).

    Per discrete step, the number of jumps is drawn Poisson(jump_intensity
    * dt); since a sum of k i.i.d. Normal(jump_mean, jump_std) draws is
    itself exactly Normal(k*jump_mean, k*jump_std^2), the compound jump for
    a step is drawn directly from that closed form rather than drawing and
    summing k individual jumps - exact, and avoids an O(n_paths * n_steps *
    max_jumps) array that doesn't scale.

    This is where news/Hormuz signals plug in later: Phase 4-5 adjust
    jump_intensity (more/bigger shocks expected) and sigma (elevated
    background volatility), not mu - the model does not claim to know
    direction from headlines, only that risk has gone up.

    Everything stays in log-space until the very end: touch detection,
    terminal value and running high/low are all computed on the
    (n_paths, n_steps) log-increment array directly, so the full price-space
    path matrix is never materialized - only the small per-path reduced
    vectors are exponentiated.

    n_steps=390 by default assumes T in "trading days" and roughly
    1-minute granularity for a 6.5-hour session; raise it for longer T.
    """
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be positive")
    if S0 <= 0 or target <= 0:
        raise ValueError("S0 and target must be positive prices")
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    if jump_intensity < 0 or jump_std < 0:
        raise ValueError("jump_intensity and jump_std cannot be negative")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
    drift = (mu - 0.5 * sigma**2 - jump_intensity * k) * dt

    # log_increments: build in place to avoid a second full-size array.
    log_increments = sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    log_increments += drift

    jump_counts = None
    if jump_intensity > 0:
        jump_counts = rng.poisson(jump_intensity * dt, size=(n_paths, n_steps))
        # Exact compound sum: k jumps -> Normal(k*mean, k*std^2). sqrt(0)=0
        # correctly gives zero added noise on cells with no jump.
        log_increments += jump_counts * jump_mean
        log_increments += np.sqrt(jump_counts.astype(np.float64)) * jump_std * rng.standard_normal((n_paths, n_steps))

    cum_log = np.cumsum(log_increments, axis=1, out=log_increments)  # relative to x0=log(S0); reuses the array

    x0 = np.log(S0)
    log_target = np.log(target)
    dt_sqrt = np.sqrt(dt)

    if target >= S0:
        corrected_threshold = (log_target - x0) - _BGK_BETA * sigma * dt_sqrt
        touched = (cum_log >= corrected_threshold).any(axis=1)
    else:
        corrected_threshold = (log_target - x0) + _BGK_BETA * sigma * dt_sqrt
        touched = (cum_log <= corrected_threshold).any(axis=1)
    touch_probability = float(touched.mean())

    # Running high/low include the starting point (relative log-return 0,
    # i.e. S0 itself) since the day's actual range starts there.
    terminal_rel = cum_log[:, -1]
    running_high_rel = np.maximum(cum_log.max(axis=1), 0.0)
    running_low_rel = np.minimum(cum_log.min(axis=1), 0.0)

    terminal = np.exp(x0 + terminal_rel)
    running_high = np.exp(x0 + running_high_rel)
    running_low = np.exp(x0 + running_low_rel)

    pct_levels = [5, 10, 25, 50, 75, 90, 95]
    terminal_percentiles = {p: float(np.percentile(terminal, p)) for p in pct_levels}
    high_percentiles = {p: float(np.percentile(running_high, p)) for p in pct_levels}
    low_percentiles = {p: float(np.percentile(running_low, p)) for p in pct_levels}

    n_jumps_expected = jump_intensity * T
    n_jumps_observed_mean = float(jump_counts.sum(axis=1).mean()) if jump_counts is not None else 0.0

    return SimulationResult(
        touch_probability=touch_probability,
        terminal_percentiles=terminal_percentiles,
        path_high_percentiles=high_percentiles,
        path_low_percentiles=low_percentiles,
        n_paths=n_paths,
        n_jumps_expected=n_jumps_expected,
        n_jumps_observed_mean=n_jumps_observed_mean,
    )


def target_for_required_return(S0: float, required_return: float, leverage: float, direction: str) -> float:
    """
    The nearest price level that would deliver `required_return` (e.g. 0.20
    for 20%) on margin at the given leverage, long or short.

    This function makes NO claim about how likely that level is - pair it
    with simulate_jump_diffusion to get the actual probability. Framing it
    this way (instead of "guaranteeing" 20%) is deliberate: the honest
    output is "here's the level that would get you there, and here's how
    often it's actually reached historically."
    """
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    move = required_return / leverage
    if direction == "long":
        return S0 * (1 + move)
    if direction == "short":
        return S0 * (1 - move)
    raise ValueError("direction must be 'long' or 'short'")

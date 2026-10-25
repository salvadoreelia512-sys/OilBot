"""
These tests are the most important ones in the repo: they're what stops the
simulator from silently drifting away from correct math. No network needed -
everything here is pure computation.
"""

import numpy as np
import pytest

from core.simulation import (
    closed_form_barrier_touch_probability,
    simulate_jump_diffusion,
    target_for_required_return,
)


DAILY_VOL = 0.022  # ~35% annualized vol / sqrt(252), a plausible oil daily vol


def test_monte_carlo_matches_closed_form_no_jumps_up_barrier():
    """With jump_intensity=0, the MC engine must converge to the exact
    closed-form first-passage probability - this is the core correctness
    check for the whole simulator."""
    S0, target, mu, T = 80.0, 82.0, 0.0, 1.0
    closed_form = closed_form_barrier_touch_probability(S0, target, mu, DAILY_VOL, T)
    mc = simulate_jump_diffusion(
        S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T,
        jump_intensity=0.0, n_paths=200_000, n_steps=200, seed=42,
    )
    assert mc.touch_probability == pytest.approx(closed_form, abs=0.01)


def test_monte_carlo_matches_closed_form_no_jumps_down_barrier():
    S0, target, mu, T = 80.0, 77.0, 0.0, 1.0
    closed_form = closed_form_barrier_touch_probability(S0, target, mu, DAILY_VOL, T)
    mc = simulate_jump_diffusion(
        S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T,
        jump_intensity=0.0, n_paths=200_000, n_steps=200, seed=7,
    )
    assert mc.touch_probability == pytest.approx(closed_form, abs=0.01)


def test_monte_carlo_matches_closed_form_with_drift():
    S0, target, mu, T = 80.0, 81.5, 0.0005, 1.0  # small positive daily drift
    closed_form = closed_form_barrier_touch_probability(S0, target, mu, DAILY_VOL, T)
    mc = simulate_jump_diffusion(
        S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T,
        jump_intensity=0.0, n_paths=200_000, n_steps=200, seed=99,
    )
    assert mc.touch_probability == pytest.approx(closed_form, abs=0.01)


def test_degenerate_jumps_equal_no_jumps():
    """jump_intensity > 0 but jump_mean=jump_std=0 means every 'jump' adds
    exactly zero to log-price. The compensator k=exp(0)-1=0 leaves drift
    untouched too, so results must be statistically identical to the
    zero-jump case. This catches compensator bugs that would otherwise
    silently bias the drift whenever jumps are turned on."""
    S0, target, mu, T = 80.0, 82.0, 0.0, 1.0
    no_jumps = simulate_jump_diffusion(
        S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T,
        jump_intensity=0.0, n_paths=200_000, n_steps=200, seed=1,
    )
    degenerate_jumps = simulate_jump_diffusion(
        S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T,
        jump_intensity=2.0, jump_mean=0.0, jump_std=0.0,
        n_paths=200_000, n_steps=200, seed=1,
    )
    assert degenerate_jumps.touch_probability == pytest.approx(no_jumps.touch_probability, abs=0.01)


def test_jump_intensity_matches_poisson_expectation():
    """Sanity check on the jump-count machinery itself: average observed
    jumps per path should track lambda*T (law of large numbers), not just
    'some made-up number of jumps happened'."""
    result = simulate_jump_diffusion(
        S0=80.0, target=95.0, mu=0.0, sigma=DAILY_VOL, T=1.0,
        jump_intensity=3.0, jump_mean=0.0, jump_std=0.01,
        n_paths=100_000, n_steps=200, seed=3,
    )
    assert result.n_jumps_expected == pytest.approx(3.0)
    assert result.n_jumps_observed_mean == pytest.approx(3.0, abs=0.05)


def test_touch_probability_decreases_as_target_moves_away():
    """Monotonicity sanity check: a farther target must never be MORE
    probable than a closer one, holding everything else fixed. This is the
    check that would catch the '20% guaranteed' framing error at the math
    level - the model should never claim a bigger move is more likely."""
    S0, mu, T = 80.0, 0.0, 1.0
    near = simulate_jump_diffusion(S0=S0, target=81.0, mu=mu, sigma=DAILY_VOL, T=T, n_paths=100_000, n_steps=100, seed=11)
    mid = simulate_jump_diffusion(S0=S0, target=84.0, mu=mu, sigma=DAILY_VOL, T=T, n_paths=100_000, n_steps=100, seed=11)
    far = simulate_jump_diffusion(S0=S0, target=96.0, mu=mu, sigma=DAILY_VOL, T=T, n_paths=100_000, n_steps=100, seed=11)
    assert near.touch_probability > mid.touch_probability > far.touch_probability


def test_adding_jumps_increases_touch_probability_for_distant_target():
    """Jump risk should make an otherwise-rare distant target MORE likely
    to be touched, not less - this is the entire point of layering
    news/Hormuz shocks onto the diffusion."""
    S0, target, mu, T = 80.0, 90.0, 0.0, 1.0
    calm = simulate_jump_diffusion(S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T, jump_intensity=0.0, n_paths=100_000, n_steps=100, seed=5)
    shocked = simulate_jump_diffusion(
        S0=S0, target=target, mu=mu, sigma=DAILY_VOL, T=T,
        jump_intensity=2.0, jump_mean=0.01, jump_std=0.02,
        n_paths=100_000, n_steps=100, seed=5,
    )
    assert shocked.touch_probability > calm.touch_probability


def test_percentile_bands_are_ordered():
    result = simulate_jump_diffusion(
        S0=80.0, target=82.0, mu=0.0, sigma=DAILY_VOL, T=1.0,
        n_paths=50_000, n_steps=100, seed=2,
    )
    p = result.terminal_percentiles
    assert p[5] < p[25] < p[50] < p[75] < p[95]
    # running high must be >= terminal percentile at same confidence band's shape (loosely: high band sits above low band)
    assert result.path_high_percentiles[50] >= result.path_low_percentiles[50]


def test_zero_distance_target_is_certain():
    result = closed_form_barrier_touch_probability(80.0, 80.0, 0.0, DAILY_VOL, 1.0)
    assert result == 1.0


@pytest.mark.parametrize("bad_kwargs", [
    {"sigma": 0}, {"sigma": -0.1}, {"T": 0}, {"T": -1},
])
def test_closed_form_rejects_invalid_inputs(bad_kwargs):
    kwargs = {"S0": 80.0, "barrier": 82.0, "mu": 0.0, "sigma": DAILY_VOL, "T": 1.0}
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        closed_form_barrier_touch_probability(**kwargs)


def test_target_for_required_return_long():
    target = target_for_required_return(S0=80.0, required_return=0.20, leverage=10.0, direction="long")
    # 20% return at 10x leverage needs a 2% underlying move
    assert target == pytest.approx(80.0 * 1.02)


def test_target_for_required_return_short():
    target = target_for_required_return(S0=80.0, required_return=0.20, leverage=10.0, direction="short")
    assert target == pytest.approx(80.0 * 0.98)


def test_target_for_required_return_no_leverage_needs_full_move():
    target = target_for_required_return(S0=80.0, required_return=0.20, leverage=1.0, direction="long")
    assert target == pytest.approx(80.0 * 1.20)


def test_target_for_required_return_rejects_bad_direction():
    with pytest.raises(ValueError):
        target_for_required_return(S0=80.0, required_return=0.2, leverage=1.0, direction="sideways")

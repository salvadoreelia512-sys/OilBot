# Oil Probability Bot — Phase 1 & 2

Price reliability layer (Phase 1) and the volatility + Monte Carlo
probability engine (Phase 2), built and unit-tested. This is the
foundation everything else (news, Hormuz/AIS, UI, paper trading) plugs
into.

## What's here

```
adapters/
  base.py          PriceAdapter interface + PriceQuote
  yahoo.py         Yahoo Finance (yfinance) - free, no key, CL=F / USO / any Nasdaq ticker
  twelvedata.py     Twelve Data REST - free tier, needs an API key
  alpaca.py        Alpaca Markets - free paper account, needs key+secret; equities/ETFs only (no CL futures)

core/
  price_service.py  Failover across adapters + staleness checking
  volatility.py      EWMA + GARCH(1,1) volatility, blending, horizon scaling
  simulation.py      Merton jump-diffusion Monte Carlo + closed-form cross-check
  config.py          Central defaults

tests/               37 tests, all pure computation - no network required
demo.py              End-to-end wiring: target price in -> probability out
```

## Run it

```
pip install -r requirements.txt
python3 -m pytest -q      # 37 passed
python3 demo.py           # end-to-end example output
```

## A note on this sandbox vs. your machine

This conversation's shell can only reach package registries (pip/npm) and
GitHub — Yahoo Finance, Twelve Data and Alpaca are all blocked from here,
so `demo.py` uses a clearly-labeled placeholder price instead of a live
one. The three adapters are complete, real implementations; run them on
your own machine (normal internet, no proxy restrictions) to see them hit
actual prices. I'd recommend doing that next, before Phase 3, just to
confirm the free tiers behave the way their docs say they do.

## Phase 1 — Price reliability

`PriceService` tries adapters in order and fails over on error. If every
source is down, it returns the last known-good quote rather than crashing
— but tagged with its real age. The critical method is
`get_quote_if_fresh()`: it returns `None` instead of a stale quote, so
nothing downstream can accidentally compute a probability — or place a
trade — on data that's actually 10 minutes old. That refusal-to-guess is
the main reliability guarantee of this whole layer.

```python
from adapters.yahoo import YahooAdapter
from adapters.twelvedata import TwelveDataAdapter
from core.price_service import PriceService

service = PriceService([YahooAdapter(), TwelveDataAdapter(api_key="...")])
quote = service.get_quote_if_fresh("CL=F")
if quote is None:
    # show "data stale" in the UI, don't trade
    ...
```

## Phase 2 — Volatility + Monte Carlo

**Volatility** (`core/volatility.py`): EWMA (fast-reacting) and GARCH(1,1)
(mean-reverting, fits volatility clustering), blended together, with a
slot for OVX implied volatility once that feed is wired up. The EWMA
assumes zero mean, which is the standard convention for short-horizon
financial returns — demeaning by the realized sample mean would let a
sustained directional move get explained away as a "mean shift" instead
of registering as the volatility spike it actually is.

**Simulation** (`core/simulation.py`): Merton jump-diffusion Monte Carlo.
Diffusion captures ordinary day-to-day wiggle; the jump component is
where news/Hormuz shocks will plug in during Phase 4-5 — as changes to
*how much and how often* price can jump, never as a directional signal.
The model does not claim to know which way a headline moves price, only
that risk went up.

Two things worth knowing about the engine:

- **Continuity correction.** A barrier checked only at discrete time
  steps misses touches that happen and revert between two steps, so naive
  discrete Monte Carlo *underestimates* touch probability relative to the
  true continuous-time value. This is corrected with the standard
  Broadie-Glasserman-Kou adjustment rather than by just cranking up the
  step count (which gets expensive fast — see below).
- **Cross-check, not just a test.** `closed_form_barrier_touch_probability`
  is the exact textbook formula for a drifted Brownian motion (no jumps).
  With `jump_intensity=0`, the Monte Carlo result must match it — that's
  what `tests/test_simulation.py` checks, and it's what would catch a
  silent correctness bug if anyone (including me, later) changes the
  simulator.

```python
from core.simulation import simulate_jump_diffusion, target_for_required_return

# "What level gets me 20% return at 10x leverage, and how likely is it?"
target = target_for_required_return(S0=68.40, required_return=0.20, leverage=10, direction="long")
result = simulate_jump_diffusion(S0=68.40, target=target, mu=0.0, sigma=0.018, T=0.7)
print(result.touch_probability)   # an honest number, not a guarantee
```

### On "guaranteed 20%"

Flagging this explicitly because it's the main thing that changed from
how you first described it: nothing in this engine can guarantee a
profit, and the math actively won't let it — `test_touch_probability_
decreases_as_target_moves_away` in the test suite locks in that a farther
target is never reported as *more* likely than a closer one. What the
tool gives you instead is honest: the nearest price level that would hit
your return target at your leverage, and a calibrated probability of
actually reaching it. Often that probability will be low, and that's the
correct, useful answer — it's telling you the trade is a long shot, which
is exactly the information worth having before you take it.

## A known limitation, worth watching

Memory scales with `n_paths * n_steps`. At `n_paths=50,000` the sandbox
this was built in ran out of memory somewhere around `n_steps=4,000` even
after the efficiency pass in this phase. The defaults (`n_paths=50,000`,
`n_steps=390`, i.e. ~1-minute granularity over a session) are comfortably
inside that limit, but if Phase 3's backtesting wants much finer
granularity or many more paths for tighter calibration, that'll need
chunking paths into batches rather than raising `n_paths` directly. Not a
problem yet, just flagging it before it quietly becomes one.

## Next: Phase 3

Backtest `simulate_jump_diffusion` against real historical oil data (once
you can pull it on your own machine) and measure calibration — does the
15% bucket actually resolve true about 15% of the time? That calibration
number is the benchmark every later feature (news, Hormuz/AIS) has to
beat before it's allowed to influence the output, per the checklist.

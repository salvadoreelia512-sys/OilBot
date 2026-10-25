"""Central defaults so Phase 1/2 settings aren't scattered across call sites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # Price reliability
    stale_after_seconds: float = 30.0
    max_retries_per_source: int = 1

    # Simulation
    n_paths: int = 50_000
    n_steps: int = 390          # ~1-minute granularity over a 6.5h session
    default_leverage: float = 1.0

    # Volatility blend weights: (ewma, garch, ovx)
    vol_weights: tuple = (0.3, 0.4, 0.3)
    ewma_lambda: float = 0.94

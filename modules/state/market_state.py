"""
MarketState — Unified Belief State (V6 Architecture)

The single source of truth for all downstream modules. Every module
reads from and writes to this one object — no module computes its own
independent view of the market.

Three layers, all continuous (not hard-labeled):
  Layer A — Structure (LF):  depth regime, spread regime, liquidity health
  Layer B — Pressure (MF):   directional flow, meta-order accumulation
  Layer C — Execution (HF):  burst detection, queue shock, cancel urgency

Key design: regimes are continuous floats [0,1], not discrete labels.
This enables soft transitions and prevents state-boundary instability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ===========================================================================
# MarketState
# ===========================================================================

@dataclass
class MarketState:
    """
    Unified market belief state — updated by all modules.

    All regime values are continuous [0, 1] where:
      0 = healthy/neutral/hold
      1 = degraded/toxic/cancel

    Usage
    -----
    >>> state = MarketState()
    >>> SignalRouter.update(state, raw_signals)
    >>> PressureMemory.update(state, signed_obi)
    >>> fsm.evaluate(state)  # reads state, updates regimes
    >>> fill_prob = FillModel.from_state(state)
    """

    # ── Layer A: Structure (LF span ~50 batches) ──
    structure_regime: float = 0.0       # 0=healthy, 1=degraded (continuous)
    latent_liquidity: float = 0.0       # depth EMA residual (negative = draining)
    latent_volatility: float = 0.0      # spread EMA residual (positive = widening)
    liquidity_z: float = 0.0            # z-score of depth trend
    spread_z: float = 0.0               # z-score of spread trend

    # ── Layer B: Pressure (MF span ~10 batches) ──
    pressure_regime: float = 0.0        # 0=neutral, 1=toxic (continuous)
    latent_pressure: float = 0.0        # pressure memory accumulator P_t
    latent_direction: float = 0.0       # -1 (sell) to +1 (buy)
    pressure_z: float = 0.0             # z-score of |P|
    pressure_flip: bool = False         # direction just reversed

    # ── Layer C: Execution (HF span ~2 batches) ──
    execution_regime: float = 0.0       # 0=hold, 1=cancel (continuous)
    latent_burst: float = 0.0           # HF anomaly magnitude
    burst_z: float = 0.0                # z-score of HF burst

    # ── Risk ──
    inventory_risk: float = 0.0         # |position| * volatility exposure
    toxicity_velocity: float = 0.0      # rate of regime change (d(structure)/dt)

    # ── Meta ──
    timestamp: int = 0                  # batch index
    confidence: float = 1.0             # state estimate confidence [0,1]

    # ── History (last N snapshots for velocity computation) ──
    _history: list = field(default_factory=list, repr=False)
    _max_history: int = field(default=20, repr=False)

    # ── State update helpers ────────────────────────────────────────

    def step(self):
        """Advance timestamp, store history snapshot."""
        self.timestamp += 1
        self._history.append(self.snapshot())
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def snapshot(self) -> dict:
        """Return a compact dict of current state (for history)."""
        return {
            "t": self.timestamp,
            "structure": self.structure_regime,
            "pressure": self.pressure_regime,
            "execution": self.execution_regime,
            "liquidity_z": self.liquidity_z,
            "pressure_z": self.pressure_z,
            "direction": self.latent_direction,
        }

    # ── Derived properties ──────────────────────────────────────────

    @property
    def is_degrading(self) -> bool:
        """Structure is worsening."""
        return self.structure_regime > 0.6

    @property
    def is_toxic(self) -> bool:
        """Pressure is extreme."""
        return self.pressure_regime > 0.7

    @property
    def should_cancel(self) -> bool:
        """Execution layer demands withdrawal."""
        return self.execution_regime > 0.8

    @property
    def velocity(self) -> float:
        """Rate of structure regime change over history."""
        if len(self._history) < 2:
            return 0.0
        past = self._history[-min(5, len(self._history))]
        dt = self.timestamp - past["t"]
        if dt == 0:
            return 0.0
        return (self.structure_regime - past["structure"]) / dt

    # ── Action generation ───────────────────────────────────────────

    def to_action(self) -> dict:
        """
        Convert MarketState to a concrete MM action.
        This is the single point where state becomes decision.
        """
        # Structure → spread + inventory limit
        spread_mult = 1.0 + 0.35 * self.structure_regime
        inv_limit = 1.0 - 0.6 * self.structure_regime

        # Pressure → skew + fill aggressiveness
        skew = self.latent_direction * 0.5 * self.pressure_regime
        fill_aggr = 1.0 - self.pressure_regime

        # Execution → size override
        if self.should_cancel:
            size_mult = 0.0
            quote = False
        else:
            size_mult = inv_limit * fill_aggr
            quote = True

        return {
            "quote": quote,
            "size_multiplier": float(max(size_mult, 0.0)),
            "spread_multiplier": float(spread_mult),
            "skew": float(skew),
            "fill_aggressiveness": float(fill_aggr),
            "structure_regime": float(self.structure_regime),
            "pressure_regime": float(self.pressure_regime),
            "execution_regime": float(self.execution_regime),
            "description": (
                f"S={self.structure_regime:.2f} P={self.pressure_regime:.2f} "
                f"E={self.execution_regime:.2f}"
            ),
        }

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self):
        self.structure_regime = 0.0
        self.latent_liquidity = 0.0
        self.latent_volatility = 0.0
        self.liquidity_z = 0.0
        self.spread_z = 0.0
        self.pressure_regime = 0.0
        self.latent_pressure = 0.0
        self.latent_direction = 0.0
        self.pressure_z = 0.0
        self.pressure_flip = False
        self.execution_regime = 0.0
        self.latent_burst = 0.0
        self.burst_z = 0.0
        self.inventory_risk = 0.0
        self.toxicity_velocity = 0.0
        self.timestamp = 0
        self.confidence = 1.0
        self._history.clear()

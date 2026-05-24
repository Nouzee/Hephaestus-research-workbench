"""
Pressure Memory — Directional Order Flow with Asymmetric Decay

Priority 2 of the Phase 2 architecture.

Captures meta-order splitting behavior via long-memory directional OBI:

    P_t = decay * P_{t-1} + signed_OBI_t

    decay = 0.995  if same direction (persistent meta-order)
    decay = 0.70   if direction flips (new meta-order, fast reset)

The asymmetric decay creates a "sticky" directional memory:
  - Same-direction flow accumulates slowly (meta-order footprint)
  - Direction reversal resets quickly (new meta-order, don't fight it)

Key signals derived from pressure:
  - pressure_level:  |P_t|          — how much accumulated directional pressure
  - pressure_delta:   P_t - P_{t-1}  — is pressure building or releasing
  - pressure_flip:    sign changed   — meta-order completion / new meta-order
  - pressure_z:       z-score vs history — anomalous pressure event

Reference: meta-order research, TSE market microstructure, long memory
in order flow (Bouchaud et al., Farmer et al.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class PressureMemoryConfig:
    """Configuration for pressure memory accumulator."""

    # Decay rates
    decay_same: float = 0.995      # same direction → slow decay (persistent)
    decay_flip: float = 0.70       # direction reversal → fast decay

    # Z-score baseline window
    baseline_window: int = 100

    # Thresholds (in z-score sigma)
    building_threshold: float = 1.5    # pressure building (WATCH)
    toxic_threshold: float = 2.5       # extreme pressure (HARD)

    seed: int = 42


# ===========================================================================
# Core: Pressure Memory
# ===========================================================================

class PressureMemory:
    """
    Directional order flow accumulator with asymmetric decay.

    Usage
    -----
    >>> pm = PressureMemory()
    >>> for signed_obi_batch in stream:
    ...     pressure, signals = pm.update(signed_obi_batch_mean)
    ...     if signals["toxic"]:
    ...         skew_quotes_away_from_pressure()
    """

    def __init__(self, config: Optional[PressureMemoryConfig] = None):
        self.config = config or PressureMemoryConfig()
        cfg = self.config

        # Internal state
        self.P = 0.0                    # current pressure accumulator
        self.prev_sign = 0              # previous direction (+1, -1, 0)
        self.n_updates = 0

        # Rolling history for z-score
        self.abs_history = np.zeros(cfg.baseline_window, dtype=np.float32)
        self.hist_ptr = 0
        self.hist_full = False

        # Derived signals
        self.P_trace: list[float] = []
        self.delta_trace: list[float] = []
        self.flip_trace: list[bool] = []

    # ── Core update ──────────────────────────────────────────────────

    def update(self, signed_obi: float) -> Tuple[float, dict]:
        """
        Update pressure memory with one batch's mean signed OBI.

        Parameters
        ----------
        signed_obi : float — mean signed imbalance for this batch

        Returns
        -------
        P      : float — current accumulated pressure
        signals: dict with keys:
            level      — |P|
            delta      — P_t - P_{t-1}
            flip       — bool, direction just reversed
            z_score    — z-score of |P| vs rolling history
            building   — bool, pressure building (>1.5 sigma)
            toxic      — bool, extreme pressure (>2.5 sigma)
            direction  — +1 (buy pressure), -1 (sell pressure), 0 (neutral)
        """
        cfg = self.config
        x = float(signed_obi)

        # Determine direction
        cur_sign = 1 if x > 0.01 else (-1 if x < -0.01 else 0)

        # Flip detection
        flipped = (self.prev_sign != 0 and cur_sign != 0
                   and cur_sign != self.prev_sign)

        # Asymmetric decay
        if flipped:
            decay = cfg.decay_flip
        else:
            decay = cfg.decay_same

        # Update accumulator
        P_prev = self.P
        self.P = decay * self.P + x

        # Update direction tracking
        if cur_sign != 0:
            self.prev_sign = cur_sign

        # Update rolling history of |P|
        abs_p = abs(self.P)
        self.abs_history[self.hist_ptr] = abs_p
        self.hist_ptr = (self.hist_ptr + 1) % cfg.baseline_window
        if self.hist_ptr == 0:
            self.hist_full = True

        # Z-score
        if self.hist_full:
            mu = self.abs_history.mean()
            sd = max(self.abs_history.std(), 1e-8)
            z = (abs_p - mu) / sd
        else:
            z = 0.0

        # Derived signals
        delta = self.P - P_prev

        signals = {
            "level": abs_p,
            "delta": delta,
            "flip": flipped,
            "z_score": float(z),
            "building": z > cfg.building_threshold,
            "toxic": z > cfg.toxic_threshold,
            "direction": cur_sign,
        }

        # Store traces
        self.P_trace.append(self.P)
        self.delta_trace.append(delta)
        self.flip_trace.append(flipped)
        self.n_updates += 1

        return self.P, signals

    # ── Batch processing (offline) ───────────────────────────────────

    def process_batch(self, signed_obi_series: np.ndarray) -> dict:
        """
        Process an entire series offline, returning trace arrays.

        Returns dict with keys: P, abs_P, delta, flip, z_score, building, toxic
        """
        n = len(signed_obi_series)
        P_arr = np.zeros(n, dtype=np.float32)
        abs_arr = np.zeros(n, dtype=np.float32)
        delta_arr = np.zeros(n, dtype=np.float32)
        flip_arr = np.zeros(n, dtype=bool)
        z_arr = np.zeros(n, dtype=np.float32)
        building_arr = np.zeros(n, dtype=bool)
        toxic_arr = np.zeros(n, dtype=bool)

        for i in range(n):
            P, sig = self.update(float(signed_obi_series[i]))
            P_arr[i] = P
            abs_arr[i] = sig["level"]
            delta_arr[i] = sig["delta"]
            flip_arr[i] = sig["flip"]
            z_arr[i] = sig["z_score"]
            building_arr[i] = sig["building"]
            toxic_arr[i] = sig["toxic"]

        return {
            "P": P_arr, "abs_P": abs_arr, "delta": delta_arr,
            "flip": flip_arr, "z_score": z_arr,
            "building": building_arr, "toxic": toxic_arr,
        }

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self):
        """Reset internal state."""
        self.P = 0.0
        self.prev_sign = 0
        self.n_updates = 0
        self.abs_history.fill(0.0)
        self.hist_ptr = 0
        self.hist_full = False
        self.P_trace.clear()
        self.delta_trace.clear()
        self.flip_trace.clear()

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.P_trace:
            return {"n_updates": 0}
        p_arr = np.array(self.P_trace)
        return {
            "n_updates": self.n_updates,
            "P_mean": float(np.mean(p_arr)),
            "P_std": float(np.std(p_arr)),
            "P_max_abs": float(np.max(np.abs(p_arr))),
            "flip_rate": float(np.mean(self.flip_trace)),
        }

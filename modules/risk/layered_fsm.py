"""
Layered FSM — 3-Layer Risk State Machine (Priority 3)

Layer A — STRUCTURE_STATE (LF inputs):
  Controls base_spread and inventory_limit.
  States: HEALTHY / FRAGILE / DEGRADING
  Inputs: LF depth, LF spread, LF volatility

Layer B — PRESSURE_STATE (MF inputs):
  Controls quote_skew and fill_aggressiveness.
  States: NEUTRAL / BUILDING / TOXIC
  Inputs: MF OBI, pressure_memory, directional persistence

Layer C — EXECUTION_TRIGGER (HF inputs):
  Controls requote/cancel decisions.
  States: HOLD / REQUOTE / CANCEL
  Inputs: HF burst, queue shock, cancel spike

Output is a combined action vector: {size, spread, skew, cancel, description}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class LayeredFSMConfig:
    """Configuration for the 3-layer risk FSM."""

    # ── Layer A: Structure (LF) ──
    lf_fragile_depth_z: float = -1.0      # depth z-score below this → FRAGILE
    lf_degrading_depth_z: float = -2.0     # depth z-score below this → DEGRADING
    lf_fragile_spread_z: float = 1.5       # spread z-score above this → FRAGILE
    lf_structure_recovery: int = 10        # consecutive HEALTHY to exit FRAGILE/DEGRADING

    # ── Layer B: Pressure (MF) ──
    mf_building_pressure_z: float = 1.5    # pressure z-score > this → BUILDING
    mf_toxic_pressure_z: float = 2.5       # pressure z-score > this → TOXIC
    mf_pressure_recovery: int = 5          # consecutive NEUTRAL to exit BUILDING

    # ── Layer C: Execution (HF) ──
    hf_requote_z: float = 2.0              # HF z-score > this → REQUOTE
    hf_cancel_z: float = 3.5               # HF z-score > this → CANCEL

    # ── Action parameters ──
    # Structure → spread
    spread_healthy: float = 1.0
    spread_fragile: float = 1.15
    spread_degrading: float = 1.35

    # Structure → inventory limit
    inv_healthy: float = 1.0
    inv_fragile: float = 0.7
    inv_degrading: float = 0.4

    # Pressure → skew (negative = skew away from buying pressure)
    skew_neutral: float = 0.0
    skew_building: float = 0.3
    skew_toxic: float = 0.6

    # Pressure → fill aggressiveness
    fill_neutral: float = 1.0
    fill_building: float = 0.5
    fill_toxic: float = 0.0

    seed: int = 42


# ===========================================================================
# Core: Layered FSM
# ===========================================================================

class LayeredFSM:
    """
    3-layer risk state machine.

    Usage
    -----
    >>> fsm = LayeredFSM()
    >>> for batch in stream:
    ...     action = fsm.evaluate(lf_features, mf_features, hf_features, pressure)
    ...     apply(action)
    """

    def __init__(self, config: Optional[LayeredFSMConfig] = None):
        self.config = config or LayeredFSMConfig()
        cfg = self.config

        # State
        self.structure_state = "HEALTHY"    # HEALTHY / FRAGILE / DEGRADING
        self.pressure_state = "NEUTRAL"     # NEUTRAL / BUILDING / TOXIC
        self.execution_trigger = "HOLD"     # HOLD / REQUOTE / CANCEL

        # Recovery counters
        self.healthy_counter = 0
        self.neutral_counter = 0

        # History
        self.state_history: list[dict] = []

    # ── Main evaluation ──────────────────────────────────────────────

    def evaluate(
        self,
        lf_depth_z: float,
        lf_spread_z: float,
        mf_obi_z: float,
        pressure_z: float,
        hf_burst_z: float,
    ) -> dict:
        """
        Evaluate all three layers and return combined action.

        Parameters
        ----------
        lf_depth_z  : z-score of LF depth component
        lf_spread_z : z-score of LF spread component
        mf_obi_z    : z-score of MF OBI component
        pressure_z  : z-score of pressure memory |P|
        hf_burst_z  : z-score of HF burst component

        Returns
        -------
        action dict: {size, spread, skew, cancel, fill_aggressiveness,
                      structure_state, pressure_state, execution_trigger}
        """
        cfg = self.config

        # ── Layer A: Structure State ─────────────────────────────────
        depth_weak = lf_depth_z < cfg.lf_fragile_depth_z
        depth_critical = lf_depth_z < cfg.lf_degrading_depth_z
        spread_wide = lf_spread_z > cfg.lf_fragile_spread_z

        prev_structure = self.structure_state

        if prev_structure == "DEGRADING":
            if not depth_critical and not spread_wide:
                self.healthy_counter += 1
                if self.healthy_counter >= cfg.lf_structure_recovery:
                    self.structure_state = "HEALTHY"
                    self.healthy_counter = 0
            else:
                self.healthy_counter = 0
        elif prev_structure == "FRAGILE":
            if depth_critical:
                self.structure_state = "DEGRADING"
                self.healthy_counter = 0
            elif not depth_weak and not spread_wide:
                self.healthy_counter += 1
                if self.healthy_counter >= cfg.lf_structure_recovery:
                    self.structure_state = "HEALTHY"
                    self.healthy_counter = 0
            else:
                self.healthy_counter = 0
        else:  # HEALTHY
            if depth_critical:
                self.structure_state = "DEGRADING"
                self.healthy_counter = 0
            elif depth_weak or spread_wide:
                self.structure_state = "FRAGILE"
                self.healthy_counter = 0

        # ── Layer B: Pressure State ──────────────────────────────────
        pressure_building = pressure_z > cfg.mf_building_pressure_z
        pressure_toxic = pressure_z > cfg.mf_toxic_pressure_z

        prev_pressure = self.pressure_state

        if prev_pressure == "TOXIC":
            if not pressure_toxic:
                self.pressure_state = "BUILDING"
            if not pressure_building:
                self.neutral_counter += 1
                if self.neutral_counter >= cfg.mf_pressure_recovery:
                    self.pressure_state = "NEUTRAL"
                    self.neutral_counter = 0
            else:
                self.neutral_counter = 0
        elif prev_pressure == "BUILDING":
            if pressure_toxic:
                self.pressure_state = "TOXIC"
                self.neutral_counter = 0
            elif not pressure_building:
                self.neutral_counter += 1
                if self.neutral_counter >= cfg.mf_pressure_recovery:
                    self.pressure_state = "NEUTRAL"
                    self.neutral_counter = 0
            else:
                self.neutral_counter = 0
        else:  # NEUTRAL
            if pressure_toxic:
                self.pressure_state = "TOXIC"
                self.neutral_counter = 0
            elif pressure_building:
                self.pressure_state = "BUILDING"
                self.neutral_counter = 0

        # ── Layer C: Execution Trigger ───────────────────────────────
        if hf_burst_z > cfg.hf_cancel_z:
            self.execution_trigger = "CANCEL"
        elif hf_burst_z > cfg.hf_requote_z:
            self.execution_trigger = "REQUOTE"
        else:
            self.execution_trigger = "HOLD"

        # ── Combine into action ──────────────────────────────────────
        # Structure controls spread + inventory limit
        spread_map = {
            "HEALTHY": cfg.spread_healthy,
            "FRAGILE": cfg.spread_fragile,
            "DEGRADING": cfg.spread_degrading,
        }
        inv_map = {
            "HEALTHY": cfg.inv_healthy,
            "FRAGILE": cfg.inv_fragile,
            "DEGRADING": cfg.inv_degrading,
        }
        # Pressure controls skew + fill aggressiveness
        skew_map = {
            "NEUTRAL": cfg.skew_neutral,
            "BUILDING": cfg.skew_building,
            "TOXIC": cfg.skew_toxic,
        }
        fill_map = {
            "NEUTRAL": cfg.fill_neutral,
            "BUILDING": cfg.fill_building,
            "TOXIC": cfg.fill_toxic,
        }

        # Size: product of structure inventory limit and pressure fill
        # When TOXIC (fill=0): withdraw entirely
        size_mult = inv_map[self.structure_state] * fill_map[self.pressure_state]

        # Spread: base spread from structure state
        spread_mult = spread_map[self.structure_state]

        # Execution override
        quote = self.execution_trigger != "CANCEL"
        if self.execution_trigger == "CANCEL":
            size_mult = 0.0

        action = {
            "quote": quote,
            "size_multiplier": size_mult,
            "spread_multiplier": spread_mult,
            "skew": skew_map[self.pressure_state],
            "fill_aggressiveness": fill_map[self.pressure_state],
            "structure_state": self.structure_state,
            "pressure_state": self.pressure_state,
            "execution_trigger": self.execution_trigger,
            "description": (
                f"S:{self.structure_state} P:{self.pressure_state} E:{self.execution_trigger}"
            ),
        }

        self.state_history.append(action)
        return action

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.state_history:
            return {"n_updates": 0}
        n = len(self.state_history)
        return {
            "n_updates": n,
            "structure_distribution": {
                s: sum(1 for h in self.state_history if h["structure_state"] == s) / n
                for s in ["HEALTHY", "FRAGILE", "DEGRADING"]
            },
            "pressure_distribution": {
                s: sum(1 for h in self.state_history if h["pressure_state"] == s) / n
                for s in ["NEUTRAL", "BUILDING", "TOXIC"]
            },
            "execution_distribution": {
                s: sum(1 for h in self.state_history if h["execution_trigger"] == s) / n
                for s in ["HOLD", "REQUOTE", "CANCEL"]
            },
        }

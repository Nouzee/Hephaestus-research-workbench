"""
State Updater — orchestrates all modules into coherent MarketState updates.

Every module becomes a stateless function that reads MarketState and raw data,
then writes its update back to the SAME state object. No module maintains its
own independent view.

Order of updates (critical for causality):
  1. SignalRouter  → updates latent_* from raw signals
  2. PressureMemory → updates latent_pressure, latent_direction
  3. LayeredFSM    → reads latent_* → updates regime_* (soft transitions)
  4. InventorySkew → reads pressure_regime → updates inventory_risk
  5. FillModel     → reads state → produces action (READ-ONLY)

This ensures: state[t] = f(state[t-1], raw_data[t]), with no hidden state
anywhere except MarketState itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from modules.state.market_state import MarketState
from modules.dictionary.causal_wavelet import CausalDecomposer, CausalWaveletConfig
from modules.dictionary.pressure_memory import PressureMemory, PressureMemoryConfig


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class StateUpdaterConfig:
    """Configuration for the state updater."""

    # EMA transition rates for regime values (higher = faster adaptation)
    tau_structure: float = 0.1     # structure regime adaptation rate
    tau_pressure: float = 0.15     # pressure regime adaptation rate
    tau_execution: float = 0.3     # execution regime adaptation rate (fastest)

    # Z-score to regime mapping: z * scale → regime [0,1]
    structure_scale: float = 0.3
    pressure_scale: float = 0.25
    execution_scale: float = 0.4

    # Wavelet config (shared decomposer for all signals)
    hf_span: int = 2
    mf_span: int = 10
    lf_span: int = 50

    seed: int = 42


# ===========================================================================
# Core: State Updater
# ===========================================================================

class StateUpdater:
    """
    Orchestrates all modules into a single MarketState update per batch.

    Usage
    -----
    >>> su = StateUpdater()
    >>> state = MarketState()
    >>> for batch in stream:
    ...     su.update(state, raw_signals, signed_obi, mid_px)
    ...     action = state.to_action()
    """

    def __init__(self, config: Optional[StateUpdaterConfig] = None):
        self.config = config or StateUpdaterConfig()
        cfg = self.config

        # Shared decomposer
        self.decomposer = CausalDecomposer(CausalWaveletConfig(
            hf_span=cfg.hf_span, mf_span=cfg.mf_span, lf_span=cfg.lf_span,
        ))

        # Pressure memory
        self.pressure = PressureMemory(PressureMemoryConfig(
            decay_same=0.995, decay_flip=0.70, baseline_window=100,
        ))

        # Per-signal decomposers
        self.decomposers = {
            name: CausalDecomposer(CausalWaveletConfig(
                hf_span=cfg.hf_span, mf_span=cfg.mf_span, lf_span=cfg.lf_span,
            ))
            for name in ["depth", "obi", "spread", "cancel"]
        }

    # ── Main update ──────────────────────────────────────────────────

    def update(
        self,
        state: MarketState,
        raw_depth_evap: float,
        raw_obi_impulse: float,
        raw_spread_shock: float,
        raw_cancel_burst: float,
        signed_obi_mean: float,
    ) -> MarketState:
        """
        One full state update cycle. All modules write to `state`.

        Returns the same state object (mutated in place).
        """
        cfg = self.config

        # ── 1. Decompose raw signals ─────────────────────────────────
        # LF components → structure latent
        _, _, depth_lf = self.decomposers["depth"].update(raw_depth_evap)
        _, _, spread_lf = self.decomposers["spread"].update(raw_spread_shock)

        # MF components → pressure latent
        _, obi_mf, _ = self.decomposers["obi"].update(raw_obi_impulse)

        # HF components → execution latent
        depth_hf, _, _ = self.decomposers["depth"].update(raw_depth_evap)
        obi_hf, _, _ = self.decomposers["obi"].update(raw_obi_impulse)

        # ── 2. Update latent state variables ─────────────────────────
        prev_state = state.snapshot() if state._history else None

        # Structure latent (LF)
        state.latent_liquidity += cfg.tau_structure * (depth_lf - state.latent_liquidity)
        state.latent_volatility += cfg.tau_structure * (spread_lf - state.latent_volatility)

        # Pressure latent (MF)
        P, p_sig = self.pressure.update(signed_obi_mean)
        state.latent_pressure = P
        state.latent_direction = float(p_sig["direction"])
        state.pressure_z = float(p_sig["z_score"])
        state.pressure_flip = bool(p_sig["flip"])

        # Execution latent (HF)
        hf_mag = np.sqrt(depth_hf**2 + obi_hf**2)
        state.latent_burst += cfg.tau_execution * (hf_mag - state.latent_burst)

        # ── 3. Map latents → regime values (soft, continuous) ────────
        # Structure: depth draining + spread widening → higher regime
        liq_score = np.clip(-state.latent_liquidity * cfg.structure_scale, 0, 1)
        vol_score = np.clip(state.latent_volatility * cfg.structure_scale, 0, 1)
        target_structure = 0.7 * liq_score + 0.3 * vol_score
        state.structure_regime += cfg.tau_structure * (target_structure - state.structure_regime)

        # Pressure: |P| z-score → regime
        target_pressure = np.clip(state.pressure_z * cfg.pressure_scale, 0, 1)
        state.pressure_regime += cfg.tau_pressure * (target_pressure - state.pressure_regime)

        # Execution: burst magnitude → regime
        target_execution = np.clip(state.latent_burst * cfg.execution_scale, 0, 1)
        state.execution_regime += cfg.tau_execution * (target_execution - state.execution_regime)

        # ── 4. Inventory risk ────────────────────────────────────────
        # Higher when pressure is directional AND structure is fragile
        state.inventory_risk = (
            0.5 * abs(state.latent_direction) * state.pressure_regime
            + 0.5 * state.structure_regime
        )

        # ── 5. Toxicity velocity ─────────────────────────────────────
        if prev_state is not None:
            dt = max(state.timestamp - prev_state["t"], 1)
            state.toxicity_velocity = (
                (state.structure_regime - prev_state["structure"]) / dt
            )

        # ── 6. Meta ──────────────────────────────────────────────────
        state.step()

        return state

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self):
        self.pressure.reset()
        for d in self.decomposers.values():
            d.reset()

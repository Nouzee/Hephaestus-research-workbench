"""
Execution Simulator — realistic PnL with fill probabilities.

Replaces the old "quote = fill" assumption. Uses FillModel to
simulate stochastic fills, producing realistic PnL that accounts
for spread widening's impact on fill rate.

Key difference from old PnLBacktest:
  - Old: spread earned = spread * capture_frac (always fills)
  - New: fill ~ Bernoulli(P(fill)) per quote, with P(fill) from FillModel
  - Spread widening now has a real cost: fewer fills
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from modules.execution.fill_model import FillModel, FillModelConfig


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ExecutionSimConfig:
    """Configuration for the execution simulator."""

    # Fill model
    fill_config: FillModelConfig = None

    # Simulation
    ticks_per_batch: int = 2048

    # Inventory: penalty for holding inventory (risk aversion)
    inventory_penalty_rate: float = 0.0  # fraction of position value penalized per batch

    seed: int = 42

    def __post_init__(self):
        if self.fill_config is None:
            self.fill_config = FillModelConfig()


# ===========================================================================
# Core: Execution Simulator
# ===========================================================================

class ExecutionSimulator:
    """
    Realistic execution PnL with stochastic fills.

    Usage
    -----
    >>> sim = ExecutionSimulator()
    >>> pnl = sim.simulate_batch(mid_px, spread, future_ret, action, market_state)
    """

    def __init__(self, config: Optional[ExecutionSimConfig] = None):
        self.config = config or ExecutionSimConfig()
        self.cfg = self.config
        self.fill_model = FillModel(self.cfg.fill_config)
        self.rng = np.random.RandomState(self.cfg.seed)

        # Cumulative tracking
        self.cumulative_pnl = 0.0
        self.cumulative_spread_pnl = 0.0
        self.cumulative_inventory_pnl = 0.0
        self.cumulative_adverse_pnl = 0.0
        self.total_fills = 0
        self.total_ticks = 0
        self.position = 0.0

        # History
        self.pnl_history: list[dict] = []

    # ── Simulate one batch ───────────────────────────────────────────

    def simulate_batch(
        self,
        mid_px: np.ndarray,           # (T,) mid prices for this batch
        spread: np.ndarray,           # (T,) spread values
        future_ret: np.ndarray,       # (T,) forward returns
        action: dict,                 # from LayeredFSM
        market_state: dict,           # imbalance, volatility, pressure
        queue_ratio: float = 0.3,     # our size / total depth
    ) -> dict:
        """
        Simulate one batch of market making with stochastic fills.

        Returns detailed PnL breakdown for this batch.
        """
        T = len(mid_px)
        size_mult = action.get("size_multiplier", 1.0)
        spread_mult = action.get("spread_multiplier", 1.0)
        quote = action.get("quote", True)

        if not quote or size_mult == 0.0:
            # Withdrawn — no PnL this batch
            result = {
                "pnl_total": 0.0, "pnl_spread": 0.0, "pnl_inventory": 0.0,
                "pnl_adverse": 0.0, "n_fills_bid": 0, "n_fills_ask": 0,
                "avg_fill_prob": 0.0, "position_end": self.position,
            }
            self.pnl_history.append(result)
            self.total_ticks += T
            return result

        # Get fill probabilities from model
        imb = float(np.mean(market_state.get("imbalance", 0.0)))
        vol_z = float(market_state.get("vol_z", 0.0))
        pressure_dir = int(market_state.get("pressure_dir", 0))
        pressure_z = float(market_state.get("pressure_z", 0.0))

        fp = self.fill_model.probability(
            spread_mult=spread_mult,
            queue_ratio=queue_ratio,
            imbalance=imb,
            vol_z=vol_z,
            pressure_dir=pressure_dir,
            pressure_z=pressure_z,
        )
        p_bid = fp["p_fill_bid"]
        p_ask = fp["p_fill_ask"]

        # Simulate per-tick fills
        fills_bid = self.rng.binomial(1, p_bid, T).astype(bool)
        fills_ask = self.rng.binomial(1, p_ask, T).astype(bool)

        # PnL per tick: spread earned on fill + adverse cost at fill time
        spread_half = spread * 0.5 * spread_mult * size_mult
        # Adverse selection cost per fill: forward return from fill time
        adverse_per_fill = np.abs(future_ret) * mid_px

        pnl_spread = 0.0
        pnl_adverse = 0.0

        for t in range(T):
            if fills_bid[t]:
                pnl_spread += spread_half[t]
                pnl_adverse -= adverse_per_fill[t]  # bought, lose if price drops
                self.position += size_mult
                self.total_fills += 1
            if fills_ask[t]:
                pnl_spread += spread_half[t]
                pnl_adverse -= adverse_per_fill[t]  # sold, lose if price rises
                self.position -= size_mult
                self.total_fills += 1

        # Close out position at end of batch (mark to last price)
        pnl_inventory = 0.0
        if self.position != 0:
            # Simple close-out: assume we unwind at mid_px[-1]
            # No additional PnL — inventory PnL already captured in adverse
            pnl_inventory = 0.0

        pnl_total = pnl_spread + pnl_adverse + pnl_inventory

        # Store
        self.cumulative_pnl += pnl_total
        self.cumulative_spread_pnl += pnl_spread
        self.cumulative_inventory_pnl += pnl_inventory
        self.cumulative_adverse_pnl += pnl_adverse
        self.total_ticks += T

        result = {
            "pnl_total": float(pnl_total),
            "pnl_spread": float(pnl_spread),
            "pnl_inventory": float(pnl_inventory),
            "pnl_adverse": float(pnl_adverse),
            "n_fills_bid": int(fills_bid.sum()),
            "n_fills_ask": int(fills_ask.sum()),
            "avg_fill_prob": float((p_bid + p_ask) / 2),
            "position_end": float(self.position),
        }
        self.pnl_history.append(result)
        return result

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.pnl_history:
            return {"n_batches": 0}

        pnl_arr = np.array([h["pnl_total"] for h in self.pnl_history])
        spread_arr = np.array([h["pnl_spread"] for h in self.pnl_history])
        adverse_arr = np.array([h["pnl_adverse"] for h in self.pnl_history])

        n = len(pnl_arr)
        return {
            "n_batches": n,
            "total_ticks": self.total_ticks,
            "total_fills": self.total_fills,
            "cumulative_pnl": float(self.cumulative_pnl),
            "cumulative_spread_pnl": float(self.cumulative_spread_pnl),
            "cumulative_inventory_pnl": float(self.cumulative_inventory_pnl),
            "cumulative_adverse_pnl": float(self.cumulative_adverse_pnl),
            "final_position": float(self.position),
            "pnl_per_batch_mean": float(np.mean(pnl_arr)),
            "pnl_per_batch_std": float(np.std(pnl_arr)),
            "sharpe": float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-8) * np.sqrt(n)),
            "spread_pnl_ratio": float(np.sum(spread_arr) / max(abs(np.sum(pnl_arr)), 1e-8)),
            "adverse_pnl_ratio": float(np.sum(adverse_arr) / max(abs(np.sum(pnl_arr)), 1e-8)),
        }

    def reset(self):
        """Reset all state."""
        self.cumulative_pnl = 0.0
        self.cumulative_spread_pnl = 0.0
        self.cumulative_inventory_pnl = 0.0
        self.cumulative_adverse_pnl = 0.0
        self.total_fills = 0
        self.total_ticks = 0
        self.position = 0.0
        self.pnl_history.clear()

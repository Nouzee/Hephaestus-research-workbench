"""
Hardened Execution Simulator — production-grade PnL with real-world frictions.

Adds three layers of realism beyond the baseline execution simulator:

  1. QUEUE POSITION: fill probability depends on our position in the queue.
     Being at the back of the queue → lower P(fill) in calm markets,
     but higher P(fill) in aggressive markets (sweeps).

  2. ADVERSE SELECTION ASYMMETRY: losing fills cost MORE than winning fills earn.
     Asymmetry ratio ~1.5:1 (consistent with HFT empirical literature).
     When you're picked off, you lose 1.5× what you'd gain on a favorable fill.

  3. INVENTORY DECAY COST: holding inventory has a carrying cost proportional
     to mid-price × half-spread. Models the cost of unwinding position.

These three frictions are what separate "academic Sharpe" from "real PnL."
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
class HardenedSimConfig:
    """Configuration for hardened execution simulation."""

    # Fill model
    fill_config: FillModelConfig = None

    # Queue position: [0, 1] — 0 = front of queue, 1 = back
    queue_position: float = 0.5

    # Queue effect: how much queue position reduces fill prob
    # P(fill | queue_pos) = P(fill) * (1 - queue_sensitivity * (queue_pos - 0.5))
    queue_sensitivity: float = 0.3

    # Adverse selection asymmetry: adverse_loss / favorable_gain
    # >1.0 means losing fills hurt more than winning fills help
    adverse_asymmetry: float = 1.5

    # Inventory decay: per-batch cost as fraction of position value
    # Models the half-spread cost of unwinding inventory
    inventory_decay_rate: float = 0.0001  # 1bp per batch

    seed: int = 42

    def __post_init__(self):
        if self.fill_config is None:
            self.fill_config = FillModelConfig()


# ===========================================================================
# Core: Hardened Execution Simulator
# ===========================================================================

class HardenedSimulator:
    """
    Production-grade execution PnL with queue, asymmetry, and inventory costs.

    Usage
    -----
    >>> sim = HardenedSimulator()
    >>> pnl = sim.simulate_batch(mid_px, spread, future_ret, action, market_state)
    """

    def __init__(self, config: Optional[HardenedSimConfig] = None):
        self.config = config or HardenedSimConfig()
        self.cfg = self.config
        self.fill_model = FillModel(self.cfg.fill_config)
        self.rng = np.random.RandomState(self.cfg.seed)

        # Cumulative tracking
        self.cumulative_pnl = 0.0
        self.cumulative_spread_pnl = 0.0
        self.cumulative_adverse_pnl = 0.0
        self.cumulative_inventory_cost = 0.0
        self.total_fills = 0
        self.total_ticks = 0
        self.position = 0.0

        # History
        self.pnl_history: list[dict] = []

    # ── Simulate one batch ───────────────────────────────────────────

    def simulate_batch(
        self,
        mid_px: np.ndarray,
        spread: np.ndarray,
        future_ret: np.ndarray,
        action: dict,
        market_state: dict,
        queue_ratio: float = 0.3,
    ) -> dict:
        """
        Hardened batch simulation with queue, asymmetry, and inventory costs.

        Returns detailed PnL breakdown.
        """
        cfg = self.cfg
        T = len(mid_px)
        size_mult = action.get("size_multiplier", 1.0)
        spread_mult = action.get("spread_multiplier", 1.0)
        quote = action.get("quote", True)

        if not quote or size_mult == 0.0:
            result = {
                "pnl_total": 0.0, "pnl_spread": 0.0, "pnl_adverse": 0.0,
                "pnl_inventory": 0.0, "n_fills": 0, "position_end": float(self.position),
            }
            self.pnl_history.append(result)
            self.total_ticks += T
            return result

        # Base fill probability
        imb = float(np.mean(market_state.get("imbalance", 0.0)))
        vol_z = float(market_state.get("vol_z", 0.0))
        pressure_dir = int(market_state.get("pressure_dir", 0))
        pressure_z = float(market_state.get("pressure_z", 0.0))

        fp = self.fill_model.probability(
            spread_mult=spread_mult, queue_ratio=queue_ratio,
            imbalance=imb, vol_z=vol_z,
            pressure_dir=pressure_dir, pressure_z=pressure_z,
        )

        # Queue position adjustment
        # Back of queue → fewer fills in calm, more fills in sweeps
        queue_adj = 1.0 - cfg.queue_sensitivity * (cfg.queue_position - 0.5)
        queue_adj += 0.1 * abs(imb)  # imbalance → more sweeps → queue matters less
        p_bid = float(np.clip(fp["p_fill_bid"] * queue_adj, 0.001, 0.999))
        p_ask = float(np.clip(fp["p_fill_ask"] * queue_adj, 0.001, 0.999))

        # Simulate fills
        f_bid = self.rng.binomial(1, p_bid, T).astype(bool)
        f_ask = self.rng.binomial(1, p_ask, T).astype(bool)

        # PnL calculation
        spread_half = spread * 0.5 * spread_mult * size_mult
        adverse_base = np.abs(future_ret) * mid_px

        pnl_spread = 0.0
        pnl_adverse = 0.0

        for t in range(T):
            if f_bid[t]:
                pnl_spread += spread_half[t]
                # Adverse asymmetry: losing fill costs more
                if future_ret[t] < 0:  # price dropped after we bought
                    pnl_adverse -= adverse_base[t] * cfg.adverse_asymmetry
                else:
                    pnl_adverse += adverse_base[t] * 0.7  # favorable, but damped
                self.position += size_mult
                self.total_fills += 1

            if f_ask[t]:
                pnl_spread += spread_half[t]
                if future_ret[t] > 0:  # price rose after we sold
                    pnl_adverse -= adverse_base[t] * cfg.adverse_asymmetry
                else:
                    pnl_adverse += adverse_base[t] * 0.7
                self.position -= size_mult
                self.total_fills += 1

        # Inventory decay cost
        inventory_cost = abs(self.position) * np.mean(mid_px) * cfg.inventory_decay_rate
        self.cumulative_inventory_cost += inventory_cost

        pnl_total = pnl_spread + pnl_adverse - inventory_cost

        # Update cumulative
        self.cumulative_pnl += pnl_total
        self.cumulative_spread_pnl += pnl_spread
        self.cumulative_adverse_pnl += pnl_adverse
        self.total_ticks += T

        result = {
            "pnl_total": float(pnl_total),
            "pnl_spread": float(pnl_spread),
            "pnl_adverse": float(pnl_adverse),
            "pnl_inventory": float(-inventory_cost),
            "n_fills": int(f_bid.sum() + f_ask.sum()),
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
        mean_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr))

        return {
            "n_batches": n,
            "total_ticks": self.total_ticks,
            "total_fills": self.total_fills,
            "cumulative_pnl": float(self.cumulative_pnl),
            "cumulative_spread_pnl": float(self.cumulative_spread_pnl),
            "cumulative_adverse_pnl": float(self.cumulative_adverse_pnl),
            "cumulative_inventory_cost": float(self.cumulative_inventory_cost),
            "final_position": float(self.position),
            "pnl_per_batch_mean": mean_pnl,
            "pnl_per_batch_std": std_pnl,
            "sharpe": mean_pnl / max(std_pnl, 1e-8) * np.sqrt(n),
            "adverse_to_spread_ratio": float(
                abs(np.sum(adverse_arr)) / max(abs(np.sum(spread_arr)), 1e-12)
            ),
        }

    def reset(self):
        self.cumulative_pnl = 0.0
        self.cumulative_spread_pnl = 0.0
        self.cumulative_adverse_pnl = 0.0
        self.cumulative_inventory_cost = 0.0
        self.total_fills = 0
        self.total_ticks = 0
        self.position = 0.0
        self.pnl_history.clear()

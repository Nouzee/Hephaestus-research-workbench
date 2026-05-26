"""
Unified Market State — single source of truth for market deconstruction.

Organizes all observed variables into four sub-states:
  flow      — order flow dynamics (who is acting)
  liquidity — order book structure (what the market can absorb)
  impact    — price response (how flow becomes price movement)
  memory    — persistent effects across time (what the market remembers)

This is NOT a trading state. It's a DESCRIPTIVE state for market science.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ===========================================================================
# MarketState
# ===========================================================================

@dataclass
class MarketState:
    """
    Unified market state for deconstruction (not trading).

    Four sub-states:
      flow:      order flow dynamics
      liquidity: order book absorptive capacity
      impact:    price response to flow
      memory:    persistent cross-time effects
    """

    # ── Flow (behavior layer) ──
    trade_arrival_rate: float = 0.0      # events per second
    signed_imbalance: float = 0.0        # -1 (sell) to +1 (buy)
    size_dispersion: float = 1.0         # CV of trade sizes
    flow_persistence: float = 0.0        # lag-1 autocorr of trade side
    cancel_burst_ratio: float = 1.0      # event compression ratio
    buy_sell_volume_ratio: float = 1.0   # buyer/seller volume

    # ── Liquidity (mechanism layer) ──
    spread_bps: float = 0.0              # bid-ask spread
    total_depth: float = 0.0             # bid+ask depth
    depth_imbalance: float = 0.0         # bid depth - ask depth / total
    queue_pressure: float = 0.0          # arrival / depth (consumption rate)
    spread_volatility: float = 0.0       # CV of spread
    liquidity_tension: float = 0.0       # spread / depth
    depth_replenish_corr: float = 0.0    # depth recovery after trades

    # ── Impact (physics layer) ──
    realized_volatility: float = 0.0     # std of returns * sqrt(T)
    immediate_impact_corr: float = 0.0   # corr(flow, return)
    nonlinear_response: float = 1.0      # large/small flow impact ratio
    volatility_persistence: float = 0.0  # lag-1 autocorr of |return|

    # ── Memory (cross-time layer) ──
    impact_memory: float = 0.0           # accumulated recent impact
    flow_memory: float = 0.0             # accumulated directional flow
    regime_stability: float = 1.0        # how stable is current regime

    # Meta
    timestamp: int = 0

    # ── Serialization ────────────────────────────────────────────────

    def to_vector(self) -> np.ndarray:
        """Export as feature vector for causal mapping."""
        return np.array([
            self.trade_arrival_rate, self.signed_imbalance, self.size_dispersion,
            self.flow_persistence, self.cancel_burst_ratio, self.buy_sell_volume_ratio,
            self.spread_bps, self.total_depth, self.depth_imbalance,
            self.queue_pressure, self.spread_volatility, self.liquidity_tension,
            self.depth_replenish_corr, self.realized_volatility,
            self.immediate_impact_corr, self.nonlinear_response,
            self.volatility_persistence,
            self.impact_memory, self.flow_memory, self.regime_stability,
        ], dtype=np.float32)

    @staticmethod
    def vector_names() -> list[str]:
        return [
            "trade_arrival_rate", "signed_imbalance", "size_dispersion",
            "flow_persistence", "cancel_burst_ratio", "buy_sell_volume_ratio",
            "spread_bps", "total_depth", "depth_imbalance",
            "queue_pressure", "spread_volatility", "liquidity_tension",
            "depth_replenish_corr", "realized_volatility",
            "immediate_impact_corr", "nonlinear_response",
            "volatility_persistence",
            "impact_memory", "flow_memory", "regime_stability",
        ]

    def snapshot(self) -> dict:
        """Compact dict representation."""
        return {
            "flow": {
                "arrival_rate": self.trade_arrival_rate,
                "signed_imb": self.signed_imbalance,
                "persistence": self.flow_persistence,
            },
            "liquidity": {
                "spread_bps": self.spread_bps,
                "depth": self.total_depth,
                "queue_pressure": self.queue_pressure,
            },
            "impact": {
                "realized_vol": self.realized_volatility,
                "nonlinear": self.nonlinear_response,
            },
            "memory": {
                "impact_memory": self.impact_memory,
                "flow_memory": self.flow_memory,
            },
        }

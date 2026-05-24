"""
Fill Probability Model — realistic quote-to-trade conversion

Priority A of the execution layer.

Estimates P(fill) for bid and ask quotes based on:
  - spread_multiplier: wider spread = further from mid = fewer fills
  - queue_ratio: your size / total queue depth
  - imbalance: order flow against your quote
  - volatility: price movement through your level
  - pressure_state: meta-order pressure against your side

Key insight: spread widening reduces fill probability non-linearly.
A 1.2x spread doesn't reduce fills by 20% — it can reduce them by 50%+
depending on queue competition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class FillModelConfig:
    """Configuration for fill probability estimation."""

    # Base fill rate (at 1x spread, neutral conditions)
    base_fill_rate: float = 0.3       # probability per batch window

    # Spread sensitivity: higher = fill drops faster as spread widens
    spread_sensitivity: float = 2.5    # P(fill) ∝ spread_mult^(-sensitivity)

    # Queue sensitivity: higher = fill drops faster with queue depth
    queue_sensitivity: float = 1.0

    # Imbalance effect: positive imbalance against your side → higher fill (but adverse!)
    imbalance_effect: float = 0.3

    # Volatility effect: higher vol → more fills
    volatility_effect: float = 0.5

    # Pressure effect: strong directional pressure against you → more fills (toxic!)
    pressure_effect: float = 0.4

    seed: int = 42


# ===========================================================================
# Core: Fill Model
# ===========================================================================

class FillModel:
    """
    Estimate P(fill) for bid and ask quotes.

    Usage
    -----
    >>> fm = FillModel()
    >>> p_bid, p_ask = fm.probability(spread_mult=1.1, queue_ratio=0.3,
    ...                                 imb=-0.2, vol_z=0.5, pressure_dir=1)
    >>> # If pressure_dir=1 (buy pressure): bid fills more, ask fills less
    """

    def __init__(self, config: Optional[FillModelConfig] = None):
        self.config = config or FillModelConfig()
        self.cfg = self.config

        # Tracking
        self.fill_history: list[dict] = []
        self.n_estimates = 0

    # ── Core probability estimation ──────────────────────────────────

    def probability(
        self,
        spread_mult: float = 1.0,
        queue_ratio: float = 0.3,
        imbalance: float = 0.0,
        vol_z: float = 0.0,
        pressure_dir: int = 0,
        pressure_z: float = 0.0,
    ) -> dict:
        """
        Compute P(fill_bid) and P(fill_ask).

        Parameters
        ----------
        spread_mult   : how wide is the quote relative to baseline (1.0 = normal)
        queue_ratio   : our size / total depth at best level
        imbalance     : signed imbalance (+ = buy pressure, - = sell)
        vol_z         : volatility z-score
        pressure_dir  : +1 (buy pressure), -1 (sell pressure), 0 (neutral)
        pressure_z    : pressure z-score magnitude

        Returns
        -------
        dict with: p_fill_bid, p_fill_ask, expected_fill_time_batches, description
        """
        cfg = self.cfg

        # Base fill probability (before adjustments)
        # Wider spread → exponentially lower fill probability
        p_base = cfg.base_fill_rate * (spread_mult ** (-cfg.spread_sensitivity))

        # Queue competition: more depth at level → harder to get filled
        queue_factor = (1.0 / max(queue_ratio, 0.01)) ** cfg.queue_sensitivity
        # Cap
        queue_factor = min(queue_factor, 3.0)

        # Volatility: higher vol → more fills (price moves through your level)
        vol_factor = 1.0 + cfg.volatility_effect * max(vol_z, -1.0)

        # Directional effects: pressure/imbalance against your side
        # Positive imbalance = buying pressure → ask fills more, bid fills less
        imb_effect_bid = -cfg.imbalance_effect * imbalance
        imb_effect_ask = +cfg.imbalance_effect * imbalance

        # Pressure memory effect
        press_effect_bid = -cfg.pressure_effect * pressure_dir * pressure_z
        press_effect_ask = +cfg.pressure_effect * pressure_dir * pressure_z

        # Combined fill probabilities
        p_bid = p_base * queue_factor * vol_factor
        p_bid *= np.exp(imb_effect_bid + press_effect_bid)
        p_bid = float(np.clip(p_bid, 0.005, 0.95))

        p_ask = p_base * queue_factor * vol_factor
        p_ask *= np.exp(imb_effect_ask + press_effect_ask)
        p_ask = float(np.clip(p_ask, 0.005, 0.95))

        # Expected fill time (batches) = 1 / p_fill
        exp_time_bid = 1.0 / max(p_bid, 0.001)
        exp_time_ask = 1.0 / max(p_ask, 0.001)

        result = {
            "p_fill_bid": p_bid,
            "p_fill_ask": p_ask,
            "expected_fill_time_bid": float(exp_time_bid),
            "expected_fill_time_ask": float(exp_time_ask),
            "description": (
                f"bid={p_bid:.2f} ask={p_ask:.2f} "
                f"(spread={spread_mult:.1f}x, queue={queue_ratio:.1f}, "
                f"imb={imbalance:+.2f}, vol_z={vol_z:+.1f})"
            ),
        }

        self.fill_history.append(result)
        self.n_estimates += 1
        return result

    # ── Batch simulation (with randomness) ───────────────────────────

    def simulate_fills(
        self,
        p_bid: float,
        p_ask: float,
        n_ticks: int = 100,
    ) -> dict:
        """
        Simulate fill events over n_ticks.

        Returns dict with: n_fills_bid, n_fills_ask, fill_rate_bid, fill_rate_ask
        """
        rng = np.random.RandomState(self.config.seed + self.n_estimates)
        fills_bid = rng.binomial(n_ticks, p_bid)
        fills_ask = rng.binomial(n_ticks, p_ask)

        return {
            "n_fills_bid": int(fills_bid),
            "n_fills_ask": int(fills_ask),
            "fill_rate_bid": float(fills_bid / n_ticks),
            "fill_rate_ask": float(fills_ask / n_ticks),
            "n_ticks": n_ticks,
        }

    # ── Spread widening impact ───────────────────────────────────────

    def spread_widening_impact(self, spread_from: float, spread_to: float) -> dict:
        """
        Estimate how much fill probability drops when widening spread.

        Returns dict with fill loss ratio and interpretation.
        """
        p_from = self.cfg.base_fill_rate * (spread_from ** (-self.cfg.spread_sensitivity))
        p_to = self.cfg.base_fill_rate * (spread_to ** (-self.cfg.spread_sensitivity))
        loss_ratio = (p_from - p_to) / max(p_from, 1e-8)

        return {
            "p_fill_before": float(p_from),
            "p_fill_after": float(p_to),
            "fill_loss_ratio": float(loss_ratio),
            "interpretation": (
                "severe fill loss" if loss_ratio > 0.5 else
                "moderate fill loss" if loss_ratio > 0.2 else
                "minor fill loss"
            ),
        }

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.fill_history:
            return {"n_estimates": 0}
        bids = [h["p_fill_bid"] for h in self.fill_history]
        asks = [h["p_fill_ask"] for h in self.fill_history]
        return {
            "n_estimates": self.n_estimates,
            "p_bid_mean": float(np.mean(bids)),
            "p_bid_std": float(np.std(bids)),
            "p_ask_mean": float(np.mean(asks)),
            "p_ask_std": float(np.std(asks)),
        }

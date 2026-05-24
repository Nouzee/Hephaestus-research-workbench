"""
Inventory Skew — Asymmetric Quoting from Pressure Direction

Priority 5 of the Phase 2 architecture.

Converts pressure memory direction into quote asymmetry:

  If pressure >> 0 (strong buy pressure):
    bid_size *= (1 - skew)    — reduce bid size (don't add to short)
    ask_size *= (1 + skew)    — increase ask size (welcome sells)
    OR: mid_skew downward     — shift quotes away from pressure

  If pressure << 0 (strong sell pressure):
    bid_size *= (1 + skew)    — increase bid size (welcome buys)
    ask_size *= (1 - skew)    — reduce ask size (don't add to long)
    OR: mid_skew upward       — shift quotes away from pressure

This is NOT about predicting direction. It's about:
  "Don't accumulate inventory on the wrong side of a meta-order."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class InventorySkewConfig:
    """Configuration for pressure-driven inventory skew."""

    # Skew sensitivity: how much to tilt per unit of pressure z-score
    skew_per_z: float = 0.15

    # Max skew (can't go beyond this fraction)
    max_skew: float = 0.5

    # Skew type: "size" (asymmetric sizes) or "price" (shift mid)
    skew_type: str = "size"     # "size" or "price"

    seed: int = 42


# ===========================================================================
# Core: Inventory Skew
# ===========================================================================

class InventorySkew:
    """
    Pressure-driven asymmetric quoting.

    Usage
    -----
    >>> skew = InventorySkew()
    >>> bid_mult, ask_mult = skew.compute(pressure_z, pressure_direction)
    >>> # bid_mult < 1 and ask_mult > 1 when buy pressure dominates
    """

    def __init__(self, config: Optional[InventorySkewConfig] = None):
        self.config = config or InventorySkewConfig()

    # ── Compute skew multipliers ─────────────────────────────────────

    def compute(
        self,
        pressure_z: float,
        pressure_direction: int,
        pressure_level: float = 0.0,
    ) -> dict:
        """
        Compute bid/ask size multipliers from pressure.

        Parameters
        ----------
        pressure_z         : z-score of |P| (how extreme is pressure)
        pressure_direction : +1 (buy), -1 (sell), 0 (neutral)
        pressure_level     : raw |P| value (for magnitude)

        Returns
        -------
        dict with: bid_mult, ask_mult, skew_amount, description
        """
        cfg = self.config

        if pressure_direction == 0 or pressure_z < 0.5:
            # No directional pressure → symmetric quoting
            return {
                "bid_mult": 1.0,
                "ask_mult": 1.0,
                "skew_amount": 0.0,
                "description": "symmetric (no pressure)",
            }

        # Skew amount: proportional to pressure z-score, capped
        skew = min(abs(pressure_z) * cfg.skew_per_z, cfg.max_skew)

        if pressure_direction > 0:
            # Buy pressure → reduce bid, increase ask
            bid_mult = 1.0 - skew
            ask_mult = 1.0 + skew
            desc = f"skew away from buys (skew={skew:.2f})"
        else:
            # Sell pressure → increase bid, reduce ask
            bid_mult = 1.0 + skew
            ask_mult = 1.0 - skew
            desc = f"skew away from sells (skew={skew:.2f})"

        return {
            "bid_mult": float(max(bid_mult, 0.1)),  # floor at 10%
            "ask_mult": float(max(ask_mult, 0.1)),
            "skew_amount": float(skew),
            "description": desc,
        }

    # ── Price skew (alternative: shift mid instead of asymmetric sizes) ──

    def price_skew(
        self,
        pressure_z: float,
        pressure_direction: int,
        mid_price: float,
    ) -> dict:
        """
        Compute an adjusted mid-price to skew quotes away from pressure.

        Positive skew → quote higher (away from buy pressure).
        """
        cfg = self.config
        if pressure_direction == 0 or pressure_z < 0.5:
            return {"adjusted_mid": mid_price, "skew_ticks": 0.0}

        skew_ticks = min(abs(pressure_z) * cfg.skew_per_z, cfg.max_skew)
        adjusted = mid_price * (1.0 + pressure_direction * skew_ticks * 0.0001)

        return {
            "adjusted_mid": float(adjusted),
            "skew_ticks": float(skew_ticks),
            "description": f"mid skewed {'up' if pressure_direction > 0 else 'down'}",
        }

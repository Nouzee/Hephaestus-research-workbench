"""
PnL Attribution — decompose trading PnL into interpretable components.

Priority B of the execution layer.

Decomposes total PnL into:
  spread_capture_pnl  — earned from bid-ask spread on filled quotes
  inventory_pnl       — mark-to-market on held inventory (favorable moves)
  adverse_selection   — loss from being picked off (unfavorable moves)
  missed_trade_pnl    — spread NOT earned because quotes were withdrawn

This decomposition answers:
  "Did the FSM actually reduce adverse selection, or just reduce spread income?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class PnLAttributionConfig:
    """Configuration for PnL attribution."""

    spread_capture_frac: float = 0.5   # fraction of spread earned per fill
    seed: int = 42


# ===========================================================================
# Core: PnL Attributor
# ===========================================================================

class PnLAttributor:
    """
    Decompose PnL into spread, inventory, adverse selection, and missed trade.

    Usage
    -----
    >>> attr = PnLAttributor()
    >>> attr.record_batch(...)   # per batch
    >>> report = attr.report()   # full decomposition
    """

    def __init__(self, config: Optional[PnLAttributionConfig] = None):
        self.config = config or PnLAttributionConfig()

        # Cumulative components
        self.total_spread_pnl = 0.0
        self.total_inventory_pnl = 0.0
        self.total_adverse_pnl = 0.0
        self.total_missed_pnl = 0.0

        # Counters
        self.n_batches = 0
        self.n_ticks_quoting = 0
        self.n_ticks_withdrawn = 0
        self.n_fills = 0

        # Per-state tracking
        self.state_pnl: dict[str, dict] = {}  # {state: {spread, inv, adverse, missed, n}}

        # History
        self.history: list[dict] = []

    # ── Record one batch ─────────────────────────────────────────────

    def record(
        self,
        spread_earned: float,        # spread income this batch (if filled)
        inventory_pnl: float,        # inventory MTM this batch
        adverse_pnl: float,          # adverse selection loss this batch
        was_quoting: bool,           # were quotes active?
        spread_potential: float,     # spread that WOULD have been earned at 1x
        state: str = "NORMAL",       # FSM state for per-state attribution
        n_fills: int = 0,
    ):
        """
        Record one batch of PnL for attribution.

        Parameters
        ----------
        spread_earned    : actual spread income
        inventory_pnl    : inventory MTM (can be + or -)
        adverse_pnl      : adverse selection loss (should be negative)
        was_quoting      : True if quotes were live
        spread_potential : spread income that would have been earned at 1x size/spread
        state            : FSM state name
        n_fills          : number of fills this batch
        """
        self.total_spread_pnl += spread_earned
        self.total_inventory_pnl += inventory_pnl
        self.total_adverse_pnl += adverse_pnl

        if not was_quoting:
            self.total_missed_pnl += spread_potential
            self.n_ticks_withdrawn += 1
        else:
            self.n_ticks_quoting += 1

        self.n_fills += n_fills
        self.n_batches += 1

        # Per-state tracking
        if state not in self.state_pnl:
            self.state_pnl[state] = {
                "spread": 0.0, "inventory": 0.0, "adverse": 0.0,
                "missed": 0.0, "n_batches": 0, "n_fills": 0,
            }
        sp = self.state_pnl[state]
        sp["spread"] += spread_earned
        sp["inventory"] += inventory_pnl
        sp["adverse"] += adverse_pnl
        if not was_quoting:
            sp["missed"] += spread_potential
        sp["n_batches"] += 1
        sp["n_fills"] += n_fills

        self.history.append({
            "spread": spread_earned, "inventory": inventory_pnl,
            "adverse": adverse_pnl, "missed": spread_potential if not was_quoting else 0.0,
            "was_quoting": was_quoting, "state": state, "n_fills": n_fills,
        })

    # ── Report ───────────────────────────────────────────────────────

    def report(self) -> dict:
        """Generate full PnL decomposition report."""
        total_pnl = (self.total_spread_pnl + self.total_inventory_pnl
                     + self.total_adverse_pnl)

        if abs(total_pnl) < 1e-12:
            return {"total_pnl": 0.0, "n_batches": self.n_batches}

        report = {
            "n_batches": self.n_batches,
            "n_fills": self.n_fills,
            "n_ticks_quoting": self.n_ticks_quoting,
            "n_ticks_withdrawn": self.n_ticks_withdrawn,

            # Absolute components
            "total_pnl": float(total_pnl),
            "spread_pnl": float(self.total_spread_pnl),
            "inventory_pnl": float(self.total_inventory_pnl),
            "adverse_pnl": float(self.total_adverse_pnl),
            "missed_pnl": float(self.total_missed_pnl),

            # As % of total |PnL|
            "spread_pnl_pct": float(self.total_spread_pnl / max(abs(total_pnl), 1e-12) * 100),
            "inventory_pnl_pct": float(self.total_inventory_pnl / max(abs(total_pnl), 1e-12) * 100),
            "adverse_pnl_pct": float(self.total_adverse_pnl / max(abs(total_pnl), 1e-12) * 100),
            "missed_pnl_pct": float(self.total_missed_pnl / max(abs(total_pnl), 1e-12) * 100),

            # Key ratios
            "adverse_to_spread_ratio": float(
                abs(self.total_adverse_pnl) / max(abs(self.total_spread_pnl), 1e-12)
            ),
            "missed_to_spread_ratio": float(
                self.total_missed_pnl / max(abs(self.total_spread_pnl), 1e-12)
            ),

            # Per-state breakdown
            "per_state": {},
        }

        for state, sp in self.state_pnl.items():
            s_total = sp["spread"] + sp["inventory"] + sp["adverse"]
            report["per_state"][state] = {
                "n_batches": sp["n_batches"],
                "n_fills": sp["n_fills"],
                "total_pnl": float(s_total),
                "spread_pnl": float(sp["spread"]),
                "adverse_pnl": float(sp["adverse"]),
                "missed_pnl": float(sp["missed"]),
                "adverse_ratio": float(
                    abs(sp["adverse"]) / max(abs(sp["spread"]), 1e-12)
                ),
            }

        self._print_report(report)
        return report

    # ── Print ────────────────────────────────────────────────────────

    def _print_report(self, r: dict):
        print(f"\n{'═'*60}")
        print(f"  PnL Attribution Report")
        print(f"{'═'*60}")
        print(f"  Batches: {r['n_batches']}  Fills: {r['n_fills']}  "
              f"Quoting: {r['n_ticks_quoting']}  Withdrawn: {r['n_ticks_withdrawn']}")
        print(f"\n  Total PnL: {r['total_pnl']:+,.1f}")
        print(f"  {'Component':<20s} {'Absolute':>12s} {'% of |PnL|':>10s}")
        print(f"  {'─'*20} {'─'*12} {'─'*10}")
        for comp in ["spread_pnl", "inventory_pnl", "adverse_pnl", "missed_pnl"]:
            pct_key = comp.replace("_pnl", "_pnl_pct")
            print(f"  {comp:<20s} {r[comp]:>+12,.1f} {r[pct_key]:>+9.1f}%")
        print(f"\n  Adverse/Spread ratio: {r['adverse_to_spread_ratio']:.3f}")
        print(f"  Missed/Spread ratio:  {r['missed_to_spread_ratio']:.3f}")
        print(f"{'═'*60}")

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self):
        self.total_spread_pnl = 0.0
        self.total_inventory_pnl = 0.0
        self.total_adverse_pnl = 0.0
        self.total_missed_pnl = 0.0
        self.n_batches = 0
        self.n_ticks_quoting = 0
        self.n_ticks_withdrawn = 0
        self.n_fills = 0
        self.state_pnl.clear()
        self.history.clear()

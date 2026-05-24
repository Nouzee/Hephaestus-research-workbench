"""
Consistency Scanner — Causal alignment between signal and execution.

Measures whether the state system's decisions are temporally aligned
with actual fill outcomes. If signal_t says "danger" but fills at t+k
are profitable, the system has a timing mismatch.

Three lag bands:
  k=0      — instant: did the state correctly reflect current conditions?
  k=1-5    — short delay: does state lead fills by a few batches?
  k=5-20   — structure delay: does state predict fill quality at longer horizons?

Output:
  consistency_score ∈ [-1, 1]
    +1 = perfect alignment (state high → fills bad, state low → fills good)
     0 = no relationship
    -1 = inverse (state high → fills good, system is wrong)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ConsistencyConfig:
    """Configuration for consistency scanning."""

    # Lag bands
    instant_lag: int = 0
    short_lag_start: int = 1
    short_lag_end: int = 5
    structure_lag_start: int = 5
    structure_lag_end: int = 20

    seed: int = 42


# ===========================================================================
# Core: Consistency Scanner
# ===========================================================================

class ConsistencyScanner:
    """
    Scan signal-to-execution temporal alignment.

    Usage
    -----
    >>> cs = ConsistencyScanner()
    >>> cs.scan(state_history, fill_pnl_history)
    >>> cs.report()
    """

    def __init__(self, config: Optional[ConsistencyConfig] = None):
        self.config = config or ConsistencyConfig()
        self.results: dict = {}

    # ── Scan ─────────────────────────────────────────────────────────

    def scan(
        self,
        structure_regime: np.ndarray,    # (N,) structure regime per batch
        pressure_regime: np.ndarray,     # (N,) pressure regime per batch
        execution_regime: np.ndarray,    # (N,) execution regime per batch
        fill_pnl: np.ndarray,            # (N,) PnL per batch from fills
        fill_rate: np.ndarray = None,    # (N,) optional fill rate per batch
    ) -> "ConsistencyScanner":
        """
        Compute consistency scores for each regime at each lag band.

        Interpretation: corr(regime[t], fill_pnl[t+k])
          - Negative = good (high regime → worse PnL later, system is right)
          - Positive = bad (high regime → better PnL later, system is wrong)
          - Zero = no relationship
        """
        cfg = self.config
        n = min(len(structure_regime), len(fill_pnl))

        struct = structure_regime[:n]
        press = pressure_regime[:n]
        exec_r = execution_regime[:n]
        pnl = fill_pnl[:n]

        self.results = {}

        for name, regime in [("structure", struct), ("pressure", press), ("execution", exec_r)]:
            self.results[name] = {}

            # Instant (k=0)
            c0 = np.corrcoef(regime, pnl)[0, 1]
            self.results[name]["instant"] = float(c0)

            # Short delay (k=1-5)
            short_corrs = []
            for k in range(cfg.short_lag_start, min(cfg.short_lag_end + 1, n - 1)):
                c = np.corrcoef(regime[:-k], pnl[k:])[0, 1]
                short_corrs.append(c)
            self.results[name]["short_delay_mean"] = float(np.mean(short_corrs))
            self.results[name]["short_delay_max"] = float(np.max(np.abs(short_corrs)))
            self.results[name]["short_delay_best_lag"] = int(np.argmax(np.abs(short_corrs)) + cfg.short_lag_start)

            # Structure delay (k=5-20)
            struct_corrs = []
            for k in range(cfg.structure_lag_start, min(cfg.structure_lag_end + 1, n - 1)):
                c = np.corrcoef(regime[:-k], pnl[k:])[0, 1]
                struct_corrs.append(c)
            self.results[name]["structure_delay_mean"] = float(np.mean(struct_corrs))
            self.results[name]["structure_delay_max"] = float(np.max(np.abs(struct_corrs)))
            self.results[name]["structure_delay_best_lag"] = int(np.argmax(np.abs(struct_corrs)) + cfg.structure_lag_start)

        # Fill rate correlation (if available)
        if fill_rate is not None:
            fr = fill_rate[:n]
            self.results["fill_rate"] = {}
            for name, regime in [("structure", struct), ("pressure", press)]:
                c = np.corrcoef(regime, fr)[0, 1]
                self.results["fill_rate"][name] = float(c)

        return self

    # ── Consistency score ────────────────────────────────────────────

    @property
    def overall_score(self) -> float:
        """
        Aggregate consistency score.

        Positive = system is directionally correct (regime leads PnL correctly).
        We negate because correct means "high regime → LOW PnL", i.e., negative corr.
        """
        if not self.results:
            return 0.0
        scores = []
        for name in ["structure", "pressure", "execution"]:
            if name in self.results:
                # Want negative correlation (high regime → bad PnL)
                inst = -self.results[name]["instant"]
                short = -self.results[name]["short_delay_mean"]
                scores.append(0.3 * inst + 0.4 * short)
        return float(np.mean(scores)) if scores else 0.0

    # ── Report ───────────────────────────────────────────────────────

    def report(self) -> dict:
        """Print and return consistency report."""
        if not self.results:
            print("No results. Run scan() first.")
            return {}

        print(f"\n{'═'*65}")
        print(f"  Causal Consistency Scan — Signal → Fill Alignment")
        print(f"{'═'*65}")
        print(f"  {'Regime':<12s} {'Instant':>8s} {'Short(1-5)':>10s} "
              f"{'BestLag':>7s} {'Struct(5-20)':>10s} {'BestLag':>7s} {'Verdict':>10s}")
        print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*7} {'─'*10} {'─'*7} {'─'*10}")

        for name in ["structure", "pressure", "execution"]:
            if name not in self.results:
                continue
            r = self.results[name]
            inst = r["instant"]
            short = r["short_delay_mean"]

            # Verdict: negative = good (high regime → low PnL)
            if short < -0.03:
                verdict = "ALIGNED"
            elif short < -0.01:
                verdict = "weak"
            elif short > 0.03:
                verdict = "MISALIGNED"
            else:
                verdict = "noisy"

            print(f"  {name:<12s} {inst:>+8.4f} {short:>+10.4f} "
                  f"{r['short_delay_best_lag']:>7d} "
                  f"{r['structure_delay_mean']:>+10.4f} "
                  f"{r['structure_delay_best_lag']:>7d} "
                  f"{verdict:>10s}")

        print(f"\n  Overall Consistency: {self.overall_score:+.4f}")
        print(f"  (> 0 = system correct, < 0 = system wrong)")

        return self.results

"""
PnL Backtest — 做市策略毒性熔断回测

Validates whether toxicity scores actually reduce adverse selection,
drawdown, and inventory risk in a simulated market-making strategy.

Strategy:
  - Baseline: earn spread/2 per tick, suffer |Δmid| when adverse selection hits
  - Toxicity-aware: withdraw quotes when toxicity > threshold (percentile)
  - Compare: total PnL, Sharpe, max drawdown, win rate

No charts — pure console math output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PnLBacktestConfig:
    """Configuration for the market-making PnL backtest."""

    # Quote capture
    spread_capture_frac: float = 0.5   # fraction of spread earned per tick as MM

    # Toxicity circuit breaker
    tox_percentile_threshold: float = 90.0  # withdraw quotes above this %ile
    min_quote_ticks: int = 1                 # minimum ticks before re-quoting

    # Risk-free rate (annualized) for Sharpe
    risk_free_rate: float = 0.0

    # Window aggregation
    window_size: int = 100  # ticks per PnL window (matches HMM window)

    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# Core: PnL Backtester
# ═══════════════════════════════════════════════════════════════════════

class PnLBacktest:
    """
    Simulate a market-making strategy with and without toxicity circuit breaker.

    Usage
    -----
    >>> bt = PnLBacktest()
    >>> bt.run(mid_px, spread, toxicity_scores)
    >>> bt.report()
    """

    def __init__(self, config: Optional[PnLBacktestConfig] = None):
        self.config = config or PnLBacktestConfig()

        # Results storage
        self.baseline_pnl: Optional[np.ndarray] = None
        self.toxicity_pnl: Optional[np.ndarray] = None
        self.baseline_equity: Optional[np.ndarray] = None
        self.toxicity_equity: Optional[np.ndarray] = None
        self.toxicity_flags: Optional[np.ndarray] = None

        # Metrics
        self.metrics_: dict = {}

    # ── Main simulation ──────────────────────────────────────────────

    def run_with_actions(
        self,
        mid_px: np.ndarray,           # (N,) tick-level mid prices
        spread: np.ndarray,           # (N,) tick-level spreads
        actions: list,                # (B,) per-batch action dicts from StateMachine
        future_ret: np.ndarray,       # (N,) forward return per tick
        batch_size: int = 2048,
    ) -> "PnLBacktest":
        """
        Run PnL simulation using state-machine actions.

        Each batch gets an action dict with:
          - quote: bool
          - size_multiplier: float (0.0 to 1.0)
          - spread_multiplier: float
          - description: str

        When size_multiplier == 0 (HARD_ALERT): withdraw, zero PnL.
        When size_multiplier < 1: reduced adverse selection but also reduced spread income.
        """
        cfg = self.config
        N = len(mid_px)
        n_batches = len(actions)

        print(f"\n{'─'*60}")
        print(f"  PnL Backtest — State-Machine-Aware MM")
        print(f"{'─'*60}")
        print(f"  Ticks: {N:,}  Batches: {n_batches}  Batch={batch_size} ticks")

        ticks_per_batch = N // n_batches if n_batches > 0 else 1

        # Baseline: always full quote
        adverse_cost = np.abs(future_ret) * mid_px
        spread_earned = spread * cfg.spread_capture_frac

        baseline_pnl_tick = spread_earned - adverse_cost
        baseline_eq = np.cumsum(baseline_pnl_tick)

        # State-machine-aware: per-tick action from batch action
        fsm_pnl_tick = np.zeros(N, dtype=np.float64)
        state_per_batch = np.zeros(n_batches, dtype=np.int32)  # 0=N,1=W,2=H,3=R

        for b in range(n_batches):
            action = actions[b]
            size_mult = action.get("size_multiplier", 1.0)
            t0 = b * ticks_per_batch
            t1 = min(t0 + ticks_per_batch, N)

            # Map state
            desc = action.get("description", "")
            if "withdraw" in desc:
                state_per_batch[b] = 2  # HARD
            elif "reduced" in desc or "half" in desc:
                state_per_batch[b] = 1  # WATCH
            elif "cautious" in desc or "minimal" in desc:
                state_per_batch[b] = 3  # RECOVERY
            else:
                state_per_batch[b] = 0  # NORMAL

            if size_mult == 0.0:
                fsm_pnl_tick[t0:t1] = 0.0
            else:
                # Spread earned scales with size (less quoted = less earned)
                # Adverse cost also scales with size (less quoted = less loss)
                fsm_pnl_tick[t0:t1] = (
                    spread_earned[t0:t1] * size_mult
                    - adverse_cost[t0:t1] * size_mult
                )

        fsm_eq = np.cumsum(fsm_pnl_tick)

        # Store
        self.baseline_pnl = baseline_pnl_tick
        self.toxicity_pnl = fsm_pnl_tick
        self.baseline_equity = baseline_eq
        self.toxicity_equity = fsm_eq
        # Store tick-level toxicity flag (True if HARD_ALERT = withdraw)
        self.toxicity_flags = (fsm_pnl_tick == 0.0)

        # State distribution
        state_names = ["NORMAL", "WATCH", "HARD", "RECOVERY"]
        for si in range(4):
            pct = np.mean(state_per_batch == si) * 100
            if pct > 0:
                # Build tick-level mask from batch-level state
                tick_mask = np.zeros(N, dtype=bool)
                for b_idx in np.where(state_per_batch == si)[0]:
                    t0 = b_idx * ticks_per_batch
                    t1 = min(t0 + ticks_per_batch, N)
                    tick_mask[t0:t1] = True
                pnl_s = fsm_pnl_tick[tick_mask].sum() if tick_mask.sum() > 0 else 0
                base_s = baseline_pnl_tick[tick_mask].sum() if tick_mask.sum() > 0 else 0
                print(f"  {state_names[si]:10s}: {pct:5.1f}%  "
                      f"FSM PnL={pnl_s:+,.1f}  Baseline={base_s:+,.1f}")

        self._compute_metrics_tick_level(N, n_batches)
        return self

    def run(
        self,
        mid_px: np.ndarray,        # (N,) raw mid prices
        spread: np.ndarray,        # (N,) raw spreads
        toxicity_scores: np.ndarray,  # (B,) toxicity per batch (upsampled to N ticks)
        future_ret: Optional[np.ndarray] = None,  # (N,) forward return per tick
    ) -> "PnLBacktest":
        """
        Run the PnL simulation at tick level.

        Model: each tick, the MM earns spread/2 but faces adverse selection
        cost proportional to future absolute mid-price movement.

        When toxicity is flagged, quotes are withdrawn → 0 PnL for that tick.

        Parameters
        ----------
        mid_px : (N,) — raw mid-price series (tick level)
        spread : (N,) — raw spread series (tick level)
        toxicity_scores : (B,) — toxicity per batch
        future_ret : (N,) optional — precomputed forward return per tick
        """
        cfg = self.config
        N = len(mid_px)
        n_batches = len(toxicity_scores)

        print(f"\n{'─'*60}")
        print(f"  PnL Backtest — MM with Toxicity Circuit Breaker")
        print(f"{'─'*60}")
        print(f"  Ticks:          {N:,}")
        print(f"  Batches:        {n_batches}")
        print(f"  Spread capt:    {cfg.spread_capture_frac:.0%}")
        print(f"  Tox thresh:     {cfg.tox_percentile_threshold:.0f}%ile")

        # Per-tick forward return (adverse selection proxy)
        if future_ret is None:
            # Default: 1-tick forward return
            future_ret = np.zeros(N, dtype=np.float64)
            future_ret[:-1] = (
                (mid_px[1:] - mid_px[:-1]) / (np.abs(mid_px[:-1]) + 1e-12)
            )

        # Adverse selection cost per tick: |forward_return| * mid_px
        # This is what you lose if you get picked off
        adverse_cost = np.abs(future_ret) * mid_px

        # Spread earned per tick
        spread_earned = spread * cfg.spread_capture_frac

        # Toxicity threshold
        tox_threshold = np.percentile(toxicity_scores, cfg.tox_percentile_threshold)
        print(f"  Tox thresh:     {tox_threshold:.2f}")

        # Upsample toxicity to tick level: each batch → BATCH_SIZE ticks
        # (caller provides batch-level toxicity; we expand to tick level)
        ticks_per_batch = N // n_batches if n_batches > 0 else 1

        # Per-tick PnL
        baseline_pnl = np.zeros(N, dtype=np.float64)
        toxicity_pnl = np.zeros(N, dtype=np.float64)
        toxicity_flags_tick = np.zeros(N, dtype=np.int32)

        for t in range(N):
            b = min(t // ticks_per_batch, n_batches - 1)
            is_toxic = toxicity_scores[b] > tox_threshold

            # Baseline: always quote
            baseline_pnl[t] = spread_earned[t] - adverse_cost[t]

            # Toxicity-aware: withdraw when toxic
            if is_toxic:
                toxicity_pnl[t] = 0.0  # no spread, no risk
                toxicity_flags_tick[t] = 1
            else:
                toxicity_pnl[t] = spread_earned[t] - adverse_cost[t]

        # Cumulative equity
        self.baseline_equity = np.cumsum(baseline_pnl)
        self.toxicity_equity = np.cumsum(toxicity_pnl)
        self.baseline_pnl = baseline_pnl
        self.toxicity_pnl = toxicity_pnl
        self.toxicity_flags = toxicity_flags_tick

        # Compute metrics
        self._compute_metrics_tick_level(N, n_batches)
        return self

    # ── Metrics ──────────────────────────────────────────────────────

    def _compute_metrics_tick_level(self, N: int, n_batches: int):
        """Compute all comparison metrics."""
        base_eq = self.baseline_equity
        tox_eq = self.toxicity_equity
        base_pnl = self.baseline_pnl
        tox_pnl = self.toxicity_pnl

        # Total PnL
        base_total = float(base_eq[-1])
        tox_total = float(tox_eq[-1])
        pnl_improvement = tox_total - base_total
        pnl_improvement_pct = (pnl_improvement / max(abs(base_total), 1e-12)) * 100 if base_total != 0 else float('inf')

        # Per-tick mean and std
        base_mean = float(np.mean(base_pnl))
        tox_mean = float(np.mean(tox_pnl))
        base_std = float(np.std(base_pnl))
        tox_std = float(np.std(tox_pnl))

        # Sharpe (per tick, then annualized to ~ daily)
        # Assume ~100K ticks per day for BTC
        ticks_per_day = 100_000
        trading_days = len(base_pnl) / ticks_per_day

        base_sharpe = (base_mean / max(base_std, 1e-12)) * np.sqrt(ticks_per_day) if base_std > 0 else 0.0
        tox_sharpe = (tox_mean / max(tox_std, 1e-12)) * np.sqrt(ticks_per_day) if tox_std > 0 else 0.0

        # Max drawdown
        base_dd = self._max_drawdown(base_eq)
        tox_dd = self._max_drawdown(tox_eq)

        # Win rate
        base_win = float(np.mean(base_pnl > 0))
        tox_win = float(np.mean(tox_pnl > 0))

        # Toxic tick statistics
        n_toxic = int(np.sum(self.toxicity_flags))
        toxic_pct = n_toxic / max(N, 1) * 100

        # PnL during toxic ticks
        toxic_mask = self.toxicity_flags > 0
        toxic_pnl_baseline = float(np.sum(base_pnl[toxic_mask]))
        toxic_pnl_withdrawn = float(np.sum(tox_pnl[toxic_mask]))

        self.metrics_ = {
            "baseline_total_pnl": base_total,
            "toxicity_total_pnl": tox_total,
            "pnl_improvement": pnl_improvement,
            "pnl_improvement_pct": pnl_improvement_pct,

            "baseline_sharpe": base_sharpe,
            "toxicity_sharpe": tox_sharpe,
            "sharpe_improvement": tox_sharpe - base_sharpe,

            "baseline_max_drawdown": base_dd,
            "toxicity_max_drawdown": tox_dd,
            "drawdown_improvement": base_dd - tox_dd,

            "baseline_win_rate": base_win,
            "toxicity_win_rate": tox_win,

            "n_ticks": N,
            "n_toxic_ticks": n_toxic,
            "toxic_tick_pct": toxic_pct,
            "toxic_baseline_pnl": float(toxic_pnl_baseline),
            "toxic_withdrawn_pnl": float(toxic_pnl_withdrawn),
            "toxic_savings": float(toxic_pnl_baseline - toxic_pnl_withdrawn),
        }

    # ── Report ───────────────────────────────────────────────────────

    def report(self) -> dict:
        """Print comparison report and return metrics dict."""
        m = self.metrics_
        if not m:
            print("No results. Call run() first.")
            return {}

        print(f"\n{'═'*60}")
        print(f"  PnL Comparison Report")
        print(f"{'═'*60}")

        print(f"\n  Total PnL:")
        print(f"    Baseline:       {m['baseline_total_pnl']:+,.2f}")
        print(f"    Toxicity-aware: {m['toxicity_total_pnl']:+,.2f}")
        print(f"    Improvement:    {m['pnl_improvement']:+,.2f}"
              f"  ({m['pnl_improvement_pct']:+.1f}%)")

        print(f"\n  Sharpe Ratio (annualized ~daily equiv):")
        print(f"    Baseline:       {m['baseline_sharpe']:.4f}")
        print(f"    Toxicity-aware: {m['toxicity_sharpe']:.4f}")
        print(f"    Improvement:    {m['sharpe_improvement']:+.4f}")

        print(f"\n  Max Drawdown:")
        print(f"    Baseline:       {m['baseline_max_drawdown']:+,.2f}")
        print(f"    Toxicity-aware: {m['toxicity_max_drawdown']:+,.2f}")
        print(f"    Improvement:    {m['drawdown_improvement']:+,.2f}")

        print(f"\n  Win Rate:")
        print(f"    Baseline:       {m['baseline_win_rate']:.2%}")
        print(f"    Toxicity-aware: {m['toxicity_win_rate']:.2%}")

        print(f"\n  Toxicity Circuit Breaker:")
        print(f"    Toxic ticks:    {m['n_toxic_ticks']:,}/{m['n_ticks']:,}"
              f" ({m['toxic_tick_pct']:.1f}%)")
        print(f"    PnL during toxic ticks (baseline):  {m['toxic_baseline_pnl']:+,.2f}")
        print(f"    PnL during toxic ticks (withdrawn): {m['toxic_withdrawn_pnl']:+,.2f}")
        print(f"    Savings from withdrawal:          {m['toxic_savings']:+,.2f}")

        print(f"{'═'*60}\n")
        return m

    # ── Utility ──────────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        """Compute maximum drawdown from equity curve."""
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / np.maximum(np.abs(running_max), 1e-12)
        return float(np.max(drawdowns))

    def summary(self) -> dict:
        return self.metrics_

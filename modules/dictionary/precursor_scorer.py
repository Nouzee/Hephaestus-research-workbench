"""
Precursor Scorer — Multi-Signal Shock Precursor Layer

Replaces single-signal Gram toxicity with a fused precursor score
combining four market-microstructure shock indicators:

  S1 — Depth Evaporation:  how fast is liquidity vanishing?
  S2 — OBI Impulse:        is one-sided order flow spiking?
  S3 — Spread Shock:       is the spread suddenly widening?
  S4 — Event Burst:        is tick frequency suddenly surging?
  S5 — Gram Acceleration:  is atom co-activation structure accelerating? (aux)

Each signal is z-scored over a rolling baseline window, then fused:
  Score_t = sum_i w_i * max(z_i(t), 0)   (only positive shocks matter)

Two-tier circuit breaker:
  WARNING — score > 1.5 sigma in >= 2 signals simultaneously
  HARD    — score > 2.5 sigma AND persists for >= 2 consecutive windows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class PrecursorConfig:
    """Configuration for the multi-signal precursor scorer."""

    # Rolling z-score baseline window (batches)
    baseline_window: int = 50

    # Signal weights (rule-based, sum does not need to be 1)
    w_depth: float = 1.0
    w_obi: float = 1.0
    w_spread: float = 1.0
    w_burst: float = 0.5
    w_gram_accel: float = 0.3   # auxiliary only

    # Two-tier thresholds (in z-score sigma)
    warn_sigma: float = 1.5        # single-signal warning
    hard_sigma: float = 2.5        # hard breaker threshold
    hard_persistence: int = 2      # consecutive windows for hard breaker

    seed: int = 42


# ===========================================================================
# Core: Precursor Scorer
# ===========================================================================

class PrecursorScorer:
    """
    Multi-signal shock precursor with two-tier circuit breaker.

    Usage
    -----
    >>> ps = PrecursorScorer()
    >>> for batch_data in stream:
    ...     score, tier = ps.score_batch(raw_signals, gram_v)
    ...     if tier == "HARD":
    ...         withdraw_all_quotes()
    """

    def __init__(self, config: Optional[PrecursorConfig] = None):
        self.config = config or PrecursorConfig()
        cfg = self.config

        # Rolling history for z-score computation
        self.n_signals = 5  # depth, obi, spread, burst, gram_accel
        self.history = np.zeros((cfg.baseline_window, self.n_signals), dtype=np.float32)
        self.history_ptr = 0
        self.history_full = False

        # Per-signal z-score time series
        self.z_scores: list[np.ndarray] = []

        # Fused score time series
        self.scores: list[float] = []
        self.tiers: list[str] = []  # "NORMAL", "WARN", "HARD"

        # Persistence tracker for hard breaker
        self.consecutive_above_hard = 0

    # ------------------------------------------------------------------
    # Per-batch signal extraction
    # ------------------------------------------------------------------

    def extract_signals(
        self,
        depth_raw: np.ndarray,         # (T,) raw total_depth
        signed_imb_raw: np.ndarray,    # (T,) raw signed_imbalance
        spread_raw: np.ndarray,        # (T,) raw spread
        duration_raw: np.ndarray,      # (T,) raw duration_ms
        gram_v: float = 0.0,           # Gram velocity (from GramTracker)
    ) -> np.ndarray:
        """
        Extract five shock indicators from a batch of raw tick data.

        All signals are designed so that POSITIVE = dangerous (shock direction).

        Returns (5,) array: [depth_evap, obi_impulse, spread_shock, event_burst, gram_accel]
        """
        eps = 1e-12

        # S1: Depth evaporation velocity
        # Negative depth change = liquidity vanishing
        # We take -roc so that positive = evaporating
        if len(depth_raw) > 5:
            depth_start = np.median(depth_raw[:10])
            depth_end = np.median(depth_raw[-10:])
            depth_evap = -(depth_end - depth_start) / max(depth_start, eps)
            depth_evap = max(depth_evap, 0.0)  # only evaporation, not accumulation
        else:
            depth_evap = 0.0

        # S2: OBI impulse (spike in signed imbalance magnitude)
        obi_abs = np.abs(signed_imb_raw)
        obi_mean = np.mean(obi_abs)
        obi_max = np.percentile(obi_abs, 95)
        # Impulse = how much the tail exceeds the mean (spike ratio)
        obi_impulse = obi_max / max(obi_mean, eps) if obi_mean > 1e-8 else 0.0
        # Normalize: typical ratio is ~3-5x, scale down
        obi_impulse = np.log1p(obi_impulse)  # log-scale for stability

        # S3: Spread shock (widening)
        spread_mean = np.mean(spread_raw)
        spread_max = np.max(spread_raw)
        spread_shock = spread_max / max(spread_mean, eps) if spread_mean > 1e-8 else 0.0
        spread_shock = np.log1p(spread_shock)

        # S4: Event burst (duration collapse = surge in activity)
        if len(duration_raw) > 5:
            dur_median = np.median(duration_raw)
            dur_min = np.percentile(duration_raw, 5)
            # Shorter duration = burst. Invert so positive = burst.
            event_burst = dur_median / max(dur_min, eps) if dur_min > 1e-8 else 0.0
            event_burst = np.log1p(event_burst)
        else:
            event_burst = 0.0

        # S5: Gram acceleration (auxiliary)
        gram_accel = max(gram_v, 0.0)  # only positive velocity

        return np.array([depth_evap, obi_impulse, spread_shock, event_burst, gram_accel],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Score a batch
    # ------------------------------------------------------------------

    def score_batch(self, signals: np.ndarray) -> tuple[float, str]:
        """
        Compute z-scored fused precursor score and circuit-breaker tier.

        Parameters
        ----------
        signals : (5,) array from extract_signals()

        Returns
        -------
        score : float — fused precursor score
        tier  : str   — "NORMAL", "WARN", or "HARD"
        """
        cfg = self.config

        # Update rolling history
        self.history[self.history_ptr] = signals
        self.history_ptr = (self.history_ptr + 1) % cfg.baseline_window
        if self.history_ptr == 0:
            self.history_full = True

        if not self.history_full:
            # Not enough history for z-score
            self.scores.append(0.0)
            self.tiers.append("NORMAL")
            self.z_scores.append(np.zeros(self.n_signals, dtype=np.float32))
            return 0.0, "NORMAL"

        # Compute per-signal z-scores
        hist_mean = self.history.mean(axis=0)
        hist_std = self.history.std(axis=0)
        hist_std = np.maximum(hist_std, 1e-8)

        z = (signals - hist_mean) / hist_std
        z_pos = np.maximum(z, 0.0)  # only positive shocks

        self.z_scores.append(z.astype(np.float32))

        # Weighted fusion
        weights = np.array([
            cfg.w_depth, cfg.w_obi, cfg.w_spread, cfg.w_burst, cfg.w_gram_accel
        ])
        score = float(np.dot(z_pos, weights))
        self.scores.append(score)

        # Two-tier circuit breaker
        n_signals_above_warn = int(np.sum(z > cfg.warn_sigma))
        n_signals_above_hard = int(np.sum(z > cfg.hard_sigma))

        if n_signals_above_hard >= 1:
            self.consecutive_above_hard += 1
        else:
            self.consecutive_above_hard = 0

        if self.consecutive_above_hard >= cfg.hard_persistence:
            tier = "HARD"
        elif n_signals_above_warn >= 2:
            tier = "WARN"
        else:
            tier = "NORMAL"

        self.tiers.append(tier)
        return score, tier

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate statistics."""
        if not self.scores:
            return {"n_scores": 0}

        scores_arr = np.array(self.scores)
        tiers_arr = np.array(self.tiers)

        n_total = len(tiers_arr)
        n_warn = int(np.sum(tiers_arr == "WARN"))
        n_hard = int(np.sum(tiers_arr == "HARD"))

        # Per-signal contribution to score (average z_pos × weight)
        if self.z_scores:
            z_stacked = np.array(self.z_scores)  # (N, 5)
            z_pos_mean = np.mean(np.maximum(z_stacked, 0), axis=0)
        else:
            z_pos_mean = np.zeros(5)

        return {
            "n_scores": len(self.scores),
            "score_mean": float(np.mean(scores_arr)),
            "score_std": float(np.std(scores_arr)),
            "score_p95": float(np.percentile(scores_arr, 95)),
            "warn_rate": n_warn / max(n_total, 1),
            "hard_rate": n_hard / max(n_total, 1),
            "signal_contributions": {
                "depth_evap": float(z_pos_mean[0]),
                "obi_impulse": float(z_pos_mean[1]),
                "spread_shock": float(z_pos_mean[2]),
                "event_burst": float(z_pos_mean[3]),
                "gram_accel": float(z_pos_mean[4]),
            },
        }

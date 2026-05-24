"""
Toxicity Scorer V5 — Causal-Aligned Multi-Signal Precursor Fusion

Replaces static Gram-deviation toxicity with causally-aligned multi-signal
precursor detection. Each signal is z-scored over a rolling baseline, then
shifted by its optimal lead (from causal_alignment.json) before fusion.

Fusion formula:
    Score_t = sum_i w_i * max(z_i(t - lead_i), 0)

Only positive shocks (signal > baseline) contribute. Negative deviations
(quieter than normal) are ignored.

Two-tier output:
    - WARN:  any predictive signal > 1.5σ  OR  score > P90
    - HARD:  >=2 predictive signals > 2.5σ  OR  score > P95 AND persistent

State machine integration:
    - Connects to risk/state_machine.py for NORMAL/WATCH/HARD_ALERT/RECOVERY
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ToxicityScorerConfig:
    """Configuration for V5 multi-signal toxicity scorer."""

    # Lag config path (from causal_alignment.py)
    lag_config_path: str = ""

    # Rolling z-score baseline (batches)
    baseline_window: int = 50

    # Signal weights (must match keys in lag_config)
    weights: dict = field(default_factory=lambda: {
        "depth_evap": 1.0,
        "obi_impulse": 1.0,
        "spread_shock": 0.5,     # diagnostic only
        "cancel_burst": 0.3,     # diagnostic only
        "gram_aux": 0.3,         # auxiliary
    })

    # Thresholds (z-score sigma)
    warn_sigma: float = 1.5
    hard_sigma: float = 2.5

    # Persistence for HARD trigger
    hard_persistence: int = 2

    # Score percentile thresholds
    score_warn_percentile: float = 90.0
    score_hard_percentile: float = 95.0

    seed: int = 42

    def __post_init__(self):
        if not self.lag_config_path:
            self.lag_config_path = str(
                Path(__file__).resolve().parent / "cache" / "causal_alignment.json"
            )


# ===========================================================================
# Core: Toxicity Scorer V5
# ===========================================================================

class ToxicityScorer:
    """
    Causal-aligned multi-signal precursor scorer.

    Usage
    -----
    >>> ts = ToxicityScorer()
    >>> for batch in stream:
    ...     result = ts.score(raw_signals, gram_v)
    ...     if result["hard_trigger"]:
    ...         withdraw_all_quotes()
    """

    def __init__(self, config: Optional[ToxicityScorerConfig] = None):
        self.config = config or ToxicityScorerConfig()
        cfg = self.config

        # Load lag configuration
        self.lag_config = self._load_lag_config()

        # Signal names in canonical order
        self.signal_names = ["depth_evap", "obi_impulse", "spread_shock",
                             "cancel_burst", "gram_aux"]
        self.n_signals = len(self.signal_names)

        # Per-signal leads
        self.leads = {}
        self.roles = {}
        for name in self.signal_names:
            if name in self.lag_config.get("alignment", {}):
                self.leads[name] = self.lag_config["alignment"][name]["optimal_lead"]
                self.roles[name] = self.lag_config["alignment"][name].get("role", "predictive")
            else:
                self.leads[name] = 0
                self.roles[name] = "diagnostic"

        # Rolling history for z-score computation
        self.history = np.zeros((cfg.baseline_window, self.n_signals), dtype=np.float32)
        self.history_ptr = 0
        self.history_full = False

        # Lead buffers: store past signal values for alignment
        max_lead = max(self.leads.values()) if self.leads else 0
        self.lead_buffers = {
            name: np.zeros(max(lead, 1), dtype=np.float32)
            for name, lead in self.leads.items()
        }

        # Output time series
        self.scores: list[float] = []
        self.warn_triggers: list[bool] = []
        self.hard_triggers: list[bool] = []
        self.feature_breakdowns: list[dict] = []
        self.states: list[str] = []

        # Persistence tracker
        self.consecutive_above_hard = 0
        self.in_recovery = False
        self.recovery_counter = 0

    # ------------------------------------------------------------------
    # Load lag config
    # ------------------------------------------------------------------

    def _load_lag_config(self) -> dict:
        path = Path(self.config.lag_config_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        # Fallback: zero leads, all equal weight
        print("[ToxicityScorer] No lag config found, using zero leads.")
        return {"alignment": {}, "weights": {}}

    # ------------------------------------------------------------------
    # Main scoring interface
    # ------------------------------------------------------------------

    def score(
        self,
        signals: dict,      # {"depth_evap": float, "obi_impulse": float, ...}
        gram_v: float = 0.0,
    ) -> dict:
        """
        Compute causally-aligned multi-signal toxicity score.

        Parameters
        ----------
        signals : dict with keys matching self.signal_names, each a float scalar
        gram_v  : Gram velocity (for gram_aux signal)

        Returns
        -------
        dict with keys: score, warn_trigger, hard_trigger, feature_breakdown, state
        """
        cfg = self.config

        # Build raw signal vector in canonical order
        raw = np.zeros(self.n_signals, dtype=np.float32)
        for i, name in enumerate(self.signal_names):
            if name == "gram_aux":
                raw[i] = float(gram_v)
            else:
                raw[i] = float(signals.get(name, 0.0))

        # Update lead buffers
        for name, lead in self.leads.items():
            if lead > 0:
                buf = self.lead_buffers[name]
                # Shift and insert new value
                buf[:-1] = buf[1:]
                buf[-1] = raw[self.signal_names.index(name)]

        # Build aligned signal vector for z-scoring
        aligned = np.zeros(self.n_signals, dtype=np.float32)
        for i, name in enumerate(self.signal_names):
            lead = self.leads.get(name, 0)
            if lead > 0 and lead <= len(self.lead_buffers[name]):
                # Use lagged value: signal[t - lead]
                aligned[i] = self.lead_buffers[name][-lead]
            else:
                aligned[i] = raw[i]

        # Update rolling history
        self.history[self.history_ptr] = aligned
        self.history_ptr = (self.history_ptr + 1) % cfg.baseline_window
        if self.history_ptr == 0:
            self.history_full = True

        # If not enough history, return zero
        if not self.history_full:
            result = {
                "score": 0.0,
                "warn_trigger": False,
                "hard_trigger": False,
                "feature_breakdown": {name: 0.0 for name in self.signal_names},
                "state": "NORMAL",
            }
            self._store_result(result)
            return result

        # Compute z-scores
        hist_mean = self.history.mean(axis=0)
        hist_std = self.history.std(axis=0)
        hist_std = np.maximum(hist_std, 1e-8)
        z = (aligned - hist_mean) / hist_std

        # Per-signal contribution (only positive = dangerous direction)
        z_pos = np.maximum(z, 0.0)

        # Build feature breakdown
        breakdown = {}
        for i, name in enumerate(self.signal_names):
            breakdown[name] = {
                "raw": float(raw[i]),
                "aligned": float(aligned[i]),
                "z_score": float(z[i]),
                "z_pos": float(z_pos[i]),
                "role": self.roles.get(name, "unknown"),
            }

        # Fused score (weighted sum of positive z-scores)
        weights_arr = np.array([
            cfg.weights.get(name, 1.0) for name in self.signal_names
        ])
        score = float(np.dot(z_pos, weights_arr))

        # Two-tier triggers
        n_predictive_above_warn = 0
        n_predictive_above_hard = 0
        for i, name in enumerate(self.signal_names):
            if self.roles.get(name) == "predictive":
                if z[i] > cfg.warn_sigma:
                    n_predictive_above_warn += 1
                if z[i] > cfg.hard_sigma:
                    n_predictive_above_hard += 1

        # Score-based thresholds
        score_warn = False
        score_hard = False
        if len(self.scores) > cfg.baseline_window:
            scores_arr = np.array(self.scores[-cfg.baseline_window:])
            p90 = np.percentile(scores_arr, cfg.score_warn_percentile)
            p95 = np.percentile(scores_arr, cfg.score_hard_percentile)
            score_warn = score > p90
            score_hard = score > p95

        # HARD persistence
        if n_predictive_above_hard >= 2 or score_hard:
            self.consecutive_above_hard += 1
        else:
            self.consecutive_above_hard = 0

        hard_trigger = self.consecutive_above_hard >= cfg.hard_persistence
        warn_trigger = (n_predictive_above_warn >= 1 or score_warn) and not hard_trigger

        # State determination
        if hard_trigger:
            state = "HARD_ALERT"
            self.in_recovery = False
        elif warn_trigger:
            state = "WATCH"
            self.in_recovery = False
        elif self.in_recovery or self.consecutive_above_hard > 0:
            state = "RECOVERY"
            self.in_recovery = True
        else:
            state = "NORMAL"
            self.in_recovery = False

        # Build result
        result = {
            "score": score,
            "warn_trigger": warn_trigger,
            "hard_trigger": hard_trigger,
            "feature_breakdown": breakdown,
            "state": state,
        }
        self._store_result(result)
        return result

    # ------------------------------------------------------------------
    # Internal storage
    # ------------------------------------------------------------------

    def _store_result(self, result: dict):
        self.scores.append(result["score"])
        self.warn_triggers.append(result["warn_trigger"])
        self.hard_triggers.append(result["hard_trigger"])
        self.states.append(result["state"])
        self.feature_breakdowns.append(result["feature_breakdown"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self.scores:
            return {"n_scores": 0}

        scores_arr = np.array(self.scores)
        n = len(scores_arr)
        states_arr = np.array(self.states)

        state_counts = {}
        for s in ["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]:
            state_counts[s] = float(np.mean(states_arr == s))

        return {
            "n_scores": n,
            "score_mean": float(np.mean(scores_arr)),
            "score_std": float(np.std(scores_arr)),
            "score_p95": float(np.percentile(scores_arr, 95)),
            "warn_rate": float(np.mean(self.warn_triggers)),
            "hard_rate": float(np.mean(self.hard_triggers)),
            "state_distribution": state_counts,
        }

"""
HMM Risk Scaling Layer — capital allocation via regime multiplier.

Fits a Gaussian HMM on market structure features (volatility, depth, spread)
and maps discovered regimes to position-size multipliers.

CRITICAL CONSTRAINT: HMM does NOT touch alpha, direction, entry, or exit.
It ONLY answers: "how much capital should I risk right now?"

Regime → multiplier mapping (learned from state statistics):
  low_vol      → 1.2  (aggressive — market is calm, spread is stable)
  mid_vol      → 1.0  (normal)
  high_vol     → 0.6  (defensive — elevated volatility)
  liquidation  → 0.2  (survival — extreme conditions, near-flat)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from hmmlearn import hmm


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class HMMScalerConfig:
    """Configuration for HMM-based position scaling."""

    n_states: int = 4              # low_vol, mid_vol, high_vol, liquidation
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42

    # Default multiplier mapping (overridden by fit)
    default_multipliers: tuple = (1.2, 1.0, 0.6, 0.2)

    # Feature extraction windows
    feature_window: int = 50       # batches for rolling z-score


# ===========================================================================
# Core: HMM Scaler
# ===========================================================================

class HMMScaler:
    """
    HMM-based position sizing. Reads market structure, outputs capital multiplier.

    Usage
    -----
    >>> scaler = HMMScaler()
    >>> scaler.fit(features)         # (N, 3) — vol, depth, spread
    >>> mult = scaler.predict(batch_features)  # float in [0.2, 1.2]
    >>> position = base_K * mult * pressure_direction
    """

    def __init__(self, config: Optional[HMMScalerConfig] = None):
        self.config = config or HMMScalerConfig()
        self.cfg = self.config

        self.model_: Optional[hmm.GaussianHMM] = None
        self.is_fitted_ = False
        self.state_multipliers_: Optional[np.ndarray] = None
        self.state_labels_: dict = {}

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray) -> "HMMScaler":
        """
        Fit HMM on market structure features.

        Parameters
        ----------
        features : (N, D) — columns: [volatility_z, depth_z, spread_z]
        """
        cfg = self.cfg
        N, D = features.shape

        print(f"[HMM Scaler] Fitting {cfg.n_states}-state HMM on {N:,} samples ...")
        t0 = time.perf_counter()

        self.model_ = hmm.GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            tol=cfg.tol,
            random_state=cfg.random_state,
        )
        self.model_.fit(features)

        # Predict states
        states = self.model_.predict(features)

        # Map states to multipliers based on state statistics
        # Higher vol → lower multiplier
        state_vol = np.array([
            np.mean(features[states == s, 0]) if np.sum(states == s) > 0 else 0
            for s in range(cfg.n_states)
        ])

        # Sort states by volatility: lowest vol → highest multiplier
        vol_order = np.argsort(state_vol)
        multipliers = np.array(cfg.default_multipliers)

        # Assign multipliers in vol order
        self.state_multipliers_ = np.zeros(cfg.n_states)
        for rank, state_idx in enumerate(vol_order):
            self.state_multipliers_[state_idx] = multipliers[rank]

        # Label states
        state_pct = np.array([
            np.mean(states == s) * 100 for s in range(cfg.n_states)
        ])
        for s in range(cfg.n_states):
            vol_mean = state_vol[s]
            mult = self.state_multipliers_[s]
            pct = state_pct[s]
            self.state_labels_[s] = {
                "vol_mean": float(vol_mean),
                "multiplier": float(mult),
                "frequency": float(pct),
            }

        self.is_fitted_ = True

        print(f"  Fitted in {time.perf_counter()-t0:.1f}s")
        for s in range(cfg.n_states):
            lbl = self.state_labels_[s]
            print(f"    State {s}: mult={lbl['multiplier']:.1f}x  "
                  f"vol={lbl['vol_mean']:.4f}  freq={lbl['frequency']:.1f}%")

        return self

    # ── Predict ──────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime multiplier for each sample.

        Returns (N,) array of multipliers in [0.2, 1.2].
        """
        if not self.is_fitted_:
            return np.ones(len(features))

        states = self.model_.predict(features)
        return self.state_multipliers_[states]

    def predict_single(self, features: np.ndarray) -> float:
        """Predict multiplier for a single sample (3,) array."""
        return float(self.predict(features.reshape(1, -1))[0])

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.is_fitted_:
            return {"fitted": False}
        return {
            "fitted": True,
            "n_states": self.cfg.n_states,
            "state_labels": self.state_labels_,
            "transition_matrix": self.model_.transmat_.tolist(),
        }

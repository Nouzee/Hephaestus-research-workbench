"""
Gram Tracker — 原子拓扑关系追踪

Phase 3 of the toxicity detection pipeline.

Tracks two types of Gram matrices over sliding time windows:
  1. D-Gram: G_D = D @ D^T          — atom dictionary similarity (structural)
  2. Alpha-Gram: G_α = (1/N) α^T α  — atom co-activation pattern (behavioral)

Key insight:
  A sudden shift in G_α without a corresponding shift in G_D suggests
  the market microstructure is being activated in an unusual way —
  this is the early signal for toxic flow / regime transition.

Also tracks:
  - Gram trace, determinant (volume of atom space), condition number
  - Pairwise cosine drift from historical baseline
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GramTrackerConfig:
    """Configuration for Gram matrix tracking."""

    window_size: int = 200          # batches of alpha history to track
    alpha_gram_smooth: float = 0.95 # EMA smoothing for current G_α estimate
    alert_threshold_sigma: float = 2.5  # Mahalanobis σ for Gram anomaly flag
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# Core: Gram Tracker
# ═══════════════════════════════════════════════════════════════════════

class GramTracker:
    """
    Sliding-window tracker for atom co-activation Gram matrices.

    Usage
    -----
    >>> gt = GramTracker()
    >>> for alpha_batch in stream:
    ...     gt.update(alpha_batch, D_current)
    ...     if gt.is_anomalous():
    ...         print("Toxic flow detected!")
    """

    def __init__(self, config: Optional[GramTrackerConfig] = None):
        self.config = config or GramTrackerConfig()
        cfg = self.config

        self.K: Optional[int] = None

        # D-Gram (current snapshot)
        self.G_D: Optional[np.ndarray] = None          # K × K

        # Alpha-Gram: EMA estimate
        self.G_alpha: Optional[np.ndarray] = None       # K × K
        self.G_alpha_smooth = cfg.alpha_gram_smooth

        # Rolling history of G_alpha for baseline estimation
        self._gram_history = deque(maxlen=cfg.window_size)

        # Baseline statistics (computed over rolling window)
        self.G_alpha_mean: Optional[np.ndarray] = None  # K × K
        self.G_alpha_std: Optional[np.ndarray] = None   # K × K (element-wise)

        # Trace / determinant history for quick diagnostics
        self.trace_history: list[float] = []
        self.det_history: list[float] = []
        self.frobenius_history: list[float] = []

        self.n_updates = 0

    # ── Update ───────────────────────────────────────────────────────

    def update(self, alpha_batch: np.ndarray, D: Optional[np.ndarray] = None):
        """
        Update Gram estimates with a new batch of coefficients.

        Parameters
        ----------
        alpha_batch : (B, K) — sparse coefficients for one batch
        D : (K, M) optional — current dictionary for D-Gram update
        """
        B, K = alpha_batch.shape
        if self.K is None:
            self.K = K

        # Alpha Gram: G_α = (1/B) α^T α  → average co-activation per sample
        G_alpha_batch = (alpha_batch.T @ alpha_batch) / B  # K × K

        # EMA update
        if self.G_alpha is None:
            self.G_alpha = G_alpha_batch
        else:
            rho = self.config.alpha_gram_smooth
            self.G_alpha = rho * self.G_alpha + (1 - rho) * G_alpha_batch

        # Rolling history
        self._gram_history.append(G_alpha_batch)

        # D-Gram
        if D is not None:
            self.G_D = D @ D.T  # K × K

        # Update baseline from rolling window
        self._update_baseline()

        # Scalar diagnostics
        self.trace_history.append(float(np.trace(self.G_alpha)))
        if K > 1:
            self.det_history.append(float(np.linalg.det(self.G_alpha)))
        self.frobenius_history.append(float(np.linalg.norm(self.G_alpha, ord='fro')))

        self.n_updates += 1

    # ── Baseline estimation ──────────────────────────────────────────

    def _update_baseline(self):
        """Compute mean and element-wise std of G_alpha over the rolling window."""
        if len(self._gram_history) < 20:
            return  # not enough data

        grams = np.array(self._gram_history)  # (W, K, K)
        self.G_alpha_mean = np.mean(grams, axis=0)
        self.G_alpha_std = np.std(grams, axis=0, ddof=1)
        self.G_alpha_std = np.maximum(self.G_alpha_std, 1e-8)  # avoid div-by-zero

    # ── Anomaly detection ────────────────────────────────────────────

    def mahalanobis_distance(self) -> float:
        """
        Element-wise Mahalanobis distance of current G_alpha from the
        rolling-window baseline distribution.

        Returns ∞ if baseline not yet established.
        """
        if self.G_alpha is None or self.G_alpha_mean is None:
            return float('inf')

        diff = self.G_alpha - self.G_alpha_mean  # K × K
        # Element-wise z-score, then Frobenius norm
        z_scores = diff / self.G_alpha_std
        return float(np.linalg.norm(z_scores, ord='fro'))

    def is_anomalous(self) -> bool:
        """
        True if current G_alpha deviates significantly from baseline.

        Uses Mahalanobis distance with σ threshold.
        """
        if len(self._gram_history) < 50:
            return False  # baseline too thin

        d_maha = self.mahalanobis_distance()
        # Normalize by sqrt(K*K) for scale-invariant threshold
        K = self.K
        d_normalized = d_maha / np.sqrt(K * K)
        return d_normalized > self.config.alert_threshold_sigma

    # ── Gram drift metrics ───────────────────────────────────────────

    def gram_drift(self) -> dict:
        """
        Compute drift metrics between current G_alpha and baseline.

        Returns
        -------
        dict with keys:
          frobenius_drift  — ||G - G_baseline||_F / ||G_baseline||_F
          cosine_drift     — 1 - cosine_similarity between vectorized Grams
          trace_drift      — |trace(G) - trace(G_baseline)| / |trace(G_baseline)|
        """
        if self.G_alpha is None or self.G_alpha_mean is None:
            return {"frobenius_drift": 0.0, "cosine_drift": 0.0, "trace_drift": 0.0}

        G = self.G_alpha
        G_base = self.G_alpha_mean

        # Frobenius drift
        fnorm_base = np.linalg.norm(G_base, ord='fro')
        frob_drift = float(np.linalg.norm(G - G_base, ord='fro') / max(fnorm_base, 1e-12))

        # Cosine drift
        g_flat = G.ravel()
        g_base_flat = G_base.ravel()
        cos_sim = np.dot(g_flat, g_base_flat) / max(
            np.linalg.norm(g_flat) * np.linalg.norm(g_base_flat), 1e-12
        )
        cos_drift = float(1.0 - cos_sim)

        # Trace drift
        trace_base = np.trace(G_base)
        trace_drift = float(np.abs(np.trace(G) - trace_base) / max(np.abs(trace_base), 1e-12))

        return {
            "frobenius_drift": frob_drift,
            "cosine_drift": cos_drift,
            "trace_drift": trace_drift,
        }

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Snapshot of current Gram state."""
        return {
            "n_updates": self.n_updates,
            "G_alpha_trace": float(np.trace(self.G_alpha)) if self.G_alpha is not None else None,
            "G_alpha_det": float(np.linalg.det(self.G_alpha)) if self.G_alpha is not None and self.K > 1 else None,
            "mahalanobis_d": self.mahalanobis_distance(),
            "is_anomalous": self.is_anomalous(),
            **self.gram_drift(),
        }

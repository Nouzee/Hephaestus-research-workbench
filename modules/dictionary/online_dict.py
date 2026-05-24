"""
Online Dictionary Learning — 动态在线字典学习 (Mairal et al. 2010)

Phase 2 of the toxicity detection pipeline.

Implements streaming mini-batch dictionary updates with:
  - Exponential forgetting factor (γ) for distribution-shift adaptation
  - Dead-atom detection & reinitialization (prevents dictionary collapse)
  - Per-atom usage tracking
  - Reconstruction error monitoring for regime-change detection

Math:
  X_t ≈ D_t @ α_t    (sparse coding per batch)
  A_t = γ·A_{t-1} + α_t^T @ α_t    (coefficient Gram, with forgetting)
  B_t = γ·B_{t-1} + X_t^T @ α_t    (cross-correlation, with forgetting)
  D_t[k] ← project_to_L2_ball( (b_k - D·a_k)/A_{kk} + d_k )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OnlineDictConfig:
    """Configuration for the online dictionary learner."""

    n_components: int = 3          # K
    alpha: float = 1.0             # L1 sparsity penalty (LASSO λ)
    gamma: float = 0.995           # forgetting factor (1.0 = no forgetting)
    max_lasso_iters: int = 100     # CD iterations per sparse coding step
    lasso_tol: float = 1e-5        # convergence tolerance for CD

    # Dead atom handling
    dead_atom_threshold: float = 0.005   # fraction of activations to be "alive"
    dead_atom_window: int = 500          # check window (batches)
    reinit_from_residual: bool = True    # reinit dead atoms from largest residual

    # Drift monitoring
    recon_spike_threshold: float = 3.0   # σ multiplier for recon-error spike alert

    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# Core: Online Dictionary Learner
# ═══════════════════════════════════════════════════════════════════════

class OnlineDictLearner:
    """
    Streaming dictionary learning with forgetting factor and dead-atom revival.

    Usage
    -----
    >>> odl = OnlineDictLearner(D_init=D_offline)
    >>> for batch in data_stream:
    ...     odl.partial_fit(batch)
    ...     print(odl.recon_error_trace[-1])
    >>> D_adapted = odl.D
    """

    def __init__(
        self,
        D_init: np.ndarray,
        config: Optional[OnlineDictConfig] = None,
    ):
        self.config = config or OnlineDictConfig()
        cfg = self.config

        self.K = D_init.shape[0]   # number of atoms
        self.M = D_init.shape[1]   # feature dimension

        # Dictionary: each column is one atom (M × K)
        self.D = D_init.T.copy().astype(np.float64)  # M × K

        # Sufficient statistics
        self.A = np.zeros((self.K, self.K), dtype=np.float64)  # K × K
        self.B = np.zeros((self.M, self.K), dtype=np.float64)  # M × K

        # Tracking
        self.n_batches = 0
        self.atom_usage = np.zeros(self.K, dtype=np.float64)     # cumulative activation count
        self.recon_error_trace: list[float] = []
        self.recon_mean = 0.0
        self.recon_std = 0.0
        self.atom_usage_history: list[np.ndarray] = []  # per-batch snapshots

    # ── Main streaming interface ─────────────────────────────────────

    def partial_fit(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Process one mini-batch: sparse code → update stats → update D.

        Parameters
        ----------
        X_batch : (B, M) np.ndarray

        Returns
        -------
        alphas : (B, K) sparse coefficient matrix
        """
        cfg = self.config
        B = len(X_batch)
        X_batch = X_batch.astype(np.float64)

        # Step 1: Sparse coding (per sample)
        alphas = np.zeros((B, self.K), dtype=np.float64)
        for i in range(B):
            alphas[i] = self._sparse_code(X_batch[i])

        # Step 2: Update sufficient statistics with forgetting
        self.A = cfg.gamma * self.A + alphas.T @ alphas
        self.B = cfg.gamma * self.B + X_batch.T @ alphas

        # Step 3: Block coordinate descent dictionary update
        for k in range(self.K):
            A_kk = self.A[k, k]
            if A_kk > 1e-12:
                # u_k = (b_k - D @ a_k) / A_kk + d_k
                u = (self.B[:, k] - self.D @ self.A[:, k]) / A_kk + self.D[:, k]
                norm = np.linalg.norm(u)
                if norm > 0:
                    self.D[:, k] = u / max(norm, 1.0)  # project onto L2 unit ball

        # Step 4: Track atom usage
        batch_usage = (np.abs(alphas) > 1e-8).sum(axis=0).astype(np.float64) / B
        self.atom_usage = cfg.gamma * self.atom_usage + (1 - cfg.gamma) * batch_usage
        self.atom_usage_history.append(batch_usage.copy())

        # Step 5: Reconstruction error
        X_recon = alphas @ self.D.T
        recon_err = np.mean((X_batch - X_recon) ** 2)
        self.recon_error_trace.append(float(recon_err))
        self._update_recon_stats(recon_err)

        self.n_batches += 1

        # Step 6: Dead atom check
        if self.n_batches > 0 and self.n_batches % cfg.dead_atom_window == 0:
            self._revive_dead_atoms(X_batch)

        return alphas.astype(np.float32)

    # ── Sparse coding: coordinate descent LASSO ──────────────────────

    def _sparse_code(self, x: np.ndarray) -> np.ndarray:
        """
        Solve min_α 0.5||x - Dα||² + λ||α||₁ via coordinate descent.

        For K=3, this converges in ~10-20 iterations.
        """
        cfg = self.config
        K = self.K
        alpha = np.zeros(K, dtype=np.float64)

        # Precompute column norms
        D_col_norms = np.sum(self.D ** 2, axis=0)  # (K,)

        for _ in range(cfg.max_lasso_iters):
            alpha_old = alpha.copy()

            for k in range(K):
                if D_col_norms[k] < 1e-12:
                    continue
                # Residual without contribution of atom k
                residual = x - self.D @ alpha + self.D[:, k] * alpha[k]
                # Soft-thresholded projection
                rho = np.dot(self.D[:, k], residual) / D_col_norms[k]
                alpha[k] = self._soft_threshold(rho, cfg.alpha / D_col_norms[k])

            if np.max(np.abs(alpha - alpha_old)) < cfg.lasso_tol:
                break

        return alpha

    @staticmethod
    def _soft_threshold(z: float, lam: float) -> float:
        """S(z, λ) = sign(z) · max(|z| - λ, 0)"""
        if z > lam:
            return z - lam
        elif z < -lam:
            return z + lam
        return 0.0

    # ── Dead atom revival ────────────────────────────────────────────

    def _revive_dead_atoms(self, X_recent: np.ndarray):
        """
        Reinitialize atoms that are rarely activated.
        Replaces the dead atom with a random sample from the recent batch,
        or with the direction of largest reconstruction residual.
        """
        cfg = self.config
        dead_mask = self.atom_usage < cfg.dead_atom_threshold

        if not dead_mask.any():
            return

        for k in np.where(dead_mask)[0]:
            if cfg.reinit_from_residual:
                # Compute residuals for recent batch
                alphas_batch = np.zeros((len(X_recent), self.K))
                for i in range(len(X_recent)):
                    alphas_batch[i] = self._sparse_code(X_recent[i])
                residuals = X_recent - alphas_batch @ self.D.T
                # Pick the sample with largest residual norm
                idx = np.argmax(np.sum(residuals ** 2, axis=1))
                new_atom = X_recent[idx].copy()
            else:
                # Random reinit
                idx = np.random.randint(0, len(X_recent))
                new_atom = X_recent[idx].copy()

            # Normalize
            norm = np.linalg.norm(new_atom)
            if norm > 1e-8:
                self.D[:, k] = new_atom / norm
            self.atom_usage[k] = 0.0  # reset usage counter

            print(f"  [OnlineDict] Revived dead atom {k} "
                  f"→ norm={np.linalg.norm(self.D[:, k]):.3f}")

    # ── Recon error statistics ───────────────────────────────────────

    def _update_recon_stats(self, err: float):
        """Online update of reconstruction error mean/std."""
        n = self.n_batches + 1
        delta = err - self.recon_mean
        self.recon_mean += delta / n
        self.recon_std = np.sqrt(
            ((n - 2) * self.recon_std ** 2 + delta * (err - self.recon_mean)) / (n - 1)
            if n > 1 else 0.0
        )

    @property
    def recon_spike(self) -> bool:
        """True if the most recent recon error is anomalously high."""
        if self.recon_std < 1e-12 or len(self.recon_error_trace) < 10:
            return False
        latest = self.recon_error_trace[-1]
        return (latest - self.recon_mean) > self.config.recon_spike_threshold * self.recon_std

    # ── D as (K, M) for compatibility ────────────────────────────────

    @property
    def D_km(self) -> np.ndarray:
        """Dictionary in (K × M) format (familiar from sklearn)."""
        return self.D.T

    @property
    def atom_usage_pct(self) -> np.ndarray:
        """Per-atom usage as percentage."""
        total = self.atom_usage.sum()
        if total < 1e-12:
            return np.zeros(self.K)
        return self.atom_usage / total * 100

    def summary(self) -> dict:
        """Return a dict summary of current learner state."""
        return {
            "n_batches": self.n_batches,
            "recon_error_latest": self.recon_error_trace[-1] if self.recon_error_trace else None,
            "recon_error_mean": float(self.recon_mean),
            "atom_usage_pct": self.atom_usage_pct.tolist(),
            "recon_spike": self.recon_spike,
        }

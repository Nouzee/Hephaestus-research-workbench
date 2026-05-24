"""
HMM Regime Classifier — 隐马尔可夫市场状态识别

Phase 4 of the toxicity detection pipeline.

Uses a Gaussian HMM on the atom coefficient space (α ∈ ℝ^K) to:
  1. Discover latent market regimes from historical tick data
  2. Estimate transition probabilities between regimes
  3. Predict current regime state (online via Viterbi / forward-backward)
  4. Estimate regime-conditional Gram statistics for toxicity baseline

Key insight:
  Toxicity isn't raw deviation from "normal" — it's deviation from
  what's expected GIVEN the current regime. HMM provides the conditioning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from hmmlearn import hmm


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HMMRegimeConfig:
    """Configuration for HMM regime classifier."""

    n_states: int = 3              # number of latent regimes
    covariance_type: str = "full"  # "full", "diag", "spherical", "tied"
    n_iter: int = 100              # EM iterations for fitting
    tol: float = 1e-4              # convergence tolerance
    random_state: int = 42
    verbose: bool = False

    # State labeling
    state_names: tuple = ("LOW_VOL", "TRENDING", "HIGH_VOL")  # user-overridable


# ═══════════════════════════════════════════════════════════════════════
# Core: HMM Regime Classifier
# ═══════════════════════════════════════════════════════════════════════

class HMMRegime:
    """
    Train a Gaussian HMM on atom coefficient space and provide
    online regime inference.

    Usage
    -----
    >>> hmm_r = HMMRegime()
    >>> hmm_r.fit(alpha_matrix)        # (N, K) historical coefficients
    >>> regime_id, probs = hmm_r.predict(alpha_new)  # (B, K) new batch
    """

    def __init__(self, config: Optional[HMMRegimeConfig] = None):
        self.config = config or HMMRegimeConfig()
        self.model_: Optional[hmm.GaussianHMM] = None
        self.is_fitted_ = False

        # Regime-conditional Gram statistics
        # For each state, store list of G_alpha matrices seen while in that state
        self.regime_grams: dict[int, list] = {}
        self.regime_gram_mean: dict[int, np.ndarray] = {}
        self.regime_gram_cov: dict[int, np.ndarray] = {}  # vectored Gram covariance

        # Summary after fit
        self.transmat_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None          # (n_states, K)
        self.covars_: Optional[np.ndarray] = None         # (n_states, K, K)
        self.startprob_: Optional[np.ndarray] = None

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, alpha_matrix: np.ndarray) -> "HMMRegime":
        """
        Fit HMM on historical sparse coefficient data.

        Parameters
        ----------
        alpha_matrix : (N, K) — stacked sparse coefficient vectors
        """
        cfg = self.config
        N, K = alpha_matrix.shape

        print(f"[HMMRegime] Fitting GaussianHMM with {cfg.n_states} states "
              f"on {N:,} samples × {K} features")
        t0 = time.perf_counter()

        self.model_ = hmm.GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            tol=cfg.tol,
            random_state=cfg.random_state,
            verbose=cfg.verbose,
        )
        self.model_.fit(alpha_matrix)

        self.transmat_ = self.model_.transmat_
        self.means_ = self.model_.means_
        self.covars_ = self.model_.covars_
        self.startprob_ = self.model_.startprob_
        self.is_fitted_ = True

        elapsed = time.perf_counter() - t0
        print(f"[HMMRegime] Fitted in {elapsed:.1f}s")
        print(f"  Transition matrix:\n{np.round(self.transmat_, 3)}")
        print(f"  State means:\n{np.round(self.means_, 3)}")
        print(f"  Stationary distribution: {np.round(self._stationary_dist(), 3)}")

        return self

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, alpha_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict most likely regime for new coefficient vectors.

        Parameters
        ----------
        alpha_new : (B, K) — new batch of sparse coefficients

        Returns
        -------
        states : (B,) int array of regime IDs
        probs  : (B, n_states) posterior probabilities
        """
        if not self.is_fitted_:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        states = self.model_.predict(alpha_new)
        probs = self.model_.predict_proba(alpha_new)
        return states, probs

    def predict_single(self, alpha_vec: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict regime for a single coefficient vector."""
        return self.predict(alpha_vec.reshape(1, -1))

    # ── Regime-conditional Gram baseline ─────────────────────────────

    def build_regime_gram_baseline(
        self,
        alpha_matrix: np.ndarray,
        batch_size: int = 2048,
    ):
        """
        For each HMM state, collect G_alpha = (1/B) α^T α for all batches
        assigned to that state. Then compute mean and covariance for
        toxicity scoring.

        Parameters
        ----------
        alpha_matrix : (N, K) — full historical coefficient matrix
        batch_size   : samples per batch for Gram computation
        """
        if not self.is_fitted_:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        N, K = alpha_matrix.shape
        n_states = self.config.n_states

        # Predict states for all samples
        all_states, _ = self.predict(alpha_matrix)

        # Initialize collectors
        self.regime_grams = {s: [] for s in range(n_states)}

        # Batch through, compute G_alpha per batch
        n_batches = N // batch_size
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            batch = alpha_matrix[start:end]
            states_batch = all_states[start:end]

            # Group samples by state within this batch
            for s in range(n_states):
                mask = states_batch == s
                if mask.sum() > 10:  # minimum samples for stable Gram
                    alpha_s = batch[mask]
                    G = (alpha_s.T @ alpha_s) / len(alpha_s)
                    self.regime_grams[s].append(G)

        # Compute mean and covariance of vectored Grams per regime
        for s in range(n_states):
            grams = self.regime_grams[s]
            if len(grams) < 5:
                print(f"  [HMMRegime] State {s}: insufficient Gram samples ({len(grams)})")
                continue

            grams_stacked = np.array([g.ravel() for g in grams])  # (N_g, K²)
            self.regime_gram_mean[s] = np.mean(grams_stacked, axis=0).reshape(K, K)
            self.regime_gram_cov[s] = np.cov(grams_stacked, rowvar=False)

            print(f"  [HMMRegime] State {s}: {len(grams)} Gram batches, "
                  f"mean Frobenius norm = {np.linalg.norm(self.regime_gram_mean[s], 'fro'):.3f}")

    # ── Stationary distribution ──────────────────────────────────────

    def _stationary_dist(self) -> np.ndarray:
        """Compute stationary distribution from transition matrix."""
        if self.transmat_ is None:
            return np.array([])

        # Eigenvector of transmat^T corresponding to eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(self.transmat_.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = pi / pi.sum()
        pi = np.maximum(pi, 0)  # numerical safety
        return pi / pi.sum()

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return HMM regime summary."""
        if not self.is_fitted_:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_states": self.config.n_states,
            "state_means": self.means_.tolist() if self.means_ is not None else None,
            "transition_matrix": self.transmat_.tolist() if self.transmat_ is not None else None,
            "stationary_dist": self._stationary_dist().tolist(),
        }

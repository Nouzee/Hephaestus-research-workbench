"""
Stochastic State Object — S_t = (X_t, Z_t, H_t, M_t)

Unified random state representation. Every component is a distribution,
not a point estimate. Replaces all deterministic dict-based state objects.

  X_t : observable microstructure features (random vector)
  Z_t : latent regime (categorical random variable)
  H_t : hazard / toxicity field (scalar random variable)
  M_t : mode projection (random vector on 1D backbone + residuals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ===========================================================================
# Stochastic State
# ===========================================================================

@dataclass
class StochasticState:
    """
    Unified random state S_t for limit order book markets.

    Every attribute is either a random variable or a probability distribution.
    No point estimates — uncertainty is part of the state representation.

    Usage
    -----
    >>> state = StochasticState.from_observation(x_raw, regime_probs, hazard, modes)
    >>> state.entropy()  # total uncertainty
    >>> s_sample = state.sample()  # draw one realization
    """

    # ── X_t: Observable features (as distribution) ──
    x_mean: np.ndarray              # (D,) mean observable
    x_cov: np.ndarray               # (D, D) covariance
    x_samples: np.ndarray = None    # (B, D) bootstrap samples (optional)

    # ── Z_t: Latent regime (categorical) ──
    z_probs: np.ndarray = None      # (K,) P(Z_t = k | X_t)
    z_entropy: float = 0.0          # H(Z_t) — regime uncertainty

    # ── H_t: Hazard field (scalar distribution) ──
    h_mean: float = 0.0             # E[H_t]
    h_std: float = 0.0              # Std[H_t]
    h_curve: np.ndarray = None      # (T,) survival curve P(τ > t)

    # ── M_t: Mode projection ──
    m_proj: np.ndarray = None       # (d,) projection onto backbone
    m_residual: np.ndarray = None   # (d,) residual from backbone
    m_variance: float = 0.0         # projection uncertainty

    # Meta
    timestamp: int = 0

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def from_observation(
        cls,
        x_raw: np.ndarray,               # (D,) observed features
        regime_probs: np.ndarray,         # (K,) P(Z_t=k|X_t)
        hazard: float,                    # E[H_t|X_t]
        mode_coeffs: np.ndarray,          # (d,) mode coefficients
        x_std: np.ndarray = None,         # (D,) per-feature std (optional)
        h_std: float = 0.0,
        **kwargs,
    ) -> "StochasticState":
        """Construct from raw observation + posterior estimates."""
        if x_std is None:
            x_std = np.ones_like(x_raw) * 0.1

        D = len(x_raw)
        x_cov = np.diag(x_std ** 2)

        z_entropy = float(-np.sum(regime_probs * np.log(regime_probs + 1e-12)))

        return cls(
            x_mean=x_raw.astype(np.float64),
            x_cov=x_cov,
            z_probs=regime_probs.astype(np.float64),
            z_entropy=z_entropy,
            h_mean=float(hazard),
            h_std=float(h_std),
            m_proj=mode_coeffs.astype(np.float64),
            m_residual=np.zeros_like(mode_coeffs),
            m_variance=0.0,
            **kwargs,
        )

    # ── Sampling ─────────────────────────────────────────────────────

    def sample_x(self, n: int = 1) -> np.ndarray:
        """Draw n samples from observable distribution X_t ~ N(x_mean, x_cov)."""
        return np.random.multivariate_normal(self.x_mean, self.x_cov, size=n)

    def sample_z(self) -> int:
        """Draw one regime from categorical P(Z_t|X_t)."""
        if self.z_probs is None:
            return 0
        return int(np.random.choice(len(self.z_probs), p=self.z_probs))

    def sample_h(self, n: int = 1) -> np.ndarray:
        """Draw n hazard samples H_t ~ N(h_mean, h_std^2) truncated to [0,1]."""
        samples = np.random.normal(self.h_mean, max(self.h_std, 1e-8), size=n)
        return np.clip(samples, 0.0, 1.0)

    # ── Information measures ─────────────────────────────────────────

    def entropy(self) -> float:
        """Total state uncertainty: H(Z_t) + H(X_t|Z_t)."""
        # H(X_t|Z_t) approximated by Gaussian entropy
        D = len(self.x_mean)
        _, logdet = np.linalg.slogdet(self.x_cov)
        h_x = 0.5 * (D * (1 + np.log(2 * np.pi)) + logdet)
        return self.z_entropy + max(h_x, 0.0)

    def kl_divergence(self, other: "StochasticState") -> float:
        """KL(S_t || S'_t) — approximate via regime + Gaussian KL."""
        # Regime KL
        if self.z_probs is not None and other.z_probs is not None:
            kl_z = float(np.sum(
                self.z_probs * np.log(self.z_probs / (other.z_probs + 1e-12) + 1e-12)
            ))
        else:
            kl_z = 0.0

        # Gaussian KL (X component)
        D = len(self.x_mean)
        cov_inv = np.linalg.inv(other.x_cov + 1e-8 * np.eye(D))
        kl_x = 0.5 * (
            np.trace(cov_inv @ self.x_cov)
            + (other.x_mean - self.x_mean) @ cov_inv @ (other.x_mean - self.x_mean)
            - D
            + np.linalg.slogdet(other.x_cov)[1]
            - np.linalg.slogdet(self.x_cov)[1]
        )
        return float(kl_z + max(kl_x, 0.0))

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "t": self.timestamp,
            "H(Z)": float(self.z_entropy),
            "E[H]": float(self.h_mean),
            "Std[H]": float(self.h_std),
            "|M_proj|": float(np.linalg.norm(self.m_proj)) if self.m_proj is not None else 0,
        }

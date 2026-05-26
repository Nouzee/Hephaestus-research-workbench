"""
Stochastic Information Geometry — Fisher metric, entropy flow, drift field.

Replaces deterministic PCA geometry with proper stochastic geometry:
  - Fisher information metric on state space
  - Entropy production rate (irreversibility measure)
  - Stochastic drift field v(S) = E[dS_t | S_t]
  - Information curvature (local compression indicator)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ===========================================================================
# Stochastic Geometry
# ===========================================================================

@dataclass
class StochasticGeometry:
    """
    Information-geometric structure of the stochastic state space.

    Usage
    -----
    >>> sg = StochasticGeometry()
    >>> sg.fit(state_sequence)
    >>> print(f"Drift norm: {sg.drift_norm:.4f}")
    >>> print(f"Entropy production: {sg.entropy_production:.4f}")
    """

    # Drift field
    drift_vector: np.ndarray = None       # (d,) mean drift E[ΔS_t]
    drift_norm: float = 0.0               # ||E[ΔS_t]||

    # Diffusion
    diffusion_matrix: np.ndarray = None   # (d, d) covariance of ΔS_t
    diffusion_trace: float = 0.0          # total noise intensity

    # Fisher information (local)
    fisher_trace: float = 0.0             # approximate trace of Fisher matrix
    information_curvature: float = 0.0    # local curvature proxy

    # Entropy flow
    entropy_state: np.ndarray = None      # (N,) H(S_t) per step
    entropy_production: float = 0.0       # mean ΔH
    entropy_volatility: float = 0.0       # std of ΔH

    # Stability
    local_lyapunov: float = 0.0           # mean expansion rate
    is_expansive: bool = False            # λ > 0?
    n_dim: int = 0

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(self, states: np.ndarray) -> "StochasticGeometry":
        """
        Estimate geometric structure from state trajectory.

        Parameters
        ----------
        states : (N, d) array of stochastic state vectors
        """
        N, d = states.shape
        self.n_dim = d

        # Drift: E[ΔS_t]
        diffs = np.diff(states, axis=0)  # (N-1, d)
        self.drift_vector = np.mean(diffs, axis=0)
        self.drift_norm = float(np.linalg.norm(self.drift_vector))

        # Diffusion: Cov[ΔS_t]
        diffs_centered = diffs - self.drift_vector
        self.diffusion_matrix = (diffs_centered.T @ diffs_centered) / (N - 1)
        self.diffusion_trace = float(np.trace(self.diffusion_matrix))

        # Fisher trace approximation: trace of inverse diffusion
        ev = np.linalg.eigvalsh(self.diffusion_matrix)
        ev_pos = np.maximum(ev, 1e-8)
        self.fisher_trace = float(np.sum(1.0 / ev_pos))
        self.information_curvature = float(np.std(ev_pos) / max(np.mean(ev_pos), 1e-8))

        # Entropy flow: H(S_t) approximated by Gaussian entropy
        entropies = np.zeros(N, dtype=np.float64)
        for t in range(N):
            # Approximate with local covariance (use rolling window if N large)
            window = states[max(0, t-10):min(N, t+10)]
            if len(window) > 2:
                cov = np.cov(window.T)
                _, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(d))
                entropies[t] = 0.5 * (d * (1 + np.log(2*np.pi)) + logdet)

        self.entropy_state = entropies
        delta_entropy = np.diff(entropies)
        self.entropy_production = float(np.mean(delta_entropy))
        self.entropy_volatility = float(np.std(delta_entropy))

        # Local Lyapunov: mean expansion rate = trace(drift gradient)
        # Approximate: ||drift|| / ||state||
        state_norms = np.linalg.norm(states[:-1], axis=1)
        drift_norms = np.linalg.norm(diffs, axis=1)
        ratios = drift_norms / np.maximum(state_norms, 1e-8)
        self.local_lyapunov = float(np.mean(ratios))
        self.is_expansive = self.local_lyapunov > 0.01

        return self

    # ── Query ────────────────────────────────────────────────────────

    def drift_at(self, s: np.ndarray) -> np.ndarray:
        """Return drift vector at state s (constant in linear approx)."""
        return self.drift_vector.copy()

    def diffusion_at(self, s: np.ndarray) -> np.ndarray:
        """Return diffusion matrix at state s (constant in linear approx)."""
        return self.diffusion_matrix.copy()

    def expansion_rate(self, direction: np.ndarray) -> float:
        """
        Local expansion rate along a given direction v.
        λ(v) = v^T D v / ||v||^2 where D is normalized diffusion.
        """
        v = direction / max(np.linalg.norm(direction), 1e-8)
        return float(v @ self.diffusion_matrix @ v) if self.diffusion_matrix is not None else 0.0

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "n_dim": self.n_dim,
            "drift_norm": float(self.drift_norm),
            "diffusion_trace": float(self.diffusion_trace),
            "local_lyapunov": float(self.local_lyapunov),
            "is_expansive": self.is_expansive,
            "entropy_production": float(self.entropy_production),
            "information_curvature": float(self.information_curvature),
            "fisher_trace": float(self.fisher_trace),
        }

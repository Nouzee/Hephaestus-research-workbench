"""
Transition Kernel Estimator — P(Z_{t+1} | Z_t)

Replaces A-matrix regression with proper empirical Markov kernel.
Outputs: transition probabilities, entropy rate, spectral gap,
         stochastic path simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ===========================================================================
# Transition Kernel
# ===========================================================================

@dataclass
class TransitionKernel:
    """
    Empirical Markov kernel P(Z_{t+1} | Z_t) for discrete regime space.

    Usage
    -----
    >>> tk = TransitionKernel()
    >>> tk.fit(regime_sequence)
    >>> P = tk.kernel  # (K, K) stochastic matrix
    >>> gap = tk.spectral_gap()  # 1 - |λ2|, stability metric
    >>> path = tk.sample_path(n_steps=100)  # stochastic simulation
    """

    kernel: np.ndarray = None          # (K, K) transition probability matrix
    stationary_dist: np.ndarray = None  # (K,) stationary distribution
    eigenvalues: np.ndarray = None      # (K,) complex eigenvalues
    entropy_rate: float = 0.0           # H(Z_{t+1}|Z_t) — conditional entropy
    spectral_gap: float = 0.0           # 1 - |λ2|
    n_states: int = 0

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(self, regime_seq: np.ndarray) -> "TransitionKernel":
        """
        Estimate empirical transition kernel from regime sequence.

        Parameters
        ----------
        regime_seq : (N,) int array of regime labels Z_t
        """
        Z = regime_seq.astype(np.int32)
        K = int(np.max(Z)) + 1
        self.n_states = K

        # Count transitions: C[i,j] = # of i→j
        C = np.zeros((K, K), dtype=np.float64)
        for t in range(len(Z) - 1):
            C[Z[t], Z[t+1]] += 1

        # Normalize to probabilities
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)
        self.kernel = C / row_sums

        # Stationary distribution: eigenvector of P^T with eigenvalue 1
        evals, evecs = np.linalg.eig(self.kernel.T)
        idx = np.argmin(np.abs(evals - 1.0))
        pi = np.real(evecs[:, idx])
        pi = np.maximum(pi, 0)
        self.stationary_dist = pi / pi.sum()

        # Eigenvalues for spectral gap
        self.eigenvalues = np.linalg.eigvals(self.kernel)

        # Spectral gap: 1 - |λ2|
        sorted_mag = np.sort(np.abs(self.eigenvalues))[::-1]
        self.spectral_gap = float(1.0 - sorted_mag[1]) if K > 1 else 1.0

        # Entropy rate: H(Z_{t+1} | Z_t)
        log_kernel = np.log(np.maximum(self.kernel, 1e-12))
        self.entropy_rate = float(-np.sum(
            self.stationary_dist[:, None] * self.kernel * log_kernel
        ))
        self.entropy_rate = 0.0 if np.isnan(self.entropy_rate) else self.entropy_rate

        return self

    # ── Query ────────────────────────────────────────────────────────

    def predict_proba(self, z_current: int) -> np.ndarray:
        """P(Z_{t+1} | Z_t = z_current) — (K,) probability vector."""
        return self.kernel[z_current].copy()

    def predict(self, z_current: int) -> int:
        """Sample Z_{t+1} ~ P(· | Z_t = z_current)."""
        return int(np.random.choice(self.n_states, p=self.kernel[z_current]))

    # ── Simulation ──────────────────────────────────────────────────

    def sample_path(self, n_steps: int, z0: int = 0) -> np.ndarray:
        """Generate a stochastic regime path of length n_steps."""
        path = np.zeros(n_steps, dtype=np.int32)
        path[0] = z0
        for t in range(1, n_steps):
            path[t] = self.predict(path[t-1])
        return path

    def sample_paths(self, n_paths: int, n_steps: int, z0: int = 0) -> np.ndarray:
        """Generate n_paths independent regime trajectories. Returns (n_paths, n_steps)."""
        paths = np.zeros((n_paths, n_steps), dtype=np.int32)
        for p in range(n_paths):
            paths[p] = self.sample_path(n_steps, z0)
        return paths

    # ── Absorbing states ─────────────────────────────────────────────

    @property
    def absorbing_states(self) -> list[int]:
        """States with P(i→i) > 0.90."""
        return [i for i in range(self.n_states)
                if self.kernel[i, i] > 0.90]

    def mean_absorption_time(self, target_state: int) -> np.ndarray:
        """Expected steps to reach target_state from each initial state."""
        # Solve (I - Q)^{-1} * 1 where Q is the submatrix without target
        K = self.n_states
        non_target = [i for i in range(K) if i != target_state]
        m = len(non_target)

        if m == 0:
            return np.zeros(K)

        Q = self.kernel[np.ix_(non_target, non_target)]
        I = np.eye(m)
        try:
            N = np.linalg.inv(I - Q)  # fundamental matrix
            times = N @ np.ones(m)
        except np.linalg.LinAlgError:
            return np.full(K, np.inf)

        result = np.zeros(K)
        for i, ni in enumerate(non_target):
            result[ni] = times[i]
        result[target_state] = 0.0
        return result

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        if self.kernel is None:
            return {"fitted": False}
        return {
            "n_states": self.n_states,
            "spectral_gap": float(self.spectral_gap),
            "entropy_rate": float(self.entropy_rate),
            "n_absorbing": len(self.absorbing_states),
            "absorbing_states": self.absorbing_states,
            "mixing_time_bound": float(1.0 / max(self.spectral_gap, 1e-8)),
        }

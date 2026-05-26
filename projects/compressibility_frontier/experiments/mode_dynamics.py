"""
Mode Dynamics — empirical dynamical system identification.

Computes the Mode Interaction Matrix A (8x8):
    z_i(t+1) = Σ_j A_ij · z_j(t)  +  ε_i(t)

Per-regime comparison: NORMAL vs FRAGILE vs HIGH_VOL

Outputs:
  A matrix          — mode-to-mode causal flow
  Driver/Response   — which modes drive the system
  Stability spectrum — eigenvalues of A
  Amplification operator — A_fragile - A_normal
"""

from __future__ import annotations

import numpy as np


class ModeDynamics:
    """
    Identify the empirical dynamical system governing mode evolution.

    Usage
    -----
    >>> md = ModeDynamics()
    >>> md.fit(z_series, mode_labels)
    >>> md.print_report()
    """

    def __init__(self):
        self.A: np.ndarray = None              # (K, K) full interaction matrix
        self.A_regimes: dict[str, np.ndarray] = {}  # per-regime
        self.eigenvalues: np.ndarray = None
        self.eigenvectors: np.ndarray = None
        self.mode_labels: list[str] = []
        self.K: int = 0

        # Classification
        self.out_strength: np.ndarray = None   # row sum = how much mode drives others
        self.in_strength: np.ndarray = None     # col sum = how much mode is driven
        self.self_excitation: np.ndarray = None # diagonal = self-feedback

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        z_series: np.ndarray,      # (N, K) mode time series
        mode_labels: list[str],
    ) -> "ModeDynamics":
        """
        Estimate A via OLS: z(t+1) = z(t) @ A^T

        Parameters
        ----------
        z_series    : (N, K) standardized mode activations
        mode_labels : length K, human-readable mode names
        """
        N, K = z_series.shape
        self.K = K
        self.mode_labels = mode_labels

        # Build: Y = z(t+1), X = z(t)
        Y = z_series[1:]    # (N-1, K)
        X = z_series[:-1]   # (N-1, K)

        # OLS: A = (X^T X)^{-1} X^T Y  → A_ij = effect of mode j on mode i
        XtX = X.T @ X
        XtY = X.T @ Y

        # Regularized inverse for stability
        ridge = 1e-4 * np.eye(K)
        self.A = np.linalg.solve(XtX + ridge, XtY).T  # (K, K)

        # Eigen-decomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)

        # Classification
        self.out_strength = np.sum(np.abs(self.A), axis=1)  # row sum
        self.in_strength = np.sum(np.abs(self.A), axis=0)   # col sum
        self.self_excitation = np.diag(self.A)

        return self

    # ── Per-regime fit ───────────────────────────────────────────────

    def fit_regimes(
        self,
        z_series: np.ndarray,
        mode_labels: list[str],
        regimes: dict[str, np.ndarray],
    ) -> "ModeDynamics":
        """Fit separate A matrices for each regime."""
        self.mode_labels = mode_labels
        self.K = len(mode_labels)

        for rname, mask in regimes.items():
            if mask.sum() < 50:
                continue
            indices = np.where(mask)[0]
            # Need contiguous-ish segments for lag regression
            # Take all consecutive pairs within this regime
            z_r = z_series[indices]

            if len(z_r) < 10:
                continue

            Y = z_r[1:]
            X = z_r[:-1]
            XtX = X.T @ X
            XtY = X.T @ Y
            ridge = 1e-4 * np.eye(self.K)
            A_r = np.linalg.solve(XtX + ridge, XtY).T
            self.A_regimes[rname] = A_r

        # Full fit on all data
        self.fit(z_series, mode_labels)
        return self

    # ── Amplification operator ───────────────────────────────────────

    def amplification_operator(self, regime_a: str, regime_b: str) -> np.ndarray:
        """Diff matrix: A_a - A_b. Shows which edges are amplified."""
        if regime_a not in self.A_regimes or regime_b not in self.A_regimes:
            return np.zeros((self.K, self.K))
        return self.A_regimes[regime_a] - self.A_regimes[regime_b]

    # ── Classification ───────────────────────────────────────────────

    def classify_modes(self) -> dict:
        """Classify each mode as Driver, Response, or Feedback loop."""
        classification = {}
        for k in range(self.K):
            out = self.out_strength[k]
            in_s = self.in_strength[k]
            ratio = out / max(in_s, 1e-12)
            diag = self.self_excitation[k]

            if diag > 0.1 and ratio > 1.3:
                cls = "SELF_EXCITING"
            elif ratio > 1.5:
                cls = "DRIVER"
            elif ratio < 0.6:
                cls = "RESPONSE"
            else:
                cls = "NEUTRAL"

            classification[self.mode_labels[k]] = {
                "class": cls,
                "out_strength": float(out),
                "in_strength": float(in_s),
                "self_excitation": float(diag),
                "drive_ratio": float(ratio),
            }

        return classification

    # ── Stability report ─────────────────────────────────────────────

    def stability_report(self) -> dict:
        """Analyze system stability from eigenvalues."""
        if self.eigenvalues is None:
            return {}

        ev = self.eigenvalues
        real_parts = np.real(ev)
        imag_parts = np.imag(ev)
        magnitudes = np.abs(ev)

        n_stable = int(np.sum(real_parts < 0))
        n_unstable = int(np.sum(real_parts > 0.02))
        n_neutral = self.K - n_stable - n_unstable
        max_real = float(np.max(real_parts))

        return {
            "n_stable": n_stable,
            "n_unstable": n_unstable,
            "n_neutral": n_neutral,
            "max_eigenvalue_real": max_real,
            "max_magnitude": float(np.max(magnitudes)),
            "is_stable": max_real < 0,
            "dominant_frequency": float(np.max(np.abs(imag_parts))),
            "eigenvalues": [
                {"real": float(np.real(ev[k])), "imag": float(np.imag(ev[k])),
                 "magnitude": float(magnitudes[k])}
                for k in range(self.K)
            ],
        }

    # ── Report ───────────────────────────────────────────────────────

    def print_report(self):
        """Full dynamics report."""
        K = self.K
        if self.A is None:
            print("Not fitted.")
            return

        print(f"\n{'═'*75}")
        print(f"  Mode Interaction Dynamics — Empirical Dynamical System")
        print(f"{'═'*75}")

        # ── A matrix ──
        print(f"\n  Interaction Matrix A (rows = effect ON mode i, cols = FROM mode j):")
        header = f"  {'':>30s}" + "".join(f"{j:>8d}" for j in range(K))
        print(header)
        for i in range(K):
            row = f"  {self.mode_labels[i][:30]:>30s}"
            for j in range(K):
                val = self.A[i, j]
                row += f"{val:>+8.3f}"
            print(row)

        # ── Driver / Response ──
        cls = self.classify_modes()
        print(f"\n  Mode Roles:")
        print(f"  {'Mode':<35s} {'Role':<16s} {'Out':>6s} {'In':>6s} {'Ratio':>6s} {'Self':>6s}")
        print(f"  {'─'*35} {'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        for label, info in cls.items():
            print(f"  {label:<35s} {info['class']:<16s} "
                  f"{info['out_strength']:>6.3f} {info['in_strength']:>6.3f} "
                  f"{info['drive_ratio']:>6.2f} {info['self_excitation']:>+6.3f}")

        # ── Stability ──
        sr = self.stability_report()
        print(f"\n  Stability Spectrum:")
        print(f"    Stable modes:   {sr['n_stable']}  (λ_real < 0)")
        print(f"    Neutral modes:  {sr['n_neutral']}  (λ_real ≈ 0)")
        print(f"    Unstable modes: {sr['n_unstable']}  (λ_real > 0)")
        print(f"    Max Re(λ):      {sr['max_eigenvalue_real']:+.4f}")
        print(f"    System stable:  {'YES' if sr['is_stable'] else 'NO — has explosive subspace'}")
        print(f"    Dominant freq:  {sr['dominant_frequency']:.4f}  "
              f"({'oscillatory' if sr['dominant_frequency'] > 0.1 else 'non-oscillatory'})")

        # ── Top feedback loops ──
        print(f"\n  Strongest Self-Exciting Modes (A_ii):")
        order = np.argsort(-np.abs(self.self_excitation))
        for idx in order[:5]:
            print(f"    {self.mode_labels[idx]:<35s} A_ii={self.self_excitation[idx]:+.4f}")

        # ── Top cross-interactions ──
        print(f"\n  Strongest Cross-Mode Interactions (|A_ij| > 0.05):")
        for i in range(K):
            for j in range(K):
                if i != j and abs(self.A[i, j]) > 0.05:
                    print(f"    {self.mode_labels[j][:25]:>25s} → "
                          f"{self.mode_labels[i][:25]:<25s}  "
                          f"A={self.A[i,j]:+.4f}")

        print(f"{'═'*75}")

    # ── Regime comparison report ─────────────────────────────────────

    def print_regime_comparison(self, regime_a: str = "FRAGILE",
                                 regime_b: str = "NORMAL"):
        """Compare A matrices across regimes."""
        amp = self.amplification_operator(regime_a, regime_b)
        if amp is None or np.allclose(amp, 0):
            print(f"  No regime comparison available.")
            return

        print(f"\n  Amplification Operator: A({regime_a}) - A({regime_b})")
        print(f"  Showing edges amplified in {regime_a} vs {regime_b}:")
        print(f"  {'From':>30s} → {'To':<30s} {'ΔA':>8s} {'Interpretation':>20s}")
        print(f"  {'─'*30}   {'─'*30} {'─'*8} {'─'*20}")

        edges = []
        for i in range(self.K):
            for j in range(self.K):
                if abs(amp[i, j]) > 0.02:
                    edges.append((i, j, amp[i, j]))

        edges.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, j, val in edges[:15]:
            interp = "amplified" if val > 0 else "suppressed"
            print(f"  {self.mode_labels[j][:30]:>30s} → "
                  f"{self.mode_labels[i][:30]:<30s} {val:>+8.4f} {interp:>20s}")

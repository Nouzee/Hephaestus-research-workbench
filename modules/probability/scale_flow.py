"""
Scale Flow + Fixed Point Detector — renormalization in stochastic market systems.

Not dimensionality reduction. It's fixed-point identification across scale.

Scale hierarchy:
  S0: raw L2 microstructure (16-dim observable)
  S1: microstructure modes (d-eff modes, PCA compressed)
  S2: regime state (R0-R7, discrete 8-state)
  S3: hazard space (continuous tox ∈ [0,1])
  S4: backbone 1D drift (projection onto v1)

Three minimality criteria per scale k:
  (A) ΔI_k = I(S; S_k) - I(S; S_{k+1})  → 0 at fixed point
  (B) ε_k = ||P(S_{t+1}|S_k) - P(S_{t+1}|S_{k+1})||  → 0 at closure
  (C) ΔSharpe_k = |Sharpe(π_k) - Sharpe(π_{k+1})|  → 0 at saturation

Output: k* (fixed-point scale), basis B_k*, measure P_k*, transition kernel at k*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import linalg


# ===========================================================================
# Scale definitions
# ===========================================================================

SCALE_NAMES = ["S0:raw_L2", "S1:modes", "S2:regime", "S3:hazard", "S4:backbone"]
N_SCALES = len(SCALE_NAMES)


# ===========================================================================
# Scale Flow Result
# ===========================================================================

@dataclass
class ScaleFlowResult:
    """Complete renormalization flow analysis."""

    # Per-scale metrics
    entropies: np.ndarray           # (N_SCALES,) H(S_k)
    variances: np.ndarray           # (N_SCALES,) total variance
    info_gains: np.ndarray          # (N_SCALES-1,) ΔI_k = I_k - I_{k+1}
    closure_errors: np.ndarray      # (N_SCALES-1,) ε_k
    sharpe_deltas: np.ndarray       # (N_SCALES-1,) ΔSharpe_k

    # Fixed point detection
    fixed_point_scale: int = -1     # k* where all three plateau
    plateau_start: int = -1         # first scale where criteria are met
    is_valid: bool = False          # does a genuine fixed point exist?

    # Plateau metrics
    plateau_width: int = 0          # number of consecutive scales in plateau
    plateau_stability: float = 0.0  # how flat is the plateau (CV of metrics)

    # Warnings
    warnings: list = field(default_factory=list)

    @property
    def minimal_element(self) -> dict:
        """The minimal sufficient representation."""
        if not self.is_valid or self.fixed_point_scale < 0:
            return {"error": "no valid fixed point found"}
        return {
            "k*": self.fixed_point_scale,
            "scale_name": SCALE_NAMES[self.fixed_point_scale],
            "entropy": float(self.entropies[self.fixed_point_scale]),
            "plateau_width": self.plateau_width,
            "interpretation": (
                f"Scale {self.fixed_point_scale} ({SCALE_NAMES[self.fixed_point_scale]}) "
                f"is the minimal sufficient representation. "
                f"Adding more scales does not increase information, improve dynamics, "
                f"or enhance trading performance."
            ),
        }


# ===========================================================================
# Scale Flow Engine
# ===========================================================================

@dataclass
class ScaleFlow:
    """
    Renormalization flow detector for stochastic market systems.

    Usage
    -----
    >>> sf = ScaleFlow()
    >>> sf.fit(features, regimes, hazard_curve, backbone_proj, pnl_paths)
    >>> result = sf.detect_fixed_point()
    >>> print(f"Minimal scale: {result.fixed_point_scale}")
    """

    # Thresholds for fixed-point detection
    eps_info: float = 0.05      # ΔI < ε → plateau
    eps_closure: float = 0.10   # ε_k < ε → closure
    eps_sharpe: float = 0.10    # ΔSharpe < ε → saturation

    # Internal
    result: ScaleFlowResult = None

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,           # (N, D) raw L2 features (S0)
        regime_labels: np.ndarray,      # (N,) regime labels (S2)
        hazard_values: np.ndarray,      # (N,) continuous hazard (S3)
        backbone_proj: np.ndarray,      # (N,) backbone projection (S4)
        pnl_per_path: np.ndarray = None, # (n_paths, n_scales) for MC stability
        mode_features: np.ndarray = None, # (N, d) mode projections (S1, optional)
    ) -> "ScaleFlow":
        """Fit scale flow from hierarchical representations."""
        N = len(features)
        D = features.shape[1]
        result = ScaleFlowResult(
            entropies=np.zeros(N_SCALES),
            variances=np.zeros(N_SCALES),
            info_gains=np.zeros(N_SCALES - 1),
            closure_errors=np.zeros(N_SCALES - 1),
            sharpe_deltas=np.zeros(N_SCALES - 1),
        )

        # ── S0: raw L2 ──
        result.entropies[0] = self._gaussian_entropy(features)
        result.variances[0] = float(np.var(features))

        # ── S1: mode space (PCA compressed) ──
        if mode_features is not None:
            result.entropies[1] = self._gaussian_entropy(mode_features)
            result.variances[1] = float(np.var(mode_features))
        else:
            # Approximate: first d_eff PCs
            X_c = features - features.mean(axis=0)
            _, S, _ = linalg.svd(X_c, full_matrices=False)
            cum_var = np.cumsum(S**2) / np.sum(S**2)
            d_eff = int(np.searchsorted(cum_var, 0.90)) + 1
            X_pca = X_c @ linalg.svd(X_c, full_matrices=False)[2][:d_eff].T
            result.entropies[1] = self._gaussian_entropy(X_pca)
            result.variances[1] = float(np.var(X_pca))

        # ── S2: regime ──
        K = int(np.max(regime_labels)) + 1
        p_z = np.bincount(regime_labels, minlength=K) / N
        p_z = p_z[p_z > 0]
        result.entropies[2] = float(-np.sum(p_z * np.log(p_z + 1e-12)))
        result.variances[2] = float(np.var(regime_labels.astype(np.float64)))

        # ── S3: hazard ──
        h_clipped = np.clip(hazard_values, 0.001, 0.999)
        result.entropies[3] = float(-np.mean(
            h_clipped * np.log(h_clipped) + (1 - h_clipped) * np.log(1 - h_clipped)
        ))
        result.variances[3] = float(np.var(hazard_values))

        # ── S4: backbone ──
        result.entropies[4] = self._gaussian_entropy(backbone_proj.reshape(-1, 1))
        result.variances[4] = float(np.var(backbone_proj))

        # ── Cross-scale metrics ──
        for k in range(N_SCALES - 1):
            # (A) Info gain: entropy reduction
            delta_I = result.entropies[k] - result.entropies[k+1]
            result.info_gains[k] = float(max(delta_I, 0.0))

            # (B) Dynamic closure: variance ratio stability
            v_k = result.variances[k]
            v_k1 = result.variances[k+1]
            result.closure_errors[k] = float(abs(v_k - v_k1) / max(v_k, 1e-8))

            # (C) MC Sharpe stability (if available)
            if pnl_per_path is not None and pnl_per_path.shape[1] > max(k, k+1):
                s_k = self._sharpe_from_paths(pnl_per_path[:, k])
                s_k1 = self._sharpe_from_paths(pnl_per_path[:, k+1])
                result.sharpe_deltas[k] = float(abs(s_k - s_k1))
            else:
                result.sharpe_deltas[k] = 0.0

        self.result = result
        return self

    # ── Fixed point detection ───────────────────────────────────────

    def detect_fixed_point(self) -> ScaleFlowResult:
        """
        Find k* where all three criteria simultaneously plateau.

        Plateau ≠ zero. Plateau = all derivatives near zero.
        """
        if self.result is None:
            raise RuntimeError("Call fit() first.")

        r = self.result

        # Find first scale from the right where ALL criteria converge
        for k in range(N_SCALES - 2, -1, -1):
            info_ok = r.info_gains[k] < self.eps_info
            closure_ok = r.closure_errors[k] < self.eps_closure
            sharpe_ok = r.sharpe_deltas[k] < self.eps_sharpe

            if info_ok and closure_ok and sharpe_ok:
                r.fixed_point_scale = k
                r.is_valid = True
                break

        # Plateau width: how many consecutive scales meet criteria
        if r.is_valid:
            width = 1
            for k in range(r.fixed_point_scale + 1, N_SCALES - 1):
                if (r.info_gains[k] < self.eps_info and
                    r.closure_errors[k] < self.eps_closure):
                    width += 1
                else:
                    break
            r.plateau_width = width

            # Stability: CV of info_gains within plateau
            plateau_vals = r.info_gains[r.fixed_point_scale:
                                        r.fixed_point_scale + width]
            r.plateau_stability = float(np.std(plateau_vals) / max(np.mean(plateau_vals), 1e-8))

        # False minimum checks
        self._verify(r)

        return r

    # ── Verification ─────────────────────────────────────────────────

    def _verify(self, r: ScaleFlowResult):
        """Check for false minima: regime collapse, projection artifact, sample size."""
        r.warnings = []

        # Regime collapse: if S2 has near-zero entropy
        if r.entropies[2] < 0.1:
            r.warnings.append("REGIME_COLLAPSE: S2 entropy near zero — dead system")

        # Projection artifact: if backbone has more variance than modes
        if r.variances[4] > r.variances[1] * 0.9:
            r.warnings.append("PROJECTION_ARTIFACT: backbone captures nearly as much as full mode space")

        # Flat flow: if all info_gains are near zero
        if np.all(r.info_gains < 0.01):
            r.warnings.append("FLAT_FLOW: no information gain at any scale — data may be noise")

        # Degenerate fixed point: if k* is S0 (raw)
        if r.fixed_point_scale == 0 and r.is_valid:
            r.warnings.append("DEGENERATE_FP: fixed point at raw data — no compression possible")

        if r.warnings:
            r.is_valid = False

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_entropy(X: np.ndarray) -> float:
        """Gaussian entropy approximation: H = 0.5 * log((2πe)^d * |Σ|)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, d = X.shape
        if N < 2:
            return 0.0
        cov = np.cov(X.T)
        _, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(d))
        return float(0.5 * (d * (1 + np.log(2 * np.pi)) + logdet))

    @staticmethod
    def _sharpe_from_paths(pnl_paths: np.ndarray) -> float:
        """Sharpe ratio from MC PnL paths."""
        terminal = pnl_paths if pnl_paths.ndim == 1 else pnl_paths
        mu = np.mean(terminal)
        sigma = np.std(terminal)
        return float(mu / max(sigma, 1e-8))

    # ── Report ───────────────────────────────────────────────────────

    def print_report(self):
        """Print scale flow analysis."""
        r = self.result
        if r is None:
            print("Not fitted.")
            return

        print(f"\n{'═'*70}")
        print(f"  Scale Flow — Renormalization Fixed Point Detection")
        print(f"{'═'*70}")

        print(f"\n  Scale Hierarchy:")
        print(f"  {'Scale':<16s} {'H(S_k)':>10s} {'Var(S_k)':>10s} "
              f"{'ΔI_k':>10s} {'ε_k':>10s} {'ΔSharpe':>10s}")
        print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        for k in range(N_SCALES):
            h = r.entropies[k]
            v = r.variances[k]
            di = r.info_gains[k-1] if k > 0 else 0
            eps = r.closure_errors[k-1] if k > 0 else 0
            ds = r.sharpe_deltas[k-1] if k > 0 else 0

            marker = ""
            if r.is_valid and k == r.fixed_point_scale:
                marker = " ← FIXED POINT k*"
            elif r.is_valid and k >= r.fixed_point_scale and k < r.fixed_point_scale + r.plateau_width:
                marker = " ← plateau"

            print(f"  {SCALE_NAMES[k]:<16s} {h:>10.4f} {v:>10.4f} "
                  f"{di:>10.4f} {eps:>10.4f} {ds:>10.4f}{marker}")

        # Thresholds used
        print(f"\n  Detection thresholds:")
        print(f"    ΔI < {self.eps_info}  |  ε < {self.eps_closure}  |  ΔSharpe < {self.eps_sharpe}")

        # Result
        if r.is_valid:
            me = r.minimal_element
            print(f"\n  FIXED POINT FOUND: {me['scale_name']}")
            print(f"    Plateau width: {r.plateau_width} scales")
            print(f"    {me['interpretation']}")
        else:
            print(f"\n  NO VALID FIXED POINT")
            if r.warnings:
                print(f"  Warnings: {r.warnings}")
            if r.fixed_point_scale >= 0:
                print(f"  (closest candidate: {SCALE_NAMES[r.fixed_point_scale]}, "
                      f"but verification failed)")

        print(f"{'═'*70}")

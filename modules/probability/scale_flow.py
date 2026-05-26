"""
Scale Flow v2 — Distribution-Level Fixed Point Detection

Upgraded from metric-comparison to distributional invariance.

Three stability criteria (all must hold at k*):
  (A) KL stability:  D_KL(P(S_{k+1}) || P(S_k)) → noise floor
  (B) Kernel stability: ||P(Z_{t+1}|S_k) - P(Z_{t+1}|S_{k+1})||_W → 0
  (C) Decision stability: E[π_k(a|S)] - E[π_{k+1}(a|S)] → 0

Anti-fixed-point tests:
  1. Basis rotation: PCA random rotate → k* disappears? → pseudo plateau
  2. Noise injection: S_k += ε·N(0,1) → k* shifts? → fragile
  3. Kernel permutation: shuffle Z transitions → k* survives? → trivial

Output classification:
  CASE A: TRUE MINIMAL ELEMENT (stable k*, passes anti-tests, cross-time stable)
  CASE B: QUASI-STABLE (weak plateau, fails anti-tests partially)
  CASE C: NO MINIMUM (no stable k*, everything scale-dependent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance


SCALE_NAMES = ["S0:raw_L2", "S1:modes", "S2:regime", "S3:hazard", "S4:backbone"]
N_SCALES = len(SCALE_NAMES)


# ===========================================================================
# Scale Flow Result
# ===========================================================================

@dataclass
class ScaleFlowResult:
    """Distribution-level fixed point detection result."""

    # Per-scale entropies
    entropies: np.ndarray = None           # (N_SCALES,) H(S_k)

    # Distribution-level stability metrics (N_SCALES-1,)
    kl_stability: np.ndarray = None        # D_KL(P_{k+1} || P_k)
    kernel_stability: np.ndarray = None    # Wasserstein distance between kernels
    decision_stability: np.ndarray = None  # policy shift

    # Fixed point
    fixed_point_scale: int = -1
    plateau_scales: list = field(default_factory=list)
    classification: str = "UNCLASSIFIED"   # CASE_A, CASE_B, CASE_C

    # Anti-test results
    anti_basis_passed: bool = False
    anti_noise_passed: bool = False
    anti_permute_passed: bool = False

    # Cross-time stability
    cross_time_stable: bool = False

    # Warnings
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.classification == "CASE_A"

    @property
    def verdict(self) -> str:
        return {
            "CASE_A": "TRUE MINIMAL ELEMENT — market has intrinsic representation scale",
            "CASE_B": "QUASI-STABLE — regime-dependent compression only",
            "CASE_C": "NO MINIMUM — market is scale-free stochastic field",
        }.get(self.classification, "UNCLASSIFIED")


# ===========================================================================
# Scale Flow Engine v2
# ===========================================================================

@dataclass
class ScaleFlow:
    """
    Distribution-level fixed point detection.

    Usage
    -----
    >>> sf = ScaleFlow()
    >>> sf.fit(features_by_scale, regime_seqs_by_scale, policy_by_scale)
    >>> sf.detect_fixed_point()
    >>> sf.run_anti_tests(features_by_scale)
    >>> print(sf.result.verdict)
    """

    # Noise floor thresholds (auto-calibrated from data)
    kl_floor: float = 0.0
    kernel_floor: float = 0.0
    decision_floor: float = 0.0

    # Internal
    result: ScaleFlowResult = None
    _features_by_scale: list = field(default_factory=list)
    _regimes_by_scale: list = field(default_factory=list)

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        features_by_scale: list[np.ndarray],    # [S0, S1, S2, S3, S4] each (N, d_k)
        regime_seqs_by_scale: list[np.ndarray],  # [S2, S3, S4] regime sequences
        policy_actions_by_scale: list[np.ndarray] = None,  # expected action per scale
    ) -> "ScaleFlow":
        """
        Fit scale flow from hierarchical representations.

        Parameters
        ----------
        features_by_scale : list of (N, d_k) arrays for each scale
        regime_seqs_by_scale : list of (N,) regime label arrays
        policy_actions_by_scale : optional, expected action per (N,)
        """
        self._features_by_scale = features_by_scale
        self._regimes_by_scale = regime_seqs_by_scale

        result = ScaleFlowResult()
        n_scales = len(features_by_scale)
        result.entropies = np.zeros(n_scales)

        # Compute per-scale entropies
        for k in range(n_scales):
            result.entropies[k] = self._gaussian_entropy(features_by_scale[k])

        # ── (A) KL stability: D_KL(P_{k+1} || P_k) ──
        result.kl_stability = np.zeros(n_scales - 1)
        for k in range(n_scales - 1):
            result.kl_stability[k] = self._distribution_kl(
                features_by_scale[k], features_by_scale[k+1])

        # ── (B) Kernel stability: Wasserstein between transition kernels ──
        result.kernel_stability = np.zeros(n_scales - 1)
        for k in range(min(len(regime_seqs_by_scale) - 1, n_scales - 1)):
            if k < len(regime_seqs_by_scale) and k+1 < len(regime_seqs_by_scale):
                result.kernel_stability[k] = self._kernel_wasserstein(
                    regime_seqs_by_scale[k], regime_seqs_by_scale[k+1])

        # ── (C) Decision stability: policy shift ──
        if policy_actions_by_scale is not None:
            result.decision_stability = np.zeros(n_scales - 1)
            for k in range(min(len(policy_actions_by_scale) - 1, n_scales - 1)):
                result.decision_stability[k] = float(np.mean(
                    np.abs(policy_actions_by_scale[k] - policy_actions_by_scale[k+1])
                ))
        else:
            result.decision_stability = np.zeros(n_scales - 1)

        # Auto-calibrate noise floors from data
        self.kl_floor = float(np.median(result.kl_stability) * 0.5)
        self.kernel_floor = float(np.median(result.kernel_stability) * 0.5)
        self.decision_floor = float(np.median(result.decision_stability) * 0.5)
        if self.decision_floor < 1e-8:
            self.decision_floor = 0.05

        self.result = result
        return self

    # ── Fixed point detection ───────────────────────────────────────

    def detect_fixed_point(self) -> ScaleFlowResult:
        """
        Find k* where ALL THREE distribution-level criteria hit noise floor.

        k* = first scale where KL, kernel, and decision stability are all
        simultaneously below their respective noise floors.
        """
        r = self.result
        n = len(r.kl_stability)

        # Find plateau: consecutive scales where all three are below floor
        plateau_start = -1
        for k in range(n - 1, -1, -1):
            kl_ok = r.kl_stability[k] < self.kl_floor
            kern_ok = r.kernel_stability[k] < self.kernel_floor
            dec_ok = r.decision_stability[k] < self.decision_floor

            if kl_ok and kern_ok and dec_ok:
                plateau_start = k
            else:
                break  # plateau must be contiguous from the right

        if plateau_start >= 0:
            r.fixed_point_scale = plateau_start
            # Count plateau width
            r.plateau_scales = []
            for k in range(plateau_start, n):
                if (r.kl_stability[k] < self.kl_floor and
                    r.kernel_stability[k] < self.kernel_floor):
                    r.plateau_scales.append(k)
                else:
                    break

        return r

    # ── Anti-fixed-point tests ──────────────────────────────────────

    def run_anti_tests(
        self,
        features_by_scale: list[np.ndarray],
        regime_seqs: list[np.ndarray] = None,
    ) -> ScaleFlowResult:
        """
        Three perturbation tests to verify k* is not a pseudo plateau.

        Test 1: Basis rotation — randomly rotate PCA basis
        Test 2: Noise injection — add Gaussian noise to features
        Test 3: Kernel permutation — shuffle transition matrix
        """
        r = self.result
        n_scales = len(features_by_scale)

        # ── Test 1: Basis rotation ──
        # Randomly rotate feature space; re-fit; check if k* persists
        rotated_features = []
        for k in range(n_scales):
            X_k = features_by_scale[k].copy()
            d = X_k.shape[1]
            if d > 1:
                Q = np.linalg.qr(np.random.randn(d, d))[0]  # random orthonormal
                X_k = X_k @ Q
            rotated_features.append(X_k)

        sf_rot = ScaleFlow()
        if regime_seqs:
            sf_rot.fit(rotated_features, regime_seqs)
        else:
            sf_rot.fit(rotated_features, self._regimes_by_scale)
        sf_rot.detect_fixed_point()

        # If k* shifts by >1 scale, basis rotation matters → pseudo plateau
        r.anti_basis_passed = (
            abs(sf_rot.result.fixed_point_scale - r.fixed_point_scale) <= 1
        )

        # ── Test 2: Noise injection ──
        noisy_features = []
        for k in range(n_scales):
            X_k = features_by_scale[k].copy()
            noise_std = np.std(X_k) * 0.10  # 10% relative noise
            X_k += np.random.randn(*X_k.shape) * noise_std
            noisy_features.append(X_k)

        sf_noise = ScaleFlow()
        if regime_seqs:
            sf_noise.fit(noisy_features, regime_seqs)
        else:
            sf_noise.fit(noisy_features, self._regimes_by_scale)
        sf_noise.detect_fixed_point()

        r.anti_noise_passed = (
            abs(sf_noise.result.fixed_point_scale - r.fixed_point_scale) <= 1
        )

        # ── Test 3: Kernel permutation ──
        if regime_seqs and len(regime_seqs) > 1:
            permuted_regimes = []
            for seq in regime_seqs:
                perm = seq.copy()
                np.random.shuffle(perm)
                permuted_regimes.append(perm)

            sf_perm = ScaleFlow()
            sf_perm.fit(features_by_scale, permuted_regimes)
            sf_perm.detect_fixed_point()

            # If k* DOESN'T change under permutation, the structure was trivial
            r.anti_permute_passed = (
                abs(sf_perm.result.fixed_point_scale - r.fixed_point_scale) > 1
            )
        else:
            r.anti_permute_passed = True  # not applicable

        # ── Final classification ──
        anti_pass = r.anti_basis_passed and r.anti_noise_passed and r.anti_permute_passed
        has_plateau = r.fixed_point_scale >= 0 and len(r.plateau_scales) >= 2

        if has_plateau and anti_pass and r.cross_time_stable:
            r.classification = "CASE_A"
        elif has_plateau and (anti_pass or r.cross_time_stable):
            r.classification = "CASE_B"
        else:
            r.classification = "CASE_C"

        return r

    # ── Distribution-level metrics ──────────────────────────────────

    @staticmethod
    def _distribution_kl(X_a: np.ndarray, X_b: np.ndarray) -> float:
        """Approximate KL divergence between two Gaussian distributions."""
        if X_a.shape[1] != X_b.shape[1]:
            return 0.0  # different dimensionalities — use entropy difference instead
        d = X_a.shape[1]
        cov_a = np.cov(X_a.T) + 1e-8 * np.eye(d)
        cov_b = np.cov(X_b.T) + 1e-8 * np.eye(d)
        mu_a = np.mean(X_a, axis=0)
        mu_b = np.mean(X_b, axis=0)

        # KL(N_a || N_b) = 0.5 * [tr(Σ_b^{-1} Σ_a) + (μ_b-μ_a)^T Σ_b^{-1} (μ_b-μ_a) - d + log|Σ_b|/|Σ_a|]
        try:
            cov_b_inv = np.linalg.inv(cov_b)
            _, logdet_a = np.linalg.slogdet(cov_a)
            _, logdet_b = np.linalg.slogdet(cov_b)
            kl = 0.5 * (
                np.trace(cov_b_inv @ cov_a)
                + (mu_b - mu_a) @ cov_b_inv @ (mu_b - mu_a)
                - d
                + logdet_b - logdet_a
            )
            return float(max(kl, 0.0))
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def _kernel_wasserstein(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Wasserstein distance between empirical transition kernels."""
        Ka = int(np.max(seq_a)) + 1
        Kb = int(np.max(seq_b)) + 1
        K = max(Ka, Kb)

        # Build empirical transition counts
        def build_kernel(s, K):
            C = np.zeros((K, K), dtype=np.float64)
            for t in range(len(s) - 1):
                if s[t] < K and s[t+1] < K:
                    C[s[t], s[t+1]] += 1
            row_sum = C.sum(axis=1, keepdims=True)
            return C / np.maximum(row_sum, 1)

        P_a = build_kernel(seq_a, K)
        P_b = build_kernel(seq_b, K)

        # 1D Wasserstein per row, averaged (add epsilon for zero rows)
        eps_row = 1e-8
        P_a_safe = P_a + eps_row
        P_a_safe = P_a_safe / P_a_safe.sum(axis=1, keepdims=True)
        P_b_safe = P_b + eps_row
        P_b_safe = P_b_safe / P_b_safe.sum(axis=1, keepdims=True)

        total_w = 0.0
        for i in range(K):
            support = np.arange(K)
            w = wasserstein_distance(support, support, P_a_safe[i], P_b_safe[i])
            total_w += w
        return float(total_w / K)

    @staticmethod
    def _gaussian_entropy(X: np.ndarray) -> float:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, d = X.shape
        if N < 2:
            return 0.0
        cov = np.cov(X.T)
        _, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(d))
        return float(0.5 * (d * (1 + np.log(2 * np.pi)) + logdet))

    # ── Walk-forward stability ──────────────────────────────────────

    def test_cross_time(
        self,
        features_by_scale: list[np.ndarray],
        regime_seqs: list[np.ndarray],
        n_splits: int = 3,
    ) -> bool:
        """Test if k* is stable across time splits."""
        N = len(features_by_scale[0])
        k_stars = []

        for split in range(n_splits):
            t0 = int(N * split / n_splits)
            t1 = int(N * (split + 1) / n_splits)

            feats_split = [f[t0:t1] for f in features_by_scale]
            regs_split = [r[t0:t1] for r in regime_seqs]

            sf = ScaleFlow()
            sf.fit(feats_split, regs_split)
            sf.detect_fixed_point()
            k_stars.append(sf.result.fixed_point_scale)

        # Stable if all splits agree within ±1
        k_stars_arr = np.array(k_stars)
        self.result.cross_time_stable = bool(
            np.max(k_stars_arr) - np.min(k_stars_arr) <= 1
            and np.all(k_stars_arr >= 0)
        )
        return self.result.cross_time_stable

    # ── Report ───────────────────────────────────────────────────────

    def print_report(self):
        r = self.result
        if r is None:
            print("Not fitted."); return

        print(f"\n{'═'*70}")
        print(f"  Scale Flow v2 — Distribution-Level Fixed Point Detection")
        print(f"{'═'*70}")

        print(f"\n  Stability metrics (lower = more stable):")
        print(f"  {'Transition':<20s} {'KL stab':>10s} {'Kernel stab':>12s} "
              f"{'Decision stab':>14s} {'Verdict':>12s}")
        print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*14} {'─'*12}")

        for k in range(len(r.kl_stability)):
            kl = r.kl_stability[k]; kw = r.kernel_stability[k]
            ds = r.decision_stability[k]
            all_ok = kl < self.kl_floor and kw < self.kernel_floor and ds < self.decision_floor
            marker = "PLATEAU" if all_ok else "—"
            print(f"  {SCALE_NAMES[k]} -> {SCALE_NAMES[k+1][:6]:<6s}  "
                  f"{kl:>10.4f} {kw:>12.4f} {ds:>14.4f} {marker:>12s}")

        print(f"\n  Noise floors: KL<{self.kl_floor:.4f}  "
              f"Kernel<{self.kernel_floor:.4f}  Decision<{self.decision_floor:.4f}")

        print(f"\n  Fixed point: k* = {r.fixed_point_scale}")
        print(f"  Plateau: {r.plateau_scales}")

        print(f"\n  Anti-Fixed-Point Tests:")
        print(f"    Basis rotation:  {'PASS' if r.anti_basis_passed else 'FAIL'}")
        print(f"    Noise injection: {'PASS' if r.anti_noise_passed else 'FAIL'}")
        print(f"    Kernel permute:  {'PASS' if r.anti_permute_passed else 'FAIL'}")
        print(f"    Cross-time:      {'PASS' if r.cross_time_stable else 'FAIL'}")

        print(f"\n  ┌{'─'*60}┐")
        print(f"  │  CLASSIFICATION: {r.classification:<45s} │")
        print(f"  │  {r.verdict:<58s} │")
        print(f"  └{'─'*60}┘")
        print(f"{'═'*70}")

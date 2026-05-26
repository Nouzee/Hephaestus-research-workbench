"""
Minimal Scale Discovery Phase (MSDP)

Empirical renormalization group experiment on financial systems.

Not optimizing a trading system. Running a convergence experiment:
  - Multi-path MC scan across all scales
  - F(k) = KL + Wasserstein + policy_drift
  - Stable interval [k1, k2] detection (not single point)
  - Null model comparison (shuffled / i.i.d. / random kernel)
  - Formal CASE A / B / C classification

Output: whether this market has an intrinsic representation scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance

from modules.probability.scale_flow import ScaleFlow, ScaleFlowResult
from modules.probability.scale_consistency import ConsistencyCurve
from modules.probability.msdp_guard import MSDPGuard, GuardResult


SCALE_NAMES = ["S0:raw_L2", "S1:modes", "S2:regime", "S3:hazard", "S4:backbone"]


# ===========================================================================
# MSDP Result
# ===========================================================================

@dataclass
class MSDPResult:
    """Complete Minimal Scale Discovery result."""

    # F(k) curve per scale
    F_curve: np.ndarray = None            # (N_SCALES-1,) stability metric
    F_null_curves: dict = field(default_factory=dict)  # {null_type: F(k)}

    # Stable interval
    stable_interval: tuple = (-1, -1)      # (k1, k2)
    interval_width: int = 0
    plateau_strength: float = 0.0          # mean flatness within interval

    # Scale flow result
    scale_flow_result: ScaleFlowResult = None
    consistency: dict = field(default_factory=dict)

    # Anti-test
    anti_passed: bool = False

    # Null comparison
    null_separation: float = 0.0           # F_real / F_null — higher = more significant
    null_passed: bool = False

    # Final classification
    classification: str = "UNCLASSIFIED"
    case_detail: str = ""

    # MC paths
    n_paths: int = 0
    n_scales: int = 0

    @property
    def verdict(self) -> str:
        base = {
            "CASE_A": "INTRINSIC SCALE EXISTS — market has a preferred representation level",
            "CASE_B": "QUASI-SCALE BAND — regime-dependent compression, not universal",
            "CASE_C": "NO INTRINSIC SCALE — market is a scale-free stochastic field",
        }
        return base.get(self.classification, "UNCLASSIFIED")

    @property
    def minimal_scale(self) -> int:
        """The recommended minimal scale for this market."""
        if self.classification == "CASE_A":
            return self.stable_interval[0]  # start of plateau
        elif self.classification == "CASE_B":
            return self.stable_interval[0] if self.stable_interval[0] >= 0 else 2
        else:
            return -1  # no recommendation

    def summary(self) -> dict:
        return {
            "classification": self.classification,
            "case_detail": self.case_detail,
            "stable_interval": list(self.stable_interval),
            "interval_width": self.interval_width,
            "plateau_strength": float(self.plateau_strength),
            "null_separation": float(self.null_separation),
            "minimal_scale": self.minimal_scale,
            "verdict": self.verdict,
        }


# ===========================================================================
# MSDP Engine
# ===========================================================================

@dataclass
class MSDP:
    """
    Minimal Scale Discovery Phase — convergence experiment on market scales.

    Usage
    -----
    >>> msdp = MSDP(n_paths=100)
    >>> msdp.run(features_by_scale, regime_seqs)
    >>> msdp.print_report()
    >>> result = msdp.result
    """

    n_paths: int = 100
    plateau_threshold: float = 0.15     # max ΔF within plateau
    min_plateau_width: int = 2          # minimum scales for valid plateau
    null_threshold: float = 0.70        # F_real / F_null must be below this

    # Guard (pre-flight check before full MSDP execution)
    guard_enabled: bool = True
    guard: MSDPGuard = field(default_factory=MSDPGuard)
    guard_result: GuardResult = None

    seed: int = 42
    result: MSDPResult = None

    # ── Run MSDP ────────────────────────────────────────────────────

    def run(
        self,
        features_by_scale: list[np.ndarray],
        regime_seqs: list[np.ndarray],
    ) -> MSDPResult:
        """
        Execute full Minimal Scale Discovery protocol.

        Parameters
        ----------
        features_by_scale : [S0, S1, S2, S3, S4] each (N, d_k)
        regime_seqs : regime label arrays per scale
        """
        rng = np.random.RandomState(self.seed)
        n_scales = len(features_by_scale)
        result = MSDPResult(n_paths=self.n_paths, n_scales=n_scales)

        # ── 1. Multi-path F(k) estimation ──
        F_samples = np.zeros((self.n_paths, n_scales - 1), dtype=np.float64)

        for path in range(self.n_paths):
            # Bootstrap sample
            N = len(features_by_scale[0])
            idx = rng.choice(N, size=int(N * 0.8), replace=True)

            feats_boot = [f[idx] for f in features_by_scale]
            regs_boot = [r[idx] if len(r) == N else r for r in regime_seqs]

            # Fit scale flow on bootstrap
            sf = ScaleFlow()
            sf.fit(feats_boot, regs_boot)

            # F(k) = normalized KL + Wasserstein + policy_drift
            kl = sf.result.kl_stability
            kw = sf.result.kernel_stability
            ds = sf.result.decision_stability

            kl_n = kl / max(kl.max(), 1e-8) if kl.max() > 1e-8 else kl
            kw_n = kw / max(kw.max(), 1e-8) if kw.max() > 1e-8 else kw
            ds_n = ds / max(ds.max(), 1e-8) if ds.max() > 1e-8 else ds

            F_samples[path] = kl_n + kw_n + ds_n

        result.F_curve = np.mean(F_samples, axis=0)
        F_std = np.std(F_samples, axis=0)

        # ── 2. Stable interval detection ──
        k1, k2 = self._find_stable_interval(result.F_curve, F_std)
        result.stable_interval = (k1, k2)
        result.interval_width = k2 - k1 + 1 if k1 >= 0 else 0

        if result.interval_width > 0:
            plateau_vals = result.F_curve[k1:k2+1]
            result.plateau_strength = float(1.0 - np.std(plateau_vals) / max(np.mean(plateau_vals), 1e-8))

        # ── 3. Full scale flow fit + anti-tests ──
        sf_full = ScaleFlow()
        sf_full.fit(features_by_scale, regime_seqs)
        sf_full.detect_fixed_point()
        sf_full.test_cross_time(features_by_scale, regime_seqs)
        sf_full.run_anti_tests(features_by_scale, regime_seqs)
        result.scale_flow_result = sf_full.result

        # Consistency
        cc = ConsistencyCurve()
        cc.compute(
            sf_full.result.entropies,
            sf_full.result.kl_stability,
            sf_full.result.kernel_stability,
            sf_full.result.decision_stability,
        )
        result.consistency = cc.summary()
        result.anti_passed = (
            sf_full.result.anti_basis_passed
            and sf_full.result.anti_noise_passed
            and sf_full.result.anti_permute_passed
        )

        # ── 4. Null model comparison ──
        result.F_null_curves = {}
        null_F_values = []

        # Null 1: i.i.d. returns (Gaussian with same mean/cov)
        iid_features = []
        for X in features_by_scale:
            mu, cov = np.mean(X, axis=0), np.cov(X.T)
            X_iid = rng.multivariate_normal(mu, cov + 1e-8*np.eye(X.shape[1]), size=N)
            iid_features.append(X_iid)
        null_F = self._compute_F(iid_features, regime_seqs)
        result.F_null_curves["iid"] = null_F
        null_F_values.append(np.mean(null_F))

        # Null 2: shuffled order book (preserve marginal, destroy time)
        shuffled_features = [X.copy() for X in features_by_scale]
        for X_s in shuffled_features:
            for d in range(X_s.shape[1]):
                rng.shuffle(X_s[:, d])
        null_F = self._compute_F(shuffled_features, regime_seqs)
        result.F_null_curves["shuffled"] = null_F
        null_F_values.append(np.mean(null_F))

        # Null 3: random kernel (uniform transition)
        random_regimes = []
        for seq in regime_seqs:
            K = int(np.max(seq)) + 1
            r_seq = rng.randint(0, K, size=len(seq))
            random_regimes.append(r_seq)
        null_F = self._compute_F(features_by_scale, random_regimes)
        result.F_null_curves["random_kernel"] = null_F
        null_F_values.append(np.mean(null_F))

        # Separation: F_real / mean(F_null) — lower = closer to null
        F_real_mean = np.mean(result.F_curve)
        F_null_mean = np.mean(null_F_values)
        result.null_separation = float(F_real_mean / max(F_null_mean, 1e-8))
        result.null_passed = result.null_separation < self.null_threshold

        # ── 4.5. MSDP Guard (mandatory pre-check) ──
        if self.guard_enabled:
            null_F_curve = result.F_null_curves.get("shuffled",
                            np.ones_like(result.F_curve))
            self.guard_result = self.guard.check(
                result.F_curve, F_samples, null_F_curve)
            if not self.guard_result.passed:
                # Guard blocked — downgrade classification
                result.classification = "CASE_C"
                result.case_detail = (
                    f"MSDP Guard blocked: {self.guard_result.block_reason}"
                )
                self.result = result
                return result

        # ── 5. Final classification ──
        has_plateau = result.interval_width >= self.min_plateau_width
        cross_time_ok = sf_full.result.cross_time_stable

        if has_plateau and result.anti_passed and cross_time_ok and result.null_passed:
            result.classification = "CASE_A"
            result.case_detail = (
                f"Stable plateau at scales [{k1}, {k2}] "
                f"(width={result.interval_width}), "
                f"anti-tests passed, cross-time stable, "
                f"null separation={result.null_separation:.3f}"
            )
        elif has_plateau and (result.anti_passed or cross_time_ok):
            result.classification = "CASE_B"
            result.case_detail = (
                f"Weak plateau at scales [{k1}, {k2}], "
                f"anti={'[OK]' if result.anti_passed else '[X]'}, "
                f"cross_time={'[OK]' if cross_time_ok else '[X]'}, "
                f"null={'[OK]' if result.null_passed else '[X]'}"
            )
        else:
            result.classification = "CASE_C"
            result.case_detail = "No stable plateau survives validation."

        self.result = result
        return result

    # ── Helpers ──────────────────────────────────────────────────────

    def _compute_F(self, features, regime_seqs) -> np.ndarray:
        """Compute F(k) for a given feature set."""
        sf = ScaleFlow()
        sf.fit(features, regime_seqs)
        kl = sf.result.kl_stability
        kw = sf.result.kernel_stability
        ds = sf.result.decision_stability
        kl_n = kl / max(kl.max(), 1e-8) if kl.max() > 1e-8 else kl
        kw_n = kw / max(kw.max(), 1e-8) if kw.max() > 1e-8 else kw
        ds_n = ds / max(ds.max(), 1e-8) if ds.max() > 1e-8 else ds
        return kl_n + kw_n + ds_n

    def _find_stable_interval(self, F_curve, F_std) -> tuple:
        """Find longest contiguous interval where ΔF < threshold."""
        n = len(F_curve)
        best_k1, best_k2 = -1, -1
        best_width = 0

        for k1 in range(n):
            for k2 in range(k1 + self.min_plateau_width - 1, n):
                segment = F_curve[k1:k2+1]
                segment_std = np.std(segment)
                if segment_std < self.plateau_threshold:
                    width = k2 - k1 + 1
                    if width > best_width:
                        best_width = width
                        best_k1, best_k2 = k1, k2

        return best_k1, best_k2

    # ── Report ───────────────────────────────────────────────────────

    def print_report(self):
        r = self.result
        if r is None:
            print("Not run. Call run() first."); return

        print(f"\n{'═'*70}")
        print(f"  MSDP — Minimal Scale Discovery Phase")
        print(f"  Empirical Renormalization Group on Financial System")
        print(f"{'═'*70}")

        print(f"\n  F(k) Stability Curve ({self.n_paths} MC paths):")
        print(f"  {'k':>4s} {'Transition':<20s} {'F(k)':>10s} {'±Std':>8s}")
        print(f"  {'─'*4} {'─'*20} {'─'*10} {'─'*8}")
        for k in range(len(r.F_curve)):
            marker = ""
            if r.stable_interval[0] <= k <= r.stable_interval[1]:
                marker = " ← PLATEAU"
            F_std = 0  # computed during bootstrap
            print(f"  {k:>4d} {SCALE_NAMES[k]} -> {SCALE_NAMES[k+1][:6]:<6s} "
                  f"{r.F_curve[k]:>10.4f}{marker}")

        print(f"\n  Stable interval: [{r.stable_interval[0]}, {r.stable_interval[1]}] "
              f"(width={r.interval_width}, strength={r.plateau_strength:.3f})")

        print(f"\n  Null Model Comparison (F_real / F_null):")
        print(f"    Real:          {np.mean(r.F_curve):.4f}")
        for null_name, null_F in r.F_null_curves.items():
            ratio = np.mean(r.F_curve) / max(np.mean(null_F), 1e-8)
            marker = "[OK]" if ratio < self.null_threshold else "[X]"
            print(f"    {null_name:<15s}: {np.mean(null_F):.4f}  "
                  f"(separation={ratio:.3f}) {marker}")

        print(f"\n  Validation:")
        print(f"    Anti-tests:  {'PASS' if r.anti_passed else 'FAIL'}")
        print(f"    Cross-time:  {'PASS' if r.scale_flow_result.cross_time_stable else 'FAIL'}")
        print(f"    Null model:  {'PASS' if r.null_passed else 'FAIL'}")
        print(f"    Consistency: {r.consistency.get('consistency_score', 0):.3f}")

        print(f"\n  ┌{'─'*60}┐")
        print(f"  │  CLASSIFICATION: {r.classification:<45s} │")
        print(f"  │  {r.verdict:<58s} │")
        print(f"  │                                                            │")
        print(f"  │  {r.case_detail:<58s} │")
        print(f"  └{'─'*60}┘")

        if r.minimal_scale >= 0:
            print(f"\n  Recommended minimal scale: {SCALE_NAMES[r.minimal_scale]}")
        print(f"{'═'*70}")

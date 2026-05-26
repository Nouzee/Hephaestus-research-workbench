"""
Scale Consistency Curve — three curves must flatten simultaneously.

  1. Information decay I(k): entropy per scale
  2. Kernel distance D_kernel(k): transition kernel shift between scales
  3. Policy shift Δπ(k): decision stability across scales

The fixed point k* is where ALL THREE curves jointly flatten.
Output: consistency_score ∈ [0,1] measuring how clean the plateau is.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ===========================================================================
# Consistency Curve
# ===========================================================================

@dataclass
class ConsistencyCurve:
    """
    Three-curve joint flattening detector.

    Usage
    -----
    >>> cc = ConsistencyCurve()
    >>> cc.compute(entropies, kl_gaps, kernel_gaps, decision_gaps)
    >>> print(f"Consistency: {cc.score:.3f}")
    """

    # Input curves
    entropies: np.ndarray = None          # H(S_k) per scale
    kl_gaps: np.ndarray = None            # D_KL(P_{k+1} || P_k)
    kernel_gaps: np.ndarray = None        # W(P_{k+1}, P_k)
    decision_gaps: np.ndarray = None      # |π_{k+1} - π_k|

    # Derived
    entropy_decay: np.ndarray = None      # ΔH_k = H_k - H_{k+1}
    consistency_score: float = 0.0        # [0,1] how clean is the plateau
    flatten_point: int = -1              # k where all three flatten
    flatten_quality: str = "UNKNOWN"      # SHARP / WEAK / NONE

    def compute(
        self,
        entropies: np.ndarray,
        kl_gaps: np.ndarray,
        kernel_gaps: np.ndarray,
        decision_gaps: np.ndarray,
    ) -> "ConsistencyCurve":
        """Compute consistency score from scale curves."""
        self.entropies = entropies
        self.kl_gaps = kl_gaps
        self.kernel_gaps = kernel_gaps
        self.decision_gaps = decision_gaps

        n = len(kl_gaps)

        # Normalize each gap curve to [0,1]
        kl_norm = kl_gaps / max(kl_gaps.max(), 1e-8)
        kw_norm = kernel_gaps / max(kernel_gaps.max(), 1e-8)
        ds_norm = decision_gaps / max(decision_gaps.max(), 1e-8) if decision_gaps.max() > 1e-8 else decision_gaps

        # Joint flatness: product of (1 - normalized gap) — high when ALL are low
        joint_flatness = (1 - kl_norm) * (1 - kw_norm) * (1 - ds_norm)

        # Best flatten point: argmax of joint flatness
        self.flatten_point = int(np.argmax(joint_flatness))
        best_flatness = float(joint_flatness[self.flatten_point])

        # Consistency score = how cleanly joint flatness peaks
        # (peak value) × (how much peak exceeds mean)
        mean_flatness = float(np.mean(joint_flatness))
        peak_ratio = best_flatness / max(mean_flatness, 1e-8)
        self.consistency_score = float(np.clip(best_flatness * min(peak_ratio, 3.0) / 3.0, 0, 1))

        if self.consistency_score > 0.7:
            self.flatten_quality = "SHARP"
        elif self.consistency_score > 0.4:
            self.flatten_quality = "WEAK"
        else:
            self.flatten_quality = "NONE"

        # Entropy decay
        if len(entropies) > 1:
            self.entropy_decay = np.abs(np.diff(entropies))

        return self

    def summary(self) -> dict:
        return {
            "flatten_point": self.flatten_point,
            "consistency_score": float(self.consistency_score),
            "flatten_quality": self.flatten_quality,
            "scale_name": SCALE_NAMES[self.flatten_point] if self.flatten_point >= 0 else "N/A",
        }


SCALE_NAMES = ["S0:raw_L2", "S1:modes", "S2:regime", "S3:hazard", "S4:backbone"]


# ===========================================================================
# Anti-Fixed-Point Test (standalone)
# ===========================================================================

@dataclass
class AntiFixedPointTest:
    """
    Three perturbation tests to verify k* is not a pseudo plateau.

    Test 1: Basis rotation — random orthonormal transform
    Test 2: Noise injection — 10% Gaussian noise
    Test 3: Kernel permutation — shuffle transition matrix

    A valid fixed point must survive all three with k* shift ≤ 1.
    """

    # Results
    basis_k_star: int = -1
    noise_k_star: int = -1
    permute_k_star: int = -1
    original_k_star: int = -1

    basis_pass: bool = False
    noise_pass: bool = False
    permute_pass: bool = False

    @property
    def all_pass(self) -> bool:
        return self.basis_pass and self.noise_pass and self.permute_pass

    def run(
        self,
        features_by_scale: list[np.ndarray],
        regime_seqs: list[np.ndarray],
        original_k_star: int,
        scale_flow_class,  # ScaleFlow class for re-fitting
    ) -> "AntiFixedPointTest":
        """Run all three anti-tests."""
        self.original_k_star = original_k_star
        n_scales = len(features_by_scale)

        # ── Test 1: Basis rotation ──
        rotated = []
        for X in features_by_scale:
            d = X.shape[1]
            if d > 1:
                Q = np.linalg.qr(np.random.randn(d, d))[0]
                rotated.append(X @ Q)
            else:
                rotated.append(X.copy())

        sf1 = scale_flow_class()
        sf1.fit(rotated, regime_seqs); sf1.detect_fixed_point()
        self.basis_k_star = sf1.result.fixed_point_scale
        self.basis_pass = abs(self.basis_k_star - original_k_star) <= 1

        # ── Test 2: Noise injection ──
        noisy = []
        rng = np.random.RandomState(42)
        for X in features_by_scale:
            sigma = np.std(X) * 0.10
            noisy.append(X + rng.randn(*X.shape) * sigma)

        sf2 = scale_flow_class()
        sf2.fit(noisy, regime_seqs); sf2.detect_fixed_point()
        self.noise_k_star = sf2.result.fixed_point_scale
        self.noise_pass = abs(self.noise_k_star - original_k_star) <= 1

        # ── Test 3: Kernel permutation ──
        permuted_regimes = []
        for seq in regime_seqs:
            p = seq.copy(); rng.shuffle(p); permuted_regimes.append(p)

        sf3 = scale_flow_class()
        sf3.fit(features_by_scale, permuted_regimes); sf3.detect_fixed_point()
        self.permute_k_star = sf3.result.fixed_point_scale
        self.permute_pass = abs(self.permute_k_star - original_k_star) > 1

        return self

    def report(self) -> dict:
        return {
            "original_k*": self.original_k_star,
            "basis_rotation_k*": self.basis_k_star,
            "noise_injection_k*": self.noise_k_star,
            "kernel_permute_k*": self.permute_k_star,
            "basis_pass": self.basis_pass,
            "noise_pass": self.noise_pass,
            "permute_pass": self.permute_pass,
            "all_pass": self.all_pass,
            "verdict": (
                "TRUE FIXED POINT — survives all perturbations"
                if self.all_pass else
                "PSEUDO PLATEAU — fails at least one perturbation test"
            ),
        }

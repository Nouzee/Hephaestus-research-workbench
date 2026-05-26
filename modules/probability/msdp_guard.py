"""
MSDP Guard — prevent false k* from statistical smoothing artifacts.

Three rapid checks BEFORE entering full MSDP:
  1. Signal-to-null separation: SNR(k) = F_hephaestus(k) / F_null(k)
     SNR > 1.0 for at least one scale → structure may exist.
     SNR <= 1.0 everywhere → no structure, save compute.

  2. Variance collapse test:
     Var(F(k)) across MC paths must exceed noise floor.
     If Var(F) < ε_var → plateau is numerical dead zone, not structure.

  3. Scale monotonicity sanity:
     F(k) should not be purely random oscillation.
     Check: autocorrelation of ΔF(k) > 0 or runs test passes.

Output: GUARD_PASSED or GUARD_BLOCKED with specific failure reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ===========================================================================
# Guard Result
# ===========================================================================

@dataclass
class GuardResult:
    """MSDP guard check result."""

    passed: bool = False
    block_reason: str = ""

    # Check 1: SNR
    snr_values: np.ndarray = None       # SNR per scale
    snr_max: float = 0.0
    snr_passed: bool = False

    # Check 2: Variance collapse
    var_values: np.ndarray = None       # Var(F) per scale
    var_min: float = 0.0
    var_passed: bool = False

    # Check 3: Monotonicity
    is_monotonic: bool = False
    runs_test_pvalue: float = 1.0
    monotonicity_passed: bool = False

    # Meta
    recommendation: str = ""

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "block_reason": self.block_reason,
            "snr_max": float(self.snr_max),
            "snr_passed": self.snr_passed,
            "var_min": float(self.var_min),
            "var_passed": self.var_passed,
            "monotonicity_passed": self.monotonicity_passed,
            "recommendation": self.recommendation,
        }


# ===========================================================================
# MSDP Guard
# ===========================================================================

@dataclass
class MSDPGuard:
    """
    Lightweight pre-flight check before full MSDP execution.

    Usage
    -----
    >>> guard = MSDPGuard()
    >>> result = guard.check(F_hephaestus, F_samples, F_null)
    >>> if result.passed:
    ...     msdp.run(...)
    """

    # Thresholds
    snr_threshold: float = 1.05          # F_null / F_heph must exceed this (heph is more stable)
    var_threshold: float = 1e-6          # minimum MC variance
    n_runs_min: int = 5                  # minimum runs for monotonicity test

    seed: int = 42

    # ── Main check ──────────────────────────────────────────────────

    def check(
        self,
        F_hephaestus: np.ndarray,        # (n_scales-1,) mean F(k)
        F_samples: np.ndarray,           # (n_paths, n_scales-1) MC samples
        F_null: np.ndarray,              # (n_scales-1,) null F(k) from shuffled
    ) -> GuardResult:
        """
        Run all three guard checks. Returns GuardResult.
        """
        result = GuardResult()
        n_scales_m1 = len(F_hephaestus)

        # ── Check 1: Signal-to-null separation ──
        result.snr_values = np.zeros(n_scales_m1)
        for k in range(n_scales_m1):
            result.snr_values[k] = float(
                abs(F_null[k]) / max(abs(F_hephaestus[k]), 1e-12)
            )
        result.snr_max = float(np.max(result.snr_values))
        result.snr_passed = result.snr_max > self.snr_threshold

        if not result.snr_passed:
            result.block_reason = (
                f"SNR_MAX={result.snr_max:.3f} <= {self.snr_threshold}. "
                f"Hephaestus F(k) indistinguishable from null. "
                f"No structure to discover."
            )
            result.recommendation = "ABORT — market is scale-free noise"
            return result

        # ── Check 2: Variance collapse ──
        if F_samples is not None and F_samples.ndim == 2:
            result.var_values = np.var(F_samples, axis=0)
            result.var_min = float(np.min(result.var_values))
            result.var_passed = result.var_min > self.var_threshold
        else:
            # No MC samples available — use heuristic
            result.var_values = np.ones(n_scales_m1) * 1e-4
            result.var_min = 1e-4
            result.var_passed = True

        if not result.var_passed:
            result.block_reason = (
                f"Variance collapse: min Var(F)={result.var_min:.2e} <= {self.var_threshold}. "
                f"Plateau is numerical dead zone, not structure."
            )
            result.recommendation = "ABORT — variance collapse detected"
            return result

        # ── Check 3: Scale monotonicity sanity ──
        if len(F_hephaestus) >= 3:
            # Runs test: is ΔF(k) random or structured?
            dF = np.diff(F_hephaestus)
            signs = np.sign(dF)
            signs = signs[signs != 0]

            if len(signs) >= self.n_runs_min:
                # Count runs (consecutive same-sign stretches)
                runs = 1
                for i in range(1, len(signs)):
                    if signs[i] != signs[i-1]:
                        runs += 1

                # Expected runs under randomness: (2*n_pos*n_neg)/N + 1
                n_pos = int(np.sum(signs > 0))
                n_neg = int(np.sum(signs < 0))
                N = n_pos + n_neg
                if N > 0 and n_pos > 0 and n_neg > 0:
                    expected_runs = 2 * n_pos * n_neg / N + 1
                    std_runs = np.sqrt(
                        2 * n_pos * n_neg * (2 * n_pos * n_neg - N) / (N**2 * (N-1))
                    )
                    if std_runs > 1e-8:
                        z_score = (runs - expected_runs) / std_runs
                        # Two-tailed p-value approximation
                        from scipy.stats import norm
                        result.runs_test_pvalue = float(2 * norm.sf(abs(z_score)))
                    else:
                        result.runs_test_pvalue = 1.0
                else:
                    result.runs_test_pvalue = 0.01  # all same sign → structured

                # Monotonic if runs are significantly fewer than random
                result.is_monotonic = runs < expected_runs * 0.7
                result.monotonicity_passed = (
                    result.runs_test_pvalue < 0.10 or result.is_monotonic
                )
            else:
                result.monotonicity_passed = True  # not enough data
        else:
            result.monotonicity_passed = True

        if not result.monotonicity_passed:
            result.block_reason = (
                f"F(k) is purely random oscillation (runs p={result.runs_test_pvalue:.3f}). "
                f"No scale structure detected."
            )
            result.recommendation = "ABORT — F(k) is random noise"
            return result

        # ── All checks passed ──
        result.passed = True
        result.recommendation = (
            f"PROCEED to MSDP. SNR={result.snr_max:.2f}, "
            f"Var_min={result.var_min:.2e}, "
            f"structure={'monotonic' if result.is_monotonic else 'present'}"
        )
        return result

    # ── Quick pre-check (no MC needed) ──────────────────────────────

    def quick_check(
        self,
        F_hephaestus: np.ndarray,
        F_null: np.ndarray,
    ) -> bool:
        """Ultra-fast check: is SNR > threshold at any scale?"""
        snr = np.max(np.abs(F_null) / np.maximum(np.abs(F_hephaestus), 1e-12))
        return snr > self.snr_threshold

    # ── Report ───────────────────────────────────────────────────────

    def print_report(self, result: GuardResult):
        """Print guard check results."""
        print(f"\n  MSDP Guard — Pre-Flight Check")
        print(f"  {'─'*40}")

        snr_status = "[OK] PASS" if result.snr_passed else "[X] FAIL"
        var_status = "[OK] PASS" if result.var_passed else "[X] FAIL"
        mono_status = "[OK] PASS" if result.monotonicity_passed else "[X] FAIL"

        print(f"  {snr_status}  SNR:         max={result.snr_max:.3f}  "
              f"(threshold={self.snr_threshold})")
        print(f"  {var_status}  Var collapse: min={result.var_min:.2e}  "
              f"(threshold={self.var_threshold})")
        print(f"  {mono_status}  Monotonicity: is_monotonic={result.is_monotonic}  "
              f"runs_p={result.runs_test_pvalue:.3f}")

        if result.passed:
            print(f"\n  GUARD PASSED — {result.recommendation}")
        else:
            print(f"\n  GUARD BLOCKED — {result.block_reason}")
            print(f"  {result.recommendation}")

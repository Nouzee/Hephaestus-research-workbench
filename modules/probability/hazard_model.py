"""
Hazard Rate Model — continuous h(X_t) = P(adverse regime | X_t)

Replaces threshold-based tox scoring with proper hazard function:
  - Continuous output in [0,1], not discrete buckets {0..6}
  - Survival probability P(τ > t) for stress-free duration
  - Cumulative hazard Λ(t) for stress accumulation
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ===========================================================================
# Hazard Model
# ===========================================================================

@dataclass
class HazardModel:
    """
    Continuous hazard function for stress regime arrival.

    h(X_t) = P(Z_{t+1} ∈ A_stress | X_t)

    Usage
    -----
    >>> hm = HazardModel()
    >>> hm.fit(features, stress_labels)
    >>> h = hm.predict(x_new)  # [0,1] continuous hazard
    >>> S = hm.survival_curve(x_new, horizon=50)  # P(τ > t)
    """

    # Fitted from data
    thresholds: dict = None          # feature thresholds from train quantiles
    stress_regimes: list = None      # which regimes constitute "stress"
    baseline_hazard: float = 0.0     # P(stress) unconditional (base rate)

    # Hazard calibration
    hazard_by_bucket: dict = None    # (spread_bucket, depth_bucket, tod) → P(stress)

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {}
        if self.stress_regimes is None:
            self.stress_regimes = [5]  # R5 by default

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,          # (N, D) — spread, depth, imb, arrival, ...
        regimes: np.ndarray,            # (N,) — Z_t labels
        time_of_day: np.ndarray,        # (N,) — 0=OPEN, 1=MID, 2=CLOSE
        feature_names: list = None,
    ) -> "HazardModel":
        """
        Calibrate hazard function from historical data.

        Hazard = P(Z_{t+1} ∈ stress_regimes | current features).
        """
        N = len(regimes)

        # Identify stress events: transition INTO stress
        stress_mask = np.isin(regimes, self.stress_regimes)
        stress_entry = np.zeros(N, dtype=bool)
        stress_entry[1:] = stress_mask[1:] & ~stress_mask[:-1]

        self.baseline_hazard = float(np.mean(stress_entry))

        # Calibrate thresholds from features
        if feature_names is None:
            feature_names = ["spread", "depth"]

        self.thresholds = {}
        for fi, name in enumerate(feature_names[:min(2, features.shape[1])]):
            vals = features[:, fi]
            self.thresholds[f"{name}_lo"] = float(np.percentile(vals, 33))
            self.thresholds[f"{name}_hi"] = float(np.percentile(vals, 67))

        # Build hazard lookup by discretized feature buckets
        self.hazard_by_bucket = {}
        sp_lo, sp_hi = self.thresholds.get("spread_lo", 0), self.thresholds.get("spread_hi", 1)
        dp_lo, dp_hi = self.thresholds.get("depth_lo", 0), self.thresholds.get("depth_hi", 1)

        for sp_b in range(3):  # low/mid/high spread
            for dp_b in range(3):  # low/mid/high depth
                for tod in range(3):  # OPEN/MID/CLOSE
                    mask = np.ones(N, dtype=bool)
                    if features.shape[1] >= 1:
                        if sp_b == 0: mask &= features[:, 0] < sp_lo
                        elif sp_b == 1: mask &= (features[:, 0] >= sp_lo) & (features[:, 0] < sp_hi)
                        else: mask &= features[:, 0] >= sp_hi
                    if features.shape[1] >= 2:
                        if dp_b == 0: mask &= features[:, 1] < dp_lo
                        elif dp_b == 1: mask &= (features[:, 1] >= dp_lo) & (features[:, 1] < dp_hi)
                        else: mask &= features[:, 1] >= dp_hi
                    if time_of_day is not None:
                        mask &= time_of_day == tod

                    if mask.sum() > 20:
                        self.hazard_by_bucket[(sp_b, dp_b, tod)] = float(
                            np.mean(stress_entry[mask])
                        )

        return self

    # ── Predict ──────────────────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,          # (N, D) or (D,)
        time_of_day: int = 1,           # 0=OPEN, 1=MID, 2=CLOSE
    ) -> np.ndarray:
        """
        Predict continuous hazard h(X_t) ∈ [0, 1].

        Returns (N,) array of stress entry probabilities.
        """
        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        N = features.shape[0]
        hazards = np.full(N, self.baseline_hazard, dtype=np.float64)

        sp_lo = self.thresholds.get("spread_lo", 0)
        sp_hi = self.thresholds.get("spread_hi", 1)
        dp_lo = self.thresholds.get("depth_lo", 0)
        dp_hi = self.thresholds.get("depth_hi", 1)

        for i in range(N):
            sp_b = 0 if features[i, 0] < sp_lo else (1 if features[i, 0] < sp_hi else 2) if features.shape[1] >= 1 else 1
            dp_b = 0 if features[i, 1] < dp_lo else (1 if features[i, 1] < dp_hi else 2) if features.shape[1] >= 2 else 1

            bucket = (sp_b, dp_b, time_of_day)
            if bucket in self.hazard_by_bucket:
                hazards[i] = self.hazard_by_bucket[bucket]

        if single:
            return hazards[0]
        return hazards

    def predict_hazard_curve(
        self,
        features: np.ndarray,
        horizon: int = 50,
    ) -> np.ndarray:
        """
        Cumulative hazard: Λ(t) = -log(S(t)) for t = 1..horizon.

        Under constant hazard assumption: Λ(t) = h * t.
        """
        h = float(self.predict(features))
        return np.cumsum(np.full(horizon, h))

    def survival_curve(
        self,
        features: np.ndarray,
        horizon: int = 50,
    ) -> np.ndarray:
        """
        Survival probability: S(t) = P(τ > t), where τ = time until stress.

        Under constant hazard: S(t) = (1-h)^t ≈ exp(-h*t).
        """
        h = float(self.predict(features))
        return np.exp(-h * np.arange(1, horizon + 1))

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "baseline_hazard": float(self.baseline_hazard),
            "n_buckets": len(self.hazard_by_bucket) if self.hazard_by_bucket else 0,
            "stress_regimes": self.stress_regimes,
            "max_hazard": float(max(self.hazard_by_bucket.values())) if self.hazard_by_bucket else 0.0,
            "min_hazard": float(min(self.hazard_by_bucket.values())) if self.hazard_by_bucket else 0.0,
        }

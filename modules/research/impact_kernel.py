"""
Impact Kernel — callable function: shock → price impact path.

Models how a flow shock propagates through the market over time.
The kernel is a function h(τ) that maps a shock at time t=0 to
the expected price impact at lag τ.

Core function:  impact_path = kernel(shock_magnitude)

Properties estimated from data:
  - Decay type: power law with estimated exponent α
  - Half-life: τ where impact falls to 50% of peak
  - Peak lag: τ where impact is maximal
  - Permanent impact: asymptotic limit (if any)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ImpactKernelConfig:
    """Configuration for impact kernel estimation."""

    max_lag: int = 30            # max τ for kernel
    min_events: int = 50         # min events for stable estimate
    decay_types: tuple = ("power_law", "exponential", "linear")

    # Default params (overridden by fit)
    peak_impact: float = 1.0
    half_life: float = 5.0
    power_exponent: float = 0.5


# ===========================================================================
# Core: Impact Kernel
# ===========================================================================

class ImpactKernel:
    """
    Callable: maps shock → price impact path.

    Usage
    -----
    >>> ik = ImpactKernel()
    >>> ik.fit(flow_events, price_changes)
    >>> path = ik(shock=1.0, steps=20)   # price impact over next 20 batches
    >>> total = ik.total_impact(1.0)      # integrated impact
    """

    def __init__(self, config: Optional[ImpactKernelConfig] = None):
        self.config = config or ImpactKernelConfig()
        self.cfg = self.config

        # Estimated from data
        self.kernel_: Optional[np.ndarray] = None     # raw h(τ)
        self.decay_type: str = "unknown"
        self.alpha: float = 0.0           # power law exponent
        self.half_life: float = 0.0        # batches
        self.peak_lag: int = 0
        self.peak_value: float = 0.0
        self.permanent_impact: float = 0.0

        self._fitted = False

    # ── Fit to data ──────────────────────────────────────────────────

    def fit(
        self,
        flow_shocks: np.ndarray,     # (N,) signed flow events
        price_changes: np.ndarray,   # (N,) price changes
    ) -> "ImpactKernel":
        """
        Estimate kernel from flow events and subsequent price changes.

        Uses event-study methodology: for each significant flow event,
        accumulate the average price path over subsequent batches.
        """
        cfg = self.config
        max_lag = cfg.max_lag
        N = min(len(flow_shocks), len(price_changes))

        # Identify significant shocks (top quartile by |magnitude|)
        abs_flow = np.abs(flow_shocks[:N])
        threshold = np.percentile(abs_flow, 75)
        event_mask = abs_flow > threshold
        event_signs = np.sign(flow_shocks[:N][event_mask])
        event_times = np.where(event_mask)[0]

        if len(event_times) < cfg.min_events:
            self._set_defaults()
            return self

        # For each event, accumulate signed price path
        kernel = np.zeros(max_lag + 1, dtype=np.float64)
        counts = np.zeros(max_lag + 1, dtype=np.int32)

        for ev_t, ev_sign in zip(event_times, event_signs):
            end_t = min(ev_t + max_lag + 1, N)
            tau_range = end_t - ev_t
            # Event study: signed price change = sign(flow) * Δp
            kernel[:tau_range] += price_changes[ev_t:end_t] * ev_sign
            counts[:tau_range] += 1

        valid = counts > 0
        kernel[valid] /= counts[valid]
        kernel[~valid] = 0.0
        self.kernel_ = kernel

        # Extract parameters
        self.peak_lag = int(np.argmax(np.abs(kernel)))
        self.peak_value = float(kernel[self.peak_lag])

        # Half-life: first τ where |kernel| < |peak|/2
        for tau in range(self.peak_lag, len(kernel)):
            if abs(kernel[tau]) < abs(self.peak_value) / 2:
                self.half_life = float(tau - self.peak_lag)
                break
        if self.half_life == 0:
            self.half_life = float(max_lag)

        # Fit power law: log|k| = -α * log(τ)
        tau = np.arange(1, len(kernel), dtype=np.float64)
        log_k = np.log(np.abs(kernel[1:]) + 1e-12)
        log_tau = np.log(tau)
        if len(tau) > 5:
            # Simple OLS
            n = len(tau)
            x_mean = np.mean(log_tau)
            y_mean = np.mean(log_k)
            slope = np.sum((log_tau - x_mean) * (log_k - y_mean)) / max(
                np.sum((log_tau - x_mean) ** 2), 1e-12)
            self.alpha = float(abs(slope))
            self.decay_type = f"power_law (α={self.alpha:.2f})"
        else:
            self.decay_type = "insufficient_data"

        # Permanent impact: mean of last 25% of kernel
        tail_start = int(len(kernel) * 0.75)
        self.permanent_impact = float(np.mean(kernel[tail_start:]))

        self._fitted = True
        return self

    def _set_defaults(self):
        """Fallback defaults when insufficient data."""
        self.kernel_ = np.array([1.0])
        self.decay_type = "default (no data)"
        self.half_life = 5.0
        self.peak_lag = 0
        self.peak_value = 1.0
        self.permanent_impact = 0.0
        self._fitted = True

    # ── Callable interface ───────────────────────────────────────────

    def __call__(self, shock: float, steps: Optional[int] = None) -> np.ndarray:
        """
        Compute price impact path for a given shock magnitude.

        Parameters
        ----------
        shock : float — signed magnitude of the flow shock
        steps : int   — number of forward steps (default: max_lag)

        Returns
        -------
        impact_path : (steps,) array — expected price impact at each lag
        """
        if not self._fitted or self.kernel_ is None:
            return np.zeros(steps or 10)

        n = min(steps or len(self.kernel_), len(self.kernel_))
        return self.kernel_[:n] * shock / max(abs(self.peak_value), 1e-12)

    def total_impact(self, shock: float) -> float:
        """Integrated (cumulative) impact of a shock."""
        path = self(shock)
        return float(np.sum(path))

    def decay_curve(self, steps: int = 30) -> np.ndarray:
        """Normalized decay curve (peak=1)."""
        if not self._fitted or self.kernel_ is None:
            return np.ones(steps)
        n = min(steps, len(self.kernel_))
        raw = np.abs(self.kernel_[:n])
        return raw / max(raw.max(), 1e-12)

    # ── Synthetic generation ─────────────────────────────────────────

    def generate_path(
        self,
        shock: float,
        steps: int = 30,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a synthetic impact path with optional noise.

        Used by MarketGenerator for sandbox simulation.
        """
        path = self(shock, steps)
        if noise_std > 0:
            path += np.random.randn(len(path)) * noise_std * abs(shock)
        return path

    # ── Report ───────────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "fitted": self._fitted,
            "decay_type": self.decay_type,
            "alpha": self.alpha,
            "half_life": self.half_life,
            "peak_lag": self.peak_lag,
            "peak_value": self.peak_value,
            "permanent_impact": self.permanent_impact,
            "kernel_length": len(self.kernel_) if self.kernel_ is not None else 0,
        }

    def print_report(self):
        r = self.report()
        print(f"\n  Impact Kernel:")
        print(f"    Decay:        {r['decay_type']}")
        print(f"    Half-life:    {r['half_life']:.1f} batches")
        print(f"    Peak lag:     {r['peak_lag']} batches")
        print(f"    Peak value:   {r['peak_value']:.4f}")
        print(f"    Permanent:    {r['permanent_impact']:.4f}")

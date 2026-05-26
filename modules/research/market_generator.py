"""
Minimal Market Generator — synthetic microstructure sandbox.

Generates realistic market paths from first principles:
  flow_shock → liquidity_response → price_impact → recovery

Not for trading. For:
  - Stress testing execution strategies
  - Understanding fragility conditions
  - Validating causal hypotheses

Minimal loop:
  1. Inject flow shock (random or specified)
  2. Compute liquidity response (queue pressure, spread widening)
  3. Compute price impact via kernel
  4. Model recovery (depth replenishment, spread normalization)
  5. Repeat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from modules.research.impact_kernel import ImpactKernel


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class MarketGeneratorConfig:
    """Configuration for the synthetic market generator."""

    # Flow dynamics
    base_arrival_rate: float = 100.0       # events/sec baseline
    flow_persistence: float = 0.7          # autocorrelation of flow direction
    flow_volatility: float = 0.3           # std of flow shock magnitude

    # Liquidity dynamics
    base_spread_bps: float = 1.5
    base_depth: float = 1.5
    depth_recovery_rate: float = 0.3       # how fast depth replenishes
    spread_recovery_rate: float = 0.2       # how fast spread normalizes

    # Impact
    impact_scale: float = 1.0              # overall impact sensitivity

    # Simulation
    n_steps: int = 1000
    seed: int = 42


# ===========================================================================
# Core: Market Generator
# ===========================================================================

class MarketGenerator:
    """
    Synthetic market sandbox — generates realistic microstructure paths.

    Usage
    -----
    >>> mg = MarketGenerator()
    >>> paths = mg.generate(n_steps=500)
    >>> # paths: flow_shocks, queue_pressure, spread, price_impact, depth
    >>> mg.stress_test(shock_magnitude=5.0)  # stress scenario
    """

    def __init__(
        self,
        config: Optional[MarketGeneratorConfig] = None,
        kernel: Optional[ImpactKernel] = None,
    ):
        self.config = config or MarketGeneratorConfig()
        self.cfg = self.config
        self.kernel = kernel
        self.rng = np.random.RandomState(self.cfg.seed)

    # ── Generate market path ─────────────────────────────────────────

    def generate(self, n_steps: Optional[int] = None) -> dict:
        """
        Generate a synthetic market path.

        Returns dict with keys:
          flow_shocks, queue_pressure, spread_bps, depth, price_impact,
          cumulative_impact, recovery_state
        """
        cfg = self.config
        n = n_steps or cfg.n_steps

        # State variables
        flow_dir = np.zeros(n, dtype=np.float32)       # signed flow direction
        flow_mag = np.zeros(n, dtype=np.float32)       # flow magnitude
        queue_pressure = np.zeros(n, dtype=np.float32)
        spread_bps = np.zeros(n, dtype=np.float32)
        depth = np.zeros(n, dtype=np.float32)
        price_impact = np.zeros(n, dtype=np.float32)
        recovery_state = np.zeros(n, dtype=np.float32)  # 0=normal, 1=stressed

        # Initial conditions
        spread_bps[0] = cfg.base_spread_bps
        depth[0] = cfg.base_depth
        queue_pressure[0] = cfg.base_arrival_rate / cfg.base_depth

        prev_flow = 0.0
        for t in range(1, n):
            # ── Step 1: Flow shock generation ──
            # AR(1) direction with persistence
            flow_dir[t] = (cfg.flow_persistence * prev_flow
                           + (1 - cfg.flow_persistence) * self.rng.randn())
            flow_mag[t] = abs(self.rng.randn() * cfg.flow_volatility)
            flow_shock = flow_dir[t] * flow_mag[t]
            prev_flow = flow_dir[t]

            # ── Step 2: Liquidity response ──
            # Queue pressure: arrival rate / available depth
            # Large flow → depth consumed → queue pressure rises
            depth_consumption = abs(flow_shock) * 0.1
            depth[t] = depth[t-1] - depth_consumption
            # Recovery: depth replenishes toward baseline
            depth[t] += cfg.depth_recovery_rate * (cfg.base_depth - depth[t])
            depth[t] = max(depth[t], 0.1)  # floor

            # Spread widening: proportional to queue pressure
            target_spread = cfg.base_spread_bps * (
                1.0 + 0.5 * (depth_consumption / max(depth[t], 0.1))
            )
            spread_bps[t] = (spread_bps[t-1]
                             + cfg.spread_recovery_rate * (target_spread - spread_bps[t-1])
                             + 0.02 * self.rng.randn())

            # Queue pressure
            queue_pressure[t] = (cfg.base_arrival_rate + abs(flow_shock) * 50) / depth[t]

            # ── Step 3: Price impact ──
            if self.kernel is not None and self.kernel._fitted:
                # Use empirical kernel
                decay = self.kernel.decay_curve(steps=30)
                impact_now = flow_shock * cfg.impact_scale
                # Convolve recent shocks with kernel
                lookback = min(t, 30)
                price_impact[t] = np.sum(
                    flow_dir[t-lookback:t] * flow_mag[t-lookback:t]
                    * decay[lookback-1::-1][:lookback]
                ) * cfg.impact_scale
            else:
                # Simple linear impact with noise
                price_impact[t] = (flow_shock * cfg.impact_scale
                                   + 0.3 * price_impact[t-1]  # AR(1) decay
                                   + 0.05 * self.rng.randn())

            # ── Step 4: Recovery state ──
            stress_level = (abs(flow_shock) / max(cfg.flow_volatility, 0.01)
                            + (spread_bps[t] / cfg.base_spread_bps - 1.0)
                            + (1.0 - depth[t] / cfg.base_depth))
            recovery_state[t] = float(np.clip(stress_level / 3.0, 0, 1))

        return {
            "flow_shocks": flow_dir * flow_mag,
            "flow_direction": flow_dir,
            "flow_magnitude": flow_mag,
            "queue_pressure": queue_pressure,
            "spread_bps": spread_bps,
            "depth": depth,
            "price_impact": price_impact,
            "recovery_state": recovery_state,
        }

    # ── Stress test ──────────────────────────────────────────────────

    def stress_test(self, shock_magnitude: float = 5.0,
                    n_steps: int = 200) -> dict:
        """
        Inject a single large shock and observe propagation.

        Returns the same dict as generate(), with the shock injected at t=50.
        """
        paths = self.generate(n_steps=n_steps)

        # Inject shock at t=50
        shock_t = min(50, n_steps - 30)
        paths["flow_shocks"][shock_t] = shock_magnitude
        paths["flow_magnitude"][shock_t] = abs(shock_magnitude)
        paths["flow_direction"][shock_t] = np.sign(shock_magnitude)

        # Re-simulate from shock onward
        cfg = self.config
        for t in range(shock_t + 1, n_steps):
            dc = abs(paths["flow_shocks"][t]) * 0.1
            paths["depth"][t] = (paths["depth"][t-1] - dc
                                 + cfg.depth_recovery_rate * (cfg.base_depth - paths["depth"][t-1]))
            paths["depth"][t] = max(paths["depth"][t], 0.1)

            paths["queue_pressure"][t] = (
                cfg.base_arrival_rate + abs(paths["flow_shocks"][t]) * 50) / paths["depth"][t]

            paths["recovery_state"][t] = float(np.clip(
                abs(paths["flow_shocks"][t]) / max(cfg.flow_volatility, 0.01) / 3.0
                + (paths["spread_bps"][t] / cfg.base_spread_bps - 1.0) / 2.0
                + (1.0 - paths["depth"][t] / cfg.base_depth), 0, 1))

        return paths

    # ── Fragility scan ───────────────────────────────────────────────

    def fragility_scan(
        self,
        flow_persistence_range: list[float] = None,
        depth_recovery_range: list[float] = None,
        n_steps: int = 500,
    ) -> dict:
        """
        Scan parameter space to find fragility boundaries.

        Returns dict mapping (persistence, recovery_rate) → max_impact.
        """
        if flow_persistence_range is None:
            flow_persistence_range = [0.3, 0.5, 0.7, 0.9]
        if depth_recovery_range is None:
            depth_recovery_range = [0.1, 0.3, 0.5, 0.7]

        results = {}
        for fp in flow_persistence_range:
            for dr in depth_recovery_range:
                # Temp config
                orig_fp = self.cfg.flow_persistence
                orig_dr = self.cfg.depth_recovery_rate
                self.cfg.flow_persistence = fp
                self.cfg.depth_recovery_rate = dr

                paths = self.generate(n_steps=n_steps)
                max_impact = float(np.max(np.abs(paths["price_impact"])))
                max_spread = float(np.max(paths["spread_bps"]))
                recovery_time = self._recovery_time(paths["recovery_state"])

                results[(fp, dr)] = {
                    "max_impact": max_impact,
                    "max_spread": max_spread,
                    "recovery_time": recovery_time,
                }

                self.cfg.flow_persistence = orig_fp
                self.cfg.depth_recovery_rate = orig_dr

        return results

    @staticmethod
    def _recovery_time(recovery_state: np.ndarray, threshold: float = 0.3) -> int:
        """Batches until recovery state drops below threshold."""
        above = np.where(recovery_state > threshold)[0]
        if len(above) == 0:
            return 0
        last_above = above[-1]
        below = np.where(recovery_state[last_above:] < threshold)[0]
        return int(below[0]) if len(below) > 0 else len(recovery_state)

    # ── Report ───────────────────────────────────────────────────────

    def print_fragility_report(self, scan_results: dict):
        """Print fragility scan results."""
        print(f"\n  Fragility Scan — max impact by (persistence, recovery_rate):")
        fps = sorted(set(k[0] for k in scan_results))
        drs = sorted(set(k[1] for k in scan_results))

        print(f"  {'pers\\\\rec':>10s}", end="")
        for dr in drs:
            print(f" {dr:>10.1f}", end="")
        print()

        for fp in fps:
            print(f"  {fp:>10.2f}", end="")
            for dr in drs:
                v = scan_results.get((fp, dr), {}).get("max_impact", 0)
                print(f" {v:>10.4f}", end="")
            print()

        # Find fragility boundary: where max_impact exceeds 2x baseline
        baseline = scan_results.get((0.3, 0.7), {}).get("max_impact", 1.0)
        print(f"\n  Fragile regions (impact > 2x baseline={baseline:.4f}):")
        for (fp, dr), r in scan_results.items():
            if r["max_impact"] > 2 * baseline:
                print(f"    persistence={fp:.1f}, recovery={dr:.1f} → "
                      f"impact={r['max_impact']:.4f}, recovery_time={r['recovery_time']}")

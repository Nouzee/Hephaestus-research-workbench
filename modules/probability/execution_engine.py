"""
Controlled Stochastic Execution Engine

Replaces rule-based execution with proper stochastic control:
  a_t = argmax_a E[R | S_t, a]  (optimal control)
  a_t ~ π(a | S_t)              (stochastic policy)

Integrated with StochasticState, TransitionKernel, HazardModel,
and StochasticGeometry to form a complete stochastic dynamical
system simulator for limit order book markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from modules.probability.stochastic_state import StochasticState
from modules.probability.transition_kernel import TransitionKernel
from modules.probability.hazard_model import HazardModel
from modules.probability.stochastic_geometry import StochasticGeometry
from modules.probability.policy import StochasticPolicy, ACTIONS, ACTION_NAMES


# ===========================================================================
# Execution Engine
# ===========================================================================

@dataclass
class ExecutionEngine:
    """
    Stochastic control system for limit order book markets.

    Integrates: state, transition kernel, hazard model, geometry, policy.

    Usage
    -----
    >>> engine = ExecutionEngine()
    >>> engine.fit(historical_data)
    >>> path = engine.simulate(n_steps=500)
    >>> metrics = engine.evaluate_policy(policy, n_paths=200)
    """

    # Components
    tk: TransitionKernel = field(default_factory=TransitionKernel)
    hm: HazardModel = field(default_factory=HazardModel)
    sg: StochasticGeometry = field(default_factory=StochasticGeometry)
    policy: StochasticPolicy = field(default_factory=StochasticPolicy)

    # State
    current_state: StochasticState = None
    state_history: list = field(default_factory=list)

    # Configuration
    fill_prob: float = 0.30
    adverse_asymmetry: float = 1.5
    inventory_penalty: float = 0.0001

    seed: int = 42

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,           # (N, D) observable features
        regimes: np.ndarray,             # (N,) regime labels
        time_of_day: np.ndarray = None,  # (N,) TOD
    ) -> "ExecutionEngine":
        """Fit all components from historical data."""
        rng = np.random.RandomState(self.seed)

        # Transition kernel
        self.tk.fit(regimes)

        # Hazard model
        if time_of_day is None:
            time_of_day = np.ones(len(regimes), dtype=np.int32)
        self.hm.fit(features[:, :2], regimes, time_of_day)  # use first 2 features

        # Geometry (on regime centroids as reduced state)
        K = self.tk.n_states
        centroids = np.array([
            np.mean(features[regimes == k], axis=0)
            for k in range(K) if np.sum(regimes == k) > 10
        ])
        if len(centroids) > 2:
            self.sg.fit(centroids)

        return self

    # ── Simulation ──────────────────────────────────────────────────

    def simulate(
        self,
        n_steps: int,
        policy: StochasticPolicy = None,
        initial_regime: int = 0,
    ) -> dict:
        """
        Generate one stochastic market path with policy-controlled execution.

        Returns dict with: states, actions, pnl, regime_path, hazard_path.
        """
        if policy is None:
            policy = self.policy

        rng = np.random.RandomState(self.seed)

        # Generate regime path from transition kernel
        regime_path = self.tk.sample_path(n_steps, z0=initial_regime)

        # Simulate step by step
        pnl_path = np.zeros(n_steps, dtype=np.float64)
        hazard_path = np.zeros(n_steps, dtype=np.float64)
        action_path = []
        inventory = 0.0

        for t in range(n_steps):
            z = regime_path[t]

            # Construct approximate state
            x_proto = np.zeros(16)
            pz = np.zeros(self.tk.n_states)
            pz[z] = 1.0

            # Hazard from transition kernel: h = P(current → stress)
            if 5 < self.tk.n_states:
                hazard = float(self.tk.kernel[z, 5])
            else:
                hazard = 0.0

            state = StochasticState.from_observation(
                x_proto, pz, hazard=hazard, mode_coeffs=np.zeros(6),
            )

            # Sample action from policy
            action = policy.sample(state)
            action_path.append(action)
            sz = action["size_multiplier"]
            sp_m = action["spread_multiplier"]

            if sz <= 0:
                pnl_path[t] = 0.0
                hazard_path[t] = hazard
                continue

            # Stochastic fill
            p_fill = self.fill_prob / max(sp_m, 0.5)
            if rng.random() > p_fill:
                pnl_path[t] = 0.0
                hazard_path[t] = hazard
                continue

            # Spread capture (using x_mean[0] as spread proxy)
            spread_earned = 100.0 * sp_m / 2 * sz  # approximate

            # Adverse selection (stochastic, proportional to hazard)
            adverse = 0.0
            if rng.random() < hazard:
                adverse = spread_earned * self.adverse_asymmetry
            else:
                adverse = spread_earned * 0.3  # baseline adverse

            # Inventory cost
            inv_cost = abs(inventory) * spread_earned * self.inventory_penalty

            pnl_path[t] = spread_earned - adverse - inv_cost
            hazard_path[t] = hazard

            # Update inventory (simplified)
            side = 1 if rng.random() > 0.5 else -1
            inventory += side * sz

        self.state_history = []  # reset

        return {
            "regime_path": regime_path,
            "pnl_path": pnl_path,
            "hazard_path": hazard_path,
            "actions": action_path,
            "cumulative_pnl": np.cumsum(pnl_path),
        }

    # ── Policy evaluation ───────────────────────────────────────────

    def evaluate_policy(
        self,
        policy: StochasticPolicy,
        n_paths: int = 200,
        n_steps: int = 500,
    ) -> dict:
        """
        Monte Carlo policy evaluation.

        Returns distribution of terminal PnL across n_paths.
        """
        rng = np.random.RandomState(self.seed)
        terminal_pnls = np.zeros(n_paths, dtype=np.float64)
        all_regime_paths = []

        for p in range(n_paths):
            self.seed = rng.randint(0, 2**31)
            path = self.simulate(n_steps, policy, initial_regime=0)
            terminal_pnls[p] = path["cumulative_pnl"][-1]
            all_regime_paths.append(path["regime_path"])

        terminal_pnls = np.array(terminal_pnls)

        return {
            "E[PnL]": float(np.mean(terminal_pnls)),
            "Std[PnL]": float(np.std(terminal_pnls)),
            "Sharpe": float(np.mean(terminal_pnls) / max(np.std(terminal_pnls), 1e-8) * np.sqrt(n_steps)),
            "VaR_95": float(np.percentile(terminal_pnls, 5)),
            "CVaR_95": float(np.mean(terminal_pnls[terminal_pnls <= np.percentile(terminal_pnls, 5)])) if np.any(terminal_pnls <= np.percentile(terminal_pnls, 5)) else 0.0,
            "P(profit)": float(np.mean(terminal_pnls > 0)),
            "min_pnl": float(np.min(terminal_pnls)),
            "max_pnl": float(np.max(terminal_pnls)),
        }

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "transition": self.tk.summary(),
            "hazard": self.hm.summary(),
            "geometry": self.sg.summary(),
            "policy": self.policy.summary(),
        }

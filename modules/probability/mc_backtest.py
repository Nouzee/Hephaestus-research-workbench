"""
Monte Carlo Policy Evaluator — E[R | π] via sample paths.

Replaces deterministic single-path backtest with proper MC estimation:
  - E[R|π] = mean over N independent sample paths
  - PnL distribution (not single scalar)
  - VaR, CVaR tail risk
  - Policy uncertainty band (±2σ)

Key: every result is a distribution, not a point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ===========================================================================
# MC Backtest Result
# ===========================================================================

@dataclass
class MCResult:
    """Container for Monte Carlo backtest results."""

    n_paths: int
    n_steps: int

    # PnL distribution
    pnl_paths: np.ndarray         # (n_paths, n_steps) — per-step PnL
    pnl_cumulative: np.ndarray    # (n_paths, n_steps) — cumulative equity
    pnl_terminal: np.ndarray      # (n_paths,) — final PnL per path

    # Risk metrics
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    var_95: float = 0.0           # 95% VaR
    cvar_95: float = 0.0          # 95% CVaR (expected shortfall)
    sharpe: float = 0.0
    prob_profit: float = 0.0      # P(PnL > 0)

    # Path statistics
    equity_mean: np.ndarray = None   # (n_steps,) mean equity curve
    equity_std: np.ndarray = None    # (n_steps,) ±1σ band

    @classmethod
    def from_paths(cls, pnl_paths: np.ndarray) -> "MCResult":
        """Construct from (n_paths, n_steps) PnL array."""
        n_paths, n_steps = pnl_paths.shape
        pnl_cum = np.cumsum(pnl_paths, axis=1)
        pnl_term = pnl_cum[:, -1]

        mean_p = float(np.mean(pnl_term))
        std_p = float(np.std(pnl_term))
        var95 = float(np.percentile(pnl_term, 5))
        cvar95 = float(np.mean(pnl_term[pnl_term <= var95])) if np.any(pnl_term <= var95) else var95
        sharpe = mean_p / max(std_p, 1e-8) * np.sqrt(n_steps)

        equity_mean = np.mean(pnl_cum, axis=0)
        equity_std = np.std(pnl_cum, axis=0)

        return cls(
            n_paths=n_paths, n_steps=n_steps,
            pnl_paths=pnl_paths, pnl_cumulative=pnl_cum, pnl_terminal=pnl_term,
            mean_pnl=mean_p, std_pnl=std_p, var_95=var95, cvar_95=cvar95,
            sharpe=sharpe, prob_profit=float(np.mean(pnl_term > 0)),
            equity_mean=equity_mean, equity_std=equity_std,
        )

    def summary(self) -> dict:
        return {
            "E[PnL]": self.mean_pnl,
            "Std[PnL]": self.std_pnl,
            "Sharpe": self.sharpe,
            "VaR_95": self.var_95,
            "CVaR_95": self.cvar_95,
            "P(profit)": self.prob_profit,
        }


# ===========================================================================
# Monte Carlo Backtest Engine
# ===========================================================================

@dataclass
class MCBacktest:
    """
    Monte Carlo policy evaluation.

    Usage
    -----
    >>> mc = MCBacktest(n_paths=500)
    >>> result = mc.evaluate(policy, state_generator, n_steps=1000)
    >>> print(f"E[PnL] = {result.mean_pnl:.2f} +- {result.std_pnl:.2f}")
    """

    n_paths: int = 500
    seed: int = 42

    def evaluate(
        self,
        policy,              # callable: state → action dict
        state_stream,        # iterable yielding StochasticState per step
        n_steps: int,
        fill_model=None,     # optional: state → fill probability
        adverse_model=None,  # optional: state → adverse cost
    ) -> MCResult:
        """
        Run n_paths independent Monte Carlo simulations.

        Each path: state_stream → policy(state) → fill → adverse → PnL.
        """
        rng = np.random.RandomState(self.seed)
        all_paths = np.zeros((self.n_paths, n_steps), dtype=np.float64)

        for p in range(self.n_paths):
            # Re-seed per path for independence
            path_rng = np.random.RandomState(rng.randint(0, 2**31))
            state_iter = iter(state_stream) if hasattr(state_stream, '__iter__') else state_stream

            for t in range(n_steps):
                try:
                    state = next(state_iter)
                except StopIteration:
                    break

                # policy can be callable or StochasticPolicy object
                if hasattr(policy, 'sample'):
                    action = policy.sample(state)
                else:
                    action = policy(state)

                sz = action.get("size_multiplier", 1.0)
                sp_m = action.get("spread_multiplier", 1.0)
                if sz <= 0:
                    all_paths[p, t] = 0.0
                    continue

                # Stochastic fill
                p_fill = 0.30 / max(sp_m, 0.5)
                if fill_model is not None:
                    p_fill = fill_model(state)

                if path_rng.random() > p_fill:
                    all_paths[p, t] = 0.0
                    continue

                # Spread capture
                spread_earned = float(np.mean(state.x_mean[:1])) * sp_m / 2 * sz if len(state.x_mean) > 0 else 0.0

                # Adverse selection (stochastic)
                adverse = 0.0
                if adverse_model is not None:
                    adverse = adverse_model(state)
                else:
                    # Simple: proportional to hazard
                    adverse = state.h_mean * spread_earned * 1.5

                all_paths[p, t] = spread_earned - adverse

        return MCResult.from_paths(all_paths)

    def compare_policies(
        self,
        policies: dict,       # {name: policy_callable}
        state_stream,
        n_steps: int,
    ) -> dict[str, MCResult]:
        """Evaluate multiple policies and return comparison."""
        results = {}
        for name, policy in policies.items():
            results[name] = self.evaluate(policy, state_stream, n_steps)
        return results

    def print_comparison(self, results: dict[str, MCResult]):
        """Print side-by-side policy comparison."""
        print(f"\n  {'Policy':<16s} {'E[PnL]':>12s} {'Std':>10s} "
              f"{'Sharpe':>8s} {'VaR95':>10s} {'P(profit)':>10s}")
        print(f"  {'─'*16} {'─'*12} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")
        for name, r in results.items():
            print(f"  {name:<16s} {r.mean_pnl:>+12,.0f} {r.std_pnl:>10,.0f} "
                  f"{r.sharpe:>8.2f} {r.var_95:>+10,.0f} {r.prob_profit:>9.1%}")

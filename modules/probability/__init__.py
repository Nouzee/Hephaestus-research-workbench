"""
Hephaestus Probability Engine v1

Stochastic process representation of limit order book markets.

  stochastic_state  — S_t = (X_t, Z_t, H_t, M_t), unified random state
  transition_kernel — P(Z_{t+1}|Z_t), Markov kernel, entropy rate, spectral gap
  hazard_model      — h(X_t) = P(adverse regime | X_t), survival probability
  mc_backtest       — E[R|pi] via Monte Carlo paths, PnL distribution
  policy            — pi(a|S_t), probabilistic action selection
"""

__all__ = ["stochastic_state", "transition_kernel", "hazard_model",
           "mc_backtest", "policy"]

"""
Execution Subpackage — realistic fill simulation + PnL attribution.

  fill_model            — P(fill) from spread, queue, imbalance, volatility
  queue_model           — queue position + depth tracking
  execution_simulator   — realistic PnL with fill probabilities
  pnl_attribution       — PnL decomposition (spread / inventory / adverse / missed)

Replaces the old "quote = fill" assumption with a realistic execution layer.
"""

__all__ = ["fill_model", "queue_model", "execution_simulator", "pnl_attribution", "hardened_simulator"]

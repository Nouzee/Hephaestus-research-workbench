"""
State Subpackage — Unified Market Belief State.

  market_state   — single MarketState object, all modules read/write same state
  state_updater  — orchestrates module updates into coherent state transitions
"""

__all__ = ["market_state", "state_updater"]

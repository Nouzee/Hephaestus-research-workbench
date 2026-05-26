"""
Research Subpackage — Market Microstructure Deconstruction Lab.

  market_state     — unified MarketState (flow + liquidity + impact + memory)
  market_decon     — batch decomposer (raw ticks → 3 layers)
  causal_graph     — formal causal graph (edges, strength, lag)
  impact_kernel    — callable shock→price_path function
  market_generator — synthetic market sandbox + fragility scanner

Not for trading. For understanding how markets generate price changes.
"""

__all__ = ["market_state", "market_decon", "causal_graph",
           "impact_kernel", "market_generator"]

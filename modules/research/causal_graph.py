"""
Causal Graph — formal directed edges between market variables.

Not a correlation matrix. A directed graph where each edge has:
  source → target
  strength  (max lead correlation)
  best_lag  (optimal lead in batches)
  direction (uni-directional or bi-directional)

Answers: "How does the market generate itself?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ===========================================================================
# Edge + Graph
# ===========================================================================

@dataclass
class CausalEdge:
    """A directed causal link between two market variables."""
    source: str
    target: str
    strength: float       # max |corr| at best_lag > 0
    best_lag: int         # optimal lead (batches)
    is_bidirectional: bool = False
    reverse_strength: float = 0.0


@dataclass
class CausalGraphConfig:
    max_lag: int = 10
    min_strength: float = 0.05   # below this, edge is filtered out


# ===========================================================================
# Core: Causal Graph Builder
# ===========================================================================

class CausalGraph:
    """
    Directed causal graph of market microstructure variables.

    Usage
    -----
    >>> cg = CausalGraph()
    >>> cg.fit(feature_matrix, var_names)
    >>> cg.report()
    >>> paths = cg.causal_paths("flow_persistence")  # downstream effects
    """

    def __init__(self, config: Optional[CausalGraphConfig] = None):
        self.config = config or CausalGraphConfig()
        self.edges: list[CausalEdge] = []
        self.var_names: list[str] = []
        self.n_vars: int = 0

        # Adjacency: adjacency[i,j] = strength of i→j
        self.adjacency: Optional[np.ndarray] = None
        self.lag_matrix: Optional[np.ndarray] = None

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, var_names: list[str]) -> "CausalGraph":
        """
        Build causal graph from (N, D) standardized feature matrix.

        For each pair (i, j): find max |corr(xi[t], xj[t+k])| for k=1..max_lag.
        If significant, add edge i→j with strength and best_lag.
        """
        cfg = self.config
        self.var_names = var_names
        self.n_vars = len(var_names)
        N, D = features.shape

        self.adjacency = np.zeros((D, D), dtype=np.float32)
        self.lag_matrix = np.zeros((D, D), dtype=np.int32)
        self.edges = []

        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                xi = features[:, i]
                xj = features[:, j]

                best_corr, best_lag = 0.0, 0
                for k in range(1, min(cfg.max_lag + 1, N // 2)):
                    c = np.corrcoef(xi[:-k], xj[k:])[0, 1]
                    if not np.isnan(c) and abs(c) > abs(best_corr):
                        best_corr, best_lag = c, k

                if abs(best_corr) >= cfg.min_strength:
                    self.adjacency[i, j] = float(abs(best_corr))
                    self.lag_matrix[i, j] = best_lag

                    # Check bidirectionality
                    rev_corr = 0.0
                    for k in range(1, min(cfg.max_lag + 1, N // 2)):
                        c = np.corrcoef(xj[:-k], xi[k:])[0, 1]
                        if not np.isnan(c) and abs(c) > abs(rev_corr):
                            rev_corr = c

                    is_bi = abs(rev_corr) >= cfg.min_strength
                    self.edges.append(CausalEdge(
                        source=var_names[i],
                        target=var_names[j],
                        strength=float(abs(best_corr)),
                        best_lag=best_lag,
                        is_bidirectional=is_bi,
                        reverse_strength=float(abs(rev_corr)) if is_bi else 0.0,
                    ))

        # Sort by strength
        self.edges.sort(key=lambda e: e.strength, reverse=True)

        return self

    # ── Query ────────────────────────────────────────────────────────

    def causal_paths(self, source: str, max_depth: int = 3) -> list[list[str]]:
        """Find all downstream causal paths from a source variable."""
        if source not in self.var_names:
            return []
        src_idx = self.var_names.index(source)
        paths = [[source]]

        for _ in range(max_depth):
            new_paths = []
            for path in paths:
                last = path[-1]
                if last not in self.var_names:
                    continue
                last_idx = self.var_names.index(last)
                for edge in self.edges:
                    if edge.source == last and edge.target not in path:
                        new_paths.append(path + [edge.target])
            paths.extend(new_paths)

        return [p for p in paths if len(p) > 1]

    def upstream_of(self, target: str) -> list[CausalEdge]:
        """All edges pointing TO this variable."""
        return [e for e in self.edges if e.target == target]

    def downstream_of(self, source: str) -> list[CausalEdge]:
        """All edges FROM this variable."""
        return [e for e in self.edges if e.source == source]

    # ── Generative chain ─────────────────────────────────────────────

    def generative_chain(self) -> str:
        """
        Extract the primary generative chain: flow → liquidity → impact.

        Returns a string description of the causal flow.
        """
        # Find strongest edge from flow vars to liquidity vars
        flow_vars = [v for v in self.var_names if v in [
            "trade_arrival_rate", "signed_imbalance", "flow_persistence",
            "size_dispersion", "cancel_burst_ratio", "buy_sell_volume_ratio"]]
        liq_vars = [v for v in self.var_names if v in [
            "spread_bps", "total_depth", "queue_pressure",
            "depth_imbalance", "liquidity_tension"]]
        impact_vars = [v for v in self.var_names if v in [
            "realized_volatility", "immediate_impact_corr", "nonlinear_response"]]

        chain = []
        for fv in flow_vars:
            for lv in liq_vars:
                edge = next((e for e in self.edges
                            if e.source == fv and e.target == lv), None)
                if edge:
                    chain.append(f"{fv} → {lv} ({edge.strength:.2f})")
                    for iv in impact_vars:
                        edge2 = next((e for e in self.edges
                                     if e.source == lv and e.target == iv), None)
                        if edge2:
                            chain.append(f"  {lv} → {iv} ({edge2.strength:.2f})")
                    break
            if chain:
                break

        return "\n".join(chain[:10]) if chain else "no clear chain found"

    # ── Report ───────────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "n_variables": self.n_vars,
            "n_edges": len(self.edges),
            "top_edges": [
                {"source": e.source, "target": e.target,
                 "strength": e.strength, "lag": e.best_lag,
                 "bidirectional": e.is_bidirectional}
                for e in self.edges[:15]
            ],
            "generative_chain": self.generative_chain(),
        }

    def print_report(self):
        r = self.report()
        print(f"\n{'═'*65}")
        print(f"  Causal Graph — Market Generative Structure")
        print(f"{'═'*65}")
        print(f"  Variables: {r['n_variables']}  Edges: {r['n_edges']}")
        print(f"\n  Top edges:")
        print(f"  {'Source':<28s} {'→ Target':<28s} {'Str':>5s} {'Lag':>4s} {'Bi?':>4s}")
        print(f"  {'─'*28} {'─'*28} {'─'*5} {'─'*4} {'─'*4}")
        for e in r["top_edges"]:
            print(f"  {e['source']:<28s} → {e['target']:<27s} "
                  f"{e['strength']:>.3f} {e['lag']:>4d} "
                  f"{'yes' if e['bidirectional'] else 'no':>4s}")
        print(f"\n  Generative Chain:")
        for line in r["generative_chain"].split("\n"):
            print(f"    {line}")
        print(f"{'═'*65}")

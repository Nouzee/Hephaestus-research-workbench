"""
Market Deconstruction Lab — decompose market behavior into generative layers.

Three-layer decomposition of every batch:
  L1 — ORDER FLOW (behavior):  who is acting, how, and with what persistence?
  L2 — LIQUIDITY (mechanism):   how does the order book respond to flow?
  L3 — PRICE IMPACT (physics):  how does liquidity tension become price movement?

PLUS:
  Causal Structure Map — which variables cause which others?
  Signal Taxonomy      — classify each as Generator / Mediator / Outcome

Answers the question: "how does the market generate price changes?"
Not: "what will the price be next?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class MarketDeconConfig:
    """Configuration for market deconstruction."""

    # Impact kernel estimation
    kernel_max_lag: int = 20        # max lag for impact decay estimation
    kernel_min_events: int = 50     # min events for stable kernel

    # Causal mapping
    granger_max_lag: int = 5        # max lag for Granger causality test
    granger_significance: float = 0.05  # p-value threshold

    # Taxonomy thresholds
    cause_lead_threshold: float = 0.05   # min |corr| at lead>0 to be "cause"
    effect_lag_threshold: float = 0.05   # min |corr| at lag>0 to be "effect"

    seed: int = 42


# ===========================================================================
# Layer Decomposer
# ===========================================================================

class LayerDecomposer:
    """
    Decompose each batch into three generative layers.

    Usage
    -----
    >>> dec = LayerDecomposer()
    >>> layers = dec.decompose_batch(raw_tick_data)
    >>> print(layers["order_flow"]["signed_imbalance"])
    """

    def __init__(self, config: Optional[MarketDeconConfig] = None):
        self.config = config or MarketDeconConfig()

    def decompose_batch(
        self,
        trade_px: np.ndarray,
        trade_sz: np.ndarray,
        trade_side: np.ndarray,
        bid_px: np.ndarray,
        bid_sz: np.ndarray,
        ask_px: np.ndarray,
        ask_sz: np.ndarray,
        duration_ms: np.ndarray,
    ) -> dict:
        """
        Decompose one batch into three layers.

        Returns dict with keys: order_flow, liquidity, price_impact.
        Each value is a dict of named scalar features.
        """
        eps = 1e-12

        # ── L1: Order Flow Layer ─────────────────────────────────
        T = len(trade_px)

        # Trade arrival intensity
        trade_arrival_rate = T / max(np.sum(duration_ms) / 1000.0, eps)

        # Directional flow
        signed_flow = np.sum(trade_side * trade_sz)
        signed_imbalance = signed_flow / max(np.sum(trade_sz), eps)

        # Trade size dispersion
        size_cv = float(np.std(trade_sz) / max(np.mean(trade_sz), eps))

        # Flow persistence (lag-1 autocorrelation of trade side)
        if T > 5:
            side_acf = np.corrcoef(trade_side[:-1], trade_side[1:])[0, 1]
            side_acf = 0.0 if np.isnan(side_acf) else float(side_acf)
        else:
            side_acf = 0.0

        # Cancellation proxy: event rate compression
        if T > 5:
            med_dur = np.median(duration_ms)
            min_dur = np.percentile(duration_ms, 5)
            cancel_burst = med_dur / max(min_dur, eps)
        else:
            cancel_burst = 1.0

        # Buyer/seller-initiated volume ratio
        buy_vol = np.sum(trade_sz[trade_side > 0]) if np.any(trade_side > 0) else 0.0
        sell_vol = np.sum(trade_sz[trade_side < 0]) if np.any(trade_side < 0) else 0.0
        buy_sell_ratio = buy_vol / max(sell_vol, eps)

        order_flow = {
            "trade_arrival_rate": float(trade_arrival_rate),
            "signed_imbalance": float(signed_imbalance),
            "size_dispersion": float(size_cv),
            "flow_persistence": float(side_acf),
            "cancel_burst_ratio": float(cancel_burst),
            "buy_sell_volume_ratio": float(buy_sell_ratio),
        }

        # ── L2: Liquidity Layer ──────────────────────────────────
        mid_px = (bid_px + ask_px) / 2.0
        spread_bps = np.mean(ask_px - bid_px) / max(np.mean(mid_px), eps) * 10000

        total_depth = np.mean(bid_sz + ask_sz)
        depth_imbalance = (np.mean(bid_sz) - np.mean(ask_sz)) / max(total_depth, eps)

        # Queue pressure: how fast is depth consumed relative to arrival
        depth_consumption = trade_arrival_rate / max(total_depth, eps)

        # Spread volatility
        spread_vol = float(np.std(ask_px - bid_px) / max(np.mean(ask_px - bid_px), eps))

        # Liquidity tension: spread * depth interaction
        liquidity_tension = spread_bps / max(total_depth, eps)

        # Depth replenishment proxy: does depth recover after trades?
        # Measure: correlation between depth changes and trade volume
        if T > 10:
            depth_changes = np.diff(bid_sz + ask_sz)
            trade_vol = trade_sz[:-1]
            replenish_corr = np.corrcoef(depth_changes, trade_vol)[0, 1]
            replenish_corr = 0.0 if np.isnan(replenish_corr) else float(replenish_corr)
        else:
            replenish_corr = 0.0

        liquidity = {
            "spread_bps": float(spread_bps),
            "total_depth": float(total_depth),
            "depth_imbalance": float(depth_imbalance),
            "queue_pressure": float(depth_consumption),
            "spread_volatility": float(spread_vol),
            "liquidity_tension": float(liquidity_tension),
            "depth_replenish_corr": float(replenish_corr),
        }

        # ── L3: Price Impact Layer ───────────────────────────────
        mid_ret = np.diff(mid_px) / (np.abs(mid_px[:-1]) + eps)
        realized_vol = float(np.std(mid_ret) * np.sqrt(T))

        # Immediate impact: how much does a unit of signed flow move price?
        if T > 2:
            # Regression: mid_ret ~ signed_flow[t-1]
            if len(mid_ret) > 1 and len(trade_side) > 1:
                n_eff = min(len(mid_ret), len(trade_side)) - 1
                if n_eff > 5:
                    impact_reg = np.corrcoef(
                        trade_side[:n_eff] * trade_sz[:n_eff],
                        mid_ret[:n_eff]
                    )[0, 1]
                    impact_reg = 0.0 if np.isnan(impact_reg) else float(impact_reg)
                else:
                    impact_reg = 0.0
            else:
                impact_reg = 0.0
        else:
            impact_reg = 0.0

        # Nonlinear response: does large flow have disproportionate impact?
        flow_mag = np.abs(trade_side * trade_sz)
        if T > 10:
            # Split into small/large flows, compare impact
            med_flow = np.median(flow_mag)
            small_mask = flow_mag[:-1] < med_flow
            large_mask = flow_mag[:-1] >= med_flow
            impact_small = float(np.mean(np.abs(mid_ret[small_mask]))) if np.any(small_mask) else 0.0
            impact_large = float(np.mean(np.abs(mid_ret[large_mask]))) if np.any(large_mask) else 0.0
            nonlinearity = impact_large / max(impact_small, eps)
        else:
            nonlinearity = 1.0

        # Volatility clustering: does vol predict vol?
        if T > 10:
            abs_ret = np.abs(mid_ret)
            vol_persist = np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1]
            vol_persist = 0.0 if np.isnan(vol_persist) else float(vol_persist)
        else:
            vol_persist = 0.0

        price_impact = {
            "realized_volatility": float(realized_vol),
            "immediate_impact_corr": float(impact_reg),
            "nonlinear_response": float(nonlinearity),
            "volatility_persistence": float(vol_persist),
        }

        return {
            "order_flow": order_flow,
            "liquidity": liquidity,
            "price_impact": price_impact,
        }


# ===========================================================================
# Causal Structure Mapper
# ===========================================================================

class CausalMapper:
    """
    Build a causal graph from time series of layer variables.

    Uses pairwise lead-lag correlation to infer causal direction:
      - If corr(X[t], Y[t+k]) is significant for k>0: X → Y (X Granger-causes Y)
      - If corr(Y[t], X[t+k]) is significant for k>0: Y → X

    Output: adjacency matrix + causal position classification.
    """

    def __init__(self, config: Optional[MarketDeconConfig] = None):
        self.config = config or MarketDeconConfig()

        # All variable names (flattened from layers)
        self.variables: list[str] = []
        self.n_vars: int = 0

        # Causal graph
        self.adjacency: Optional[np.ndarray] = None    # (n_vars, n_vars): i→j strength
        self.causal_position: dict[str, str] = {}       # generator / mediator / outcome

    def fit(self, feature_matrix: np.ndarray, var_names: list[str]) -> "CausalMapper":
        """
        Build causal graph from (N, D) feature matrix.

        Parameters
        ----------
        feature_matrix : (N, D) — time series of each variable
        var_names      : length D — variable names
        """
        cfg = self.config
        self.variables = var_names
        self.n_vars = len(var_names)
        N, D = feature_matrix.shape

        self.adjacency = np.zeros((D, D), dtype=np.float32)
        max_lag = cfg.granger_max_lag

        # For each pair (i, j): test i→j (i leads j)
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                xi = feature_matrix[:, i]
                xj = feature_matrix[:, j]

                # Test i → j: corr(xi[t], xj[t+k]) for k=1..max_lag
                lead_corrs = []
                for k in range(1, min(max_lag + 1, N // 2)):
                    c = np.corrcoef(xi[:-k], xj[k:])[0, 1]
                    if not np.isnan(c):
                        lead_corrs.append(c)

                if lead_corrs:
                    max_abs = max(abs(c) for c in lead_corrs)
                    self.adjacency[i, j] = float(max_abs)

        # Classify each variable
        self._classify()

        return self

    def _classify(self):
        """Classify each variable as Generator, Mediator, or Outcome."""
        out_strength = self.adjacency.sum(axis=1)   # how much i causes others
        in_strength = self.adjacency.sum(axis=0)     # how much i is caused by others
        total = out_strength + in_strength + 1e-12

        for idx, name in enumerate(self.variables):
            out_ratio = out_strength[idx] / total[idx]

            if out_ratio > 0.6:
                self.causal_position[name] = "generator"
            elif out_ratio > 0.35:
                self.causal_position[name] = "mediator"
            else:
                self.causal_position[name] = "outcome"

    def report(self) -> dict:
        """Generate causal structure report."""
        if self.adjacency is None:
            return {"error": "Not fitted"}

        # Find strongest causal links
        links = []
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i != j and self.adjacency[i, j] > 0.03:
                    links.append({
                        "from": self.variables[i],
                        "to": self.variables[j],
                        "strength": float(self.adjacency[i, j]),
                    })

        links.sort(key=lambda l: l["strength"], reverse=True)

        # Count by position
        position_counts = {"generator": 0, "mediator": 0, "outcome": 0}
        for v, pos in self.causal_position.items():
            position_counts[pos] += 1

        return {
            "n_variables": self.n_vars,
            "top_links": links[:20],
            "position_counts": position_counts,
            "classification": self.causal_position,
        }

    def print_report(self):
        """Print human-readable causal structure report."""
        r = self.report()
        if "error" in r:
            print(r["error"])
            return

        print(f"\n{'═'*65}")
        print(f"  Causal Structure Map")
        print(f"{'═'*65}")

        # Position summary
        print(f"\n  Signal Taxonomy ({r['n_variables']} variables):")
        for pos in ["generator", "mediator", "outcome"]:
            vars_in_pos = [v for v, p in r["classification"].items() if p == pos]
            if vars_in_pos:
                print(f"    {pos.upper()} ({len(vars_in_pos)}): {', '.join(vars_in_pos[:8])}"
                      f"{'...' if len(vars_in_pos) > 8 else ''}")

        # Top causal links
        print(f"\n  Top Causal Links (i → j):")
        print(f"  {'From':<28s} {'→':>3s} {'To':<28s} {'Strength':>8s}")
        print(f"  {'─'*28} {'─'*3} {'─'*28} {'─'*8}")
        for link in r["top_links"][:15]:
            print(f"  {link['from']:<28s}  →  {link['to']:<28s} {link['strength']:>8.4f}")

        print(f"{'═'*65}")


# ===========================================================================
# Impact Kernel Estimator
# ===========================================================================

class ImpactKernel:
    """
    Estimate how order flow events decay in their price impact over time.

    Kernel: h(τ) = E[Δp(t+τ) | flow_event(t)]
    where τ is the lag in batches after the event.

    The shape of h(τ) tells us:
      - Power-law decay → long memory, meta-order splitting
      - Exponential decay → fast mean reversion
      - Concave → market is resilient (liquidity replenishes)
    """

    def __init__(self, config: Optional[MarketDeconConfig] = None):
        self.config = config or MarketDeconConfig()
        self.kernel: Optional[np.ndarray] = None   # h(τ) for τ=0..max_lag
        self.decay_type: str = "unknown"

    def estimate(
        self,
        flow_events: np.ndarray,      # (N,) signed flow magnitude
        price_changes: np.ndarray,    # (N,) subsequent price changes
    ) -> "ImpactKernel":
        """
        Estimate the impact kernel from flow events and price responses.

        Parameters
        ----------
        flow_events   : (N,) — signed order flow at each time step
        price_changes : (N,) — price change at each time step
        """
        cfg = self.config
        max_lag = cfg.kernel_max_lag
        N = min(len(flow_events), len(price_changes))

        # Find significant flow events (top quartile by magnitude)
        abs_flow = np.abs(flow_events[:N])
        threshold = np.percentile(abs_flow, 75)
        event_mask = abs_flow > threshold
        event_signs = np.sign(flow_events[:N][event_mask])

        if event_mask.sum() < cfg.kernel_min_events:
            print(f"  [ImpactKernel] Too few events ({event_mask.sum()})")
            return self

        # For each event at time t, accumulate Δp(t+τ) * sign(flow)
        n_events = event_mask.sum()
        event_times = np.where(event_mask)[0]
        self.kernel = np.zeros(max_lag + 1, dtype=np.float64)
        counts = np.zeros(max_lag + 1, dtype=np.int32)

        for ev_t, ev_sign in zip(event_times, event_signs):
            end_t = min(ev_t + max_lag + 1, N)
            tau_range = end_t - ev_t
            self.kernel[:tau_range] += price_changes[ev_t:end_t] * ev_sign
            counts[:tau_range] += 1

        # Average
        valid = counts > 0
        self.kernel[valid] /= counts[valid]
        self.kernel[~valid] = 0.0

        # Classify decay type
        self._classify_decay()

        return self

    def _classify_decay(self):
        """Classify the decay shape of the kernel."""
        if self.kernel is None:
            return

        k = self.kernel
        half_life = None
        for tau in range(1, len(k)):
            if abs(k[tau]) < abs(k[0]) / 2:
                half_life = tau
                break

        # Fit power law vs exponential
        tau = np.arange(1, len(k))
        log_k = np.log(np.abs(k[1:]) + 1e-12)
        log_tau = np.log(tau)

        # Power law: log(k) ~ -α * log(τ) → linear in log-log
        # Exponential: log(k) ~ -λ * τ → linear in semi-log
        if len(tau) > 5:
            slope_ll, _, _, _, _ = stats.linregress(log_tau, log_k)
            slope_sl, _, _, _, _ = stats.linregress(tau, log_k)

            r2_ll = np.corrcoef(log_tau, log_k)[0, 1] ** 2
            r2_sl = np.corrcoef(tau, log_k)[0, 1] ** 2

            if r2_ll > r2_sl * 1.2:
                self.decay_type = f"power_law (α≈{abs(slope_ll):.2f})"
            elif r2_sl > r2_ll * 1.2:
                self.decay_type = f"exponential (λ≈{abs(slope_sl):.4f})"
            else:
                self.decay_type = "mixed"

        if half_life is not None:
            self.decay_type += f", half_life≈{half_life}"

    def report(self) -> dict:
        if self.kernel is None:
            return {"error": "Not estimated"}
        return {
            "kernel": self.kernel.tolist(),
            "decay_type": self.decay_type,
            "peak_impact": float(self.kernel[0]),
            "peak_lag": int(np.argmax(np.abs(self.kernel))),
        }

    def print_report(self):
        r = self.report()
        if "error" in r:
            print(r["error"])
            return
        print(f"\n  Impact Kernel:")
        print(f"    Decay type:    {r['decay_type']}")
        print(f"    Peak impact:   {r['peak_impact']:.6f}")
        print(f"    Peak lag:      {r['peak_lag']} batches")
        print(f"    Kernel (first 10 lags): "
              f"{[f'{v:.6f}' for v in r['kernel'][:10]]}")

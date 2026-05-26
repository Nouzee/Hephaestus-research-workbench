"""
Market Deconstruction V1 — unified pipeline.

  1. Load raw tick data
  2. Decompose into MarketState per batch
  3. Build CausalGraph (formal edges, not correlation table)
  4. Fit ImpactKernel (callable shock→path function)
  5. Run MarketGenerator (synthetic sandbox)
  6. Fragility scan — find conditions where the market breaks
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.research.market_state import MarketState
from modules.research.market_decon import LayerDecomposer
from modules.research.causal_graph import CausalGraph, CausalGraphConfig
from modules.research.impact_kernel import ImpactKernel, ImpactKernelConfig
from modules.research.market_generator import MarketGenerator, MarketGeneratorConfig


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Market Deconstruction V1")
print("  State → Causal Graph → Impact Kernel → Generator → Fragility")
print("=" * 65)


# ===========================================================================
# [1] Load + decompose into MarketState
# ===========================================================================

print("\n[1] Loading + decomposing into MarketState ...")
t0 = time.perf_counter()

raw = pl.read_parquet(SOURCE)
offset = 50
N_raw = raw.shape[0] - offset
n_batches = N_raw // BATCH_SIZE

dec = LayerDecomposer()
states: list[MarketState] = []

for b in range(n_batches):
    s = offset + b * BATCH_SIZE
    e = offset + (b+1) * BATCH_SIZE
    batch = raw.slice(s, e - s)

    layers = dec.decompose_batch(
        batch["trade_px"].to_numpy().astype(np.float64),
        batch["trade_sz"].to_numpy().astype(np.float64),
        batch["trade_side"].to_numpy().astype(np.float64),
        batch["bid_px"].to_numpy().astype(np.float64),
        batch["bid_sz"].to_numpy().astype(np.float64),
        batch["ask_px"].to_numpy().astype(np.float64),
        batch["ask_sz"].to_numpy().astype(np.float64),
        batch["duration_ms"].to_numpy().astype(np.float64),
    )

    state = MarketState(
        trade_arrival_rate=layers["order_flow"]["trade_arrival_rate"],
        signed_imbalance=layers["order_flow"]["signed_imbalance"],
        size_dispersion=layers["order_flow"]["size_dispersion"],
        flow_persistence=layers["order_flow"]["flow_persistence"],
        cancel_burst_ratio=layers["order_flow"]["cancel_burst_ratio"],
        buy_sell_volume_ratio=layers["order_flow"]["buy_sell_volume_ratio"],
        spread_bps=layers["liquidity"]["spread_bps"],
        total_depth=layers["liquidity"]["total_depth"],
        depth_imbalance=layers["liquidity"]["depth_imbalance"],
        queue_pressure=layers["liquidity"]["queue_pressure"],
        spread_volatility=layers["liquidity"]["spread_volatility"],
        liquidity_tension=layers["liquidity"]["liquidity_tension"],
        depth_replenish_corr=layers["liquidity"]["depth_replenish_corr"],
        realized_volatility=layers["price_impact"]["realized_volatility"],
        immediate_impact_corr=layers["price_impact"]["immediate_impact_corr"],
        nonlinear_response=layers["price_impact"]["nonlinear_response"],
        volatility_persistence=layers["price_impact"]["volatility_persistence"],
    )
    states.append(state)

del raw
print(f"  {n_batches} batches → {len(states)} MarketStates  "
      f"time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Build feature matrix + Causal Graph
# ===========================================================================

print("\n[2] Building Causal Graph ...")
t0 = time.perf_counter()

feat_matrix = np.array([s.to_vector() for s in states], dtype=np.float32)
var_names = MarketState.vector_names()

# Standardize
fm = feat_matrix
fm_mean = fm.mean(axis=0)
fm_std = np.maximum(fm.std(axis=0), 1e-8)
fm_z = (fm - fm_mean) / fm_std

# Remove extreme outliers (>10 sigma)
fm_z = np.clip(fm_z, -10, 10)

cg = CausalGraph(CausalGraphConfig(max_lag=10, min_strength=0.05))
cg.fit(fm_z[:, :17], var_names[:17])  # core 17 vars, skip memory vars
cg.print_report()

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Fit Impact Kernel
# ===========================================================================

print("\n[3] Fitting Impact Kernel ...")
t0 = time.perf_counter()

# Use signed_imbalance as flow shock, realized_volatility as price change
flow_idx = var_names.index("signed_imbalance")
price_idx = var_names.index("realized_volatility")

ik = ImpactKernel(ImpactKernelConfig(max_lag=30, min_events=50))
ik.fit(fm_z[:, flow_idx], fm_z[:, price_idx])
ik.print_report()

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Market Generator + Fragility Scan
# ===========================================================================

print("\n[4] Market Generator — synthetic sandbox ...")
t0 = time.perf_counter()

# Calibrate generator from data
base_arrival = float(np.mean([s.trade_arrival_rate for s in states]))
base_spread = float(np.mean([s.spread_bps for s in states]))
base_depth = float(np.mean([s.total_depth for s in states]))
flow_persist = float(np.mean([s.flow_persistence for s in states]))
flow_vol = float(np.std([s.signed_imbalance for s in states]))

print(f"  Calibrated: arrival={base_arrival:.1f}/s  spread={base_spread:.2f}bps  "
      f"depth={base_depth:.2f}  persistence={flow_persist:.2f}  flow_vol={flow_vol:.2f}")

mg = MarketGenerator(
    config=MarketGeneratorConfig(
        base_arrival_rate=base_arrival,
        base_spread_bps=base_spread,
        base_depth=base_depth,
        flow_persistence=flow_persist,
        flow_volatility=flow_vol,
        n_steps=500,
    ),
    kernel=ik,
)

# Generate baseline path
paths = mg.generate(n_steps=500)
print(f"  Baseline: max_impact={np.max(np.abs(paths['price_impact'])):.4f}  "
      f"max_spread={np.max(paths['spread_bps']):.2f}bps  "
      f"mean_recovery={np.mean(paths['recovery_state']):.2f}")

# Stress test
stress = mg.stress_test(shock_magnitude=5.0, n_steps=200)
print(f"  Stress (5σ shock): max_impact={np.max(np.abs(stress['price_impact'])):.4f}  "
      f"peak_recovery={np.max(stress['recovery_state']):.2f}")

# Fragility scan
print(f"\n  Running fragility scan ...")
scan = mg.fragility_scan(
    flow_persistence_range=[0.3, 0.5, 0.7, 0.9],
    depth_recovery_range=[0.1, 0.3, 0.5, 0.7],
    n_steps=500,
)
mg.print_fragility_report(scan)

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Market Deconstruction V1 complete.")
print(f"{'═'*65}")
print(f"\n  Modules built:")
print(f"    modules/research/market_state.py     — unified MarketState (20 vars)")
print(f"    modules/research/causal_graph.py      — {cg.n_vars} nodes, {len(cg.edges)} edges")
print(f"    modules/research/impact_kernel.py     — {ik.decay_type}, half-life={ik.half_life:.1f}")
print(f"    modules/research/market_generator.py  — synthetic sandbox + fragility scan")
print(f"    modules/research/market_decon.py      — batch decomposer (17 raw → 3 layers)")
print(f"{'═'*65}")

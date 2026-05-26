"""
Market Deconstruction Pipeline — decompose BTC L2 data into generative layers.

Runs the Market Deconstruction Lab on the full dataset:
  1. Load raw tick-level order book data
  2. Decompose each batch into Order Flow / Liquidity / Price Impact layers
  3. Build causal structure map between all variables
  4. Estimate impact kernel (how flow events decay into price impact)
  5. Classify every variable as Generator / Mediator / Outcome

Output: market generative structure, not trading signals.
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.research.market_decon import (
    LayerDecomposer, CausalMapper, ImpactKernel, MarketDeconConfig
)


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Market Deconstruction Lab")
print("  Generative Layer Decomposition + Causal Mapping")
print("=" * 65)


# ===========================================================================
# [1] Load raw tick-level data
# ===========================================================================

print("\n[1] Loading raw tick-level data ...")
t0 = time.perf_counter()

# Load ALL raw columns needed for decomposition
raw = pl.read_parquet(SOURCE)
offset = 50  # align with MatrixBuilder NaN drop
N_raw = raw.shape[0] - offset

# Slice to manageable size for decomposition
n_batches = N_raw // BATCH_SIZE
print(f"  Raw ticks: {N_raw:,}  Batches: {n_batches}  "
      f"Time: {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Decompose each batch into three layers
# ===========================================================================

print(f"\n[2] Decomposing {n_batches} batches into 3 layers ...")
t0 = time.perf_counter()

dec = LayerDecomposer()

# Collect layer features across all batches
all_order_flow = []
all_liquidity = []
all_price_impact = []

# Process in chunks for memory
chunk_size = 500
for chunk_start in range(0, n_batches, chunk_size):
    chunk_end = min(chunk_start + chunk_size, n_batches)

    for b in range(chunk_start, chunk_end):
        s = offset + b * BATCH_SIZE
        e = offset + (b + 1) * BATCH_SIZE

        # Extract raw columns for this batch
        batch_raw = raw.slice(s, e - s)

        trade_px = batch_raw["trade_px"].to_numpy().astype(np.float64)
        trade_sz = batch_raw["trade_sz"].to_numpy().astype(np.float64)
        trade_side = batch_raw["trade_side"].to_numpy().astype(np.float64)
        bid_px = batch_raw["bid_px"].to_numpy().astype(np.float64)
        bid_sz = batch_raw["bid_sz"].to_numpy().astype(np.float64)
        ask_px = batch_raw["ask_px"].to_numpy().astype(np.float64)
        ask_sz = batch_raw["ask_sz"].to_numpy().astype(np.float64)
        duration_ms = batch_raw["duration_ms"].to_numpy().astype(np.float64)

        layers = dec.decompose_batch(
            trade_px, trade_sz, trade_side,
            bid_px, bid_sz, ask_px, ask_sz,
            duration_ms,
        )

        all_order_flow.append(layers["order_flow"])
        all_liquidity.append(layers["liquidity"])
        all_price_impact.append(layers["price_impact"])

    print(f"  [{chunk_end}/{n_batches}] batches processed")

del raw
print(f"  Time: {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Build feature matrix for causal mapping
# ===========================================================================

print(f"\n[3] Building causal structure map ...")
t0 = time.perf_counter()

# Flatten layer dicts into arrays
of_keys = list(all_order_flow[0].keys())
liq_keys = list(all_liquidity[0].keys())
pi_keys = list(all_price_impact[0].keys())

all_var_names = (
    [f"OF:{k}" for k in of_keys] +
    [f"LQ:{k}" for k in liq_keys] +
    [f"PI:{k}" for k in pi_keys]
)

n_samples = len(all_order_flow)
feature_matrix = np.zeros((n_samples, len(all_var_names)), dtype=np.float32)

col = 0
for k in of_keys:
    feature_matrix[:, col] = [b[k] for b in all_order_flow]
    col += 1
for k in liq_keys:
    feature_matrix[:, col] = [b[k] for b in all_liquidity]
    col += 1
for k in pi_keys:
    feature_matrix[:, col] = [b[k] for b in all_price_impact]
    col += 1

# Standardize
fm_mean = feature_matrix.mean(axis=0)
fm_std = np.maximum(feature_matrix.std(axis=0), 1e-8)
feature_matrix_z = (feature_matrix - fm_mean) / fm_std

# Causal mapping
mapper = CausalMapper()
mapper.fit(feature_matrix_z, all_var_names)
mapper.print_report()

print(f"  Time: {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Impact kernel estimation
# ===========================================================================

print(f"\n[4] Estimating impact kernel ...")
t0 = time.perf_counter()

# Use signed flow as events, mid-price return as response
flow_events = feature_matrix[:, of_keys.index("signed_imbalance")]
# Use price impact "realized_volatility" as response proxy
# Actually, better: reconstruct batch-level mid returns
# For now use the nonlinear response as a price change proxy
price_response = np.diff(feature_matrix[:, pi_keys.index("realized_volatility")])
price_response = np.append(price_response, 0.0)  # pad

kernel = ImpactKernel()
kernel.estimate(
    flow_events=flow_events,
    price_changes=price_response,
)
kernel.print_report()

print(f"  Time: {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [5] Layer-level summary statistics
# ===========================================================================

print(f"\n[5] Layer statistics ...")

for layer_name, keys, data in [
    ("ORDER FLOW", of_keys, all_order_flow),
    ("LIQUIDITY", liq_keys, all_liquidity),
    ("PRICE IMPACT", pi_keys, all_price_impact),
]:
    print(f"\n  {layer_name}:")
    print(f"  {'Variable':<30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for k in keys:
        vals = np.array([b[k] for b in data])
        print(f"  {k:<30s} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
              f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

# Cross-layer correlations
print(f"\n  Cross-Layer Correlations (|r| > 0.15):")
for i, name_i in enumerate(all_var_names):
    for j, name_j in enumerate(all_var_names):
        if i < j:
            r = np.corrcoef(feature_matrix_z[:, i], feature_matrix_z[:, j])[0, 1]
            if abs(r) > 0.15:
                layer_i = name_i.split(":")[0]
                layer_j = name_j.split(":")[0]
                if layer_i != layer_j:  # cross-layer only
                    print(f"    {name_i:<30s} × {name_j:<30s}  r={r:+.3f}")

print(f"\n{'═'*65}")
print(f"  Market Deconstruction complete.")
print(f"{'═'*65}")

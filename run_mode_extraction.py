"""
Mode Extraction Pipeline — solve the latent dynamical basis of this market.

  1. Load MarketState features + PnL components
  2. Fit SVD → extract 8 stable modes φ_i
  3. Project → z_i(t) time series
  4. Classify each mode (timescale, economic role, regime sensitivity)
  5. Mode collapse report (which vanish in FRAGILE?)
  6. Mode → Action mapping
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from sklearn.decomposition import sparse_encode

from projects.compressibility_frontier.metrics.state_segmenter import segment_all
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

print("=" * 65)
print("  Mode Extraction — Latent Dynamical Basis of Markets")
print("=" * 65)


# ===========================================================================
# [1] Load MarketState features + PnL components
# ===========================================================================

print("\n[1] Loading MarketState + computing PnL components ...")
t0 = time.perf_counter()

# Standardized features
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

# Dictionary + sparse encode
D0 = np.load(str(DICT_PATH)).astype(np.float64)
alpha = sparse_encode(
    X.astype(np.float64), D0,
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)

# Raw data for regime features + PnL
raw = pl.read_parquet(SOURCE,
    columns=["mid_px", "spread", "total_depth", "signed_imbalance",
             "duration_ms", "trade_px", "trade_sz", "bid_px", "bid_sz",
             "ask_px", "ask_sz", "trade_side"])
offset = raw.shape[0] - N
n_batches = N // BATCH_SIZE

dec = LayerDecomposer()

# Collect all features + PnL per batch
all_features = []
realized_vol = np.zeros(n_batches, dtype=np.float32)
spread_bps_arr = np.zeros(n_batches, dtype=np.float32)
total_depth_arr = np.zeros(n_batches, dtype=np.float32)
pnl_spread_arr = np.zeros(n_batches, dtype=np.float64)
pnl_adverse_arr = np.zeros(n_batches, dtype=np.float64)

mid_ret = np.zeros(N, dtype=np.float64)
FWD = 50
raw_mid = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
mid_ret[:-FWD] = np.abs(
    (raw_mid[FWD:] - raw_mid[:-FWD]) / (np.abs(raw_mid[:-FWD]) + 1e-12))

rng = np.random.RandomState(42)

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

    # Build feature vector (9 core features)
    feat = np.array([
        layers["order_flow"]["trade_arrival_rate"],
        layers["order_flow"]["signed_imbalance"],
        layers["order_flow"]["flow_persistence"],
        layers["liquidity"]["spread_bps"],
        layers["liquidity"]["total_depth"],
        layers["liquidity"]["queue_pressure"],
        layers["liquidity"]["liquidity_tension"],
        layers["price_impact"]["realized_volatility"],
        layers["price_impact"]["nonlinear_response"],
    ], dtype=np.float32)
    all_features.append(feat)

    realized_vol[b] = layers["price_impact"]["realized_volatility"]
    spread_bps_arr[b] = layers["liquidity"]["spread_bps"]
    total_depth_arr[b] = layers["liquidity"]["total_depth"]

    # PnL components per batch (simple baseline MM)
    batch_spread = batch["spread"].to_numpy().astype(np.float64)
    sh = batch_spread * 0.5
    adv = np.abs(mid_ret[s-offset:e-offset]) * raw_mid[s-offset:e-offset]
    f_bid = rng.binomial(1, 0.85, BATCH_SIZE)
    f_ask = rng.binomial(1, 0.85, BATCH_SIZE)
    spread_pnl = np.sum(sh[f_bid.astype(bool)]) + np.sum(sh[f_ask.astype(bool)])
    adverse_pnl = -np.sum(adv[f_bid.astype(bool)]) - np.sum(adv[f_ask.astype(bool)])
    pnl_spread_arr[b] = spread_pnl
    pnl_adverse_arr[b] = adverse_pnl

del raw
feat_matrix = np.array(all_features, dtype=np.float32)
var_names = [
    "trade_arrival_rate", "signed_imbalance", "flow_persistence",
    "spread_bps", "total_depth", "queue_pressure",
    "liquidity_tension", "realized_volatility", "nonlinear_response",
]

# Standardize
f_mean = feat_matrix.mean(axis=0)
f_std = np.maximum(feat_matrix.std(axis=0), 1e-8)
feat_z = (feat_matrix - f_mean) / f_std
feat_z = np.clip(feat_z, -10, 10)

print(f"  Features: {feat_z.shape}  PnL range: [{pnl_spread_arr.min():.0f}, {pnl_spread_arr.max():.0f}]")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Extract stable modes
# ===========================================================================

print(f"\n[2] Extracting 8 latent modes via SVD ...")
t0 = time.perf_counter()

me = ModeExtractor(n_modes=8)
me.fit(feat_z, var_names)
z_series = me.project(feat_z)

# Regime segmentation
regimes = segment_all(realized_vol, spread_bps_arr, total_depth_arr)

# Classify modes
me.classify_modes(z_series, pnl_spread_arr, pnl_adverse_arr, realized_vol, regimes)
me.print_modes()

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Mode collapse analysis
# ===========================================================================

print(f"\n[3] Mode collapse in FRAGILE state ...")
t0 = time.perf_counter()

collapse = me.collapse_report()

print(f"  Collapsing modes: {collapse['collapsing_modes']}")
print(f"  Amplifying modes: {collapse['amplifying_modes']}")
print(f"  {collapse['collapse_interpretation']}")

# Quantitative: variance of each mode in FRAGILE vs HEALTHY
if "FRAGILE" in regimes and "HEALTHY" in regimes:
    mask_f = regimes["FRAGILE"]
    mask_h = regimes["HEALTHY"]
    print(f"\n  Mode variance: FRAGILE vs HEALTHY:")
    print(f"  {'Mode':<30s} {'Var(F)':>8s} {'Var(H)':>8s} {'Ratio':>8s} {'Change':>12s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")
    for k in range(me.n_modes):
        var_f = float(np.var(z_series[mask_f, k])) if mask_f.sum() > 10 else 0
        var_h = float(np.var(z_series[mask_h, k])) if mask_h.sum() > 10 else 0
        ratio = var_f / max(var_h, 1e-12)
        if ratio < 0.7:
            change = f"COLLAPSED ({ratio:.2f}x)"
        elif ratio > 1.5:
            change = f"AMPLIFIED ({ratio:.2f}x)"
        else:
            change = "stable"
        print(f"  {me.mode_labels[k]:<30s} {var_f:>8.4f} {var_h:>8.4f} {ratio:>8.2f} {change:>12s}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Mode → Action mapping
# ===========================================================================

print(f"\n[4] Mode → Action Mapping ...")
am = me.action_map()
for mode, info in am.items():
    print(f"  {mode:<45s} → {info['action']}")


# ===========================================================================
# [5] Summary
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Mode Extraction complete.")
print(f"{'═'*65}")
print(f"\n  Key findings:")
print(f"    Total explained variance (8 modes): {me.explained_var.sum():.1%}")
print(f"    Effective rank (90% var): {np.searchsorted(np.cumsum(me.explained_var), 0.90)+1}")
print(f"    Collapsing modes: {collapse['collapsing_modes']}")
print(f"    Amplifying modes: {collapse['amplifying_modes']}")
print(f"{'═'*65}")

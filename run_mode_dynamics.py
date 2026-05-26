"""
Mode Dynamics Pipeline — empirical dynamical system identification.

  1. Load 8D mode time series from Mode Extractor
  2. Fit Mode Interaction Matrix A (8×8): z(t+1) = A · z(t)
  3. Classify modes: Driver / Response / Self-Exciting
  4. Stability spectrum (eigenvalues of A)
  5. Per-regime comparison (A_NORMAL vs A_FRAGILE vs A_HIGH_VOL)
  6. Amplification operator: which edges intensify under stress?
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from projects.compressibility_frontier.metrics.state_segmenter import segment_all
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor
from projects.compressibility_frontier.experiments.mode_dynamics import ModeDynamics

BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Mode Dynamics — Empirical Dynamical System Identification")
print("=" * 65)


# ===========================================================================
# [1] Build mode time series z(t)
# ===========================================================================

print("\n[1] Building 8D mode time series ...")
t0 = time.perf_counter()

builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

raw = pl.read_parquet(SOURCE,
    columns=["mid_px", "spread", "total_depth", "signed_imbalance",
             "duration_ms", "trade_px", "trade_sz", "bid_px", "bid_sz",
             "ask_px", "ask_sz", "trade_side"])
offset = raw.shape[0] - N
n_batches = N // BATCH_SIZE

dec = LayerDecomposer()
all_features = []
realized_vol = np.zeros(n_batches, dtype=np.float32)
spread_bps_arr = np.zeros(n_batches, dtype=np.float32)
total_depth_arr = np.zeros(n_batches, dtype=np.float32)

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
feat_z = np.clip((feat_matrix - f_mean) / f_std, -10, 10)

# SVD mode extraction
me = ModeExtractor(n_modes=8)
me.fit(feat_z, var_names)
z_series = me.project(feat_z)

# Standardize each mode to unit variance (required for SDE)
z_mean = z_series.mean(axis=0)
z_std = np.maximum(z_series.std(axis=0), 1e-8)
z_series = (z_series - z_mean) / z_std

# Quick classify for labels
regimes = segment_all(realized_vol, spread_bps_arr, total_depth_arr)
me.classify_modes(z_series, np.ones(n_batches), np.ones(n_batches), realized_vol, regimes)
mode_labels = me.mode_labels

print(f"  z_series: {z_series.shape}  std={z_series.std(axis=0).mean():.2f}")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Fit Mode Interaction Matrix A
# ===========================================================================

print(f"\n[2] Fitting Mode Interaction Matrix A (8x8) ...")
t0 = time.perf_counter()

md = ModeDynamics()
md.fit(z_series, mode_labels)
md.print_report()
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Per-regime dynamics
# ===========================================================================

print(f"\n[3] Per-regime comparison ...")
t0 = time.perf_counter()

md_regime = ModeDynamics()
md_regime.fit_regimes(z_series, mode_labels, regimes)

for rname in ["NORMAL", "FRAGILE", "HIGH_VOL"]:
    rkey = None
    for k in md_regime.A_regimes:
        if rname in k.upper():
            rkey = k
            break
    if rkey is None:
        continue
    A_r = md_regime.A_regimes[rkey]
    ev = np.linalg.eigvals(A_r)
    stable = int(np.sum(np.real(ev) < 0))
    max_real = float(np.max(np.real(ev)))
    print(f"  {rname}: stable_modes={stable}/8  max_Re(λ)={max_real:+.4f}  "
          f"{'STABLE' if max_real < 0 else 'HAS EXPLOSIVE'}")


# ===========================================================================
# [4] Amplification operator
# ===========================================================================

print(f"\n[4] Amplification Operator: A(FRAGILE) - A(NORMAL) ...")
md_regime.print_regime_comparison("FRAGILE", "NORMAL")

print(f"\n[5] Amplification Operator: A(HIGH_VOL) - A(NORMAL) ...")
md_regime.print_regime_comparison("HIGH_VOL", "NORMAL")


# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Mode Dynamics complete.")
print(f"{'═'*65}")

sr = md.stability_report()
cls = md.classify_modes()

drivers = [k for k, v in cls.items() if v["class"] == "DRIVER"]
responses = [k for k, v in cls.items() if v["class"] == "RESPONSE"]
self_ex = [k for k, v in cls.items() if v["class"] == "SELF_EXCITING"]

print(f"\n  System type: {'STABLE attractor' if sr['is_stable'] else 'Has explosive subspace'}")
print(f"  Drivers:     {drivers}")
print(f"  Responses:   {responses}")
print(f"  Self-excite: {self_ex}")
print(f"{'═'*65}")

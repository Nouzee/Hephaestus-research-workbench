"""
Compressibility Frontier V1 — measure market structural density.

Loads existing MarketState + dictionary, segments by regime,
computes four compressibility metrics per regime.

Answers: "How compressible is the market under different conditions?"
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_state import MarketState
from modules.research.market_decon import LayerDecomposer
from sklearn.decomposition import sparse_encode

from projects.compressibility_frontier.metrics.compressibility_metrics import (
    reconstruction_residual, effective_rank, atom_usage_entropy,
    temporal_redundancy,
)
from projects.compressibility_frontier.metrics.state_segmenter import (
    segment_all, regime_summary,
)
from projects.compressibility_frontier.experiments.compressibility_scan import (
    run_scan, print_scan_table,
)
from projects.compressibility_frontier.reports.report_builder import (
    build_report, print_report,
)


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"
REPORT_DIR = ROOT / "projects" / "compressibility_frontier" / "reports"

print("=" * 65)
print("  Compressibility Frontier V1")
print("  Market Structural Density Under Different Regimes")
print("=" * 65)


# ===========================================================================
# [1] Load existing infrastructure (frozen — NOT modified)
# ===========================================================================

print("\n[1] Loading MarketState + Dictionary ...")
t0 = time.perf_counter()

# Standardized features
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

# Dictionary
D0 = np.load(str(DICT_PATH)).astype(np.float64)
K = D0.shape[0]

# Sparse encode
alpha = sparse_encode(
    X.astype(np.float64), D0,
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)

# Build MarketState per batch (for regime segmentation)
raw = pl.read_parquet(SOURCE)
offset = raw.shape[0] - N
n_batches = N // BATCH_SIZE

dec = LayerDecomposer()
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
    realized_vol[b] = layers["price_impact"]["realized_volatility"]
    spread_bps_arr[b] = layers["liquidity"]["spread_bps"]
    total_depth_arr[b] = layers["liquidity"]["total_depth"]

del raw

print(f"  X: {X.shape}  D: {D0.shape}  alpha: {alpha.shape}")
print(f"  Regime features: {n_batches} batches  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Regime segmentation
# ===========================================================================

print("\n[2] Segmenting market by regime ...")
t0 = time.perf_counter()

regimes = segment_all(realized_vol, spread_bps_arr, total_depth_arr)
summary = regime_summary(regimes)

for name, info in sorted(summary.items()):
    print(f"  {name:<16s}: {info['batches']:>5d} batches ({info['pct']:>5.1f}%)")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Global compressibility (all data)
# ===========================================================================

print("\n[3] Global compressibility (all 3.9M ticks) ...")
t0 = time.perf_counter()

resid = reconstruction_residual(X, D0, alpha)
erank = effective_rank(X)
entropy = atom_usage_entropy(alpha)
redundancy = temporal_redundancy(X)

# Normalize X for rank computation
X_z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-8)

print(f"  Reconstruction residual: {resid:.4f}  "
      f"({'highly compressible' if resid < 0.5 else 'noisy'})")
print(f"  Effective rank:          {erank:.2f}  "
      f"(of {M} features, {erank/M*100:.0f}% active)")
print(f"  Atom usage entropy:      {entropy:.3f}  "
      f"({'concentrated' if entropy < 0.5 else 'uniform'})")
print(f"  Temporal redundancy:     {redundancy:.3f}  "
      f"({'repetitive' if redundancy > 0.5 else 'varied'})")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Regime-specific compressibility scan
# ===========================================================================

print(f"\n[4] Compressibility scan per regime ...")
t0 = time.perf_counter()

# Downsample X and alpha to batch-level (mean per batch of 2048 ticks)
X_batch = np.array([
    X[b*BATCH_SIZE:(b+1)*BATCH_SIZE].mean(axis=0)
    for b in range(n_batches)
], dtype=np.float32)
A_batch = np.array([
    alpha[b*BATCH_SIZE:(b+1)*BATCH_SIZE].mean(axis=0)
    for b in range(n_batches)
], dtype=np.float32)

scan_results = {}
for regime_name, mask in regimes.items():
    n_samples = int(np.sum(mask))
    if n_samples < 20:
        continue

    X_seg = X_batch[mask]
    A_seg = A_batch[mask]

    from projects.compressibility_frontier.metrics.compressibility_metrics import (
        compressibility_summary,
    )
    metrics = compressibility_summary(X_seg, D0, A_seg)
    metrics["n_samples"] = n_samples
    scan_results[regime_name] = metrics

print_scan_table(scan_results)
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [5] Report
# ===========================================================================

print(f"\n[5] Building report ...")
t0 = time.perf_counter()

report = build_report(scan_results, regimes, str(REPORT_DIR))
print_report(report)
print(f"  time={time.perf_counter()-t0:.1f}s")

# Cleanup
del X, alpha
import gc; gc.collect()


# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Compressibility Frontier V1 complete.")
print(f"{'═'*65}")

if report["rankings"]:
    r = report["rankings"]
    print(f"\n  Key Finding:")
    print(f"    Most compressible:  {r[0]['regime']} (composite={r[0]['composite']:.3f})")
    print(f"    Least compressible: {r[-1]['regime']} (composite={r[-1]['composite']:.3f})")
    print(f"    Range: {report['compression_range']['min']:.3f} — "
          f"{report['compression_range']['max']:.3f}")

print(f"\n  Files created:")
print(f"    projects/compressibility_frontier/metrics/compressibility_metrics.py")
print(f"    projects/compressibility_frontier/metrics/state_segmenter.py")
print(f"    projects/compressibility_frontier/experiments/compressibility_scan.py")
print(f"    projects/compressibility_frontier/reports/report_builder.py")
print(f"    projects/compressibility_frontier/reports/compressibility_report.json")
print(f"{'═'*65}")

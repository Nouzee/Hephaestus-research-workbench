"""
Observation Operator Sweep — test market structure invariance.

Projects the same raw tick data through multiple observation operators
(tick_50, tick_200, tick_2048, trade_50, volume_100) and measures
whether market structure (effective rank, compressibility, entropy)
is invariant to how you observe it.

Answers:
  - Which structures are real (invariant)?
  - Which are scale artifacts (projection-dependent)?
  - What is the intrinsic dimension of this market?
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.compressibility_frontier.operators.observation_operators import (
    OPERATORS, apply_operator, extract_features,
)
from projects.compressibility_frontier.experiments.cross_scale_scanner import (
    run_cross_scale_scan, print_cross_scale_table,
)

SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
DICT_PATH = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache" / "dict_atoms_3.npy"

print("=" * 65)
print("  Observation Operator Sweep")
print("  Market Structure Invariance Test")
print("=" * 65)


# ===========================================================================
# [1] Load raw tick data
# ===========================================================================

print("\n[1] Loading raw tick data ...")
t0 = time.perf_counter()

raw_df = pl.read_parquet(SOURCE, columns=[
    "mid_px", "spread", "total_depth", "signed_imbalance",
    "duration_ms", "trade_px", "trade_sz",
])

# Sample 500K ticks for speed (full 3.9M is redundant for structure analysis)
n_sample = 500_000
raw_df = raw_df.slice(0, n_sample)

raw_data = {
    "mid_px": raw_df["mid_px"].to_numpy().astype(np.float64),
    "spread": raw_df["spread"].to_numpy().astype(np.float64),
    "total_depth": raw_df["total_depth"].to_numpy().astype(np.float64),
    "signed_imbalance": raw_df["signed_imbalance"].to_numpy().astype(np.float64),
    "duration_ms": raw_df["duration_ms"].to_numpy().astype(np.float64),
    "trade_px": raw_df["trade_px"].to_numpy().astype(np.float64),
    "trade_sz": raw_df["trade_sz"].to_numpy().astype(np.float64),
}

N = len(raw_data["mid_px"])
print(f"  {N:,} ticks loaded  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Run cross-scale scan
# ===========================================================================

print(f"\n[2] Running cross-scale observation sweep ...")
t0 = time.perf_counter()

# Load dictionary for reconstruction metric
dict_path = DICT_PATH
if dict_path.exists():
    D0 = np.load(str(dict_path))
else:
    # Fallback: random dictionary
    D0 = np.random.randn(3, 9).astype(np.float64)

results = run_cross_scale_scan(raw_data, D0)
print_cross_scale_table(results)
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Structure stability classification
# ===========================================================================

print(f"\n[3] Structure Stability Report ...")

cls = results.get("_classification", {})

# Per-operator dimensionality
print(f"\n  Intrinsic dimension under each operator:")
for op_name in OPERATORS:
    r = results.get(op_name, {})
    if "error" in r:
        continue
    erank = r["effective_rank"]
    n_feat = 9
    pct = erank / n_feat * 100
    bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
    print(f"    {op_name:<14s}: rank={erank:>5.2f}/{n_feat} ({pct:>5.1f}%) [{bar}]")

# SVD spectrum comparison
print(f"\n  SVD spectrum decay (first 5 singular values, normalized):")
print(f"  {'Operator':<14s} {'σ1':>6s} {'σ2':>6s} {'σ3':>6s} {'σ4':>6s} {'σ5':>6s} {'decay':>8s}")
print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
for op_name in OPERATORS:
    r = results.get(op_name, {})
    sv = r.get("sv_spectrum", [])
    if len(sv) >= 5:
        decay = sv[0] / max(sv[4], 1e-12) if sv[4] > 0.001 else 999
        print(f"  {op_name:<14s} {sv[0]:>6.3f} {sv[1]:>6.3f} {sv[2]:>6.3f} "
              f"{sv[3]:>6.3f} {sv[4]:>6.3f} {decay:>8.1f}x")

# Final classification
if cls:
    print(f"\n  {'═'*50}")
    print(f"  Final Classification: {cls['type']}")
    print(f"  {'═'*50}")
    print(f"  Rank CV across operators: {cls['rank_cv']:.3f}")
    print(f"  Rank range: [{cls['rank_range'][0]:.2f}, {cls['rank_range'][1]:.2f}]")
    print(f"  {cls['interpretation']}")

    if cls["type"] == "INVARIANT":
        print(f"\n  → Market has a stable intrinsic geometry.")
        print(f"  → Structure is NOT a projection artifact.")
        print(f"  → Cross-scale consistency is strong — trust the geometry.")
    elif cls["type"] == "SCALE_DEPENDENT":
        print(f"\n  → Some structure is real, some is scale-specific.")
        print(f"  → Alpha signals that only exist at one scale are fragile.")
    else:
        print(f"\n  → What you see depends on how you look.")
        print(f"  → Most 'alpha' is observation artifact.")

print(f"\n{'═'*65}")
print(f"  Observation Sweep complete.")
print(f"{'═'*65}")

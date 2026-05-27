"""
Demo Pipeline — minimal runnable example of the Hephaestus workflow.

No real data required. Uses synthetic data to demonstrate:
  Feature extraction → Regime clustering → Risk scoring → Backtest → Attribution

Prerequisites: pip install -r requirements.txt
"""

import numpy as np

print("=" * 60)
print("  Hephaestus Demo Pipeline")
print("  Feature → Regime → Filter → Backtest → Attribution")
print("=" * 60)

# ── 1. Synthetic L2 data generation ──
print("\n[1] Generating synthetic L2 microstructure data ...")
np.random.seed(42)
N_TICKS = 20_000
N_WINDOWS = N_TICKS // 100

# Simulate: spread, depth, imbalance, arrival, volatility
spread = np.abs(np.random.randn(N_WINDOWS) * 0.5 + 1.5)       # spread in bps
depth = np.abs(np.random.randn(N_WINDOWS) * 5 + 15)            # total depth
imbalance = np.random.randn(N_WINDOWS) * 0.3                    # bid/ask imbalance
arrival = np.abs(np.random.randn(N_WINDOWS) * 2 + 10)          # trade arrival rate
real_vol = np.abs(np.random.randn(N_WINDOWS) * 0.1 + 0.3)     # realized volatility

features = np.column_stack([spread, depth, imbalance, arrival, real_vol])
print(f"  Generated {N_WINDOWS} windows × {features.shape[1]} features")

# ── 2. Regime clustering ──
print("\n[2] Regime clustering (KMeans, K=4) ...")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(features)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
regimes = km.fit_predict(X_scaled)

for r in range(4):
    pct = np.mean(regimes == r) * 100
    print(f"  Regime {r}: {pct:.1f}%")

# ── 3. Risk scoring ──
print("\n[3] Quantile-based risk scoring ...")
# Tox proxy: spread / depth ratio
tox_raw = spread / np.maximum(depth, 1e-8)
p30 = np.percentile(tox_raw, 30)
p70 = np.percentile(tox_raw, 70)

tox_quantile = np.where(tox_raw <= p30, 0, np.where(tox_raw <= p70, 1, 2))
for q in range(3):
    pct = np.mean(tox_quantile == q) * 100
    print(f"  Tox Q{q}: {pct:.1f}%  (thresholds: p30={p30:.4f}, p70={p70:.4f})")

# ── 4. CORE state filter ──
print("\n[4] CORE state filtering ...")
# CORE = high-tox (Q2) states
core_mask = tox_quantile == 2

# Per-state statistics
from collections import defaultdict
state_pnl = defaultdict(float)
state_fills = defaultdict(int)

rng = np.random.RandomState(42)
for i in range(N_WINDOWS):
    if not core_mask[i]: continue
    state_key = f"R{regimes[i]}_Q{tox_quantile[i]}"
    # Simulate fills with tox-dependent profitability
    n_fills = rng.poisson(3 if tox_quantile[i] == 2 else 1)
    pnl_per_fill = spread[i] / 2 - np.abs(rng.randn() * real_vol[i]) * 2
    state_pnl[state_key] += pnl_per_fill * n_fills
    state_fills[state_key] += n_fills

print(f"  CORE states (high-tox, profitable):")
for sk in sorted(state_pnl.keys()):
    ev = state_pnl[sk] / max(state_fills[sk], 1)
    print(f"    {sk}: EV/fill={ev:+.2f}  fills={state_fills[sk]}")

# ── 5. Backtest comparison ──
print("\n[5] Backtest: Baseline vs CORE-filtered ...")
bl_pnl = 0.0
core_pnl = 0.0

for i in range(N_WINDOWS):
    n_ticks = 100
    for t in range(n_ticks):
        # Baseline: always quote
        if rng.random() < 0.30:
            bl_pnl += spread[i] / 2 - np.abs(rng.randn() * real_vol[i]) * 1.5
        # CORE: only quote in high-tox
        if core_mask[i] and rng.random() < 0.30:
            core_pnl += spread[i] / 2 - np.abs(rng.randn() * real_vol[i]) * 1.5

print(f"  Baseline:    {bl_pnl:>+12.1f}")
print(f"  CORE filter: {core_pnl:>+12.1f}")
print(f"  Improvement: {core_pnl - bl_pnl:>+12.1f}  ({(core_pnl-bl_pnl)/max(abs(bl_pnl),1)*100:+.1f}%)")

# ── 6. Attribution ──
print(f"\n[6] Attribution by tox quantile:")
for q in range(3):
    mask = tox_quantile == q
    avg_spread = np.mean(spread[mask])
    avg_vol = np.mean(real_vol[mask])
    print(f"  Q{q}: spread={avg_spread:.2f}  vol={avg_vol:.3f}  "
          f"windows={mask.sum()}")

print(f"\n{'='*60}")
print(f"  Demo complete. Full pipeline: feature → regime → filter → backtest.")
print(f"  Replace synthetic data with real L2 parquet for production use.")
print(f"{'='*60}")

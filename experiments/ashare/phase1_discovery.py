"""
A-Share Phase 1 — Regime Segmentation

Pipeline:
  1. Load 000333 train data → feature matrix (N windows × 16 features)
  2. Spectral clustering → regime labels
  3. Regime analysis: persistence, occupancy, transition frequencies
  4. Find: attractor states, irreversible transitions, metastable basins
"""

import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import (
    build_feature_matrix, FEATURE_NAMES,
)

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100
MAX_DAYS = 10  # first 10 days (memory-safe for spectral clustering)

print("=" * 65)
print("  A-Share Phase 1 — Regime Segmentation")
print("  000333 L2 Microstructure Regime Discovery")
print("=" * 65)


# ===========================================================================
# [1] Build feature matrix
# ===========================================================================

print("\n[1] Building feature matrix ...")
t0 = time.perf_counter()

X, day_boundaries, meta = build_feature_matrix(
    TRAIN_DIR, window_size=WINDOW_SIZE, max_days=MAX_DAYS,
)
N, D = X.shape
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Clustering — Spectral + HDBSCAN
# ===========================================================================

print(f"\n[2] Clustering {N} windows into regimes ...")
t0 = time.perf_counter()

# KMeans clustering for regime discovery
from sklearn.cluster import KMeans

n_regimes = 8
km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10, max_iter=300)
labels = km.fit_predict(X)

# Count per regime
regime_counts = {}
for r in range(n_regimes):
    n_r = int(np.sum(labels == r))
    if n_r > 0:
        regime_counts[r] = n_r

active_regimes = sorted(regime_counts.keys())
print(f"  Regimes found: {len(active_regimes)}")
for r in active_regimes:
    pct = regime_counts[r] / N * 100
    print(f"    Regime {r}: {regime_counts[r]:>6d} windows ({pct:>5.1f}%)")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Regime characterization — what defines each regime?
# ===========================================================================

print(f"\n[3] Regime characterization ...")
t0 = time.perf_counter()

# Per-regime mean of each feature (in standardized space)
print(f"\n  Regime profiles (mean z-score per feature):")
header = f"  {'Feature':<22s} " + "".join(f"R{r:>7d}" for r in active_regimes)
print(header)
print(f"  {'─'*22} " + "".join("───────" for _ in active_regimes))

for f_idx, fname in enumerate(FEATURE_NAMES):
    row = f"  {fname:<22s} "
    for r in active_regimes:
        mask = labels == r
        val = float(np.mean(X[mask, f_idx]))
        row += f"{val:>+7.3f}"
    print(row)

# Name regimes based on dominant features
regime_names = {}
for r in active_regimes:
    mask = labels == r
    profile = np.mean(X[mask], axis=0)
    top_idx = np.argsort(np.abs(profile))[::-1][:3]
    top_feats = [FEATURE_NAMES[i] for i in top_idx]
    regime_names[r] = f"R{r}:{top_feats[0].split('_')[0]}"

print(f"\n  Regime names:")
for r in active_regimes:
    print(f"    {regime_names[r]}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Transition analysis
# ===========================================================================

print(f"\n[4] Transition analysis ...")
t0 = time.perf_counter()

# Build transition matrix P(i→j)
n_r = len(active_regimes)
trans_count = np.zeros((n_r, n_r), dtype=np.int32)

for t in range(N - 1):
    i = labels[t]
    j = labels[t + 1]
    if i != j:  # only count actual transitions
        ri = active_regimes.index(i)
        rj = active_regimes.index(j)
        trans_count[ri, rj] += 1

# Normalize to probabilities
trans_prob = np.zeros((n_r, n_r), dtype=np.float64)
for ri in range(n_r):
    row_sum = trans_count[ri].sum()
    if row_sum > 0:
        trans_prob[ri] = trans_count[ri] / row_sum

print(f"\n  Transition Matrix P(i → j):")
header = f"  {'':>12s}" + "".join(f"{regime_names[active_regimes[j]]:<20s}" for j in range(n_r))
print(header)
for ri in range(n_r):
    rname = regime_names[active_regimes[ri]]
    row = f"  {rname:<12s}"
    for rj in range(n_r):
        row += f"{trans_prob[ri, rj]:>20.4f}"
    print(row)

# Persistence: fraction of time staying in same regime
persistence = np.zeros(n_r, dtype=np.float64)
for ri in range(n_r):
    r = active_regimes[ri]
    same = np.sum((labels[:-1] == r) & (labels[1:] == r))
    total_r = np.sum(labels == r)
    persistence[ri] = same / max(total_r, 1)

print(f"\n  Regime Persistence (stay probability):")
for ri in range(n_r):
    r = active_regimes[ri]
    bar = "#" * int(persistence[ri] * 30) + "." * (30 - int(persistence[ri] * 30))
    print(f"    {regime_names[r]:<20s}: {persistence[ri]:.3f}  {bar} "
          f"({'ATTRACTOR' if persistence[ri] > 0.85 else 'transient'})")

# Irreversibility: |P(i→j) - P(j→i)|
print(f"\n  Irreversible transitions (|P(i→j) - P(j→i)| > 0.1):")
for ri in range(n_r):
    for rj in range(ri+1, n_r):
        asym = abs(trans_prob[ri, rj] - trans_prob[rj, ri])
        if asym > 0.1:
            print(f"    {regime_names[active_regimes[ri]]} <-> {regime_names[active_regimes[rj]]}: "
                  f"asym={asym:.3f}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [5] Intraday regime structure
# ===========================================================================

print(f"\n[5] Intraday regime structure ...")
t0 = time.perf_counter()

# Per-day regime distribution
day_labels = []
for d_idx in range(len(day_boundaries)):
    start = 0 if d_idx == 0 else day_boundaries[d_idx - 1]
    end = day_boundaries[d_idx]
    day_labels.append(labels[start:end])

print(f"\n  Regime occupancy by day (first 5 days):")
header = f"  {'Day':>5s}" + "".join(f"R{r:>7d}" for r in active_regimes)
print(header)
for d_idx in range(min(5, len(day_labels))):
    dl = day_labels[d_idx]
    row = f"  {d_idx+1:>5d}"
    for r in active_regimes:
        pct = np.mean(dl == r) * 100
        row += f"{pct:>7.1f}"
    print(row)

# Regime at open vs close
# First and last 10% of each day
open_regimes = []
close_regimes = []
for dl in day_labels:
    n = len(dl)
    open_10 = int(n * 0.1)
    close_10 = int(n * 0.9)
    if open_10 > 0 and close_10 < n:
        open_regimes.extend(dl[:open_10])
        close_regimes.extend(dl[close_10:])

if open_regimes and close_regimes:
    print(f"\n  Regime at OPEN vs CLOSE:")
    print(f"  {'Regime':>12s} {'Open%':>8s} {'Close%':>8s} {'Shift':>8s}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    for r in active_regimes:
        op_pct = np.mean(np.array(open_regimes) == r) * 100
        cl_pct = np.mean(np.array(close_regimes) == r) * 100
        shift = cl_pct - op_pct
        marker = " ↑CLOSE" if shift > 3 else (" ↑OPEN" if shift < -3 else "")
        print(f"  R{r:>11d} {op_pct:>7.1f}% {cl_pct:>7.1f}% {shift:>+7.1f}%{marker}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Phase 1 complete.")
print(f"{'═'*65}")
print(f"\n  Data: 000333 L2, {MAX_DAYS} days, {N} windows")
print(f"  Regimes: {len(active_regimes)} discovered")
print(f"  Attractor candidates: "
      f"{[regime_names[active_regimes[ri]] for ri in range(n_r) if persistence[ri] > 0.85]}")
print(f"{'═'*65}")

"""
A-Share Full Pipeline — 3-Month Regime Discovery + Transition + R3 Precursor

Steps:
  1. Load ALL 59 training days → feature matrix
  2. KMeans 8-regime clustering
  3. Full transition matrix + attractor verification
  4. R3 event study — precursor chain identification
  5. Intraday slicing — open/mid/close
  6. Weekly stability check
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from projects.ashare.regime_segmentation import (
    L2FeatureExtractor, FEATURE_NAMES,
)
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100
N_REGIMES = 8
SUBSAMPLE_FRAC = 0.15  # fraction of windows for KMeans fit (memory control)

print("=" * 65)
print("  A-Share Full Pipeline — 3-Month Regime Discovery")
print("=" * 65)


# ===========================================================================
# [1] Load ALL data + extract features
# ===========================================================================

print("\n[1] Loading ALL 59 training days ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))

all_features = []
day_boundaries = []
day_timestamps = []
day_mid_prices = []

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf)
    ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]
    n_windows = N // WINDOW_SIZE
    if n_windows < 5:
        continue

    msg_dict = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_dict = {col: ob_df[col].to_numpy() for col in ob_df.columns}

    for w in range(n_windows):
        s, e = w * WINDOW_SIZE, (w+1) * WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_dict.items()},
            {k: v[s:e] for k, v in msg_dict.items()},
        )
        all_features.append(list(feats.values()))
        day_timestamps.append(msg_dict["Time (sec)"][s])
        day_mid_prices.append(float(np.mean(
            (ob_dict["OfferPrice1"][s:e] + ob_dict["BidPrice1"][s:e]) / 2.0)))

    day_boundaries.append(len(all_features))

    if (day_idx + 1) % 15 == 0:
        print(f"  [{day_idx+1}/{len(msg_files)}] days, {len(all_features):,} windows")

X = np.array(all_features, dtype=np.float32)
day_bounds = np.array(day_boundaries, dtype=np.int32)
N, D = X.shape

# Standardize
X_mean = X.mean(axis=0)
X_std = np.maximum(X.std(axis=0), 1e-8)
X_z = np.clip((X - X_mean) / X_std, -10, 10)

print(f"  Total: {N:,} windows × {D} features  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Regime Clustering (subsample fit, full predict)
# ===========================================================================

print(f"\n[2] KMeans {N_REGIMES}-regime clustering ...")
t0 = time.perf_counter()

# Fit on subsample for speed
rng = np.random.RandomState(42)
n_fit = max(int(N * SUBSAMPLE_FRAC), 10000)
fit_idx = rng.choice(N, n_fit, replace=False)

km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(X_z[fit_idx])
labels = km.predict(X_z)

# Regime stats
regime_counts = {r: int(np.sum(labels == r)) for r in range(N_REGIMES)}
active = sorted([r for r, c in regime_counts.items() if c > 0])

print(f"  Regimes ({len(active)}):")
for r in active:
    pct = regime_counts[r] / N * 100
    # Persistence
    same = np.sum((labels[:-1] == r) & (labels[1:] == r))
    total_r = np.sum(labels == r)
    persist = same / max(total_r, 1)
    marker = "ATTRACTOR" if persist > 0.85 else ("sticky" if persist > 0.7 else "transient")
    print(f"    R{r}: {regime_counts[r]:>8,} ({pct:>5.1f}%)  persist={persist:.3f}  [{marker}]")

# Name regimes by top 3 features
regime_names = {}
for r in active:
    profile = np.mean(X_z[labels == r], axis=0)
    top3 = np.argsort(np.abs(profile))[::-1][:3]
    names = [FEATURE_NAMES[i].split("_")[0] for i in top3]
    regime_names[r] = f"R{r}:{'+'.join(names)}"
    print(f"    -> {regime_names[r]}: "
          f"{FEATURE_NAMES[top3[0]]}={profile[top3[0]]:+.1f}σ  "
          f"{FEATURE_NAMES[top3[1]]}={profile[top3[1]]:+.1f}σ  "
          f"{FEATURE_NAMES[top3[2]]}={profile[top3[2]]:+.1f}σ")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Full Transition Matrix
# ===========================================================================

print(f"\n[3] Transition Matrix + Attractor Analysis ...")
t0 = time.perf_counter()

n_r = len(active)
trans_count = np.zeros((n_r, n_r), dtype=np.int32)
stay_count = np.zeros(n_r, dtype=np.int32)

for t in range(N - 1):
    i = labels[t]
    j = labels[t + 1]
    ri = active.index(i)
    if i == j:
        stay_count[ri] += 1
    else:
        rj = active.index(j)
        trans_count[ri, rj] += 1

# Normalize
trans_prob = np.zeros((n_r, n_r), dtype=np.float64)
for ri in range(n_r):
    row_sum = trans_count[ri].sum()
    if row_sum > 0:
        trans_prob[ri] = trans_count[ri] / row_sum

# Print transition matrix
print(f"\n  Transition Matrix P(i->j | i!=j):")
header = f"  {'From':<20s}" + "".join(f"{regime_names[active[j]]:<22s}" for j in range(n_r))
print(header)
for ri in range(n_r):
    row = f"  {regime_names[active[ri]]:<20s}"
    for rj in range(n_r):
        if ri == rj:
            row += f"{'─':>22s}"
        else:
            row += f"{trans_prob[ri, rj]:>22.4f}"
    print(row)

# Average stay duration (geometric: 1/(1-persist))
print(f"\n  Average stay duration (batches):")
for ri in range(n_r):
    r = active[ri]
    same = np.sum((labels[:-1] == r) & (labels[1:] == r))
    total_r = max(np.sum(labels == r), 1)
    persist = same / total_r
    stay = 1.0 / max(1.0 - persist, 0.001)
    marker = "ABSORBING" if stay > 20 else ("sticky" if stay > 10 else "fleeting")
    print(f"    {regime_names[r]:<25s}: {stay:>6.1f} batches  [{marker}]")

# Verify R0/R1→R5→R4→R3 chain
print(f"\n  Key transition chain verification:")
chains_to_check = []

# Find stress regime (highest persistence)
stress_r = max(active, key=lambda r: np.sum((labels[:-1]==r)&(labels[1:]==r))/max(np.sum(labels==r),1))
print(f"    Stress attractor: {regime_names[stress_r]} "
      f"(persist={np.sum((labels[:-1]==stress_r)&(labels[1:]==stress_r))/max(np.sum(labels==stress_r),1):.3f})")

# Find most common path into stress
stress_ri = active.index(stress_r)
incoming = trans_count[:, stress_ri]
top_in = np.argsort(incoming)[::-1][:3]
print(f"    Top 3 incoming to stress:")
for idx in top_in:
    print(f"      {regime_names[active[idx]]} -> {regime_names[stress_r]}: "
          f"{incoming[idx]:,} transitions")

# Find most common path out of stress
outgoing = trans_count[stress_ri, :]
top_out = np.argsort(outgoing)[::-1][:3]
print(f"    Top 3 outgoing from stress:")
for idx in top_out:
    print(f"      {regime_names[stress_r]} -> {regime_names[active[idx]]}: "
          f"{outgoing[idx]:,} transitions")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] R3 Precursor Chain — Event Study
# ===========================================================================

print(f"\n[4] R3 (Stress) Precursor Event Study ...")
t0 = time.perf_counter()

PRE_WINDOW = 50
POST_WINDOW = 20

# Find entries into stress (R3 or equivalent high-persistence regime)
stress_regime = max(active, key=lambda r: np.sum((labels[:-1]==r)&(labels[1:]==r))/max(np.sum(labels==r),1))

# Find all entry events: first occurrence of stress after non-stress
entry_events = []
for t in range(1, N):
    if labels[t] == stress_regime and labels[t-1] != stress_regime:
        if t >= PRE_WINDOW and t + POST_WINDOW < N:
            entry_events.append(t)

print(f"  Stress entries found: {len(entry_events)} events")

if len(entry_events) >= 10:
    # Align features around entry
    n_events = len(entry_events)
    aligned = np.zeros((n_events, PRE_WINDOW + POST_WINDOW, D), dtype=np.float32)

    for ev_idx, t_entry in enumerate(entry_events):
        s = t_entry - PRE_WINDOW
        e = t_entry + POST_WINDOW
        aligned[ev_idx] = X_z[s:e]

    mean_aligned = np.mean(aligned, axis=0)  # (PRE+POST, D)

    # For each feature, find when it peaks relative to entry
    print(f"\n  Precursor timing (peak relative to stress entry, t=0):")
    print(f"  {'Feature':<24s} {'Peak@':>6s} {'Value':>8s} {'Type':>16s}")
    print(f"  {'─'*24} {'─'*6} {'─'*8} {'─'*16}")

    key_features = [
        "depth_collapse_ratio", "spread_bps", "cancel_intensity",
        "queue_refill_speed", "arrival_rate", "realized_volatility",
        "signed_imbalance", "local_elasticity",
    ]

    precursor_chain = []
    for fname in key_features:
        if fname not in FEATURE_NAMES:
            continue
        fidx = FEATURE_NAMES.index(fname)
        trace = mean_aligned[:, fidx]  # (PRE+POST,)

        # Find peak in pre-window
        pre_trace = trace[:PRE_WINDOW]
        peak_idx = np.argmax(np.abs(pre_trace))
        peak_val = trace[peak_idx]
        peak_rel = peak_idx - PRE_WINDOW  # negative = before entry

        if peak_rel < -5:
            ptype = "EARLY PRECURSOR"
        elif peak_rel < -2:
            ptype = "precursor"
        elif peak_rel <= 2:
            ptype = "coincident"
        else:
            ptype = "consequence"

        print(f"  {fname:<24s} {peak_rel:>+6d} {peak_val:>+8.3f} {ptype:>16s}")

        if "PRECURSOR" in ptype or ptype == "precursor":
            precursor_chain.append((fname, peak_rel))

    precursor_chain.sort(key=lambda x: x[1])  # sort by timing
    print(f"\n  R3 Precursor Chain (earliest→latest):")
    for fname, t_rel in precursor_chain:
        print(f"    t={t_rel:+d}: {fname}")
    if not precursor_chain:
        print(f"    (no clear precursors found)")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [5] Intraday Slicing
# ===========================================================================

print(f"\n[5] Intraday regime structure ...")
t0 = time.perf_counter()

# Split each day into thirds: open (first 30%), mid (30-70%), close (last 30%)
open_labels = []
mid_labels = []
close_labels = []

for d_idx in range(len(day_bounds)):
    start = 0 if d_idx == 0 else day_bounds[d_idx-1]
    end = day_bounds[d_idx]
    day_l = labels[start:end]
    n_day = len(day_l)

    if n_day < 30:
        continue

    third = n_day // 3
    open_labels.extend(day_l[:third])
    mid_labels.extend(day_l[third:2*third])
    close_labels.extend(day_l[2*third:])

print(f"\n  Regime distribution by session:")
print(f"  {'Regime':<25s} {'Open':>8s} {'Mid':>8s} {'Close':>8s} {'Pattern':>14s}")
print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")

for r in active:
    op = np.mean(np.array(open_labels) == r) * 100
    md = np.mean(np.array(mid_labels) == r) * 100
    cl = np.mean(np.array(close_labels) == r) * 100

    if cl > op * 1.3:
        pattern = "CLOSE-HEAVY"
    elif op > cl * 1.3:
        pattern = "OPEN-HEAVY"
    elif md > op and md > cl:
        pattern = "MID-PEAK"
    else:
        pattern = "uniform"

    print(f"  {regime_names[r]:<25s} {op:>7.1f}% {md:>7.1f}% {cl:>7.1f}% {pattern:>14s}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [6] Weekly Stability
# ===========================================================================

print(f"\n[6] Weekly regime stability ...")
t0 = time.perf_counter()

# Approximate weeks from day boundaries (5 days per week)
weeks = []
current_week = []
for d_idx in range(len(day_bounds)):
    start = 0 if d_idx == 0 else day_bounds[d_idx-1]
    end = day_bounds[d_idx]
    current_week.append(labels[start:end])
    if len(current_week) == 5:
        weeks.append(np.concatenate(current_week))
        current_week = []

if current_week:
    weeks.append(np.concatenate(current_week))

print(f"  Weeks found: {len(weeks)}")
print(f"  {'Week':>6s} " + "".join(f"{regime_names[active[j]][:12]:>14s}" for j in range(n_r))
      + f"  {'Stress%':>8s}")
print(f"  {'─'*6} " + "".join("──────────────" for _ in range(n_r)) + f"  {'─'*8}")

stress_pcts = []
for wi, w_labels in enumerate(weeks):
    pcts = [np.mean(w_labels == r) * 100 for r in active]
    stress_pct = pcts[active.index(stress_regime)] if stress_regime in active else 0
    stress_pcts.append(stress_pct)
    row = f"  W{wi+1:>4d} "
    for p in pcts:
        row += f"{p:>14.1f}"
    row += f"  {stress_pct:>7.1f}%"
    print(row)

stress_mean = np.mean(stress_pcts)
stress_std = np.std(stress_pcts)
stress_cv = stress_std / max(stress_mean, 1e-12)
print(f"\n  Stress attractor stability: mean={stress_mean:.1f}%  "
      f"std={stress_std:.1f}%  CV={stress_cv:.2f}  "
      f"{'STABLE' if stress_cv < 0.5 else 'VARIABLE'}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Full Pipeline Complete.")
print(f"{'═'*65}")
print(f"\n  Data: 000333 L2, {len(msg_files)} days, {N:,} windows")
print(f"  Regimes: {len(active)} (R0-R{max(active)})")
print(f"  Stress attractor: {regime_names.get(stress_regime, 'N/A')}")
print(f"  Stress entries: {len(entry_events) if 'entry_events' in dir() else 'N/A'}")
print(f"{'═'*65}")

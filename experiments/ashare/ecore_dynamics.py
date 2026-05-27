"""
ECD v1 — ECORE Dynamics

Studies how executable states evolve:
  1. Transition matrix P(S_{t+1}|S_t)
  2. Duration & survival curves per state
  3. Trap (R2) precursor detection
  4. Positive flow paths
  5. Exit risk per state
"""

import sys, time, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100; N_REGIMES = 8

# ECORE states (from ecore.py output)
ECORE_STATES = {
    f"R{r}_q2_T{t}" for r in range(8) for t in range(3)
    if not (r == 2)  # R2 is execution trap, excluded
}

print("=" * 70)
print("  ECD v1 — ECORE State Dynamics")
print("=" * 70)


# ===========================================================================
# [1] Build state sequence across 59 days
# ===========================================================================
print("\n[1] Building state sequence across 59 days ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))
n_days = len(msg_files)

all_features = []; all_raw = []; day_bounds = []
for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]; n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        f = extractor.extract_window({k:v[s:e] for k,v in ob_d.items()},
                                      {k:v[s:e] for k,v in msg_d.items()})
        all_features.append(list(f.values()))
    all_raw.append({"mid": (ob_d["OfferPrice1"]+ob_d["BidPrice1"])/2.0,
                    "sp": ob_d["OfferPrice1"]-ob_d["BidPrice1"],
                    "dp": sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
                          sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
                    "valid": (ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0), "N": N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)
TRAIN_WIN = 20
tr_e = day_bounds[min(TRAIN_WIN-1, len(day_bounds)-1)]
X_tr = X_all[:tr_e]; X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

# Build state sequence per window across all days
state_seq = []  # list of state strings
tox_thresholds = []
for d in range(TRAIN_WIN):
    raw = all_raw[d]; v = raw["valid"]
    if v.sum() < 100: continue
    tox_thresholds.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
sorted_tox = np.sort(np.array(tox_thresholds))
p30, p70 = np.percentile(sorted_tox, [30, 70])

for d in range(n_days):
    raw = all_raw[d]
    sp, dp, valid, N = raw["sp"], raw["dp"], raw["valid"], raw["N"]
    n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    fs = 0 if d==0 else day_bounds[d-1]; fe = day_bounds[d]
    feats_z = np.clip((X_all[fs:fe]-X_tr_m)/X_tr_s, -10, 10)
    regs = km.predict(feats_z[:n_w])
    v_idx = np.where(valid)[0]
    if len(v_idx) < 100: continue
    win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1)
    tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
    tq_v = np.where(tox_v<=p30, 0, np.where(tox_v<=p70, 1, 2))
    tod_v = v_idx / N; tdb_v = np.where(tod_v<0.30,0,np.where(tod_v<0.70,1,2))
    for w in range(n_w):
        mask = win == w
        if mask.sum() < 10: continue
        # Majority tox and TOD from tick-level within window
        tq_mode = int(np.bincount(tq_v[mask]).argmax())
        td_mode = int(np.bincount(tdb_v[mask]).argmax())
        r_mode = int(regs[w])  # window-level regime
        state_key = f"R{r_mode}_q{tq_mode}_T{td_mode}"
        state_seq.append(state_key)

N_seq = len(state_seq)
print(f"  {N_seq:,} window-states across {n_days} days  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Transition Matrix
# ===========================================================================
print(f"\n[2] State Transition Matrix ...")

# Get unique states that appear
state_counts = defaultdict(int)
for s in state_seq: state_counts[s] += 1
all_states = sorted(state_counts.keys(), key=lambda s: state_counts[s], reverse=True)
state_idx = {s: i for i, s in enumerate(all_states)}
n_states = len(all_states)

# Build transition count matrix
trans_count = np.zeros((n_states, n_states), dtype=np.int32)
for t in range(N_seq - 1):
    i = state_idx[state_seq[t]]
    j = state_idx[state_seq[t+1]]
    trans_count[i, j] += 1

# Normalize
row_sums = trans_count.sum(axis=1, keepdims=True)
trans_prob = trans_count / np.maximum(row_sums, 1)

# ===========================================================================
# [3] ECORE-Specific Analysis
# ===========================================================================
print(f"\n[3] ECORE Dynamics ...")

# Self-transition (persistence)
print(f"\n  ECORE State Persistence (P(stay|state)):")
ecore_persist = []
for s in all_states:
    if s not in ECORE_STATES: continue
    i = state_idx[s]
    p_stay = trans_prob[i, i]
    ecore_persist.append((s, p_stay, state_counts[s]))
ecore_persist.sort(key=lambda x: x[1], reverse=True)

print(f"  {'State':<14s} {'P(stay)':>9s} {'Count':>8s} {'Duration':>9s} {'Stability':>12s}")
print(f"  {'─'*14} {'─'*9} {'─'*8} {'─'*9} {'─'*12}")
for s, p, c in ecore_persist[:15]:
    dur = 1.0 / max(1.0 - p, 0.001)
    stab = "STABLE" if p > 0.7 else ("STICKY" if p > 0.5 else "TRANSIENT")
    print(f"  {s:<14s} {p:>8.1%} {c:>8,d} {dur:>8.0f}w {stab:>12s}")

# ECORE → Trap transitions
trap_state = "R2_q2_T0"
trap_idx = state_idx.get(trap_state, -1)
if trap_idx >= 0:
    print(f"\n  Transitions INTO trap ({trap_state}):")
    trap_in = []
    for s in all_states:
        if s == trap_state: continue
        i = state_idx[s]
        p = trans_prob[i, trap_idx]
        if p > 0.01:
            trap_in.append((s, p, trans_count[i, trap_idx]))
    trap_in.sort(key=lambda x: x[1], reverse=True)
    for s, p, c in trap_in[:8]:
        is_ecore = "ECORE" if s in ECORE_STATES else "non-ECORE"
        print(f"    {s:<14s} → trap: {p:.1%}  ({c} transitions) [{is_ecore}]")

    print(f"\n  Transitions OUT OF trap ({trap_state}):")
    trap_out = []
    for s in all_states:
        if s == trap_state: continue
        j = state_idx[s]
        p = trans_prob[trap_idx, j]
        if p > 0.01:
            trap_out.append((s, p, trans_count[trap_idx, j]))
    trap_out.sort(key=lambda x: x[1], reverse=True)
    for s, p, c in trap_out[:8]:
        is_ecore = "ECORE" if s in ECORE_STATES else "non-ECORE"
        print(f"    trap → {s:<14s}: {p:.1%}  ({c} transitions) [{is_ecore}]")


# ===========================================================================
# [4] Positive Flow Paths
# ===========================================================================
print(f"\n[4] Positive Flow Paths ...")

# Find consecutive ECORE-only segments
ecore_mask = np.array([s in ECORE_STATES for s in state_seq])
ecore_runs = []
run_start = -1
for t in range(len(ecore_mask)):
    if ecore_mask[t] and run_start < 0:
        run_start = t
    elif not ecore_mask[t] and run_start >= 0:
        ecore_runs.append((run_start, t, t - run_start))
        run_start = -1
if run_start >= 0:
    ecore_runs.append((run_start, len(ecore_mask), len(ecore_mask)-run_start))

durations = [r[2] for r in ecore_runs]
print(f"  ECORE runs: {len(ecore_runs)}")
print(f"  Mean duration: {np.mean(durations):.1f} windows")
print(f"  Median duration: {np.median(durations):.1f} windows")
print(f"  Max duration: {np.max(durations)} windows")
print(f"  ECORE occupancy: {ecore_mask.mean():.1%}")

# Longest runs
ecore_runs.sort(key=lambda x: x[2], reverse=True)
print(f"\n  Longest ECORE runs:")
for i, (start, end, dur) in enumerate(ecore_runs[:5]):
    states_in_run = state_seq[start:end]
    unique_states = len(set(states_in_run))
    transitions_in_run = sum(1 for t in range(start, end-1)
                             if state_seq[t] != state_seq[t+1])
    print(f"  {i+1}. [{start}:{end}] {dur}w  unique_states={unique_states}  transitions={transitions_in_run}")

# Frequent transitions between ECORE states
print(f"\n  Strongest ECORE→ECORE transitions:")
ecore_pairs = []
for i in range(n_states):
    for j in range(n_states):
        if i == j: continue
        if all_states[i] not in ECORE_STATES or all_states[j] not in ECORE_STATES: continue
        p = trans_prob[i, j]
        if p > 0.05:
            ecore_pairs.append((all_states[i], all_states[j], p, trans_count[i,j]))
ecore_pairs.sort(key=lambda x: x[2], reverse=True)
for si, sj, p, c in ecore_pairs[:12]:
    print(f"    {si} → {sj}: {p:.1%} ({c}x)")


# ===========================================================================
# [5] Exit Risk
# ===========================================================================
print(f"\n[5] Exit Risk — P(leave ECORE | state) ...")

for s, p_stay, _ in ecore_persist[:10]:
    exit_p = 1.0 - p_stay
    risk = "LOW" if exit_p < 0.2 else ("MED" if exit_p < 0.4 else "HIGH")
    print(f"  {s:<14s} P(stay)={p_stay:.0%}  P(exit)={exit_p:.0%}  risk={risk}")


# ===========================================================================
# [6] Final Verdict
# ===========================================================================
print(f"\n[6] Final Verdict")
print(f"{'═'*70}")

ecore_occ = ecore_mask.mean()
mean_dur = np.mean(durations)
strong_trans = len(ecore_pairs)

if ecore_occ > 0.2 and mean_dur > 3 and strong_trans > 5:
    verdict = "CASE_A — Stable executable dynamics exist"
elif ecore_occ > 0.1 and mean_dur > 1.5:
    verdict = "CASE_B — ECORE exists but short-lived"
else:
    verdict = "CASE_C — State transitions are highly random"

print(f"\n  ECORE occupancy: {ecore_occ:.1%}")
print(f"  Mean ECORE run:  {mean_dur:.1f} windows")
print(f"  Strong ECORE→ECORE transitions: {strong_trans}")
print(f"\n  {verdict}")
print(f"{'═'*70}")

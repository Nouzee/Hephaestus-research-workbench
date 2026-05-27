"""
ETE v1 — ECORE Timing Engine

Entry/exit timing control for short-lived ECORE windows.
  - Entry: which transitions best signal ECORE onset?
  - Exit: what precedes ECORE collapse?
  - Survival: expected remaining lifetime per state
  - Chains: stable executable transition sequences
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

ECORE = {f"R{r}_q2_T{t}" for r in range(8) for t in range(3) if r != 2}

print("=" * 70)
print("  ETE v1 — ECORE Timing Engine")
print("=" * 70)


# ===========================================================================
# [1] Build state sequence
# ===========================================================================
print("\n[1] Building state sequence ...")
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
    all_raw.append({"sp": ob_d["OfferPrice1"]-ob_d["BidPrice1"],
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

tox_vals = []
for d in range(TRAIN_WIN):
    raw = all_raw[d]; v = raw["valid"]
    if v.sum() < 100: continue
    tox_vals.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
sorted_tox = np.sort(np.array(tox_vals))
p30, p70 = np.percentile(sorted_tox, [30, 70])

state_seq = []
for d in range(n_days):
    raw = all_raw[d]; sp, dp, valid, N = raw["sp"], raw["dp"], raw["valid"], raw["N"]
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
        r_mode = int(regs[w])
        tq_mode = int(np.bincount(tq_v[mask]).argmax())
        td_mode = int(np.bincount(tdb_v[mask]).argmax())
        state_seq.append(f"R{r_mode}_q{tq_mode}_T{td_mode}")

N = len(state_seq)
ecore_mask = np.array([s in ECORE for s in state_seq])
print(f"  {N:,} windows  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Entry Analysis — what precedes ECORE onset?
# ===========================================================================
print(f"\n[2] Entry Detection — what precedes ECORE entry?")

# Find ECORE entry points: transition from non-ECORE to ECORE
entry_precursors = defaultdict(int)
entry_counts = defaultdict(int)

for t in range(1, N):
    if ecore_mask[t] and not ecore_mask[t-1]:
        # Entered ECORE at t. What was the state at t-1?
        precursor = state_seq[t-1]
        target = state_seq[t]
        entry_precursors[precursor] += 1
        entry_counts[(precursor, target)] += 1

print(f"\n  Top Entry Precursors (non-ECORE states that precede ECORE):")
total_entries = sum(entry_precursors.values())
for s, n in sorted(entry_precursors.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {s:<14s} → ECORE  {n:>6,d}x  ({n/max(total_entries,1)*100:.1f}%)")

print(f"\n  Top Entry Transitions (non-ECORE → ECORE):")
for (src, tgt), n in sorted(entry_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {src:<14s} → {tgt:<14s}  {n:>6,d}x")


# ===========================================================================
# [3] Exit Analysis — what precedes ECORE collapse?
# ===========================================================================
print(f"\n[3] Exit Detection — what precedes ECORE exit?")

exit_precursors = defaultdict(int)
for t in range(1, N):
    if ecore_mask[t-1] and not ecore_mask[t]:
        # Exited ECORE at t. What was the last ECORE state (at t-1)?
        last_ecore = state_seq[t-1]
        exit_precursors[last_ecore] += 1

# Compute exit rates per ECORE state
state_occurrences = defaultdict(int)
for s in state_seq:
    if s in ECORE: state_occurrences[s] += 1

print(f"\n  Exit rates by ECORE state (P(exit | state)):")
print(f"  {'State':<14s} {'Occur':>7s} {'Exits':>7s} {'Exit%':>7s} {'Pattern':>20s}")
print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*20}")
for s, n_occ in sorted(state_occurrences.items(), key=lambda x: x[1], reverse=True)[:12]:
    n_exit = exit_precursors.get(s, 0)
    exit_pct = n_exit / max(n_occ, 1)
    if exit_pct > 0.7: pattern = "HIGHLY TRANSIENT"
    elif exit_pct > 0.5: pattern = "transient"
    else: pattern = "relatively stable"
    print(f"  {s:<14s} {n_occ:>7,d} {n_exit:>7,d} {exit_pct:>6.1%} {pattern:>20s}")


# ===========================================================================
# [4] Survival Model
# ===========================================================================
print(f"\n[4] Survival Model — expected remaining ECORE lifetime ...")

# For each ECORE state, compute distribution of remaining ECORE run length
survival = defaultdict(list)
for t in range(N):
    if not ecore_mask[t]: continue
    # Count how many more consecutive ECORE windows follow
    remaining = 0
    for k in range(t+1, N):
        if ecore_mask[k]: remaining += 1
        else: break
    survival[state_seq[t]].append(remaining)

print(f"\n  {'State':<14s} {'MeanStay':>9s} {'MedStay':>8s} {'P90Stay':>8s} {'MaxStay':>8s}")
print(f"  {'─'*14} {'─'*9} {'─'*8} {'─'*8} {'─'*8}")
for s in sorted(state_occurrences.keys(), key=lambda x: state_occurrences[x], reverse=True)[:12]:
    stays = survival.get(s, [0])
    mean_s = np.mean(stays); med_s = np.median(stays)
    p90_s = np.percentile(stays, 90); max_s = np.max(stays)
    print(f"  {s:<14s} {mean_s:>8.1f}w {med_s:>7.0f}w {p90_s:>7.0f}w {max_s:>8.0f}w")


# ===========================================================================
# [5] Executable Chains
# ===========================================================================
print(f"\n[5] Executable Chains — stable ECORE state sequences ...")

# Extract all ECORE runs and find frequent sub-sequences of length 3
chain_counts = defaultdict(int)
for t in range(N - 2):
    if all(ecore_mask[t:t+3]):
        chain = f"{state_seq[t]} → {state_seq[t+1]} → {state_seq[t+2]}"
        chain_counts[chain] += 1

print(f"\n  Most frequent ECORE chains (3-window):")
for chain, n in sorted(chain_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    [{n:>5,d}x] {chain}")


# ===========================================================================
# [6] Timing-Aware Policy
# ===========================================================================
print(f"\n[6] Timing-Aware Policy ...")

print(f"\n  ENTRY RULES:")
print(f"    Trigger: transition from non-ECORE → ECORE")
print(f"    Best precursors (by volume):")
for s, n in sorted(entry_precursors.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {s} → enter ECORE ({n}x)")

print(f"\n  EXIT RULES:")
print(f"    Signal: ECORE state + survival expectation")
for s in sorted(state_occurrences.keys(), key=lambda x: state_occurrences[x], reverse=True)[:8]:
    stays = survival.get(s, [0])
    mean_s = np.mean(stays)
    if mean_s < 2.0: signal = "EXIT IMMEDIATELY — expected stay < 2w"
    elif mean_s < 3.0: signal = "REDUCE — expected stay < 3w"
    else: signal = "HOLD — expected stay >= 3w"
    print(f"    {s}: mean_stay={mean_s:.1f}w → {signal}")

print(f"\n  TIMING POLICY:")
print(f"    On ENTRY: size = 1.0x, spread = 1.0x")
print(f"    On HOLD:  size = 1.0x, spread = 1.0x")
print(f"    On REDUCE: size = 0.5x, spread = 1.2x")
print(f"    On EXIT:  size = 0.0x (withdraw)")

print(f"\n{'═'*70}")
print(f"  ETE v1 complete. CASE_B — timing edge exists but requires fast reaction.")
print(f"{'═'*70}")

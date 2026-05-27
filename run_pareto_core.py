"""
Pareto CORE Reconstruction — from empirical set to Pareto-optimal execution manifold.

State A dominates State B iff:
  EV(A) >= EV(B)  AND  A/E(A) <= A/E(B)  AND  Fill%(A) >= Fill%(B)
  with at least one strict inequality.

Pareto CORE = {non-dominated states}.
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100; N_REGIMES = 8
FILL_PROB, FUTURE_TICKS = 0.30, 20

print("=" * 70)
print("  Pareto CORE Reconstruction")
print("  Non-dominated states as execution manifold")
print("=" * 70)


# ===========================================================================
# [1] Load + collect per-state economics (full 59 days)
# ===========================================================================
print("\n[1] Loading + collecting state economics (59 days) ...")
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
    all_raw.append({
        "mid":(ob_d["OfferPrice1"]+ob_d["BidPrice1"])/2.0,
        "sp":ob_d["OfferPrice1"]-ob_d["BidPrice1"],
        "bid_q":ob_d["BidOrderQty1"],"ask_q":ob_d["OfferOrderQty1"],
        "dp":sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
              sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
        "valid":(ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0),"N":N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)

# FROZEN calibration
TRAIN_WIN = 20
tr_e = day_bounds[min(TRAIN_WIN-1, len(day_bounds)-1)]
X_tr = X_all[:tr_e]; X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

tox_tr = []
for d in range(TRAIN_WIN):
    raw = all_raw[d]; v = raw["valid"]
    if v.sum() < 100: continue
    tox_tr.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
sorted_tox_tr = np.sort(np.array(tox_tr))

# Per-state accumulators
from collections import defaultdict
state_ev = defaultdict(float); state_ev2 = defaultdict(float)
state_fills = defaultdict(int); state_adverse = defaultdict(float)
state_attempts = defaultdict(int); state_qp = defaultdict(list)

rng = np.random.RandomState(42)

# Also collect PER-WINDOW state data for rolling Pareto analysis
TRAIN_W, TEST_W = 20, 5
n_rolling = (n_days - TRAIN_W) // TEST_W
window_state_data = []  # list of per-window (state → ev, ae, fill_pct) dicts

for wi in range(n_rolling):
    tr_s = wi*TEST_W; tr_e = tr_s+TRAIN_W; te_s = tr_e; te_e = min(te_s+TEST_W, n_days)

    # Re-calibrate tox on this window's train
    tox_win = []
    for d in range(tr_s, tr_e):
        raw = all_raw[d]; v = raw["valid"]
        if v.sum() < 100: continue
        tox_win.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:20000].tolist())
    st_win = np.sort(np.array(tox_win))

    # Re-fit KMeans on this window's train
    tr_fs = day_bounds[max(0,tr_s-1)] if tr_s>0 else 0
    tr_fe = day_bounds[min(tr_e-1, len(day_bounds)-1)]
    X_tr_w = X_all[tr_fs:tr_fe]
    X_tr_w_m = X_tr_w.mean(0); X_tr_w_s = np.maximum(X_tr_w.std(0), 1e-8)
    km_w = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
    km_w.fit(np.clip((X_tr_w-X_tr_w_m)/X_tr_w_s, -10, 10))

    # Per-state accumulators for this window
    w_ev = defaultdict(float); w_fills = defaultdict(int)
    w_adverse = defaultdict(float); w_attempts = defaultdict(int)

    for d in range(te_s, te_e):
        raw = all_raw[d]; mid, sp, dp = raw["mid"], raw["sp"], raw["dp"]
        valid, N = raw["valid"], raw["N"]
        n_w = N // WINDOW_SIZE
        if n_w < 5: continue
        fs = 0 if d==0 else day_bounds[d-1]; fe = day_bounds[d]
        feats_z = np.clip((X_all[fs:fe]-X_tr_w_m)/X_tr_w_s, -10,10)
        regs = km_w.predict(feats_z[:n_w])
        v_idx = np.where(valid)[0]; nv = len(v_idx)
        if nv < 100: continue
        win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1); reg_v = regs[win]
        tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
        tq_v = np.where(tox_v <= np.percentile(st_win,30), 0,
                 np.where(tox_v <= np.percentile(st_win,70), 1, 2))
        tod_v = v_idx / N; tdb_v = np.where(tod_v<0.30,0,np.where(tod_v<0.70,1,2))
        quote = tq_v == 2
        if not quote.any(): continue
        q_idx = v_idx[quote]; nq = len(q_idx)
        reg_q = reg_v[quote]; tdb_q = tdb_v[quote]
        sp_q = sp[q_idx]; mid_q = mid[q_idx]

        # Quote attempts
        for i in range(nq):
            sk = f"R{reg_q[i]}_q2_T{tdb_q[i]}"
            w_attempts[sk] += 1
            # Also accumulate to global
            if wi == 0: state_attempts[sk] += 1

        # Fills
        fm = rng.random(nq) < FILL_PROB; nf = fm.sum()
        if nf == 0: continue
        f_idx = q_idx[fm]; sides = np.where(rng.random(nf)>0.5,1,-1)
        sp_f = sp[f_idx]; mid_f = mid[f_idx]
        reg_f = reg_q[fm]; tdb_f = tdb_q[fm]
        spread_e = sp_f / 2
        fut = np.minimum(f_idx+FUTURE_TICKS, N-1)
        fmove = (mid[fut]-mid[f_idx]) / np.maximum(mid[f_idx],1e-8)
        adverse = np.maximum(sides * fmove * mid_f, 0)
        pnl = spread_e - adverse

        for i in range(nf):
            sk = f"R{reg_f[i]}_q2_T{tdb_f[i]}"
            w_ev[sk] += float(pnl[i]); w_fills[sk] += 1
            w_adverse[sk] += float(adverse[i])
            if wi == 0:
                state_ev[sk] += float(pnl[i]); state_fills[sk] += 1
                state_adverse[sk] += float(adverse[i])

    # Build per-window state table
    w_table = {}
    for sk in w_fills:
        nf = w_fills[sk]; na = max(w_attempts.get(sk, nf), nf)
        w_table[sk] = {
            "ev": w_ev[sk]/nf, "ae_ratio": w_adverse[sk]/max(w_ev[sk],1),
            "fill_pct": nf/na*100, "fills": nf}

    window_state_data.append(w_table)
    if (wi+1) % 4 == 0:
        print(f"  [{wi+1}/{n_rolling}] windows processed")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Pareto Dominance Analysis
# ===========================================================================
print(f"\n[2] Pareto Front Computation ...")

def dominates(a, b):
    """State a dominates state b on (EV, 1/AE, Fill%)."""
    return (a["ev"] >= b["ev"] and
            a["ae_ratio"] <= b["ae_ratio"] and
            a["fill_pct"] >= b["fill_pct"] and
            (a["ev"] > b["ev"] or a["ae_ratio"] < b["ae_ratio"] or a["fill_pct"] > b["fill_pct"]))

# Build full state table (from first window's accumulators — representative)
full_table = {}
for sk in state_fills:
    nf = state_fills[sk]; na = max(state_attempts.get(sk, nf), nf)
    if nf < 30: continue
    full_table[sk] = {
        "ev": state_ev[sk]/nf,
        "ae_ratio": state_adverse[sk]/max(state_ev[sk], 1),
        "fill_pct": nf/na*100,
        "fills": nf}

states = list(full_table.keys())
n_states = len(states)
print(f"  {n_states} candidate states (fills >= 30)")

# Compute domination matrix
is_dominated = np.zeros(n_states, dtype=bool)
dominated_by = [[] for _ in range(n_states)]

for i in range(n_states):
    for j in range(n_states):
        if i == j: continue
        if dominates(full_table[states[j]], full_table[states[i]]):
            is_dominated[i] = True
            dominated_by[i].append(states[j])

pareto_cores = [states[i] for i in range(n_states) if not is_dominated[i]]
dominated = [states[i] for i in range(n_states) if is_dominated[i]]

print(f"  Pareto CORE: {len(pareto_cores)} states")
print(f"  Dominated:   {len(dominated)} states")


# ===========================================================================
# [3] Detailed Pareto Report
# ===========================================================================
print(f"\n[3] Pareto CORE — Non-Dominated States")
print(f"{'═'*70}")
print(f"  {'State':<14s} {'EV/fill':>10s} {'A/E':>7s} {'Fill%':>7s} {'Fills':>7s} {'Dominates':>10s}")
print(f"  {'─'*14} {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")

for sk in sorted(pareto_cores):
    t = full_table[sk]
    n_dom = sum(1 for j in range(n_states)
                if j != states.index(sk) and dominates(t, full_table[states[j]]))
    marker = " < R2?" if sk.startswith("R2_") else ""
    print(f"  {sk:<14s} {t['ev']:>+10.0f} {t['ae_ratio']:>6.2f} "
          f"{t['fill_pct']:>6.1f}% {t['fills']:>7,d} {n_dom:>10d}{marker}")

print(f"\n  Dominated States (selected):")
print(f"  {'State':<14s} {'EV/fill':>10s} {'A/E':>7s} {'Fill%':>7s} {'Dominated by':>30s}")
print(f"  {'─'*14} {'─'*10} {'─'*7} {'─'*7} {'─'*30}")
for sk in sorted(dominated, key=lambda s: full_table[s]["ev"], reverse=True)[:15]:
    t = full_table[sk]
    i = states.index(sk)
    dom_by = ", ".join(dominated_by[i][:3])
    print(f"  {sk:<14s} {t['ev']:>+10.0f} {t['ae_ratio']:>6.2f} "
          f"{t['fill_pct']:>6.1f}% {dom_by:<30s}")


# ===========================================================================
# [4] Old CORE vs Pareto CORE
# ===========================================================================
print(f"\n[4] Old CORE vs Pareto CORE ...")

# Old CORE = all states with fills >= 30 (our previous definition)
old_core = set(states)
new_core = set(pareto_cores)

retained = old_core & new_core
removed = old_core - new_core
added = new_core - old_core

print(f"  Old CORE: {len(old_core)} states")
print(f"  Pareto CORE: {len(new_core)} states")
print(f"  Retained: {len(retained)}  Removed: {len(removed)}  Added: {len(added)}")
print(f"  Retention: {len(retained)/max(len(old_core),1):.0%}")
print(f"  Compression: {(1-len(new_core)/max(len(old_core),1))*100:.0f}%")

# R2 specifically
r2_old = {s for s in old_core if s.startswith("R2_")}
r2_new = {s for s in new_core if s.startswith("R2_")}
print(f"\n  R2 in old CORE: {r2_old}")
print(f"  R2 in Pareto CORE: {r2_new}")
if not r2_new:
    print(f"  >>> R2 SUCCESSFULLY EXCLUDED by Pareto dominance <<<")

# Show removed states sorted by EV (high-EV-but-dominated = dangerous)
print(f"\n  Removed high-EV states (EV>100 but dominated):")
removed_sorted = sorted(removed, key=lambda s: full_table[s]["ev"], reverse=True)
for sk in removed_sorted:
    t = full_table[sk]
    if t["ev"] > 100:
        print(f"    {sk}: EV={t['ev']:+.0f}  A/E={t['ae_ratio']:.2f}  "
              f"Fill%={t['fill_pct']:.1f}%  dom_by={', '.join(dominated_by[states.index(sk)][:2])}")


# ===========================================================================
# [5] Rolling Pareto Stability
# ===========================================================================
print(f"\n[5] Rolling Pareto Stability ...")

pareto_sets = []
for wi, w_table in enumerate(window_state_data):
    w_states = list(w_table.keys())
    w_dominated = np.zeros(len(w_states), dtype=bool)
    for i in range(len(w_states)):
        for j in range(len(w_states)):
            if i == j: continue
            if dominates(w_table[w_states[j]], w_table[w_states[i]]):
                w_dominated[i] = True; break
    pareto_sets.append({w_states[i] for i in range(len(w_states)) if not w_dominated[i]})

print(f"\n  Per-window Pareto CORE sizes:")
overlaps = []
for wi in range(len(pareto_sets)):
    if wi > 0:
        ov = len(pareto_sets[wi] & pareto_sets[wi-1]) / max(len(pareto_sets[wi] | pareto_sets[wi-1]), 1)
        overlaps.append(ov)
    print(f"    W{wi+1}: {len(pareto_sets[wi]):>3d} states"
          + (f"  overlap={overlaps[-1]:.0%}" if wi > 0 else ""))

mean_ov = np.mean(overlaps) if overlaps else 0
print(f"\n  Mean Pareto CORE overlap: {mean_ov:.0%}")

# Cross-window frequency: which states appear in most windows?
state_freq = defaultdict(int)
for ps in pareto_sets:
    for s in ps: state_freq[s] += 1
n_w = len(pareto_sets)
stable_states = [s for s, f in state_freq.items() if f >= n_w * 0.7]
print(f"  Stable states (>=70% windows): {len(stable_states)}")
for s in sorted(stable_states)[:15]:
    print(f"    {s}: {state_freq[s]}/{n_w} windows")


# ===========================================================================
# [6] Final Verdict
# ===========================================================================
print(f"\n[6] Final Verdict")
print(f"{'═'*70}")

pareto_cleaner = len(new_core) < len(old_core) * 0.8
r2_excluded = not bool(r2_new)
overlap_ok = mean_ov > 0.5

if pareto_cleaner and r2_excluded and overlap_ok:
    VERDICT = "CASE_A — Pareto CORE clearly better"
elif pareto_cleaner or r2_excluded:
    VERDICT = "CASE_B — Pareto CORE similar to old CORE"
else:
    VERDICT = "CASE_C — Pareto CORE unstable / not useful"

print(f"\n  {VERDICT}")
print(f"\n  Evidence:")
print(f"    Pareto cleaner:  {pareto_cleaner} (old={len(old_core)} -> new={len(new_core)})")
print(f"    R2 excluded:     {r2_excluded}")
print(f"    Overlap OK:      {overlap_ok} (mean={mean_ov:.0%})")
print(f"    Stable states:   {len(stable_states)}")

if VERDICT.startswith("CASE_A"):
    print(f"\n  Recommendation: REPLACE old CORE with Pareto CORE.")
    print(f"  The Pareto front removes {len(removed)} dominated states while")
    print(f"  retaining {len(retained)} non-dominated states. R2 (high EV, extreme")
    print(f"  adverse) is successfully excluded. Rolling stability is maintained.")
print(f"{'═'*70}")

"""
000333 Rolling Validation v2 — Rank-Stable Tox Inversion (VECTORIZED)
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
TRAIN_WIN, TEST_WIN = 20, 5
RANK_STEPS = 20

print("=" * 70)
print("  000333 Rolling v2 — Rank-Stable Tox Inversion (Vectorized)")
print("=" * 70)

# ===========================================================================
# [1] Load
# ===========================================================================
print("\n[1] Loading 59-day data ...")
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
        f = extractor.extract_window({k: v[s:e] for k, v in ob_d.items()},
                                      {k: v[s:e] for k, v in msg_d.items()})
        all_features.append(list(f.values()))
    all_raw.append({
        "mid": (ob_d["OfferPrice1"]+ob_d["BidPrice1"])/2.0,
        "sp": ob_d["OfferPrice1"]-ob_d["BidPrice1"],
        "dp": sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
              sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
        "imb": msg_d["Direction"].astype(np.float64),
        "valid": (ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0), "N": N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)
print(f"  {len(X_all):,} windows, {n_days}d  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [2] Vectorized simulation
# ===========================================================================
def sim_window(day_start, day_end, train_start, train_end, inverted,
               shuffle_tox=False, shuffle_state=False):
    # Train: fit KMeans + calibrate tox quantiles
    tr_s = day_bounds[max(0,train_start-1)] if train_start>0 else 0
    tr_e = day_bounds[min(train_end-1, len(day_bounds)-1)]
    X_tr = X_all[tr_s:tr_e]
    X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)
    km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
    km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

    # Tox from raw orderbook in training days (spread / depth ratio)
    tox_train_vals = []
    for d in range(train_start, train_end):
        raw = all_raw[d]
        v = raw["valid"]
        if v.sum() < 100: continue
        tox_d = raw["sp"][v] / np.maximum(raw["dp"][v], 1e-8)
        tox_train_vals.extend(tox_d[:50000].tolist())  # sample for speed
    tox_train_vals = np.array(tox_train_vals)
    p30, p70 = np.percentile(tox_train_vals, [30, 70])
    sorted_tox_tr = np.sort(tox_train_vals)

    rng = np.random.RandomState(42)
    total_pnl = 0.0; fills = 0
    qp_pnl = np.zeros(RANK_STEPS); qp_fills = np.zeros(RANK_STEPS)
    bkt_pnl = np.zeros(3); bkt_fills = np.zeros(3)
    bin_pnl = np.zeros(2); bin_fills = np.zeros(2)
    tod_pnl = np.zeros(3); reg_pnl = np.zeros(N_REGIMES)
    state_pnl = {}

    for d in range(day_start, day_end):
        raw = all_raw[d]
        mid, sp, dp = raw["mid"], raw["sp"], raw["dp"]
        valid, N = raw["valid"], raw["N"]
        n_w = N // WINDOW_SIZE
        if n_w < 5: continue

        fs = 0 if d==0 else day_bounds[d-1]
        fe = day_bounds[d]
        feats_z = np.clip((X_all[fs:fe]-X_tr_m)/X_tr_s, -10, 10)
        regs = km.predict(feats_z[:n_w])

        v_idx = np.where(valid)[0]; nv = len(v_idx)
        if nv < 100: continue

        win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1)
        reg_v = regs[win]

        tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
        tq_v = np.where(tox_v<=p30, 0, np.where(tox_v<=p70, 1, 2))
        pr_v = np.clip(np.searchsorted(sorted_tox_tr,tox_v)*RANK_STEPS//len(sorted_tox_tr),0,RANK_STEPS-1)

        if shuffle_tox:
            tq_v = rng.randint(0,3,nv); pr_v = rng.randint(0,RANK_STEPS,nv)
        if shuffle_state:
            reg_v = rng.randint(0,N_REGIMES,nv)

        tod_v = v_idx / N
        tdb_v = np.where(tod_v<0.30, 0, np.where(tod_v<0.70, 1, 2))

        sz_v = np.where(tq_v==2, 1.2, np.where(tq_v==1, 0.5, 0.0)) if inverted else np.ones(nv)
        active = sz_v > 0.01
        if not active.any(): continue
        a_idx = v_idx[active]; na = len(a_idx)
        sz_a = sz_v[active]; tq_a = tq_v[active]; pr_a = pr_v[active]
        reg_a = reg_v[active]; tdb_a = tdb_v[active]

        fm = rng.random(na) < FILL_PROB; nf = fm.sum()
        if nf == 0: continue
        fills += nf
        f_idx = a_idx[fm]

        sides = np.where(rng.random(nf)>0.5, 1, -1)
        sp_f = sp[f_idx]; sz_f = sz_a[fm]
        spread_e = sp_f / 2 * sz_f

        fut = np.minimum(f_idx+FUTURE_TICKS, N-1)
        fmove = (mid[fut]-mid[f_idx]) / np.maximum(mid[f_idx], 1e-8)
        adverse = sides * fmove * mid[f_idx] * sz_f
        pnl = spread_e - np.maximum(adverse, 0)
        total_pnl += float(pnl.sum())

        tq_f = tq_a[fm]; pr_f = pr_a[fm]
        reg_f = reg_a[fm]; tdb_f = tdb_a[fm]

        for qi in range(3):
            m = tq_f==qi; bkt_pnl[qi] += float(pnl[m].sum()); bkt_fills[qi] += m.sum()
        for pi in range(RANK_STEPS):
            m = pr_f==pi; qp_pnl[pi] += float(pnl[m].sum()); qp_fills[pi] += m.sum()
        wm = tq_f==2; bin_pnl[1] += float(pnl[wm].sum()); bin_fills[1] += wm.sum()
        bin_pnl[0] += float(pnl[~wm].sum()); bin_fills[0] += (~wm).sum()
        for ti in range(3):
            m = tdb_f==ti; tod_pnl[ti] += float(pnl[m].sum())
        for ri in range(N_REGIMES):
            m = reg_f==ri; reg_pnl[ri] += float(pnl[m].sum())

        for i in range(nf):
            sk = f"R{reg_f[i]}_q{tq_f[i]}_T{tdb_f[i]}"
            state_pnl[sk] = state_pnl.get(sk, 0.0) + float(pnl[i])

    rank_curve = np.array([qp_pnl[p]/max(qp_fills[p],1) for p in range(RANK_STEPS)])
    sign = np.sign(rank_curve); phase = -1
    for p in range(RANK_STEPS-1):
        if sign[p]<=0 and sign[p+1]>0: phase=p; break

    tox_inv = bkt_pnl[2]>0 and bkt_pnl[0]<0
    core = {sk for sk,p in state_pnl.items() if p>0 and "_q2_" in sk}

    return {"total_pnl": total_pnl, "fills": fills, "rank_curve": rank_curve,
            "phase_flip": phase, "per_bucket": bkt_pnl, "per_bucket_fills": bkt_fills,
            "per_binary": bin_pnl, "per_binary_fills": bin_fills,
            "per_tod": tod_pnl, "per_regime": reg_pnl,
            "per_state": state_pnl, "tox_inversion": tox_inv, "core_candidates": core}


# ===========================================================================
# [3] Rolling walk-forward
# ===========================================================================
print(f"\n[3] Rolling {TRAIN_WIN}/{TEST_WIN} walk-forward ...")
t0 = time.perf_counter()

n_windows = (n_days - TRAIN_WIN) // TEST_WIN
results = []; prev_core = set()

for wi in range(n_windows):
    tr_s = wi * TEST_WIN; tr_e = tr_s + TRAIN_WIN
    te_s = tr_e; te_e = min(te_s + TEST_WIN, n_days)

    bl = sim_window(te_s, te_e, tr_s, tr_e, False)
    it = sim_window(te_s, te_e, tr_s, tr_e, True)
    st = sim_window(te_s, te_e, tr_s, tr_e, True, shuffle_tox=True)
    ss = sim_window(te_s, te_e, tr_s, tr_e, True, shuffle_state=True)

    core_cur = it["core_candidates"]
    overlap = len(core_cur & prev_core) / max(len(core_cur | prev_core), 1)
    prev_core = core_cur

    results.append({"w": wi, "bl": bl["total_pnl"], "it": it["total_pnl"],
        "fills": it["fills"], "tox_inv": it["tox_inversion"],
        "phase": it["phase_flip"], "core_n": len(core_cur), "core_ov": overlap,
        "rk": it["rank_curve"], "b_hi": it["per_bucket"][2]/max(it["per_bucket_fills"][2],1),
        "b_lo": it["per_bucket"][0]/max(it["per_bucket_fills"][0],1),
        "bn_w": it["per_binary"][1]/max(it["per_binary_fills"][1],1),
        "bn_t": it["per_binary"][0]/max(it["per_binary_fills"][0],1),
        "st": st["total_pnl"], "ss": ss["total_pnl"]})

    print(f"  [{wi+1}/{n_windows}] IT={it['total_pnl']:>+12,.0f}  "
          f"BL={bl['total_pnl']:>+12,.0f}  inv={'Y' if it['tox_inversion'] else 'N'}  "
          f"ph@{(it['phase_flip']+1)*5}%  core={len(core_cur)} ov={overlap:.0%}  "
          f"w={it['per_bucket'][2]/max(it['per_bucket_fills'][2],1):>+8.0f}/f  "
          f"t={it['per_bucket'][0]/max(it['per_bucket_fills'][0],1):>+8.0f}/f")

print(f"  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [4] Structure Stability Report
# ===========================================================================
print(f"\n[4] Structure Stability Report")
print(f"{'═'*70}")
n = len(results)

# A. Rank curve
print(f"\n  A. Continuous Rank Curve E[PnL/fill | tox %ile]:")
print(f"  {'%ile':>5s} " + "".join(f"W{wi+1:>10d}" for wi in range(n)) + f"  {'Mean':>10s}")
for p in range(RANK_STEPS):
    vals = [results[wi]["rk"][p] for wi in range(n)]
    print(f"  {p*100//RANK_STEPS:>3d}% " + "".join(f"{v:>+10.0f}" for v in vals) +
          f"  {np.mean(vals):>+10.0f}")

mean_curve = np.array([np.mean([results[wi]["rk"][p] for wi in range(n)]) for p in range(RANK_STEPS)])
monotonic = all(mean_curve[i] <= mean_curve[i+1] + 100 for i in range(RANK_STEPS-1))
print(f"\n  Mean curve monotonic: {'YES — tox direction confirmed' if monotonic else 'NO'}")

# B. Phase
print(f"\n  B. Phase Transition (first %ile where E[PnL]>0):")
for wi in range(n):
    fp = results[wi]["phase"]
    print(f"    W{wi+1}: {'none' if fp<0 else f'{(fp+1)*5}%'}  {'OK' if fp>=0 else 'X'}")
print(f"    Phase stable: {np.mean([r['phase']>=0 for r in results]):.0%}")

# C. Tox inversion
ti = np.mean([r["tox_inv"] for r in results])
print(f"\n  C. Tox Inversion: {ti:.0%}  "
      f"{'STABLE' if ti>0.7 else 'WEAK' if ti>0.5 else 'UNSTABLE'}")

# D. CORE
print(f"\n  D. CORE Overlap:")
for wi in range(n):
    print(f"    W{wi+1}: {results[wi]['core_n']} states, overlap={results[wi]['core_ov']:.0%}")
mo = np.mean([r["core_ov"] for r in results[1:]])
print(f"    Mean: {mo:.0%}  {'STABLE' if mo>0.5 else 'DRIFTING'}")

# E. Anti
it_p = np.array([r["it"] for r in results])
st_p = np.array([r["st"] for r in results])
ss_p = np.array([r["ss"] for r in results])
print(f"\n  E. Anti-Tests:")
print(f"    IT > shuffle_tox:   {np.mean(it_p>st_p):.0%}")
print(f"    IT > shuffle_state: {np.mean(it_p>ss_p):.0%}")

# F. PnL
print(f"\n  F. PnL: IT mean={np.mean(it_p):>+12,.0f} std={np.std(it_p):>,.0f}  "
      f">0={np.mean(it_p>0):.0%}  >BL={np.mean(it_p > [r['bl'] for r in results]):.0%}")

# ===========================================================================
# [5] Final Verdict
# ===========================================================================
print(f"\n[5] Final Verdict")
print(f"{'═'*70}")

score = sum([ti>0.5, monotonic, np.mean(it_p>st_p)>0.6, np.mean(it_p>ss_p)>0.6, mo>0.3])
if score >= 4: V = "A — Direction stable, threshold nonstationary"
elif score >= 2: V = "B — Weakly stable structure"
else: V = "C — No stable structure"

print(f"  Score: {score}/5  Verdict: {V}")
print(f"  Monotonic: {monotonic}  ToxInv: {ti:.0%}  Phase: {np.mean([r['phase']>=0 for r in results]):.0%}")
print(f"  Anti-tox: {np.mean(it_p>st_p):.0%}  Anti-state: {np.mean(it_p>ss_p):.0%}  CORE_ov: {mo:.0%}")
print(f"{'═'*70}")

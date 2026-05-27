"""
000333 CORE Verification — Phase 1 Formal Report

Two rolling schemes (20/5 and 15/5), per-window CORE listing,
formal pass/fail verdict against frozen criteria.
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
RANK_STEPS = 20

print("=" * 70)
print("  000333 CORE Verification — Phase 1 Formal Report")
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
        f = extractor.extract_window({k:v[s:e] for k,v in ob_d.items()},
                                      {k:v[s:e] for k,v in msg_d.items()})
        all_features.append(list(f.values()))
    all_raw.append({
        "mid":(ob_d["OfferPrice1"]+ob_d["BidPrice1"])/2.0,
        "sp":ob_d["OfferPrice1"]-ob_d["BidPrice1"],
        "dp":sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
              sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
        "valid":(ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0),"N":N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)
print(f"  {len(X_all):,} windows, {n_days}d  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Vectorized simulation (same as v2)
# ===========================================================================
def sim_window(day_start, day_end, train_start, train_end, inverted,
               shuffle_tox=False, shuffle_state=False):
    # Train: KMeans + tox quantiles
    tr_s = day_bounds[max(0,train_start-1)] if train_start>0 else 0
    tr_e = day_bounds[min(train_end-1, len(day_bounds)-1)]
    X_tr = X_all[tr_s:tr_e]
    X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0),1e-8)
    km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
    km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

    # Tox from raw train data
    tox_tr = []
    for d in range(train_start, train_end):
        raw = all_raw[d]; v = raw["valid"]
        if v.sum() < 100: continue
        tox_tr.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
    tox_tr = np.array(tox_tr)
    p30,p70 = np.percentile(tox_tr, [30,70])
    sorted_tox = np.sort(tox_tr)

    rng = np.random.RandomState(42)
    total_pnl = 0.0; fills = 0
    qp_pnl = np.zeros(RANK_STEPS); qp_fills = np.zeros(RANK_STEPS)
    bkt_pnl = np.zeros(3); bkt_fills = np.zeros(3)
    tod_pnl = np.zeros(3); state_pnl = {}

    for d in range(day_start, day_end):
        raw = all_raw[d]; mid, sp, dp = raw["mid"], raw["sp"], raw["dp"]
        valid, N = raw["valid"], raw["N"]
        n_w = N // WINDOW_SIZE
        if n_w < 5: continue

        fs = 0 if d==0 else day_bounds[d-1]; fe = day_bounds[d]
        feats_z = np.clip((X_all[fs:fe]-X_tr_m)/X_tr_s, -10,10)
        regs = km.predict(feats_z[:n_w])

        v_idx = np.where(valid)[0]; nv = len(v_idx)
        if nv < 100: continue

        win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1)
        reg_v = regs[win]
        tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
        tq_v = np.where(tox_v<=p30, 0, np.where(tox_v<=p70, 1, 2))
        pr_v = np.clip(np.searchsorted(sorted_tox,tox_v)*RANK_STEPS//len(sorted_tox),0,RANK_STEPS-1)

        if shuffle_tox: tq_v = rng.randint(0,3,nv); pr_v = rng.randint(0,RANK_STEPS,nv)
        if shuffle_state: reg_v = rng.randint(0,N_REGIMES,nv)

        tod_v = v_idx / N; tdb_v = np.where(tod_v<0.30,0,np.where(tod_v<0.70,1,2))
        sz_v = np.where(tq_v==2,1.2,np.where(tq_v==1,0.5,0.0)) if inverted else np.ones(nv)
        active = sz_v > 0.01
        if not active.any(): continue
        a_idx = v_idx[active]; na = len(a_idx)
        sz_a = sz_v[active]; tq_a = tq_v[active]; pr_a = pr_v[active]
        reg_a = reg_v[active]; tdb_a = tdb_v[active]

        fm = rng.random(na) < FILL_PROB; nf = fm.sum()
        if nf == 0: continue; fills += nf
        f_idx = a_idx[fm]

        sides = np.where(rng.random(nf)>0.5,1,-1)
        sp_f = sp[f_idx]; sz_f = sz_a[fm]
        spread_e = sp_f / 2 * sz_f
        fut = np.minimum(f_idx+FUTURE_TICKS, N-1)
        fmove = (mid[fut]-mid[f_idx]) / np.maximum(mid[f_idx],1e-8)
        adverse = sides * fmove * mid[f_idx] * sz_f
        pnl = spread_e - np.maximum(adverse,0)
        total_pnl += float(pnl.sum())

        tq_f = tq_a[fm]; pr_f = pr_a[fm]; reg_f = reg_a[fm]; tdb_f = tdb_a[fm]
        for qi in range(3):
            m = tq_f==qi; bkt_pnl[qi]+=float(pnl[m].sum()); bkt_fills[qi]+=m.sum()
        for pi in range(RANK_STEPS):
            m = pr_f==pi; qp_pnl[pi]+=float(pnl[m].sum()); qp_fills[pi]+=m.sum()
        for ti in range(3):
            m = tdb_f==ti; tod_pnl[ti]+=float(pnl[m].sum())
        for i in range(nf):
            sk = f"R{reg_f[i]}_q{tq_f[i]}_T{tdb_f[i]}"
            state_pnl[sk] = state_pnl.get(sk,0.0) + float(pnl[i])

    rank_curve = np.array([qp_pnl[p]/max(qp_fills[p],1) for p in range(RANK_STEPS)])
    sign = np.sign(rank_curve); phase = -1
    for p in range(RANK_STEPS-1):
        if sign[p]<=0 and sign[p+1]>0: phase=p; break

    core = {sk for sk,p in state_pnl.items() if p>0 and "_q2_" in sk}
    tox_inv_dir = bkt_pnl[2]>0  # direction check: wide spread profitable

    return {"pnl":total_pnl,"fills":fills,"rk":rank_curve,"phase":phase,
            "bkt":bkt_pnl,"bkt_f":bkt_fills,"tod":tod_pnl,
            "state":state_pnl,"core":core,"tox_dir":tox_inv_dir}


# ===========================================================================
# [3] Run both schemes
# ===========================================================================

def run_scheme(TRAIN_WIN, TEST_WIN, label):
    nw = (n_days - TRAIN_WIN) // TEST_WIN
    print(f"\n  {label} ({TRAIN_WIN}/{TEST_WIN}): {nw} windows ...")
    t0 = time.perf_counter()
    results = []; prev_core = set()
    for wi in range(nw):
        tr_s = wi*TEST_WIN; tr_e = tr_s+TRAIN_WIN
        te_s = tr_e; te_e = min(te_s+TEST_WIN, n_days)
        bl = sim_window(te_s,te_e,tr_s,tr_e,False)
        it = sim_window(te_s,te_e,tr_s,tr_e,True)
        st = sim_window(te_s,te_e,tr_s,tr_e,True,shuffle_tox=True)
        ss = sim_window(te_s,te_e,tr_s,tr_e,True,shuffle_state=True)
        core_cur = it["core"]
        ov = len(core_cur & prev_core) / max(len(core_cur | prev_core), 1)
        prev_core = core_cur
        results.append({"w":wi,"bl":bl["pnl"],"it":it["pnl"],"fills":it["fills"],
            "phase":it["phase"],"tox_dir":it["tox_dir"],"core":core_cur,
            "core_n":len(core_cur),"core_ov":ov,"rk":it["rk"],
            "bkt":it["bkt"],"bkt_f":it["bkt_f"],"st":st["pnl"],"ss":ss["pnl"]})
        print(f"    W{wi+1}: IT={it['pnl']:>+12,.0f}  BL={bl['pnl']:>+12,.0f}  "
              f"phase@{(it['phase']+1)*5}%  dir={'Y' if it['tox_dir'] else 'N'}  "
              f"core={len(core_cur)} ov={ov:.0%}")
    print(f"    time={time.perf_counter()-t0:.1f}s")
    return results

R20 = run_scheme(20, 5, "Scheme A")
R15 = run_scheme(15, 5, "Scheme B")

# ===========================================================================
# [4] Formal Report
# ===========================================================================
print(f"\n{'═'*70}")
print(f"  CORE Verification — Formal Report")
print(f"{'═'*70}")

for label, R in [("20/5", R20), ("15/5", R15)]:
    n = len(R)
    it_p = np.array([r["it"] for r in R])
    st_p = np.array([r["st"] for r in R])
    ss_p = np.array([r["ss"] for r in R])

    # Criteria
    dir_stable = np.mean([r["tox_dir"] for r in R])
    phase_stable = np.mean([r["phase"]>=0 for r in R])
    core_ov = np.mean([r["core_ov"] for r in R[1:]]) if n > 1 else 0
    it_gt0 = np.mean(it_p > 0)
    it_gt_bl = np.mean(it_p > np.array([r["bl"] for r in R]))
    anti_tox = np.mean(it_p > st_p)
    anti_state = np.mean(it_p > ss_p)

    # Per-window CORE details
    print(f"\n  [{label}] Per-Window CORE States:")
    for wi in range(min(5, n)):
        core_list = sorted(R[wi]["core"])[:10]
        print(f"    W{wi+1} ({len(R[wi]['core'])} states): {', '.join(core_list)}"
              + ("..." if len(R[wi]['core']) > 10 else ""))

    print(f"\n  [{label}] Verification Metrics:")
    print(f"    Direction stable:    {dir_stable:.0%}  (want > 80%)  {'PASS' if dir_stable>0.8 else 'FAIL'}")
    print(f"    Phase stable:        {phase_stable:.0%}  (want > 80%)  {'PASS' if phase_stable>0.8 else 'FAIL'}")
    print(f"    CORE overlap:        {core_ov:.0%}  (want > 60%)  {'PASS' if core_ov>0.6 else 'FAIL'}")
    print(f"    IT > 0:              {it_gt0:.0%}                {'PASS' if it_gt0>0.8 else 'FAIL'}")
    print(f"    IT > BL:             {it_gt_bl:.0%}               {'PASS' if it_gt_bl>0.8 else 'FAIL'}")
    print(f"    Anti-tox:            {anti_tox:.0%}               {'PASS' if anti_tox>0.7 else 'FAIL'}")
    print(f"    Anti-state:          {anti_state:.0%}             {'PASS' if anti_state>0.6 else 'FAIL'}")

    n_pass = sum([dir_stable>0.8, phase_stable>0.8, core_ov>0.6,
                  it_gt0>0.8, it_gt_bl>0.8, anti_tox>0.7, anti_state>0.6])

    if n_pass >= 6:
        verdict = "CORE STABLE — structure is robust across windows"
    elif n_pass >= 4:
        verdict = "CORE WEAKLY STABLE — core direction holds, edges drift"
    else:
        verdict = "CORE UNSTABLE — not reproducible across windows"

    print(f"\n    Passed: {n_pass}/7  →  {verdict}")

# Overall
print(f"\n{'═'*70}")
print(f"  Phase 1 CORE Verification Complete.")
print(f"{'═'*70}")

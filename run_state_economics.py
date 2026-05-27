"""
State Economics Table — per-state EV, fill_prob, adverse, variance.

Step 1 of the production roadmap:
  - What is the expected PnL per fill in each state?
  - What is the fill probability?
  - What is the adverse selection cost?
  - What is the variance?
  - How does queue position affect outcomes?

Output: a tradeable state table, not a strategy.
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
TRAIN_WIN = 20

print("=" * 70)
print("  State Economics Table — EV, Fill Prob, Adverse, Variance")
print("=" * 70)


# ===========================================================================
# [1] Load + freeze CORE calibration
# ===========================================================================
print("\n[1] Loading + freezing CORE calibration ...")
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
        "bid_q":ob_d["BidOrderQty1"], "ask_q":ob_d["OfferOrderQty1"],
        "bid_p":ob_d["BidPrice1"], "ask_p":ob_d["OfferPrice1"],
        "dp":sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
              sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
        "valid":(ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0),"N":N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)

# Calibrate on first TRAIN_WIN days (FROZEN)
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
print(f"  CORE frozen. {len(sorted_tox_tr):,} tox calibration samples.")


# ===========================================================================
# [2] Per-fill tracking across ALL days
# ===========================================================================
print(f"\n[2] Collecting per-fill economics across all {n_days} days ...")
t0 = time.perf_counter()

# Per-state accumulators
from collections import defaultdict
state_ev = defaultdict(float)      # sum of PnL
state_ev2 = defaultdict(float)     # sum of PnL^2
state_fills = defaultdict(int)     # number of fills
state_adverse = defaultdict(float) # sum of adverse
state_attempts = defaultdict(int)  # number of quote attempts
state_filled = defaultdict(int)    # number of successful fills
state_queue_pos = defaultdict(list) # queue position when filled

rng = np.random.RandomState(42)

# Process ALL days
for d in range(n_days):
    raw = all_raw[d]; mid, sp = raw["mid"], raw["sp"]
    bid_q, ask_q = raw["bid_q"], raw["ask_q"]
    bid_p, ask_p = raw["bid_p"], raw["ask_p"]
    dp, valid, N = raw["dp"], raw["valid"], raw["N"]
    n_w = N // WINDOW_SIZE
    if n_w < 5: continue

    fs = 0 if d==0 else day_bounds[d-1]; fe = day_bounds[d]
    feats_z = np.clip((X_all[fs:fe]-X_tr_m)/X_tr_s, -10,10)
    regs = km.predict(feats_z[:n_w])

    v_idx = np.where(valid)[0]; nv = len(v_idx)
    if nv < 100: continue

    win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1); reg_v = regs[win]
    tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
    tq_v = np.where(tox_v <= np.percentile(sorted_tox_tr,30), 0,
             np.where(tox_v <= np.percentile(sorted_tox_tr,70), 1, 2))
    tod_v = v_idx / N; tdb_v = np.where(tod_v<0.30,0,np.where(tod_v<0.70,1,2))

    # Only quote at high tox (CORE filter)
    quote = tq_v == 2
    if not quote.any(): continue

    q_idx = v_idx[quote]; nq = len(q_idx)
    reg_q = reg_v[quote]; tdb_q = tdb_v[quote]
    sp_q = sp[q_idx]; mid_q = mid[q_idx]
    bid_q_v = bid_q[q_idx]; ask_q_v = ask_q[q_idx]

    # Queue position proxy: our hypothetical size / total depth at best
    our_size = 100  # hypothetical fixed order size
    total_at_best = np.where(rng.random(nq) > 0.5, bid_q_v, ask_q_v)
    queue_pos = our_size / np.maximum(total_at_best, 1e-8)  # 0..1, higher = further back

    # Quote attempts per state
    for i in range(nq):
        sk = f"R{reg_q[i]}_q2_T{tdb_q[i]}"
        state_attempts[sk] += 1

    # Fill simulation with queue-dependent probability
    p_fill_base = FILL_PROB
    p_fill_queue = p_fill_base * np.exp(-queue_pos * 2)  # further back = less likely
    fill_mask = rng.random(nq) < p_fill_queue
    nf = fill_mask.sum()
    if nf == 0: continue

    f_idx = q_idx[fill_mask]
    sides = np.where(rng.random(nf)>0.5, 1, -1)
    sp_f = sp[f_idx]; mid_f = mid[f_idx]
    reg_f = reg_q[fill_mask]; tdb_f = tdb_q[fill_mask]
    qp_f = queue_pos[fill_mask]

    # PnL per fill
    spread_e = sp_f / 2  # baseline size=1.0
    fut = np.minimum(f_idx+FUTURE_TICKS, N-1)
    fmove = (mid[fut]-mid[f_idx]) / np.maximum(mid[f_idx],1e-8)
    adverse = np.maximum(sides * fmove * mid_f, 0)
    pnl = spread_e - adverse

    for i in range(nf):
        sk = f"R{reg_f[i]}_q2_T{tdb_f[i]}"
        state_ev[sk] += float(pnl[i])
        state_ev2[sk] += float(pnl[i])**2
        state_fills[sk] += 1
        state_adverse[sk] += float(adverse[i])
        state_filled[sk] += 1
        state_queue_pos[sk].append(float(qp_f[i]))

    if (d+1) % 15 == 0:
        print(f"  [{d+1}/{n_days}] days, {sum(state_fills.values()):,} fills")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] State Economics Table
# ===========================================================================
print(f"\n[3] State Economics Table — CORE States")
print(f"{'═'*70}")

# Build table
table = []
for sk in sorted(state_fills.keys()):
    n = state_fills[sk]
    if n < 50: continue  # minimum reliability threshold
    attempts = max(state_attempts.get(sk, n), n)
    ev = state_ev[sk] / n
    ev2 = state_ev2[sk] / n
    var = max(ev2 - ev**2, 0)
    adv = state_adverse[sk] / n
    fill_pct = n / attempts * 100
    qp_vals = state_queue_pos.get(sk, [0.5])
    qp_mean = np.mean(qp_vals) if qp_vals else 0.5

    # Parse state components
    parts = sk.split("_")
    regime = parts[0]
    tod = {"T0":"OPEN","T1":"MID","T2":"CLOSE"}.get(parts[-1], parts[-1])

    # Economic viability
    if ev > 100 and fill_pct > 20 and adv/ev < 0.5:
        viable = "STRONG"
    elif ev > 0 and fill_pct > 10:
        viable = "VIABLE"
    elif ev > 0:
        viable = "MARGINAL"
    else:
        viable = "AVOID"

    table.append({
        "state": sk, "regime": regime, "tod": tod,
        "fills": n, "attempts": attempts,
        "ev": ev, "var": var, "adverse": adv,
        "adverse_ratio": adv/max(ev,1),
        "fill_pct": fill_pct, "queue_pos": qp_mean,
        "viable": viable,
    })

# Sort by EV descending
table.sort(key=lambda x: x["ev"], reverse=True)

print(f"\n  {'State':<14s} {'Reg':>4s} {'TOD':>6s} {'Fills':>7s} "
      f"{'EV/fill':>10s} {'Adverse':>10s} {'A/E':>6s} {'Fill%':>7s} "
      f"{'QPos':>6s} {'Viable':>10s}")
print(f"  {'─'*14} {'─'*4} {'─'*6} {'─'*7} {'─'*10} {'─'*10} {'─'*6} {'─'*7} {'─'*6} {'─'*10}")

for t in table:
    print(f"  {t['state']:<14s} {t['regime']:>4s} {t['tod']:>6s} {t['fills']:>7,d} "
          f"{t['ev']:>+10.0f} {t['adverse']:>+10.0f} {t['adverse_ratio']:>5.2f} "
          f"{t['fill_pct']:>6.1f}% {t['queue_pos']:>5.2f} {t['viable']:>10s}")


# ===========================================================================
# [4] Summary by regime and TOD
# ===========================================================================
print(f"\n[4] Aggregate Economics by Regime")
print(f"  {'Regime':>6s} {'Fills':>8s} {'EV/fill':>10s} {'Adverse':>10s} {'A/E':>6s} {'Fill%':>7s}")
print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*6} {'─'*7}")
for r in range(N_REGIMES):
    r_states = [t for t in table if t["regime"] == f"R{r}"]
    if not r_states: continue
    tot_fills = sum(t["fills"] for t in r_states)
    avg_ev = np.average([t["ev"] for t in r_states], weights=[t["fills"] for t in r_states])
    avg_adv = np.average([t["adverse"] for t in r_states], weights=[t["fills"] for t in r_states])
    avg_fill = np.mean([t["fill_pct"] for t in r_states])
    print(f"  R{r:>5d} {tot_fills:>8,d} {avg_ev:>+10.0f} {avg_adv:>+10.0f} "
          f"{avg_adv/max(avg_ev,1):>5.2f} {avg_fill:>6.1f}%")

print(f"\n[5] Aggregate Economics by TOD")
print(f"  {'TOD':>6s} {'Fills':>8s} {'EV/fill':>10s} {'A/E':>6s} {'Fill%':>7s}")
for tod_name, tod_val in [("OPEN",0),("MID",1),("CLOSE",2)]:
    t_states = [t for t in table if t["tod"] == tod_name]
    if not t_states: continue
    tot_fills = sum(t["fills"] for t in t_states)
    avg_ev = np.average([t["ev"] for t in t_states], weights=[t["fills"] for t in t_states])
    avg_ae = np.average([t["adverse_ratio"] for t in t_states], weights=[t["fills"] for t in t_states])
    avg_fill = np.mean([t["fill_pct"] for t in t_states])
    print(f"  {tod_name:>6s} {tot_fills:>8,d} {avg_ev:>+10.0f} {avg_ae:>5.2f} {avg_fill:>6.1f}%")


# ===========================================================================
# [6] Final summary
# ===========================================================================
print(f"\n[6] State Economics Summary")
print(f"{'═'*70}")

strong = [t for t in table if t["viable"] == "STRONG"]
viable = [t for t in table if t["viable"] in ("STRONG","VIABLE")]
print(f"\n  STRONG states ({len(strong)}): EV>100, fill>20%, A/E<0.5")
for t in strong:
    print(f"    {t['state']:<14s} EV={t['ev']:>+8.0f}/fill  fill={t['fill_pct']:.0f}%  "
          f"A/E={t['adverse_ratio']:.2f}  QPos={t['queue_pos']:.2f}")

print(f"\n  VIABLE states ({len(viable)}): EV>0, fill>10%")
print(f"\n  Total fills analyzed: {sum(t['fills'] for t in table):,}")
print(f"  Weighted avg EV: {np.average([t['ev'] for t in table], weights=[t['fills'] for t in table]):+.0f}/fill")
print(f"  Weighted avg adverse ratio: {np.average([t['adverse_ratio'] for t in table], weights=[t['fills'] for t in table]):.2f}")
print(f"{'═'*70}")

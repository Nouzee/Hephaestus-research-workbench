"""
EVL v1 — Execution Validation Layer

Replaces Bernoulli fill (P=0.30) with L2 queue simulation using 10-level orderbook.
No live trading — paper execution using existing data.

Per tick where CORE would quote:
  1. Enter hypothetical limit order at back of queue
  2. Track queue position via cumulative depth changes
  3. Fill triggered when: queue position = 0 AND trade occurs at our price
  4. Record markout at 10/50/100/500 ticks post-fill

Output: realized fill prob, queue decay, latency markout, adverse calibration.
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
FUTURE_TICKS = 20
OUR_SIZE = 100  # hypothetical order size (shares)

print("=" * 70)
print("  EVL v1 — L2 Queue Execution Validation")
print("=" * 70)


# ===========================================================================
# [1] Load data + freeze CORE calibration
# ===========================================================================
print("\n[1] Loading + CORE calibration ...")
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
        "mid": (ob_d["OfferPrice1"]+ob_d["BidPrice1"])/2.0,
        "sp": ob_d["OfferPrice1"]-ob_d["BidPrice1"],
        "bid_p": ob_d["BidPrice1"], "ask_p": ob_d["OfferPrice1"],
        "bid_q": ob_d["BidOrderQty1"], "ask_q": ob_d["OfferOrderQty1"],
        "dp": sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
              sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
        "direction": msg_d["Direction"].astype(np.float64),
        "size": msg_d["Size"].astype(np.float64),
        "price": msg_d["Price"].astype(np.float64),
        "valid": (ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0), "N": N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)
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
sorted_tox = np.sort(np.array(tox_tr))

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] L2 Queue Simulation Engine
# ===========================================================================
print(f"\n[2] Running L2 queue simulation on 10 days ...")

TOTAL_DAYS = 10  # subset for EVL

# Accumulators
per_state_quotes = defaultdict(int)       # quotes placed
per_state_fills = defaultdict(int)        # fills realized
per_state_bernoulli_fills = defaultdict(int)  # fills under old model
per_state_markout = defaultdict(list)     # markout after fill
per_state_queue_wait = defaultdict(list)  # wait time in queue

for d in range(TOTAL_DAYS):
    raw = all_raw[d]
    mid, sp, bid_p, ask_p = raw["mid"], raw["sp"], raw["bid_p"], raw["ask_p"]
    bid_q, ask_q, dp = raw["bid_q"], raw["ask_q"], raw["dp"]
    direction, size, price = raw["direction"], raw["size"], raw["price"]
    valid, N = raw["valid"], raw["N"]
    n_w = N // WINDOW_SIZE
    if n_w < 5: continue

    fs = 0 if d==0 else day_bounds[d-1]; fe = day_bounds[d]
    feats_z = np.clip((X_all[fs:fe]-X_tr_m)/X_tr_s, -10, 10)
    regs = km.predict(feats_z[:n_w])

    v_idx = np.where(valid)[0]; nv = len(v_idx)
    if nv < 100: continue

    win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1); reg_v = regs[win]
    tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
    tq_v = np.where(tox_v <= np.percentile(sorted_tox,30), 0,
             np.where(tox_v <= np.percentile(sorted_tox,70), 1, 2))
    tod_v = v_idx / N; tdb_v = np.where(tod_v<0.30,0,np.where(tod_v<0.70,1,2))

    # Only quote in CORE (high tox)
    quote_mask = tq_v == 2
    q_idx = v_idx[quote_mask]; nq = len(q_idx)
    reg_q = reg_v[quote_mask]; tdb_q = tdb_v[quote_mask]

    # Queue simulation state per tick
    bid_queue = 0.0    # our position at best bid (0 = front)
    ask_queue = 0.0    # our position at best ask

    for i in range(nq):
        t = q_idx[i]
        r = int(reg_q[i]); tod = int(tdb_q[i])
        sk = f"R{r}_q2_T{tod}"
        per_state_quotes[sk] += 1

        # Old Bernoulli model: what WOULD have happened
        if np.random.random() < 0.30:
            per_state_bernoulli_fills[sk] += 1

        # ── L2 Queue Model ──

        # Enter order at back of queue
        if bid_q[t] > 0:
            bid_pos_initial = bid_q[t] + OUR_SIZE  # behind existing depth
        else:
            bid_pos_initial = float('inf')

        if ask_q[t] > 0:
            ask_pos_initial = ask_q[t] + OUR_SIZE
        else:
            ask_pos_initial = float('inf')

        # Track forward: when does our order reach the front?
        # Walk forward through subsequent ticks, consuming depth
        side = 1 if np.random.random() > 0.5 else 0  # 0=bid, 1=ask
        pos_remaining = bid_pos_initial if side == 0 else ask_pos_initial
        fill_side_price = bid_p[t] if side == 0 else ask_p[t]

        filled = False; fill_t = -1
        lookahead = min(FUTURE_TICKS * 5, N - t - 1)  # look up to 100 ticks ahead

        for k in range(1, lookahead):
            tk = t + k
            if not valid[tk]: continue

            # Depth consumption by trades at our price level
            if side == 0:  # bid
                depth_ahead = bid_q[tk]
                # Trades at bid price consume bid depth
                if price[tk] == fill_side_price and direction[tk] < 0:
                    pos_remaining -= size[tk]  # someone sold at bid, consuming queue
                # Queue also shrinks from cancellations
                if bid_q[tk] < bid_q[t]:
                    pos_remaining -= (bid_q[t] - bid_q[tk])

                # Did we get filled?
                if pos_remaining <= 0 and price[tk] == fill_side_price and direction[tk] < 0:
                    filled = True; fill_t = tk; break
            else:  # ask
                depth_ahead = ask_q[tk]
                if price[tk] == fill_side_price and direction[tk] > 0:
                    pos_remaining -= size[tk]
                if ask_q[tk] < ask_q[t]:
                    pos_remaining -= (ask_q[t] - ask_q[tk])

                if pos_remaining <= 0 and price[tk] == fill_side_price and direction[tk] > 0:
                    filled = True; fill_t = tk; break

        if filled:
            per_state_fills[sk] += 1
            # Markout after fill
            mark_end = min(fill_t + 100, N - 1)
            if mark_end > fill_t:
                markout_bps = (mid[mark_end] - mid[fill_t]) / max(mid[fill_t], 1e-8) * 10000
                per_state_markout[sk].append(float(markout_bps) * (1 if side==0 else -1))
            per_state_queue_wait[sk].append(fill_t - t)

    if (d+1) % 3 == 0:
        print(f"  [{d+1}/{TOTAL_DAYS}] days, {sum(per_state_quotes.values()):,} quotes, "
              f"{sum(per_state_fills.values()):,} real fills")


# ===========================================================================
# [3] Execution Gap Report
# ===========================================================================
print(f"\n[3] Execution Gap — Bernoulli vs L2 Queue")
print(f"{'═'*70}")

print(f"\n  {'State':<14s} {'Quotes':>7s} {'Bern.Fills':>10s} {'Real Fills':>10s} "
      f"{'Bern.P(f)':>9s} {'Real P(f)':>9s} {'Gap':>8s} {'Markout(bps)':>13s} {'Wait(tk)':>8s}")
print(f"  {'─'*14} {'─'*7} {'─'*10} {'─'*10} {'─'*9} {'─'*9} {'─'*8} {'─'*13} {'─'*8}")

results = []
for sk in sorted(per_state_quotes.keys()):
    nq = per_state_quotes[sk]
    if nq < 30: continue
    bf = per_state_bernoulli_fills.get(sk, 0)
    rf = per_state_fills.get(sk, 0)
    bp = bf/nq; rp = rf/nq; gap = rp - bp
    mo = np.mean(per_state_markout.get(sk, [0])) if per_state_markout.get(sk) else 0
    wt = np.mean(per_state_queue_wait.get(sk, [0])) if per_state_queue_wait.get(sk) else 0
    gap_label = "OK" if abs(gap) < 0.10 else ("OVER" if gap > 0 else "UNDER")
    print(f"  {sk:<14s} {nq:>7,d} {bf:>10,d} {rf:>10,d} "
          f"{bp:>8.1%} {rp:>8.1%} {gap:>+7.1%} {mo:>+12.1f} {wt:>8.0f} {gap_label:>6s}")

    results.append({"state": sk, "quotes": nq, "bern_fill": bp, "real_fill": rp,
                    "markout_mean": mo, "wait_mean": wt})


# ===========================================================================
# [4] Execution Validation Summary
# ===========================================================================
print(f"\n[4] Execution Validation Summary")
print(f"{'═'*70}")

total_q = sum(per_state_quotes.values())
total_bf = sum(per_state_bernoulli_fills.values())
total_rf = sum(per_state_fills.values())

print(f"\n  Total quotes:      {total_q:,}")
print(f"  Bernoulli fills:   {total_bf:,}  ({total_bf/max(total_q,1)*100:.1f}%)")
print(f"  Real (queue) fills:{total_rf:,}  ({total_rf/max(total_q,1)*100:.1f}%)")
print(f"  Fill model gap:    {total_rf/max(total_bf,1):.2f}x")

avg_markout = np.mean([r["markout_mean"] for r in results]) if results else 0
avg_wait = np.mean([r["wait_mean"] for r in results]) if results else 0
print(f"\n  Avg markout:       {avg_markout:+.2f} bps")
print(f"  Avg queue wait:    {avg_wait:.0f} ticks")

# Verdict
gap_ratio = total_rf / max(total_bf, 1)
if 0.7 < gap_ratio < 1.3:
    verdict = "CASE_A — Bernoulli model close to queue reality"
elif 0.4 < gap_ratio < 2.0:
    verdict = "CASE_B — Partial gap, execution-aware CORE needed"
else:
    verdict = "CASE_C — Fill model is significantly mis-specified"

print(f"\n  {verdict}")
print(f"{'═'*70}")

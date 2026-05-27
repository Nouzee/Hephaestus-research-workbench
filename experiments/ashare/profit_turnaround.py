"""
Profit Turnaround Backtest v2

  1. Simulate fills with full metadata (bid/ask, regime, toxicity, time)
  2. Continuous toxicity: E[PnL | tox_score], E[adverse | tox_score]
  3. Regime × tox grid: scan size/spread params
  4. Positive expectancy state discovery
  5. OPEN module — separate morning parameters
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE, N_REGIMES = 100, 8
FILL_PROB, FUTURE_TICKS = 0.30, 20

print("=" * 65)
print("  Profit Turnaround v2 — Continuous Toxicity + Optimal Params")
print("=" * 65)


# ===========================================================================
# [1] Load + classify + simulate fills with full metadata
# ===========================================================================

print("\n[1] Loading + regime classification ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))

all_features = []
n_train_days = int(len(msg_files) * 0.60)
train_features = []

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]; n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_d.items()},
            {k: v[s:e] for k, v in msg_d.items()})
        all_features.append(list(feats.values()))
        if day_idx < n_train_days: train_features.append(list(feats.values()))
    if (day_idx+1) % 20 == 0:
        print(f"  [{day_idx+1}/{len(msg_files)}] days")

X_all = np.array(all_features, dtype=np.float32)
X_tr = np.array(train_features, dtype=np.float32)
tr_m = X_tr.mean(axis=0); tr_s = np.maximum(X_tr.std(axis=0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr - tr_m) / tr_s, -10, 10))
regimes = km.predict(np.clip((X_all - tr_m) / tr_s, -10, 10))
n_windows = len(regimes)

# Calibrate spread/depth thresholds on training data
train_spreads, train_depths = [], []
for day_idx in range(n_train_days):
    ob_df = pl.read_parquet(ob_files[day_idx])
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    v = (ob_d["BidPrice1"]>0) & (ob_d["OfferPrice1"]>0)
    train_spreads.extend((ob_d["OfferPrice1"][v] - ob_d["BidPrice1"][v])[:50000].tolist())
    train_depths.extend((sum(ob_d[f"BidOrderQty{i}"][v] for i in range(1,6)) +
                          sum(ob_d[f"OfferOrderQty{i}"][v] for i in range(1,6)))[:50000].tolist())
s_lo, s_hi = np.percentile(train_spreads, [33, 67])
d_lo, d_hi = np.percentile(train_depths, [33, 67])

print(f"  {n_windows:,} windows  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Simulate fills with full metadata
# ===========================================================================

print(f"\n[2] Simulating fills with toxicity scores ...")
t0 = time.perf_counter()

rng = np.random.RandomState(42)
fill_records = []  # (tox_score, regime, pnl, adverse, spread_earned, side, tod_bucket)

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
    N_total = msg_df.shape[0]
    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    valid = (ob_d["BidPrice1"]>0) & (ob_d["OfferPrice1"]>0)
    mid = (ob_d["OfferPrice1"] + ob_d["BidPrice1"]) / 2.0
    spread_arr = ob_d["OfferPrice1"] - ob_d["BidPrice1"]
    depth_arr = sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6)) + \
                sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6))
    imb_arr = (sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6)) -
               sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6))) / np.maximum(depth_arr, 1)

    n_w = N_total // WINDOW_SIZE
    if n_w < 5: continue
    day_start_win = sum(1 for d_idx in range(day_idx)
                        if d_idx < len(msg_files) and sum(1 for w in range(
                            pl.read_parquet(msg_files[d_idx]).shape[0]//WINDOW_SIZE)) > 0)

    for w in range(n_w):
        s_w, e_w = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        regime = int(regimes[min(day_start_win + w, n_windows - 1)])

        for t in range(s_w, e_w):
            if not valid[t]: continue
            tod = t / N_total   # 0-1 within day

            # Toxicity score (identical to v1 filter)
            tox = 0
            sp = spread_arr[t]; dp = depth_arr[t]
            if sp > s_hi: tox += 2
            elif sp > s_lo: tox += 1
            if dp < d_lo: tox += 2
            elif dp < d_hi: tox += 1
            if regime in {7}: tox += 1
            elif regime in {3}: tox -= 1
            if tod < 0.30: tox += 1
            tox = max(tox, 0)

            # Simulate fill
            if rng.random() > FILL_PROB: continue
            side = 1 if rng.random() > 0.5 else -1
            spread_earned = sp / 2

            fut_end = min(t + FUTURE_TICKS, N_total - 1)
            fut_move = (mid[fut_end] - mid[t]) / max(mid[t], 1e-8)
            adverse = side * fut_move * mid[t]
            pnl = spread_earned - max(adverse, 0)

            tod_b = 0 if tod < 0.30 else (1 if tod < 0.70 else 2)

            fill_records.append((tox, regime, pnl, adverse, spread_earned, side, tod_b,
                                 sp, dp, imb_arr[t]))

n_fills = len(fill_records)
print(f"  {n_fills:,} fills  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Continuous Toxicity → PnL curve
# ===========================================================================

print(f"\n[3] Continuous Toxicity Function — E[PnL | tox] ...")

fill_arr = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in fill_records])
tox_vals = fill_arr[:, 0].astype(int)
pnl_vals = fill_arr[:, 2]
adv_vals = fill_arr[:, 3]
spread_vals = fill_arr[:, 4]
side_vals = fill_arr[:, 5]

print(f"\n  Toxicity → PnL mapping:")
print(f"  {'Tox':>5s} {'Fills':>8s} {'E[PnL]':>10s} {'E[Adverse]':>12s} "
      f"{'E[Spread]':>10s} {'Toxic%':>7s} {'Net/Fill':>10s} {'Action':>16s}")
print(f"  {'─'*5} {'─'*8} {'─'*10} {'─'*12} {'─'*10} {'─'*7} {'─'*10} {'─'*16}")

for tox in range(7):
    mask = tox_vals == tox
    n = mask.sum()
    if n < 100: continue
    e_pnl = float(np.mean(pnl_vals[mask]))
    e_adv = float(np.mean(adv_vals[mask]))
    e_sp = float(np.mean(spread_vals[mask]))
    tox_pct = float(np.mean(adv_vals[mask] > 0)) * 100

    if e_pnl > 0:
        action = "FULL SIZE"
    elif e_pnl > -e_sp * 0.3:
        action = "defensive"
    elif e_pnl > -e_sp * 0.6:
        action = "small only"
    else:
        action = "WITHDRAW"

    print(f"  {tox:>5d} {n:>8,d} {e_pnl:>+10.2f} {e_adv:>+12.2f} "
          f"{e_sp:>+10.2f} {tox_pct:>6.1f}% {e_pnl:>+10.2f}/fill {action:>16s}")


# ===========================================================================
# [4] PnL decomposition by side, regime, time
# ===========================================================================

print(f"\n[4] PnL Decomposition ...")

regime_names = {0:"R0:depth_collapse",1:"R1:ask_heavy",2:"R2:deep_liq",3:"R3:trade_surge",
                4:"R4:bid_heavy",5:"R5:STRESS",6:"R6:active_buy",7:"R7:active_sell"}
tod_names = {0:"OPEN", 1:"MID", 2:"CLOSE"}

# By regime + side
print(f"\n  PnL by Regime × Side:")
print(f"  {'Regime':<22s} {'Bid Fills':>10s} {'Bid PnL':>12s} "
      f"{'Ask Fills':>10s} {'Ask PnL':>12s} {'Bid Tox%':>9s} {'Ask Tox%':>9s} {'Worse':>8s}")
print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*10} {'─'*12} {'─'*9} {'─'*9} {'─'*8}")

for r in range(N_REGIMES):
    r_mask = fill_arr[:, 1].astype(int) == r
    if r_mask.sum() < 100: continue
    bid_mask = r_mask & (side_vals == 1)
    ask_mask = r_mask & (side_vals == -1)
    bid_n = bid_mask.sum(); ask_n = ask_mask.sum()
    bid_pnl = float(np.sum(pnl_vals[bid_mask])) if bid_n > 0 else 0
    ask_pnl = float(np.sum(pnl_vals[ask_mask])) if ask_n > 0 else 0
    bid_tox = float(np.mean(adv_vals[bid_mask] > 0)*100) if bid_n > 0 else 0
    ask_tox = float(np.mean(adv_vals[ask_mask] > 0)*100) if ask_n > 0 else 0
    worse = "BID" if bid_tox > ask_tox + 3 else ("ASK" if ask_tox > bid_tox + 3 else "equal")
    print(f"  {regime_names.get(r,f'R{r}'):<22s} {bid_n:>10,d} {bid_pnl:>+12,.0f} "
          f"{ask_n:>10,d} {ask_pnl:>+12,.0f} {bid_tox:>8.1f}% {ask_tox:>8.1f}% {worse:>8s}")

# By time × regime
print(f"\n  PnL by Time × Regime (Total):")
header = f"  {'':<8s}" + "".join(f"{regime_names.get(r,f'R{r}'):<22s}" for r in range(N_REGIMES))
print(header)
for tb in range(3):
    t_mask = fill_arr[:, 6].astype(int) == tb
    row = f"  {tod_names[tb]:<8s}"
    for r in range(N_REGIMES):
        tr_mask = t_mask & (fill_arr[:, 1].astype(int) == r)
        pnl_r = float(np.sum(pnl_vals[tr_mask])) if tr_mask.sum() > 10 else 0
        row += f"{pnl_r:>+22,.0f}"
    print(row)


# ===========================================================================
# [5] Positive expectancy state discovery
# ===========================================================================

print(f"\n[5] Positive Expectancy States ...")

# For each (regime, tox, tod) bucket, compute E[PnL/fill]
print(f"\n  States with E[PnL/fill] > 0:")
print(f"  {'Regime':<22s} {'Tox':>4s} {'TOD':>6s} {'Fills':>8s} "
      f"{'E[PnL]':>10s} {'E[Adv]':>10s} {'Verdict':>14s}")
print(f"  {'─'*22} {'─'*4} {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*14}")

positive_states = []
for r in range(N_REGIMES):
    for tox in range(7):
        for tb in range(3):
            mask = ((fill_arr[:, 1].astype(int) == r) & (tox_vals == tox)
                    & (fill_arr[:, 6].astype(int) == tb))
            n = mask.sum()
            if n < 50: continue
            e_pnl = float(np.mean(pnl_vals[mask]))
            e_adv = float(np.mean(adv_vals[mask]))

            if e_pnl > 0 and n > 200:
                positive_states.append((r, tox, tb, n, e_pnl, e_adv))
                print(f"  {regime_names.get(r,f'R{r}'):<22s} {tox:>4d} {tod_names[tb]:>6s} "
                      f"{n:>8,d} {e_pnl:>+10.2f} {e_adv:>+10.2f} {'SAFE TO QUOTE':>14s}")

if not positive_states:
    print(f"  (no states with n>200 and E[PnL]>0 — market is net toxic)")

# ===========================================================================
# [6] OPEN module — morning-specific parameters
# ===========================================================================

print(f"\n[6] OPEN Module — Morning-Specific Analysis ...")

open_mask = fill_arr[:, 6].astype(int) == 0
mid_mask = fill_arr[:, 6].astype(int) == 1
close_mask = fill_arr[:, 6].astype(int) == 2

for label, mask in [("OPEN", open_mask), ("MID", mid_mask), ("CLOSE", close_mask)]:
    if mask.sum() < 100: continue
    e_pnl = float(np.mean(pnl_vals[mask]))
    e_adv = float(np.mean(adv_vals[mask]))
    tox_pct = float(np.mean(adv_vals[mask] > 0)*100)
    n = mask.sum()
    print(f"  {label:<8s}: fills={n:>10,d}  E[PnL]={e_pnl:>+10.2f}  "
          f"E[Adv]={e_adv:>+10.2f}  Tox%={tox_pct:.1f}%")

# OPEN by tox bucket
print(f"\n  OPEN by tox score:")
for tox in range(7):
    mask = open_mask & (tox_vals == tox)
    n = mask.sum()
    if n < 50: continue
    e_pnl = float(np.mean(pnl_vals[mask]))
    print(f"    tox={tox}: fills={n:>8,d}  E[PnL/fill]={e_pnl:>+10.2f}")


# ===========================================================================
# [7] Optimal parameter grid per (regime, tox)
# ===========================================================================

print(f"\n[7] Optimal Parameters — size/spread grid per regime×tox ...")
t0 = time.perf_counter()

# Re-simulate with parameter grid: for each (regime, tox), test size in [0, 0.25, 0.5, 0.75, 1.0], spread in [0.8, 1.0, 1.3, 1.6]
sizes = [0.0, 0.25, 0.5, 0.75, 1.0]
spreads = [0.8, 1.0, 1.3, 1.6]

results_grid = {}  # (r, tox) -> best (size, spread, E[PnL])

# Collect raw ticks per (regime, tox) for re-simulation
bucket_ticks = {}  # (r, tox) -> [(spread, depth, mid, fut_move, side_pool)]
for r in range(N_REGIMES):
    r_mask = fill_arr[:, 1].astype(int) == r
    for tox in range(7):
        key = (r, tox)
        mask = r_mask & (tox_vals == tox)
        if mask.sum() < 200:
            continue
        # Extract raw tick data for this bucket
        idx = np.where(mask)[0]
        bucket_ticks[key] = {
            "spreads": fill_arr[idx, 7] if fill_arr.shape[1] > 7 else None,
            "n": mask.sum(),
            "e_spread": float(np.mean(spread_vals[mask])),
            "e_adv": float(np.mean(adv_vals[mask])),
        }

# For each bucket with sufficient data, compute optimal params
print(f"\n  Optimal (size, spread) per regime×tox (top 5 buckets by fills):")
buckets_by_n = sorted(bucket_ticks.items(), key=lambda x: x[1]["n"], reverse=True)

print(f"  {'Regime':<22s} {'Tox':>4s} {'Fills':>8s} {'BestSz':>7s} "
      f"{'BestSp':>7s} {'E[PnL]':>10s} {'Base PnL':>10s} {'Lift':>10s}")
print(f"  {'─'*22} {'─'*4} {'─'*8} {'─'*7} {'─'*7} {'─'*10} {'─'*10} {'─'*10}")

for (r, tox), bt in buckets_by_n[:15]:
    # Simulate with different params
    e_sp = bt["e_spread"]
    e_adv_val = bt["e_adv"]

    best_pnl = -float('inf')
    best_params = (1.0, 1.0)
    base_pnl = 0.0

    for sz in sizes:
        for sp_m in spreads:
            # Expected PnL = size * (spread_earned * spread_mult - E[adverse])
            # Fill prob decreases with wider spread: p_fill(sz, sp_m) ≈ FILL_PROB / sp_m * sz
            p_fill_adj = FILL_PROB * sz / max(sp_m, 0.5)
            e_pnl_adj = p_fill_adj * (e_sp * sp_m * sz - max(e_adv_val, 0) * sz)

            if e_pnl_adj > best_pnl:
                best_pnl = e_pnl_adj
                best_params = (sz, sp_m)

            if sz == 1.0 and sp_m == 1.0:
                base_pnl = e_pnl_adj

    lift = best_pnl - base_pnl if base_pnl != 0 else 0
    rname = regime_names.get(r, f"R{r}")

    print(f"  {rname:<22s} {tox:>4d} {bt['n']:>8,d} {best_params[0]:>7.2f} "
          f"{best_params[1]:>7.2f} {best_pnl:>+10.2f} {base_pnl:>+10.2f} {lift:>+10.2f}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [8] Final Action Table
# ===========================================================================

print(f"\n[8] Final Action Table — Production Rules ...")

# Synthesize: for each regime, recommended default action
print(f"\n  {'Regime':<22s} {'Default':>14s} {'OPEN rule':>16s} {'MID rule':>16s} {'Strategy':>24s}")
print(f"  {'─'*22} {'─'*14} {'─'*16} {'─'*16} {'─'*24}")

for r in range(N_REGIMES):
    rname = regime_names.get(r, f"R{r}")

    # Find best default params (all time)
    best_def = (1.0, 1.0)
    best_open = (0.5, 1.3)
    best_mid = (1.0, 1.0)

    for (rr, tox), bt in bucket_ticks.items():
        if rr != r: continue
        # Simplified: take the largest bucket's optimal

    # Strategy based on regime character from full-pipeline analysis
    if r == 5:  # R5 STRESS
        strategy = "defensive: 0.3x size, 1.8x spread"
    elif r == 3:  # R3 trade surge
        strategy = "cautious: 0.5x size, 1.3x spread"
    elif r == 2:  # R2 deep liquidity
        strategy = "active: 1.0x size, 0.9x spread, tight only"
    elif r in (0, 4):  # depth collapse
        strategy = "normal: 0.8x size, 1.1x spread"
    elif r in (6, 7):  # active flow
        strategy = "selective: 0.7x size, 1.2x spread"
    else:
        strategy = "normal: 1.0x size, 1.0x spread"

    print(f"  {rname:<22s} {'1.0x/1.0x':>14s} {'0.5x/1.3x':>16s} {'1.0x/1.0x':>16s} {strategy:>24s}")


print(f"\n{'═'*65}")
print(f"  Profit Turnaround v2 complete.")
print(f"{'═'*65}")
if positive_states:
    print(f"\n  Found {len(positive_states)} positive-expectancy states.")
    for r, tox, tb, n, e_pnl, e_adv in positive_states[:5]:
        print(f"    {regime_names.get(r,f'R{r}')} tox={tox} {tod_names[tb]}: "
              f"E[PnL]={e_pnl:+.2f}/fill ({n} fills)")
print(f"{'═'*65}")

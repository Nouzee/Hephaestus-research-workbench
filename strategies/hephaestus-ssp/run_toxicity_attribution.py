"""
Toxicity Attribution — PnL Decomposition + Toxic Fill Map

Steps:
  1. Simulate tick-by-tick fills on 59-day dataset
  2. Tag every fill: regime, spread bucket, depth bucket, time-of-day
  3. Compute future adverse move → toxic/clean classification
  4. PnL attribution by regime / time / side
  5. Toxicity heatmap: P(toxic | regime, spread, depth, imbalance)
  6. Toxicity-aware backtest vs baseline
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100
N_REGIMES = 8
FILL_PROB = 0.30
FUTURE_TICKS = 20  # ticks to look ahead for adverse move

print("=" * 65)
print("  Toxicity Attribution — PnL Decomposition + Toxic Fill Map")
print("=" * 65)


# ===========================================================================
# [1] Load + classify regimes
# ===========================================================================

print("\n[1] Loading data + regime classification ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))

all_features = []
all_window_meta = []  # (day_idx, start_tick_idx)

n_train_days = int(len(msg_files) * 0.60)
train_features = []

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf)
    ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]
    n_w = N // WINDOW_SIZE
    if n_w < 5:
        continue

    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}

    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_d.items()},
            {k: v[s:e] for k, v in msg_d.items()},
        )
        all_features.append(list(feats.values()))
        all_window_meta.append((day_idx, s))
        if day_idx < n_train_days:
            train_features.append(list(feats.values()))

    if (day_idx+1) % 15 == 0:
        print(f"  [{day_idx+1}/{len(msg_files)}] days")

X_all = np.array(all_features, dtype=np.float32)
X_tr = np.array(train_features, dtype=np.float32)
tr_mean = X_tr.mean(axis=0)
tr_std = np.maximum(X_tr.std(axis=0), 1e-8)

km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr - tr_mean) / tr_std, -10, 10))
regimes = km.predict(np.clip((X_all - tr_mean) / tr_std, -10, 10))

n_windows = len(regimes)
print(f"  {n_windows:,} windows  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Simulate fills + tag every fill with regime/features/outcome
# ===========================================================================

print(f"\n[2] Simulating fills + computing toxicity per fill ...")
t0 = time.perf_counter()

rng = np.random.RandomState(42)

# Per-fill records
fill_records = []  # list of dicts

# PnL accumulators by regime
regime_pnl_spread = np.zeros(N_REGIMES, dtype=np.float64)
regime_pnl_adverse = np.zeros(N_REGIMES, dtype=np.float64)
regime_fills = np.zeros(N_REGIMES, dtype=np.int32)
regime_toxic = np.zeros(N_REGIMES, dtype=np.int32)

# Time-of-day buckets: 0=open(0-30%), 1=mid(30-70%), 2=close(70-100%)
time_pnl_spread = np.zeros(3, dtype=np.float64)
time_pnl_adverse = np.zeros(3, dtype=np.float64)
time_fills = np.zeros(3, dtype=np.int32)
time_toxic = np.zeros(3, dtype=np.int32)

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf)
    ob_df = pl.read_parquet(of)
    N_total = msg_df.shape[0]

    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}

    mid_arr = (ob_d["OfferPrice1"] + ob_d["BidPrice1"]) / 2.0
    spread_arr = ob_d["OfferPrice1"] - ob_d["BidPrice1"]
    # Filter: invalid ticks where bid or ask is missing (price=0)
    valid_ticks = (ob_d["BidPrice1"] > 0) & (ob_d["OfferPrice1"] > 0)
    # Only use valid ticks for window stats
    valid_mid = mid_arr[valid_ticks]
    valid_spread = spread_arr[valid_ticks]
    direction = msg_d["Direction"].astype(np.float64)
    size_arr = msg_d["Size"].astype(np.float64)

    n_w = N_total // WINDOW_SIZE
    if n_w < 5:
        continue

    # Find this day's regime indices
    day_start_win = sum(1 for d_idx, _ in all_window_meta if d_idx < day_idx)
    day_end_win = day_start_win + n_w

    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        regime = int(regimes[day_start_win + w])

        # Time-of-day (fraction within this day)
        tod_frac = s / N_total
        if tod_frac < 0.30:
            tod_bucket = 0
        elif tod_frac < 0.70:
            tod_bucket = 1
        else:
            tod_bucket = 2

        # Current market conditions for this window (use valid ticks only)
        w_valid = valid_ticks[s:e]
        w_mid = mid_arr[s:e][w_valid] if w_valid.sum() > 5 else mid_arr[s:e]
        w_spread = spread_arr[s:e][w_valid] if w_valid.sum() > 5 else spread_arr[s:e]
        avg_spread = float(np.mean(w_spread)) if len(w_spread) > 0 else 0.0
        avg_depth = float(np.mean(
            sum(ob_d[f"BidOrderQty{i}"][s:e] for i in range(1, 6)) +
            sum(ob_d[f"OfferOrderQty{i}"][s:e] for i in range(1, 6))
        ))
        avg_imb = float(np.mean(
            (sum(ob_d[f"BidOrderQty{i}"][s:e] for i in range(1, 6)) -
             sum(ob_d[f"OfferOrderQty{i}"][s:e] for i in range(1, 6))) /
            max(avg_depth, 1e-8)
        ))

        for t in range(s, e):
            # Skip invalid ticks (missing bid/ask)
            if not valid_ticks[t]:
                continue
            # Random fill at this tick
            if rng.random() > FILL_PROB:
                continue

            side = 1 if rng.random() > 0.5 else -1  # 1=bid fill, -1=ask fill
            fill_px = mid_arr[t] - side * spread_arr[t] / 2

            # Future adverse move
            future_end = min(t + FUTURE_TICKS, N_total - 1)
            future_move = (mid_arr[future_end] - mid_arr[t]) / max(mid_arr[t], 1e-8)
            adverse_cost = side * future_move * mid_arr[t]  # + = adverse (lost)
            is_toxic = adverse_cost > 0  # price moved against position

            # Spread earned
            spread_earned = spread_arr[t] / 2

            # Record
            fill_records.append({
                "regime": regime,
                "tod_bucket": tod_bucket,
                "spread": avg_spread,
                "depth": avg_depth,
                "imbalance": avg_imb,
                "side": side,
                "spread_earned": spread_earned,
                "adverse_cost": adverse_cost,
                "is_toxic": is_toxic,
                "day_idx": day_idx,
            })

            # Accumulate
            regime_pnl_spread[regime] += spread_earned
            regime_pnl_adverse[regime] -= adverse_cost  # negative = loss
            regime_fills[regime] += 1
            if is_toxic:
                regime_toxic[regime] += 1

            time_pnl_spread[tod_bucket] += spread_earned
            time_pnl_adverse[tod_bucket] -= adverse_cost
            time_fills[tod_bucket] += 1
            if is_toxic:
                time_toxic[tod_bucket] += 1

n_fills = len(fill_records)
print(f"  {n_fills:,} fills simulated  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] PnL Attribution Report
# ===========================================================================

print(f"\n[3] PnL Attribution ...")

print(f"\n  {'═'*60}")
print(f"  TOTAL PnL DECOMPOSITION")
print(f"  {'═'*60}")

total_spread = float(np.sum(regime_pnl_spread))
total_adverse = float(np.sum(regime_pnl_adverse))
total_pnl = total_spread + total_adverse

print(f"  Spread capture:      {total_spread:>+15,.0f}  ({total_spread/abs(total_pnl)*100:+.0f}%)")
print(f"  Adverse selection:   {total_adverse:>+15,.0f}  ({total_adverse/abs(total_pnl)*100:+.0f}%)")
print(f"  NET PnL:             {total_pnl:>+15,.0f}")
print(f"  Toxic fill rate:     {np.sum(regime_toxic)/max(n_fills,1)*100:.1f}%")

print(f"\n  {'═'*60}")
print(f"  PNL BY REGIME")
print(f"  {'═'*60}")

regime_names = {
    0: "R0:depth_collapse", 1: "R1:ask_heavy", 2: "R2:deep_liquidity",
    3: "R3:trade_surge", 4: "R4:bid_heavy", 5: "R5:STRESS",
    6: "R6:active_buy", 7: "R7:active_sell",
}

print(f"  {'Regime':<22s} {'Fills':>8s} {'Spread':>12s} {'Adverse':>12s} "
      f"{'Net':>12s} {'Toxic%':>8s} {'Net/Fill':>10s} {'Verdict':>12s}")
print(f"  {'─'*22} {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*10} {'─'*12}")

for r in range(N_REGIMES):
    nf = regime_fills[r]
    if nf == 0:
        continue
    sp = regime_pnl_spread[r]
    adv = regime_pnl_adverse[r]
    net = sp + adv
    tox_pct = regime_toxic[r] / nf * 100
    net_per = net / nf

    if net < 0 and tox_pct > 50:
        verdict = "TOXIC"
    elif net < 0:
        verdict = "negative"
    elif tox_pct < 40:
        verdict = "CLEAN"
    else:
        verdict = "mixed"

    print(f"  {regime_names.get(r,f'R{r}'):<22s} {nf:>8,d} {sp:>+12,.0f} {adv:>+12,.0f} "
          f"{net:>+12,.0f} {tox_pct:>7.1f}% {net_per:>+10.1f} {verdict:>12s}")

print(f"\n  {'═'*60}")
print(f"  PNL BY TIME OF DAY")
print(f"  {'═'*60}")

tod_names = {0: "OPEN (0-30%)", 1: "MID (30-70%)", 2: "CLOSE (70-100%)"}
for tb in range(3):
    nf = time_fills[tb]
    if nf == 0:
        continue
    sp = time_pnl_spread[tb]
    adv = time_pnl_adverse[tb]
    net = sp + adv
    tox_pct = time_toxic[tb] / nf * 100
    print(f"  {tod_names[tb]:<22s} {nf:>8,d} {sp:>+12,.0f} {adv:>+12,.0f} "
          f"{net:>+12,.0f} {tox_pct:>7.1f}% {net/nf:>+10.1f}/fill")


# ===========================================================================
# [4] Toxic Fill Map — P(toxic | condition)
# ===========================================================================

print(f"\n[4] Toxic Fill Map — P(toxic | condition) ...")

fill_arr = np.array([(r["is_toxic"], r["regime"], r["spread"], r["depth"],
                       r["imbalance"], r["tod_bucket"])
                      for r in fill_records])
is_toxic_arr = fill_arr[:, 0].astype(bool)

def toxicity_rate(condition_mask):
    if condition_mask.sum() < 20:
        return None
    return float(np.mean(is_toxic_arr[condition_mask]))

# By regime
print(f"\n  P(toxic | regime):")
for r in range(N_REGIMES):
    mask = fill_arr[:, 1] == r
    rate = toxicity_rate(mask)
    if rate is not None:
        bar = "T" * int(rate * 30) + "." * (30 - int(rate * 30))
        print(f"    {regime_names.get(r, f'R{r}'):<22s}: {rate:.1%}  {bar}")

# By spread bucket
spread_vals = fill_arr[:, 2]
spread_lo = np.percentile(spread_vals, 33)
spread_hi = np.percentile(spread_vals, 67)
print(f"\n  P(toxic | spread bucket):")
for label, mask in [("LOW spread", spread_vals < spread_lo),
                     ("MID spread", (spread_vals >= spread_lo) & (spread_vals < spread_hi)),
                     ("HIGH spread", spread_vals >= spread_hi)]:
    rate = toxicity_rate(mask)
    if rate is not None:
        print(f"    {label:<22s}: {rate:.1%}")

# By depth bucket
depth_vals = fill_arr[:, 3]
depth_lo = np.percentile(depth_vals, 33)
depth_hi = np.percentile(depth_vals, 67)
print(f"\n  P(toxic | depth bucket):")
for label, mask in [("LOW depth", depth_vals < depth_lo),
                     ("MID depth", (depth_vals >= depth_lo) & (depth_vals < depth_hi)),
                     ("HIGH depth", depth_vals >= depth_hi)]:
    rate = toxicity_rate(mask)
    if rate is not None:
        print(f"    {label:<22s}: {rate:.1%}")

# Key question: R2 deep liquidity — is it "thick but toxic"?
r2_mask = fill_arr[:, 1] == 2
r2_depth = fill_arr[r2_mask, 3] if r2_mask.sum() > 0 else np.array([])
if len(r2_depth) > 50:
    r2_depth_med = np.median(r2_depth)
    r2_thin = r2_mask & (fill_arr[:, 3] < r2_depth_med)
    r2_thick = r2_mask & (fill_arr[:, 3] >= r2_depth_med)
    r2_thin_rate = toxicity_rate(r2_thin)
    r2_thick_rate = toxicity_rate(r2_thick)
    if r2_thin_rate and r2_thick_rate:
        print(f"\n  R2 depth effect: thin={r2_thin_rate:.1%}  thick={r2_thick_rate:.1%}  "
              f"{'THICK = MORE TOXIC' if r2_thick_rate > r2_thin_rate * 1.1 else 'no depth effect'}")

# R5 stress: toxic rate?
r5_mask = fill_arr[:, 1] == 5
r5_rate = toxicity_rate(r5_mask)
if r5_rate is not None:
    print(f"\n  R5 STRESS toxic rate: {r5_rate:.1%}  "
          f"{'EXTREMELY TOXIC' if r5_rate > 0.55 else 'toxic' if r5_rate > 0.5 else 'elevated'}")

print(f"\n{'═'*65}")
print(f"  Attribution complete.")
print(f"{'═'*65}")

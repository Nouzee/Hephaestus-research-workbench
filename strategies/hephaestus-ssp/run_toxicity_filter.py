"""
Toxicity Filter Backtest

Builds a per-tick toxicity score from:
  - spread bucket (HIGH=33% toxic, LOW=9%)
  - depth bucket (LOW=28%, HIGH=17%)
  - regime (R7=33%, R3=13%, R5=15%)
  - time of day (OPEN=27%, CLOSE=20%)

Toxicity-aware strategy:
  LOW tox  → normal quote (size=1.0x, spread=1.0x)
  MID tox  → defensive (size=0.5x, spread=1.3x)
  HIGH tox → withdraw (size=0.0x)

Compares: BASELINE (always quote) vs TOXICITY-FILTERED
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100
N_REGIMES = 8
FUTURE_TICKS = 20
FILL_PROB = 0.30

print("=" * 65)
print("  Toxicity Filter Backtest")
print("=" * 65)


# ===========================================================================
# [1] Load + regime classify
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
    msg_df = pl.read_parquet(mf)
    ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]
    n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_d.items()},
            {k: v[s:e] for k, v in msg_d.items()})
        all_features.append(list(feats.values()))
        if day_idx < n_train_days:
            train_features.append(list(feats.values()))
    if (day_idx+1) % 20 == 0:
        print(f"  [{day_idx+1}/{len(msg_files)}] days")

X_all = np.array(all_features, dtype=np.float32)
X_tr = np.array(train_features, dtype=np.float32)
tr_mean = X_tr.mean(axis=0); tr_std = np.maximum(X_tr.std(axis=0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr - tr_mean) / tr_std, -10, 10))
regimes = km.predict(np.clip((X_all - tr_mean) / tr_std, -10, 10))

n_windows = len(regimes)
print(f"  {n_windows:,} windows  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Toxicity model: compute per-window toxicity calibration
# ===========================================================================

print(f"\n[2] Building toxicity score model ...")
t0 = time.perf_counter()

# Collect per-window stats for toxicity calibration (on TRAIN only)
train_windows = sum(1 for i in range(n_windows)
                    if i < int(n_train_days / len(msg_files) * n_windows))

spread_vals = []
depth_vals = []
for day_idx, (mf, of) in enumerate(zip(msg_files[:n_train_days], ob_files[:n_train_days])):
    msg_df = pl.read_parquet(mf)
    ob_df = pl.read_parquet(of)
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    valid = (ob_d["BidPrice1"] > 0) & (ob_d["OfferPrice1"] > 0)
    spread_arr = ob_d["OfferPrice1"][valid] - ob_d["BidPrice1"][valid]
    depth_arr = sum(ob_d[f"BidOrderQty{i}"][valid] for i in range(1, 6)) + \
                sum(ob_d[f"OfferOrderQty{i}"][valid] for i in range(1, 6))
    spread_vals.extend(spread_arr[:100000].tolist())  # sample
    depth_vals.extend(depth_arr[:100000].tolist())

spread_lo = np.percentile(spread_vals, 33)
spread_hi = np.percentile(spread_vals, 67)
depth_lo = np.percentile(depth_vals, 33)
depth_hi = np.percentile(depth_vals, 67)

print(f"  Spread thresholds: LOW<{spread_lo:.0f}  MID  HIGH>{spread_hi:.0f}")
print(f"  Depth thresholds:  LOW<{depth_lo:.0f}  MID  HIGH>{depth_hi:.0f}")

# Toxicity score components (from empirical findings):
# spread: HIGH=33% tox, LOW=9%  → score +2/+1/+0
# depth:  LOW=28% tox, HIGH=17%  → score +2/+1/+0
# regime: R7=33%, R3=13%         → score +1/+0/-1
# time:   OPEN=27%, CLOSE=20%     → score +1/+0

regime_tox_bias = {0:0, 1:0, 2:0, 3:-1, 4:0, 5:0, 6:0, 7:+1}  # R3 safe, R7 toxic

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Backtest: Baseline vs Toxicity-Filtered
# ===========================================================================

print(f"\n[3] Running backtest — Baseline vs Toxicity-Filtered ...")
t0 = time.perf_counter()

rng = np.random.RandomState(42)

def compute_toxicity_score(spread_val, depth_val, regime, tod_frac):
    """Per-tick toxicity score: 0(low) to 6(high). Only past info used."""
    score = 0
    if spread_val > spread_hi: score += 2
    elif spread_val > spread_lo: score += 1
    if depth_val < depth_lo: score += 2
    elif depth_val < depth_hi: score += 1
    score += regime_tox_bias.get(regime, 0)
    if tod_frac < 0.30: score += 1  # OPEN more toxic
    return score

def run_backtest(toxicity_aware):
    """Tick-by-tick MM simulation. Returns per-window PnL, per-regime PnL."""
    pnl_total = 0.0
    per_regime_pnl = {r: 0.0 for r in range(N_REGIMES)}
    per_regime_ticks = {r: 0 for r in range(N_REGIMES)}
    per_tod_pnl = {0:0.0, 1:0.0, 2:0.0}

    results = []

    for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
        msg_df = pl.read_parquet(mf)
        ob_df = pl.read_parquet(of)
        N_total = msg_df.shape[0]
        msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
        ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}

        valid_ticks = (ob_d["BidPrice1"] > 0) & (ob_d["OfferPrice1"] > 0)
        mid_arr = (ob_d["OfferPrice1"] + ob_d["BidPrice1"]) / 2.0
        spread_arr = ob_d["OfferPrice1"] - ob_d["BidPrice1"]
        depth_arr = sum(ob_d[f"BidOrderQty{i}"] for i in range(1, 6)) + \
                    sum(ob_d[f"OfferOrderQty{i}"] for i in range(1, 6))

        n_w = N_total // WINDOW_SIZE
        if n_w < 5: continue

        day_start_win = sum(1 for d_idx in range(day_idx)
                           if d_idx < len(msg_files))  # approximate

        for w in range(n_w):
            s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            regime = int(regimes[min(day_start_win + w, n_windows-1)])

            window_pnl = 0.0
            n_quoted = 0
            n_skipped = 0

            for t in range(s, e):
                if not valid_ticks[t]: continue
                tod_frac = t / N_total

                # Toxicity score
                tox_score = compute_toxicity_score(
                    spread_arr[t], depth_arr[t], regime, tod_frac)

                if toxicity_aware:
                    if tox_score >= 5:
                        # HIGH tox — withdraw entirely
                        n_skipped += 1
                        continue
                    elif tox_score >= 3:
                        sz_m, sp_m = 0.5, 1.3  # defensive
                    else:
                        sz_m, sp_m = 1.0, 1.0  # normal
                else:
                    sz_m, sp_m = 1.0, 1.0

                # Random fill
                if rng.random() > FILL_PROB / max(sp_m, 0.5):
                    continue

                side = 1 if rng.random() > 0.5 else -1
                spread_earned = spread_arr[t] * sp_m / 2 * sz_m

                # Adverse: future move
                fut_end = min(t + FUTURE_TICKS, N_total - 1)
                fut_move = (mid_arr[fut_end] - mid_arr[t]) / max(mid_arr[t], 1e-8)
                adverse = side * fut_move * mid_arr[t] * sz_m

                pnl = spread_earned - max(adverse, 0)
                window_pnl += pnl
                n_quoted += 1

            pnl_total += window_pnl
            per_regime_pnl[regime] += window_pnl
            per_regime_ticks[regime] += (e - s)
            tod_bucket = 0 if (s/N_total)<0.30 else (1 if (s/N_total)<0.70 else 2)
            per_tod_pnl[tod_bucket] += window_pnl

            results.append(window_pnl)

    return {
        "total_pnl": pnl_total,
        "per_window": np.array(results),
        "per_regime_pnl": per_regime_pnl,
        "per_regime_ticks": per_regime_ticks,
        "per_tod_pnl": per_tod_pnl,
    }

# Run both
bl = run_backtest(False)
tx = run_backtest(True)
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Comparison
# ===========================================================================

print(f"\n[4] Results — Baseline vs Toxicity-Filtered")

pnl_arr_bl = bl["per_window"]
pnl_arr_tx = tx["per_window"]

bl_total = float(np.sum(pnl_arr_bl))
tx_total = float(np.sum(pnl_arr_tx))
bl_dd = float(np.max(np.maximum.accumulate(np.cumsum(pnl_arr_bl)) - np.cumsum(pnl_arr_bl)))
tx_dd = float(np.max(np.maximum.accumulate(np.cumsum(pnl_arr_tx)) - np.cumsum(pnl_arr_tx)))
bl_mean = float(np.mean(pnl_arr_bl)); tx_mean = float(np.mean(pnl_arr_tx))
bl_std = float(np.std(pnl_arr_bl)); tx_std = float(np.std(pnl_arr_tx))
bl_sharpe = bl_mean / max(bl_std, 1e-8) * np.sqrt(len(pnl_arr_bl))
tx_sharpe = tx_mean / max(tx_std, 1e-8) * np.sqrt(len(pnl_arr_tx))

print(f"\n  {'Metric':<25s} {'Baseline':>14s} {'Tox-Filtered':>14s} {'Delta':>14s}")
print(f"  {'─'*25} {'─'*14} {'─'*14} {'─'*14}")
metrics = [
    ("Total PnL", bl_total, tx_total),
    ("Max Drawdown", bl_dd, tx_dd),
    ("Mean PnL/window", bl_mean, tx_mean),
    ("Std PnL/window", bl_std, tx_std),
    ("Sharpe", bl_sharpe, tx_sharpe),
]
for name, b, t in metrics:
    d = t - b
    better = "TOX" if (d > 0 and "Drawdown" not in name) or (d < 0 and "Drawdown" in name) else "BL"
    if abs(b) > 1e4:
        print(f"  {name:<25s} {b:>+14,.0f} {t:>+14,.0f} {d:>+14,.0f} {better:>5s}")
    elif abs(b) > 10:
        print(f"  {name:<25s} {b:>14.2f} {t:>14.2f} {d:>+14.2f} {better:>5s}")
    else:
        print(f"  {name:<25s} {b:>14.6f} {t:>14.6f} {d:>+14.6f} {better:>5s}")

# Per-regime
print(f"\n  PnL by Regime:")
print(f"  {'Regime':<22s} {'Baseline':>14s} {'Tox-Filter':>14s} {'Delta':>14s} {'Improve':>10s}")
print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
regime_names = {0:"R0:depth_collapse",1:"R1:ask_heavy",2:"R2:deep_liq",3:"R3:trade_surge",
                4:"R4:bid_heavy",5:"R5:STRESS",6:"R6:active_buy",7:"R7:active_sell"}
for r in range(N_REGIMES):
    b = bl["per_regime_pnl"].get(r, 0)
    t = tx["per_regime_pnl"].get(r, 0)
    d = t - b
    impr = (d / max(abs(b), 1e-12) * 100) if abs(b) > 0 else 0
    print(f"  {regime_names.get(r,f'R{r}'):<22s} {b:>+14,.0f} {t:>+14,.0f} "
          f"{d:>+14,.0f} {impr:>+9.1f}%")

# Per time-of-day
print(f"\n  PnL by Time of Day:")
tod_names = {0:"OPEN", 1:"MID", 2:"CLOSE"}
for tb in range(3):
    b = bl["per_tod_pnl"].get(tb, 0)
    t = tx["per_tod_pnl"].get(tb, 0)
    d = t - b
    impr = (d / max(abs(b), 1e-12) * 100) if abs(b) > 0 else 0
    print(f"  {tod_names[tb]:<10s}: BL={b:>+14,.0f}  TX={t:>+14,.0f}  "
          f"Δ={d:>+14,.0f}  ({impr:+.1f}%)")

# Test-period only
test_start = int(len(pnl_arr_bl) * 0.60)
test_bl = float(np.sum(pnl_arr_bl[test_start:]))
test_tx = float(np.sum(pnl_arr_tx[test_start:]))
print(f"\n  Test Period (last 40%): BL={test_bl:>+14,.0f}  "
      f"TX={test_tx:>+14,.0f}  Δ={test_tx-test_bl:>+14,.0f}")

print(f"\n{'═'*65}")
print(f"  Toxicity Filter complete.")
print(f"{'═'*65}")

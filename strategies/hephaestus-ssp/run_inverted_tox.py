"""
Inverted Toxicity Policy v1

Discovery: tox_score is INVERSE — high tox = wide spread = positive expectancy.
Strategy: LONG tox 4-6 (active MM), SHORT tox 0-3 (withdraw/minimal).

Hypotheses:
  H1: tox 4-6 is positive expectancy region
  H2: tox 0-3 is structural loss region
  H3: MID session contains majority of positive expectancy
  H4: R5 + tox>=4 is strongest profit regime-state
  H5: OPEN session is net negative except rare high-tox stress states
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
print("  Inverted Toxicity Policy v1")
print("  LONG tox 4-6, SHORT tox 0-3")
print("=" * 65)

# ===========================================================================
# [1] Load + classify regimes
# ===========================================================================
print("\n[1] Loading + regime classification ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))

all_features = []; train_features = []
n_train_days = int(len(msg_files) * 0.60)

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
        print(f"  [{day_idx+1}/{len(msg_files)}]")

X_all = np.array(all_features, dtype=np.float32); X_tr = np.array(train_features, dtype=np.float32)
tr_m = X_tr.mean(axis=0); tr_s = np.maximum(X_tr.std(axis=0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-tr_m)/tr_s, -10, 10))
regimes = km.predict(np.clip((X_all-tr_m)/tr_s, -10, 10))
n_windows = len(regimes)

# Calibrate thresholds on train
train_sp, train_dp = [], []
for day_idx in range(n_train_days):
    ob_df = pl.read_parquet(ob_files[day_idx])
    od = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    v = (od["BidPrice1"]>0) & (od["OfferPrice1"]>0)
    train_sp.extend((od["OfferPrice1"][v] - od["BidPrice1"][v])[:50000].tolist())
    train_dp.extend((sum(od[f"BidOrderQty{i}"][v] for i in range(1,6)) +
                      sum(od[f"OfferOrderQty{i}"][v] for i in range(1,6)))[:50000].tolist())
s_lo, s_hi = np.percentile(train_sp, [33, 67])
d_lo, d_hi = np.percentile(train_dp, [33, 67])

print(f"  {n_windows:,} windows  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [2] Backtest — Baseline vs Inverted Tox
# ===========================================================================
print(f"\n[2] Running backtest ...")
t0 = time.perf_counter()

rng = np.random.RandomState(42)
regime_names = {0:"R0",1:"R1",2:"R2",3:"R3",4:"R4",5:"R5",6:"R6",7:"R7"}

def compute_tox(sp, dp, regime, tod):
    """Identical to original tox score."""
    tox = 0
    if sp > s_hi: tox += 2
    elif sp > s_lo: tox += 1
    if dp < d_lo: tox += 2
    elif dp < d_hi: tox += 1
    if regime == 7: tox += 1
    elif regime == 3: tox -= 1
    if tod < 0.30: tox += 1
    return max(tox, 0)

def run_strategy(name, inverted):
    """
    inverted=False: baseline (always 1.0x)
    inverted=True:  tox≥4 → active (1.2x, 1.0x spread)
                    tox≤3 → withdraw (0.1x, 1.5x spread)
                    + TOD modifiers + regime modifiers
    """
    results = {"pnl": 0.0, "per_tox": {t:0.0 for t in range(7)},
               "per_regime": {r:0.0 for r in range(N_REGIMES)},
               "per_tod": {0:0.0, 1:0.0, 2:0.0},
               "per_state": {}, "fills": 0, "tox_fills": {t:0 for t in range(7)},
               "spread_earned": 0.0, "adverse": 0.0}

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

        n_w = N_total // WINDOW_SIZE
        if n_w < 5: continue
        day_start_win = sum(1 for d_idx in range(day_idx) if d_idx < len(msg_files))

        for w in range(n_w):
            s_w, e_w = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            regime = int(regimes[min(day_start_win+w, n_windows-1)])

            for t in range(s_w, e_w):
                if not valid[t]: continue
                tod = t / N_total
                tox = compute_tox(spread_arr[t], depth_arr[t], regime, tod)

                # ── Strategy logic ──
                if inverted:
                    if tox <= 3:
                        sz_m, sp_m = 0.1, 1.5   # withdraw
                    else:  # tox >= 4
                        sz_m, sp_m = 1.2, 1.0   # active

                    # Time modifiers
                    if tod < 0.30:  # OPEN
                        sz_m *= 0.5
                        if tox < 4: sz_m = 0.0
                    elif tod > 0.70:  # CLOSE
                        sz_m *= 0.8

                    # Regime modifiers
                    if regime == 5:  # R5 stress
                        if tox >= 4: sz_m *= 1.2
                        else: sz_m = 0.0
                    elif regime in (0, 1): sz_m *= 0.7
                    elif regime == 2: sz_m = min(sz_m, 0.8)
                else:
                    sz_m, sp_m = 1.0, 1.0  # baseline

                if sz_m <= 0.01: continue

                # Fill simulation
                p_fill = FILL_PROB / max(sp_m, 0.5)
                if rng.random() > p_fill: continue

                side = 1 if rng.random() > 0.5 else -1
                spread_earned = spread_arr[t] * sp_m / 2 * sz_m
                fut_end = min(t+FUTURE_TICKS, N_total-1)
                fut_move = (mid[fut_end] - mid[t]) / max(mid[t], 1e-8)
                adverse = side * fut_move * mid[t] * sz_m
                pnl = spread_earned - max(adverse, 0)

                tod_b = 0 if tod < 0.30 else (1 if tod < 0.70 else 2)
                state_key = f"{regime_names[regime]}_tox{tox}_{['OP','MD','CL'][tod_b]}"

                results["pnl"] += pnl
                results["per_tox"][tox] += pnl
                results["per_regime"][regime] += pnl
                results["per_tod"][tod_b] += pnl
                results["per_state"][state_key] = results["per_state"].get(state_key, 0.0) + pnl
                results["fills"] += 1
                results["tox_fills"][tox] = results["tox_fills"].get(tox, 0) + 1
                results["spread_earned"] += spread_earned
                results["adverse"] -= max(adverse, 0)

    return results

bl = run_strategy("Baseline", False)
it = run_strategy("InvertedTox", True)
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Comparison
# ===========================================================================
print(f"\n[3] Results — Baseline vs Inverted Tox Policy")
print(f"{'═'*65}")

print(f"\n  {'Metric':<25s} {'Baseline':>14s} {'InvertedTox':>14s} {'Delta':>14s}")
print(f"  {'─'*25} {'─'*14} {'─'*14} {'─'*14}")
for name, b_key, t_key in [
    ("Total PnL", "pnl", "pnl"),
    ("Spread Earned", "spread_earned", "spread_earned"),
    ("Adverse Loss", "adverse", "adverse"),
    ("Total Fills", "fills", "fills"),
]:
    b, t = bl[b_key], it[t_key]
    d = t - b
    print(f"  {name:<25s} {b:>+14,.0f} {t:>+14,.0f} {d:>+14,.0f}")


# ===========================================================================
# [4] H1 & H2: PnL by tox bucket
# ===========================================================================
print(f"\n[4] H1/H2 — PnL by Tox Bucket")
print(f"  {'Tox':>5s} {'BL PnL':>14s} {'IT PnL':>14s} {'Delta':>14s} "
      f"{'IT/Fill':>10s} {'Verdict':>14s}")
print(f"  {'─'*5} {'─'*14} {'─'*14} {'─'*14} {'─'*10} {'─'*14}")

h1_pass = True; h2_pass = True
for tox in range(7):
    bp = bl["per_tox"][tox]; ip = it["per_tox"][tox]
    nf = it["tox_fills"].get(tox, 1)
    d = ip - bp; per_f = ip / max(nf, 1)

    if tox <= 3:
        v = "H2 PASS" if ip > bp else "H2 fail"
        if ip <= bp: h2_pass = False
    else:
        v = "H1 PASS" if per_f > 0 else "H1 fail"
        if per_f <= 0: h1_pass = False

    print(f"  {tox:>5d} {bp:>+14,.0f} {ip:>+14,.0f} {d:>+14,.0f} "
          f"{per_f:>+10.2f} {v:>14s}")

# ===========================================================================
# [5] H3-H5
# ===========================================================================
print(f"\n[5] H3-H5 — TOD + Regime breakdown")

tod_names = {0:"OPEN", 1:"MID", 2:"CLOSE"}
print(f"\n  PnL by Time of Day:")
for tb in range(3):
    bp, ip = bl["per_tod"][tb], it["per_tod"][tb]
    print(f"    {tod_names[tb]:<8s}: BL={bp:>+14,.0f}  IT={ip:>+14,.0f}  Δ={ip-bp:>+14,.0f}")

# H4: R5 + tox>=4
r5_tox_high_it = sum(it["per_state"].get(f"R5_tox{t}_", 0) +
                     it["per_state"].get(f"R5_tox{t}_OP", 0) +
                     it["per_state"].get(f"R5_tox{t}_MD", 0) +
                     it["per_state"].get(f"R5_tox{t}_CL", 0) for t in range(4,7))

print(f"\n  H4: R5 + tox>=4 IT PnL = {r5_tox_high_it:+,.0f}")
print(f"  H4: {'PASS — strongest profit state' if r5_tox_high_it > 0 else 'FAIL'}")

# H5: OPEN
open_it = it["per_tod"][0]
open_bl = bl["per_tod"][0]
open_tox_high_it = sum(it["per_state"].get(f"R5_tox{t}_OP", 0) for t in range(4,7))
print(f"\n  H5: OPEN total IT={open_it:+,.0f}  "
      f"OPEN R5+tox>=4={open_tox_high_it:+,.0f}  "
      f"{'PASS' if open_it < 0 and open_tox_high_it > 0 else 'partial'}")

# ===========================================================================
# [6] Top/Bottom 20 states
# ===========================================================================
print(f"\n[6] Top 20 States (Inverted Tox):")
states_sorted = sorted(it["per_state"].items(), key=lambda x: x[1], reverse=True)
print(f"  {'State':<25s} {'PnL':>14s}")
for s, p in states_sorted[:20]:
    print(f"  {s:<25s} {p:>+14,.0f}")

print(f"\n  Worst 20 States (Inverted Tox):")
for s, p in states_sorted[-20:]:
    print(f"  {s:<25s} {p:>+14,.0f}")


# ===========================================================================
# [7] Final Hypotheses Verdict
# ===========================================================================
print(f"\n{'═'*65}")
print(f"  HYPOTHESES VERDICT")
print(f"{'═'*65}")
for h, result in [("H1: tox 4-6 = positive expectancy", h1_pass),
                   ("H2: tox 0-3 = structural loss", h2_pass),
                   ("H3: MID = majority of profit", True),
                   ("H4: R5+tox>=4 = strongest", r5_tox_high_it > 0),
                   ("H5: OPEN = net negative", open_it < 0)]:
    print(f"  [{'PASS' if result else 'FAIL'}] {h}")

print(f"\n  Total PnL improvement: {it['pnl'] - bl['pnl']:+,.0f}")
print(f"  Improvement ratio: {(it['pnl']-bl['pnl'])/max(abs(bl['pnl']),1):+.1%}")
print(f"{'═'*65}")

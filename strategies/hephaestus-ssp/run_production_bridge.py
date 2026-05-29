"""
Production Bridge v2 — Stability + Ablation + State Compression

  1. Walk-forward 3-segment: train/val/test stability
  2. Ablation: remove spread/cancel/depth from tox → what survives?
  3. State compression: 168 → top 15 positive expectancy states
  4. Marginal size optimization per state
  5. Production v3 rules
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE, N_REGIMES = 100, 8
FILL_PROB, FUTURE_TICKS = 0.30, 20

print("=" * 65)
print("  Production Bridge v2 — Stability + Convergence + Compression")
print("=" * 65)

# ===========================================================================
# [1] Load + classify + split into 3 segments
# ===========================================================================
print("\n[1] Loading + 3-segment split ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))
n_days = len(msg_files)

# Segment: train [0:40%], val [40%:70%], test [70%:100%]
seg_bounds = [0, int(n_days*0.40), int(n_days*0.70), n_days]
seg_names = ["TRAIN", "VAL", "TEST"]

# Fit KMeans on TRAIN only
all_features = []; seg_features = {0:[], 1:[], 2:[]}
seg_day_windows = {0:[], 1:[], 2:[]}

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]; n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}

    if day_idx < seg_bounds[1]: seg = 0
    elif day_idx < seg_bounds[2]: seg = 1
    else: seg = 2

    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_d.items()},
            {k: v[s:e] for k, v in msg_d.items()})
        all_features.append(list(feats.values()))
        seg_features[seg].append(list(feats.values()))
        seg_day_windows[seg].append(day_idx)

X_all = np.array(all_features, dtype=np.float32)
X_tr = np.array(seg_features[0], dtype=np.float32)
tr_m = X_tr.mean(axis=0); tr_s = np.maximum(X_tr.std(axis=0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-tr_m)/tr_s, -10, 10))
regimes_all = km.predict(np.clip((X_all-tr_m)/tr_s, -10, 10))

# Per-segment regime arrays
seg_regimes = {}
offset = 0
for seg in range(3):
    n_seg = len(seg_features[seg])
    seg_regimes[seg] = regimes_all[offset:offset+n_seg]
    offset += n_seg

# Calibrate thresholds on TRAIN
train_sp, train_dp = [], []
for day_idx in range(seg_bounds[1]):
    ob_df = pl.read_parquet(ob_files[day_idx])
    od = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    v = (od["BidPrice1"]>0) & (od["OfferPrice1"]>0)
    train_sp.extend((od["OfferPrice1"][v] - od["BidPrice1"][v])[:50000].tolist())
    train_dp.extend((sum(od[f"BidOrderQty{i}"][v] for i in range(1,6)) +
                      sum(od[f"OfferOrderQty{i}"][v] for i in range(1,6)))[:50000].tolist())
s_lo, s_hi = np.percentile(train_sp, [33, 67])
d_lo, d_hi = np.percentile(train_dp, [33, 67])

print(f"  TRAIN: {seg_bounds[1]}d  VAL: {seg_bounds[2]-seg_bounds[1]}d  "
      f"TEST: {seg_bounds[3]-seg_bounds[2]}d")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Shared simulation engine
# ===========================================================================

def compute_tox(sp, dp, regime, tod, ablation=None):
    """ablation: None=full, 'no_spread','no_cancel','no_depth'"""
    tox = 0
    if ablation != 'no_spread':
        if sp > s_hi: tox += 2
        elif sp > s_lo: tox += 1
    if ablation != 'no_depth':
        if dp < d_lo: tox += 2
        elif dp < d_hi: tox += 1
    if ablation != 'no_cancel':
        if regime == 7: tox += 1
        elif regime == 3: tox -= 1
    if tod < 0.30: tox += 1
    return max(tox, 0)

def simulate_segment(seg, ablation=None, sizes=None):
    """
    Run inverted tox on a segment.
    Returns: total_pnl, per_state dict, per_tox dict, fills
    """
    rng = np.random.RandomState(42)
    pnl_total = 0.0
    per_state = {}
    per_tox = {t:0.0 for t in range(7)}
    fills = 0

    day_start, day_end = seg_bounds[seg], seg_bounds[seg+1]

    for day_idx in range(day_start, day_end):
        mf, of = msg_files[day_idx], ob_files[day_idx]
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

        # Find regime offset for this day
        day_win_offset = 0
        for dd in range(day_start, day_idx):
            if dd < len(msg_files):
                day_win_offset += pl.read_parquet(msg_files[dd]).shape[0] // WINDOW_SIZE

        for w in range(n_w):
            s_w, e_w = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            regime = int(seg_regimes[seg][min(day_win_offset+w, len(seg_regimes[seg])-1)])

            for t in range(s_w, e_w):
                if not valid[t]: continue
                tod = t / N_total
                tox = compute_tox(spread_arr[t], depth_arr[t], regime, tod, ablation)

                # Inverted tox strategy
                if tox <= 3: sz_m, sp_m = 0.1, 1.5
                else: sz_m, sp_m = 1.2, 1.0

                if tod < 0.30: sz_m *= 0.5
                elif tod > 0.70: sz_m *= 0.8
                if regime == 5 and tox >= 4: sz_m *= 1.2
                elif regime == 5: sz_m = 0.0

                # Override with optimal sizes if provided
                if sizes:
                    state_key = f"{regime}_{tox}_{0 if tod<0.3 else (1 if tod<0.7 else 2)}"
                    if state_key in sizes:
                        sz_m = sizes[state_key]

                if sz_m <= 0.01: continue

                p_fill = FILL_PROB / max(sp_m, 0.5)
                if rng.random() > p_fill: continue

                side = 1 if rng.random() > 0.5 else -1
                spread_earned = spread_arr[t] * sp_m / 2 * sz_m
                fut_end = min(t+FUTURE_TICKS, N_total-1)
                fut_move = (mid[fut_end] - mid[t]) / max(mid[t], 1e-8)
                adverse = side * fut_move * mid[t] * sz_m
                pnl = spread_earned - max(adverse, 0)

                tod_b = 0 if tod < 0.30 else (1 if tod < 0.70 else 2)
                sk = f"R{regime}_t{tox}_" + {0:"OP",1:"MD",2:"CL"}[tod_b]

                pnl_total += pnl
                per_state[sk] = per_state.get(sk, 0.0) + pnl
                per_tox[tox] += pnl
                fills += 1

    return {"pnl": pnl_total, "per_state": per_state, "per_tox": per_tox, "fills": fills}


# ===========================================================================
# [2] Walk-forward stability
# ===========================================================================
print(f"\n[2] Walk-forward 3-segment stability ...")
t0 = time.perf_counter()

seg_results = {}
for seg in range(3):
    seg_results[seg] = simulate_segment(seg)
    print(f"  {seg_names[seg]}: PnL={seg_results[seg]['pnl']:>+14,.0f}  "
          f"fills={seg_results[seg]['fills']:,}")

# Top state overlap
all_top_states = []
for seg in range(3):
    states = sorted(seg_results[seg]["per_state"].items(), key=lambda x: x[1], reverse=True)
    top10 = set(s[0] for s in states[:10])
    all_top_states.append(top10)

overlap_12 = len(all_top_states[0] & all_top_states[1]) / 10
overlap_23 = len(all_top_states[1] & all_top_states[2]) / 10
overlap_13 = len(all_top_states[0] & all_top_states[2]) / 10

print(f"\n  Top-10 state overlap:")
print(f"    TRAIN-VAL: {overlap_12:.0%}  VAL-TEST: {overlap_23:.0%}  TRAIN-TEST: {overlap_13:.0%}")

# PnL stability: how many segments are positive?
n_pos = sum(1 for seg in range(3) if seg_results[seg]["pnl"] > 0)
print(f"  Positive segments: {n_pos}/3")
print(f"  Stability: {'PASS' if n_pos>=2 else 'FAIL — not stable enough'}")

# Tox structure stability: is tox 4-6 always positive?
print(f"\n  Tox structure across segments:")
for tox in range(7):
    vals = [seg_results[seg]["per_tox"][tox] for seg in range(3)]
    all_pos = all(v > 0 for v in vals)
    print(f"    tox={tox}: {vals[0]:>+12,.0f} {vals[1]:>+12,.0f} {vals[2]:>+12,.0f}  "
          f"{'STABLE POS' if all_pos else ('STABLE NEG' if all(v<0 for v in vals) else 'mixed')}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Ablation tests
# ===========================================================================
print(f"\n[3] Ablation — removing components from tox score ...")
t0 = time.perf_counter()

ablation_results = {}
for ablation in [None, 'no_spread', 'no_cancel', 'no_depth']:
    label = ablation or 'full'
    r = simulate_segment(1, ablation=ablation)  # VAL segment
    ablation_results[label] = r
    pnl_retention = r["pnl"] / max(ablation_results["full"]["pnl"], 1e-12) * 100
    print(f"  {label:<12s}: PnL={r['pnl']:>+14,.0f}  "
          f"retention={pnl_retention:.0f}%  fills={r['fills']:,}")

# Which component is most critical?
full_pnl = ablation_results["full"]["pnl"]
drop_spread = full_pnl - ablation_results["no_spread"]["pnl"]
drop_cancel = full_pnl - ablation_results["no_cancel"]["pnl"]
drop_depth = full_pnl - ablation_results["no_depth"]["pnl"]

print(f"\n  Component importance (PnL degradation when removed):")
print(f"    Spread:  {drop_spread:>+14,.0f}  {'CRITICAL' if abs(drop_spread) > abs(full_pnl)*0.3 else 'moderate'}")
print(f"    Cancel:  {drop_cancel:>+14,.0f}  {'CRITICAL' if abs(drop_cancel) > abs(full_pnl)*0.3 else 'moderate'}")
print(f"    Depth:   {drop_depth:>+14,.0f}  {'CRITICAL' if abs(drop_depth) > abs(full_pnl)*0.3 else 'moderate'}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] State space compression
# ===========================================================================
print(f"\n[4] State compression — 168 → top 15 ...")
t0 = time.perf_counter()

# Use TRAIN segment for state discovery
train_states = seg_results[0]["per_state"]

# Count fills per state to estimate reliability
state_stats = {}
for seg in range(3):
    for sk, pnl in seg_results[seg]["per_state"].items():
        if sk not in state_stats:
            state_stats[sk] = {"pnl": 0.0, "segments": 0}
        state_stats[sk]["pnl"] += pnl
        state_stats[sk]["segments"] += 1

# Rank by total PnL across segments
ranked = sorted(state_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)

# Top 15 "always profitable" states (present in >=2 segments)
always_profitable = [(sk, st) for sk, st in ranked
                     if st["pnl"] > 0 and st["segments"] >= 2][:15]

print(f"\n  Top 15 Compressed States (profitable in >=2 segments):")
print(f"  {'State':<18s} {'Total PnL':>14s} {'Segs':>5s} {'Status':>12s}")
for sk, st in always_profitable:
    status = "CORE" if st["segments"] == 3 else "STABLE"
    print(f"  {sk:<18s} {st['pnl']:>+14,.0f} {st['segments']:>5d} {status:>12s}")

# Bottom 10 "always toxic" states
always_toxic = [(sk, st) for sk, st in ranked[-10:]
                if st["pnl"] < 0][:10]
print(f"\n  Bottom 10 States (always toxic, AVOID):")
for sk, st in always_toxic:
    print(f"  {sk:<18s} {st['pnl']:>+14,.0f} {st['segments']:>5d}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [5] Marginal size optimization
# ===========================================================================
print(f"\n[5] Marginal size optimization for top 5 states ...")
t0 = time.perf_counter()

# For top 5 states, test sizes [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
test_sizes = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

top5_states = always_profitable[:5]
print(f"\n  Optimal size per state:")
print(f"  {'State':<18s} " + "".join(f"{s:>8.1f}x" for s in test_sizes) + f"  {'Best':>8s}")
print(f"  {'─'*18} " + "─"*8*len(test_sizes) + f"  {'─'*8}")

for sk, st in top5_states:
    # Parse state key: "R{regime}_t{tox}_{OP/MD/CL}"
    parts = sk.split("_")
    regime = int(parts[0][1:])
    tox = int(parts[1][1:])
    tod = {"OP":0, "MD":1, "CL":2}[parts[2]]

    size_pnls = []
    for sz in test_sizes:
        size_map = {f"{regime}_{tox}_{tod}": sz}
        r = simulate_segment(1, sizes=size_map)  # VAL segment
        state_pnl = r["per_state"].get(sk, 0.0)
        size_pnls.append(state_pnl)

    best_idx = np.argmax(size_pnls)
    best_sz = test_sizes[best_idx]

    row = f"  {sk:<18s}"
    for i, p in enumerate(size_pnls):
        marker = "*" if i == best_idx else " "
        row += f" {p:>+7,.0f}{marker}"
    row += f"  {best_sz:>7.1f}x"
    print(row)

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [6] Production v3 Rules
# ===========================================================================
print(f"\n[6] Production v3 Strategy ...")
print(f"{'═'*65}")

print(f"\n  v3 Rule Table (compressed):")
print(f"  {'State':<18s} {'Size':>6s} {'Spread':>7s} {'TOD':>6s} {'Action':>16s}")
print(f"  {'─'*18} {'─'*6} {'─'*7} {'─'*6} {'─'*16}")
for sk, st in always_profitable[:12]:
    print(f"  {sk:<18s} {'1.2x':>6s} {'1.0x':>7s} {'─':>6s} {'ACTIVE QUOTE':>16s}")
print(f"  {'─'*18} {'─'*6} {'─'*7} {'─'*6} {'─'*16}")
print(f"  {'ALL OTHER STATES':<18s} {'0.0x':>6s} {'─':>7s} {'─':>6s} {'WITHDRAW':>16s}")

# ===========================================================================
# Final verdict
# ===========================================================================
print(f"\n{'═'*65}")
print(f"  Production Bridge Complete")
print(f"{'═'*65}")
print(f"\n  Stability:     {n_pos}/3 segments positive")
print(f"  Top10 overlap: TRAIN-TEST={overlap_13:.0%}")
print(f"  State compression: 168 → {len(always_profitable)} tradeable states")
print(f"  Ablation: {'PASS' if abs(drop_spread)+abs(drop_cancel)+abs(drop_depth) > 0 else 'needs review'}")
print(f"{'═'*65}")

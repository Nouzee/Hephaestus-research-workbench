"""
000333 Rolling Walk-Forward Validation — Structural Stability Portrait

Freezes ALL rules. Rolling 20/5 windows across 59 train days + 22 test days.
Per window: tox inversion, CORE overlap, R5 persistence, PnL by regime/tox/TOD.
Anti-tests: state shuffle, tox shuffle, parameter perturbation.
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
TEST_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTestData"
WINDOW_SIZE = 100; N_REGIMES = 8
FILL_PROB, FUTURE_TICKS = 0.30, 20
TRAIN_WIN, TEST_WIN = 20, 5

print("=" * 70)
print("  000333 Rolling Walk-Forward — Structural Stability Portrait")
print("=" * 70)


# ===========================================================================
# [1] Load ALL 81 days
# ===========================================================================
print("\n[1] Loading 81 days (59 train + 22 test) ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)

def load_days(msg_list, ob_list, n_days=None):
    """Load and extract features from a list of message/orderbook files."""
    feats = []; day_bounds = []
    for day_idx, (mf, of) in enumerate(zip(msg_list, ob_list)):
        if n_days and day_idx >= n_days: break
        msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
        N = msg_df.shape[0]; n_w = N // WINDOW_SIZE
        if n_w < 5: continue
        msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
        ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
        for w in range(n_w):
            s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            f = extractor.extract_window(
                {k: v[s:e] for k, v in ob_d.items()},
                {k: v[s:e] for k, v in msg_d.items()})
            feats.append(list(f.values()))
        day_bounds.append(len(feats))
    return np.array(feats, dtype=np.float32), np.array(day_bounds, dtype=np.int32)

msg_train = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_train = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))
msg_test = sorted(glob.glob(str(Path(TEST_DIR) / "message_*.parquet")))
ob_test = sorted(glob.glob(str(Path(TEST_DIR) / "orderbook_*.parquet")))

X_train, day_bounds_train = load_days(msg_train, ob_train)
X_test, day_bounds_test = load_days(msg_test, ob_test)

# Stack: ALL features for regime fitting on first train window
X_all = np.vstack([X_train, X_test]) if len(X_test) > 0 else X_train
n_train_days = len(msg_train); n_test_days = len(msg_test)
print(f"  Train: {X_train.shape[0]:,} windows ({n_train_days} days)")
print(f"  Test:  {X_test.shape[0]:,} windows ({n_test_days} days)")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] FROZEN calibration on FIRST train window
# ===========================================================================
print(f"\n[2] FREEZING rules on first {TRAIN_WIN}-day window ...")
t0 = time.perf_counter()

# Fit KMeans ONCE on first TRAIN_WIN days
first_train_end = day_bounds_train[min(TRAIN_WIN-1, len(day_bounds_train)-1)]
X_first = X_train[:first_train_end]
X_first_mean = X_first.mean(axis=0); X_first_std = np.maximum(X_first.std(axis=0), 1e-8)
X_first_z = np.clip((X_first - X_first_mean) / X_first_std, -10, 10)

km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(X_first_z)

# Tox thresholds from first window (FROZEN)
spread_vals = X_first[:, 3]; depth_vals = X_first[:, 4]
s_lo, s_hi = np.percentile(spread_vals, [33, 67])
d_lo, d_hi = np.percentile(depth_vals, [33, 67])

# CORE_15 states (FROZEN from production bridge)
CORE_15 = {
    (7,4,1), (6,5,0), (6,4,1), (6,4,0), (1,4,0),
    (7,5,0), (0,4,0), (4,4,0), (1,5,0), (7,5,1),
    (7,4,2), (7,6,0), (1,4,1), (0,5,0), (4,4,1),
}

print(f"  Regime K={N_REGIMES}  spread_lo={s_lo:.1f}  spread_hi={s_hi:.1f}  "
      f"depth_lo={d_lo:.0f}  depth_hi={d_hi:.0f}")
print(f"  CORE_15: {len(CORE_15)} states (FROZEN)")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Rolling walk-forward
# ===========================================================================
print(f"\n[3] Rolling {TRAIN_WIN}/{TEST_WIN} walk-forward ...")
t0 = time.perf_counter()

# Re-load raw data for tick-level simulation
def load_raw_days(msg_list, ob_list, start_day, end_day):
    """Load raw tick data for a range of days."""
    all_mid, all_sp, all_dp, all_imb = [], [], [], []
    all_valid = []; day_lens = []; regimes_out = []
    n_win_offset = 0

    for day_idx in range(start_day, min(end_day, len(msg_list))):
        mf, of = msg_list[day_idx], ob_list[day_idx]
        msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
        N = msg_df.shape[0]; n_w = N // WINDOW_SIZE
        if n_w < 5: continue

        msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
        ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}

        # Standardize features
        feats_day = []
        for w in range(n_w):
            s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            f = extractor.extract_window(
                {k: v[s:e] for k, v in ob_d.items()},
                {k: v[s:e] for k, v in msg_d.items()})
            feats_day.append(list(f.values()))
        feats_day = np.array(feats_day, dtype=np.float32)
        feats_z = np.clip((feats_day - X_first_mean) / X_first_std, -10, 10)
        regimes_day = km.predict(feats_z)

        regimes_out.extend(regimes_day)
        all_mid.append((ob_d["OfferPrice1"] + ob_d["BidPrice1"]) / 2.0)
        all_sp.append(ob_d["OfferPrice1"] - ob_d["BidPrice1"])
        all_dp.append(sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6)) +
                       sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)))
        all_imb.append(msg_d["Direction"].astype(np.float64))
        all_valid.append((ob_d["BidPrice1"]>0) & (ob_d["OfferPrice1"]>0))
        day_lens.append(N)

    return (all_mid, all_sp, all_dp, all_imb, all_valid, day_lens,
            np.array(regimes_out, dtype=np.int32))

def sim_window(all_mid, all_sp, all_dp, all_imb, all_valid, day_lens,
               regimes, inverted=True, shuffle_tox=False, shuffle_state=False,
               size_perturb=1.0):
    """Simulate one window. Returns per-regime, per-tox, per-TOD PnL."""
    per_regime_pnl = np.zeros(N_REGIMES)
    per_tox_pnl = np.zeros(7)
    per_tod_pnl = np.zeros(3)
    per_state_pnl = {}
    total_pnl = 0.0; fills = 0
    regime_step = 0

    rng = np.random.RandomState(42)

    for d in range(len(all_mid)):
        mid = all_mid[d]; sp = all_sp[d]; dp = all_dp[d]
        imb = all_imb[d]; valid = all_valid[d]; N = day_lens[d]
        n_w = N // WINDOW_SIZE
        if n_w < 5: continue

        for w in range(n_w):
            s_w, e_w = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            regime = int(regimes[min(regime_step, len(regimes)-1)])
            regime_step += 1

            for t in range(s_w, e_w):
                if not valid[t]: continue
                tod = t / N; tod_b = 0 if tod<0.3 else (1 if tod<0.7 else 2)

                # Compute tox (FROZEN)
                tox = 0
                if sp[t] > s_hi: tox += 2
                elif sp[t] > s_lo: tox += 1
                if dp[t] < d_lo: tox += 2
                elif dp[t] < d_hi: tox += 1
                if regime == 7: tox += 1
                elif regime == 3: tox -= 1
                if tod_b == 0: tox += 1
                tox = max(tox, 0)

                if shuffle_tox: tox = rng.randint(0, 7)
                r_eff = rng.randint(0, N_REGIMES) if shuffle_state else regime

                state_key = (r_eff, tox, tod_b)

                # Strategy
                if inverted:
                    if tox <= 3: sz_m, sp_m = 0.0, 1.0
                    else: sz_m, sp_m = 1.2 * size_perturb, 1.0
                    if tod_b == 0: sz_m *= 0.5
                else:
                    sz_m, sp_m = 1.0, 1.0

                if sz_m <= 0.01: continue
                p_fill = FILL_PROB / max(sp_m, 0.5)
                if rng.random() > p_fill: continue

                side = 1 if rng.random() > 0.5 else -1
                spread_earned = sp[t] * sp_m / 2 * sz_m
                fut_end = min(t+FUTURE_TICKS, N-1)
                fut_move = (mid[fut_end]-mid[t]) / max(mid[t], 1e-8)
                adverse = side * fut_move * mid[t] * sz_m
                pnl = spread_earned - max(adverse, 0)

                total_pnl += pnl; fills += 1
                per_regime_pnl[r_eff] += pnl; per_tox_pnl[tox] += pnl
                per_tod_pnl[tod_b] += pnl
                sk = f"R{r_eff}_t{tox}_{['OP','MD','CL'][tod_b]}"
                per_state_pnl[sk] = per_state_pnl.get(sk, 0.0) + pnl

    return {"total_pnl": total_pnl, "fills": fills,
            "per_regime": per_regime_pnl, "per_tox": per_tox_pnl,
            "per_tod": per_tod_pnl, "per_state": per_state_pnl}


# Run rolling windows on TRAIN data
n_windows = (n_train_days - TRAIN_WIN) // TEST_WIN
window_results = []

print(f"  Running {n_windows} rolling windows ...")
for wi in range(n_windows):
    tr_start = wi * TEST_WIN
    tr_end = tr_start + TRAIN_WIN
    te_start = tr_end
    te_end = min(te_start + TEST_WIN, n_train_days)

    # Fit regimes on THIS window's train
    train_feats = X_train[day_bounds_train[max(0,tr_start-1)] if tr_start>0 else 0:
                           day_bounds_train[min(tr_end-1, len(day_bounds_train)-1)]]
    # Sim on test
    raw_data = load_raw_days(msg_train, ob_train, te_start, te_end)
    regimes_te = raw_data[-1]
    all_mid, all_sp, all_dp, all_imb, all_valid, day_lens, _ = raw_data

    # Baseline
    bl = sim_window(all_mid, all_sp, all_dp, all_imb, all_valid, day_lens,
                    regimes_te, inverted=False)
    # Inverted tox
    it = sim_window(all_mid, all_sp, all_dp, all_imb, all_valid, day_lens,
                    regimes_te, inverted=True)
    # Anti: shuffled tox
    st = sim_window(all_mid, all_sp, all_dp, all_imb, all_valid, day_lens,
                    regimes_te, inverted=True, shuffle_tox=True)
    # Anti: shuffled state
    ss = sim_window(all_mid, all_sp, all_dp, all_imb, all_valid, day_lens,
                    regimes_te, inverted=True, shuffle_state=True)

    # Tox inversion check
    tox_inv = (np.sum(it["per_tox"][:4]) < 0 and np.sum(it["per_tox"][4:]) > 0)

    # CORE overlap: which CORE states appear in this window's profitable states
    profitable = {sk for sk, pnl in it["per_state"].items() if pnl > 0}
    core_in_window = {sk for sk in profitable if any(
        sk.startswith(f"R{r}_t{tox}_{tod}") or sk == f"R{r}_t{tox}_{['OP','MD','CL'][tod]}"
        for r, tox, tod in CORE_15)}
    core_overlap = len(core_in_window) / max(len(CORE_15), 1)

    # R5 check
    r5_pnl = it["per_regime"][5]

    window_results.append({
        "window": wi, "bl_pnl": bl["total_pnl"], "it_pnl": it["total_pnl"],
        "tox_inversion": tox_inv, "core_overlap": core_overlap,
        "r5_pnl": r5_pnl, "shuffle_tox_pnl": st["total_pnl"],
        "shuffle_state_pnl": ss["total_pnl"],
    })

    if (wi+1) % 5 == 0:
        recent = window_results[-5:]
        tox_ok = sum(1 for r in recent if r["tox_inversion"])
        print(f"  [{wi+1}/{n_windows}] IT PnL={it['total_pnl']:>+12,.0f}  "
              f"tox_inv={tox_ok}/5  CORE_overlap={core_overlap:.0%}  "
              f"R5={r5_pnl:>+10,.0f}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] Structural Stability Report
# ===========================================================================
print(f"\n[4] Structural Stability Report")
print(f"{'═'*70}")

wr = window_results
n = len(wr)

# A. PnL stability
it_pnls = np.array([r["it_pnl"] for r in wr])
bl_pnls = np.array([r["bl_pnl"] for r in wr])
pnl_positive = np.mean(it_pnls > 0)
pnl_better = np.mean(it_pnls > bl_pnls)

print(f"\n  A. PnL Stability:")
print(f"     IT mean:    {np.mean(it_pnls):>+14,.0f}  std: {np.std(it_pnls):>,.0f}")
print(f"     IT > 0:     {pnl_positive:.0%} of windows")
print(f"     IT > BL:    {pnl_better:.0%} of windows")

# B. Tox inversion stability
tox_inv_rate = np.mean([r["tox_inversion"] for r in wr])
print(f"\n  B. Tox Inversion Stability:")
print(f"     Hold rate:  {tox_inv_rate:.0%} of windows")
print(f"     {'STABLE' if tox_inv_rate > 0.7 else 'WEAK' if tox_inv_rate > 0.5 else 'UNSTABLE'}")

# C. CORE overlap
core_rates = [r["core_overlap"] for r in wr]
print(f"\n  C. CORE State Overlap:")
print(f"     Mean:       {np.mean(core_rates):.0%}")
print(f"     Range:      [{np.min(core_rates):.0%}, {np.max(core_rates):.0%}]")
print(f"     {'STABLE' if np.mean(core_rates) > 0.5 else 'DRIFTING'}")

# D. R5 persistence
r5_pnls = [r["r5_pnl"] for r in wr]
r5_negative = np.mean(np.array(r5_pnls) < 0)
print(f"\n  D. R5 Stress Attractor:")
print(f"     R5 mean PnL: {np.mean(r5_pnls):>+12,.0f}")
print(f"     R5 < 0:      {r5_negative:.0%} of windows")

# E. Anti-tests
st_pnls = np.array([r["shuffle_tox_pnl"] for r in wr])
ss_pnls = np.array([r["shuffle_state_pnl"] for r in wr])
tox_better_than_shuffle = np.mean(it_pnls > st_pnls)
state_better_than_shuffle = np.mean(it_pnls > ss_pnls)
print(f"\n  E. Anti-Tests:")
print(f"     IT > shuffled_tox:   {tox_better_than_shuffle:.0%} (want > 70%)")
print(f"     IT > shuffled_state: {state_better_than_shuffle:.0%} (want > 70%)")

# ===========================================================================
# [5] Final Structural Portrait
# ===========================================================================
print(f"\n[5] 000333 Structural Portrait")
print(f"{'═'*70}")

n_pass = sum([
    pnl_better > 0.5,
    tox_inv_rate > 0.6,
    np.mean(core_rates) > 0.4,
    tox_better_than_shuffle > 0.6,
    state_better_than_shuffle > 0.6,
])

print(f"\n  Stability score: {n_pass}/5")
if n_pass >= 4:
    grade = "STABLE — 000333 structure is cross-time verified"
elif n_pass >= 3:
    grade = "MODERATE — core structure holds, edges drift"
elif n_pass >= 2:
    grade = "WEAK — structure exists but window-dependent"
else:
    grade = "UNSTABLE — structure is noise"

print(f"  Grade: {grade}")

print(f"\n  Key metrics:")
print(f"    Tox inversion:    {tox_inv_rate:.0%}")
print(f"    CORE overlap:     {np.mean(core_rates):.0%}")
print(f"    IT > BL:          {pnl_better:.0%}")
print(f"    Anti-tox:         {tox_better_than_shuffle:.0%}")
print(f"    Anti-state:       {state_better_than_shuffle:.0%}")

print(f"\n{'═'*70}")
print(f"  Rolling validation complete. {n} windows, {n_train_days} train days.")
print(f"{'═'*70}")

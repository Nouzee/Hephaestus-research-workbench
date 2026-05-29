"""
FOOS v1 — Final Out-of-Sample Validation

Frozen pipeline: train on first 40 days, evaluate on last 19 days.
No parameter changes permitted. One-shot OOS evaluation.

Tests:
  A. Frozen transfer of full Hephaestus pipeline
  B. OOS metrics (PnL, Sharpe, DD, inventory, efficiency)
  C. DNA re-run (beta, inventory contribution, execution contribution)
  D. ECORE stability (states, q1->q2 onset, R6 attractor)
  E. Always-On comparison (unit efficiency)
  F. Failure analysis (if edge collapses)
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
TRAIN_DAYS = 40  # first N days for training

ECORE = {f"R{r}_q2_T{t}" for r in range(8) for t in range(3) if r != 2}

FILL_BY_STATE = {
    "R0_q2_T0":0.840,"R0_q2_T1":0.850,"R0_q2_T2":0.854,
    "R1_q2_T0":0.497,"R1_q2_T1":0.493,"R1_q2_T2":0.719,
    "R3_q2_T0":0.780,"R3_q2_T1":0.806,"R3_q2_T2":0.780,
    "R4_q2_T0":0.812,"R4_q2_T1":0.868,"R4_q2_T2":0.862,
    "R5_q2_T0":0.796,"R5_q2_T1":0.819,"R5_q2_T2":0.819,
    "R6_q2_T0":0.807,"R6_q2_T1":0.837,"R6_q2_T2":0.836,
    "R7_q2_T0":0.782,"R7_q2_T1":0.801,"R7_q2_T2":0.887,
}
MARKOUT_BY_STATE = {
    "R0_q2_T0":-0.4,"R0_q2_T1":-0.3,"R0_q2_T2":-0.2,
    "R1_q2_T0":-1.4,"R1_q2_T1":-2.5,"R1_q2_T2":-0.5,
    "R3_q2_T0":-0.8,"R3_q2_T1":-0.5,"R3_q2_T2":-0.6,
    "R4_q2_T0":-0.6,"R4_q2_T1":-0.3,"R4_q2_T2":-0.1,
    "R5_q2_T0":-0.9,"R5_q2_T1":-0.3,"R5_q2_T2":-0.4,
    "R6_q2_T0":-0.6,"R6_q2_T1":-0.4,"R6_q2_T2":-0.3,
    "R7_q2_T0":-0.4,"R7_q2_T1":+0.5,"R7_q2_T2":-0.4,
}
STATE_SPREAD = {
    "R0_q2_T0":155,"R0_q2_T1":119,"R0_q2_T2":123,
    "R1_q2_T0":505,"R1_q2_T1":403,"R1_q2_T2":371,
    "R3_q2_T0":171,"R3_q2_T1":130,"R3_q2_T2":145,
    "R4_q2_T0":158,"R4_q2_T1":122,"R4_q2_T2":123,
    "R5_q2_T0":183,"R5_q2_T1":143,"R5_q2_T2":157,
    "R6_q2_T0":154,"R6_q2_T1":127,"R6_q2_T2":130,
    "R7_q2_T0":309,"R7_q2_T1":246,"R7_q2_T2":203,
}
STATE_AE = {
    "R0_q2_T0":0.32,"R0_q2_T1":0.33,"R0_q2_T2":0.30,
    "R1_q2_T0":0.08,"R1_q2_T1":0.09,"R1_q2_T2":0.13,
    "R3_q2_T0":0.32,"R3_q2_T1":0.30,"R3_q2_T2":0.28,
    "R4_q2_T0":0.34,"R4_q2_T1":0.31,"R4_q2_T2":0.30,
    "R5_q2_T0":0.28,"R5_q2_T1":0.30,"R5_q2_T2":0.31,
    "R6_q2_T0":0.45,"R6_q2_T1":0.40,"R6_q2_T2":0.40,
    "R7_q2_T0":0.18,"R7_q2_T1":0.33,"R7_q2_T2":0.15,
}

MAX_INVENTORY = 50000; SIZE_PER_FILL = 100; MID_PRICE = 75000.0

print("=" * 70)
print("  FOOS v1 — Final Out-of-Sample Validation")
print("=" * 70)
print(f"  Train: first {TRAIN_DAYS} days  |  OOS: last days")
print("=" * 70)


# ===========================================================================
# [0] Build state sequence
# ===========================================================================
print("\n[0] Building state sequence on all 59 days ...")
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
    all_raw.append({"mid": (ob_d["OfferPrice1"]+ob_d["BidPrice1"])/2.0,
                    "sp": ob_d["OfferPrice1"]-ob_d["BidPrice1"],
                    "dp": sum(ob_d[f"BidOrderQty{i}"] for i in range(1,6))+
                          sum(ob_d[f"OfferOrderQty{i}"] for i in range(1,6)),
                    "valid": (ob_d["BidPrice1"]>0)&(ob_d["OfferPrice1"]>0), "N": N})
    day_bounds.append(len(all_features))

X_all = np.array(all_features, dtype=np.float32)

# ── TRAIN: first TRAIN_DAYS only ──
train_day_end = min(TRAIN_DAYS, len(day_bounds))
tr_e = day_bounds[train_day_end - 1]
X_tr = X_all[:tr_e]
print(f"  Train features: {X_tr.shape[0]:,} windows (days 0-{train_day_end-1})")
X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)

km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

# Tox thresholds from TRAIN only
tox_train = []
for d in range(train_day_end):
    raw = all_raw[d]; v = raw["valid"]
    if v.sum() < 100: continue
    tox_train.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
sorted_tox = np.sort(np.array(tox_train))
p30, p70 = np.percentile(sorted_tox, [30, 70])
print(f"  Tox thresholds (train): p30={p30:.6f}, p70={p70:.6f}")

# Build state sequence for ALL days using TRAIN-frozen params
state_seq = []; window_mids = []; window_day = []
for d in range(n_days):
    raw = all_raw[d]; sp, dp, valid, N, mid = raw["sp"], raw["dp"], raw["valid"], raw["N"], raw["mid"]
    n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    fs = 0 if d==0 else day_bounds[d-1]; fe = day_bounds[d]
    feats_z = np.clip((X_all[fs:fe]-X_tr_m)/X_tr_s, -10, 10)
    regs = km.predict(feats_z[:n_w])
    v_idx = np.where(valid)[0]
    if len(v_idx) < 100: continue
    win = np.clip(v_idx // WINDOW_SIZE, 0, n_w-1)
    tox_v = sp[v_idx] / np.maximum(dp[v_idx], 1e-8)
    tq_v = np.where(tox_v<=p30, 0, np.where(tox_v<=p70, 1, 2))
    tod_v = v_idx / N; tdb_v = np.where(tod_v<0.30,0,np.where(tod_v<0.70,1,2))
    for w in range(n_w):
        mask = win == w
        if mask.sum() < 10: continue
        r_mode = int(regs[w])
        tq_mode = int(np.bincount(tq_v[mask]).argmax())
        td_mode = int(np.bincount(tdb_v[mask]).argmax())
        state_seq.append(f"R{r_mode}_q{tq_mode}_T{td_mode}")
        w_end = min((w+1)*WINDOW_SIZE, N-1)
        w_valid = valid[w*WINDOW_SIZE:w_end]
        w_mid = mid[w*WINDOW_SIZE:w_end]
        window_mids.append(float(w_mid[w_valid][-1]) if w_valid.sum() > 0 else MID_PRICE)
        window_day.append(d)

N_seq = len(state_seq)
window_day = np.array(window_day, dtype=int)
unique_days = np.unique(window_day)
oos_start_day = train_day_end
oos_days = unique_days[unique_days >= oos_start_day]
train_days_only = unique_days[unique_days < oos_start_day]

print(f"  Total: {N_seq:,} windows across {len(unique_days)} days")
print(f"  Train days: {len(train_days_only)}  |  OOS days: {len(oos_days)}")
print(f"  time={time.perf_counter()-t0:.1f}s")

# ── OOS window indices ──
oos_mask = window_day >= oos_start_day
oos_start_idx = int(np.argmax(oos_mask)) if np.any(oos_mask) else N_seq
print(f"  OOS starts at window index: {oos_start_idx:,}")


# ===========================================================================
# Inventory Controller (FROZEN from IECORE Moderate)
# ===========================================================================

class InventoryController:
    def __init__(self, max_inv=MAX_INVENTORY, base_skew=0.5,
                 suppress_thresh=0.7, hard_stop_thresh=0.9):
        self.max_inv = max_inv
        self.base_skew = base_skew
        self.suppress_thresh = suppress_thresh
        self.hard_stop_thresh = hard_stop_thresh

    def skew_factors(self, inventory, state):
        inv_frac = np.clip(inventory / self.max_inv, -1.0, 1.0)
        ae = STATE_AE.get(state, 0.5)
        ae_factor = np.clip(1.0 - ae, 0.2, 0.9)
        skew = self.base_skew * ae_factor
        if inv_frac > 0:
            bid_mult = np.clip(1.0 - skew * inv_frac, 0.0, 1.0)
            ask_mult = np.clip(1.0 + skew * inv_frac, 1.0, 2.0)
        else:
            bid_mult = np.clip(1.0 + skew * abs(inv_frac), 1.0, 2.0)
            ask_mult = np.clip(1.0 - skew * abs(inv_frac), 0.0, 1.0)
        if inv_frac > self.hard_stop_thresh: bid_mult = 0.0
        elif inv_frac > self.suppress_thresh: bid_mult *= 0.3
        if inv_frac < -self.hard_stop_thresh: ask_mult = 0.0
        elif inv_frac < -self.suppress_thresh: ask_mult *= 0.3
        shade_bps = skew * abs(inv_frac) * 2.0
        if inv_frac > 0:
            return bid_mult, ask_mult, +shade_bps, -shade_bps
        else:
            return bid_mult, ask_mult, -shade_bps, +shade_bps


# ===========================================================================
# Backtest Engine (with DNA tracking)
# ===========================================================================

class Backtest:
    def __init__(self, controller=None):
        self.ctrl = controller
        self.cash = 0.0; self.inventory = 0
        self.records = []
        self.quotes = 0; self.fills = 0
        self.state_fills = defaultdict(int); self.state_quotes = defaultdict(int)

    def equity(self, mid): return self.cash + self.inventory * mid

    def record_fill(self, side, price, size, state):
        if side == 'bid': self.cash -= price * size; self.inventory += size
        else: self.cash += price * size; self.inventory -= size
        self.fills += 1; self.state_fills[state] += 1

    def simulate_fills(self, state, mid):
        base_p = FILL_BY_STATE.get(state, 0.0)
        spread = STATE_SPREAD.get(state, 50.0)
        if self.ctrl:
            bm, am, bs, as_ = self.ctrl.skew_factors(self.inventory, state)
            p_bid = base_p * bm; p_ask = base_p * am
            bid_px = mid - spread/2.0 + bs/10000.0 * mid
            ask_px = mid + spread/2.0 + as_/10000.0 * mid
        else:
            p_bid = base_p; p_ask = base_p
            bid_px = mid - spread/2.0; ask_px = mid + spread/2.0
        fills = []
        for _ in range(WINDOW_SIZE):
            if np.random.random() < p_bid:
                fills.append(('bid', bid_px))
            if np.random.random() < p_ask:
                fills.append(('ask', ask_px))
        return fills

    def snapshot(self, w_idx, mid, day):
        self.records.append({"w_idx": w_idx, "equity": self.equity(mid),
                             "cash": self.cash, "inventory": self.inventory,
                             "mid": mid, "day": day})


def run_strategy(bt, windows, state_seq, window_mids, window_day, oos_only=False):
    """Run ECORE+ETE strategy. If oos_only, skip windows before oos_start_idx."""
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        t_seq = state_seq[te_s:te_e]; t_mid = window_mids[te_s:te_e]
        t_day = window_day[te_s:te_e]
        if len(t_seq) < 10: continue

        # Determine if this test window is in OOS
        test_days = np.unique(t_day)
        is_oos = np.all(test_days >= oos_start_day)
        is_train = np.all(test_days < oos_start_day)

        if oos_only and not is_oos:
            # Still need to run fills to maintain inventory state continuity
            # but don't record results from train windows
            continue

        in_pos = False; pos_st = None
        for t in range(len(t_seq)):
            cs = t_seq[t]; cm = t_mid[t]; cd = t_day[t]; cq = cs.split("_")[1]
            if t > 0:
                ps = t_seq[t-1]; pq = ps.split("_")[1]
                if not in_pos and pq == "q1" and cq == "q2" and cs in ECORE:
                    in_pos = True; pos_st = cs
                elif in_pos and (cq != "q2" or cs not in ECORE):
                    in_pos = False; pos_st = None
            if in_pos and pos_st in FILL_BY_STATE:
                fills = bt.simulate_fills(pos_st, cm)
                for side, px in fills:
                    bt.record_fill(side, px, SIZE_PER_FILL, pos_st)
                bt.quotes += WINDOW_SIZE; bt.state_quotes[pos_st] += WINDOW_SIZE
            bt.snapshot(te_s + t, cm, cd)


def run_always_on(bt, windows, state_seq, window_mids, window_day, oos_only=False):
    """Always-On baseline: quote in all q2 states + q1 with lower fill."""
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        t_seq = state_seq[te_s:te_e]; t_mid = window_mids[te_s:te_e]
        t_day = window_day[te_s:te_e]
        if len(t_seq) < 10: continue

        test_days = np.unique(t_day)
        is_oos = np.all(test_days >= oos_start_day)
        if oos_only and not is_oos:
            continue

        for t in range(len(t_seq)):
            cs = t_seq[t]; cm = t_mid[t]; cd = t_day[t]; cq = cs.split("_")[1]
            if cq == "q2" and cs in FILL_BY_STATE:
                fills = bt.simulate_fills(cs, cm)
                for side, px in fills:
                    bt.record_fill(side, px, SIZE_PER_FILL, cs)
                bt.quotes += WINDOW_SIZE; bt.state_quotes[cs] += WINDOW_SIZE
            elif cq == "q1":
                spread = STATE_SPREAD.get(cs, 50) * 0.6
                p_fill = 0.30
                bid_px = cm - spread/2.0; ask_px = cm + spread/2.0
                for _ in range(WINDOW_SIZE):
                    if np.random.random() < p_fill:
                        bt.record_fill('bid', bid_px, SIZE_PER_FILL, cs)
                    if np.random.random() < p_fill:
                        bt.record_fill('ask', ask_px, SIZE_PER_FILL, cs)
                bt.quotes += WINDOW_SIZE; bt.state_quotes[cs] += WINDOW_SIZE
            bt.snapshot(te_s + t, cm, cd)


# ===========================================================================
# DNA decomposition
# ===========================================================================

def dna_decompose(records):
    n = len(records)
    total_pnl = np.zeros(n - 1)
    inv_pnl = np.zeros(n - 1)
    exec_pnl = np.zeros(n - 1)
    mid_returns = np.zeros(n - 1)
    inventories = np.array([r["inventory"] for r in records])
    mids = np.array([r["mid"] for r in records])
    days = np.array([r["day"] for r in records])
    for t in range(1, n):
        total_pnl[t-1] = records[t]["equity"] - records[t-1]["equity"]
        inv_pnl[t-1] = inventories[t-1] * (mids[t] - mids[t-1])
        exec_pnl[t-1] = total_pnl[t-1] - inv_pnl[t-1]
        mid_returns[t-1] = mids[t] - mids[t-1]
    return {"total_pnl": total_pnl, "inv_pnl": inv_pnl, "exec_pnl": exec_pnl,
            "mid_returns": mid_returns, "inventories": inventories,
            "mids": mids, "days": days}


def compute_daily_pnl(window_pnl, days):
    daily = defaultdict(float)
    for i, pnl in enumerate(window_pnl):
        daily[days[i + 1]] += pnl
    return np.array(list(daily.values()))


def daily_sharpe(daily_pnl):
    mu = np.mean(daily_pnl); std = np.std(daily_pnl)
    return mu / max(std, 1e-8) * np.sqrt(252)


# ===========================================================================
# [1] Run OOS — ECORE+ETE+IECORE
# ===========================================================================
print("\n" + "=" * 70)
print("  [1] OOS — ECORE+ETE+IECORE (Moderate, frozen)")
print("=" * 70)

TRAIN_W, TEST_W = 20, 5
n_windows = (n_days - TRAIN_W) // TEST_W

windows = []
for wi in range(n_windows):
    tsd = wi * TEST_W; ted = tsd + TRAIN_W; tesd = ted
    teed = min(tesd + TEST_W, n_days)
    tsw = 0 if tsd == 0 else day_bounds[tsd - 1]
    tew = day_bounds[min(ted - 1, len(day_bounds) - 1)]
    tesw = tew; teew = day_bounds[min(teed - 1, len(day_bounds) - 1)] if teed <= n_days else N_seq
    windows.append((tsw, tew, tesw, min(teew, N_seq)))

# Identify OOS windows
oos_windows = []
train_windows = []
for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
    if te_s < oos_start_idx: continue
    t_day = window_day[te_s:min(te_e, N_seq)]
    if len(t_day) > 0 and np.any(t_day >= oos_start_day):
        oos_windows.append(wi)
    elif len(t_day) > 0:
        train_windows.append(wi)

print(f"  Train test-windows: {len(train_windows)}  |  OOS test-windows: {len(oos_windows)}")

ctrl_mod = InventoryController(max_inv=MAX_INVENTORY, base_skew=0.5,
                               suppress_thresh=0.7, hard_stop_thresh=0.9)

# Run ECORE+IECORE
t0 = time.perf_counter()
bt_ecore = Backtest(controller=ctrl_mod)
run_strategy(bt_ecore, windows, state_seq, window_mids, window_day, oos_only=False)

# Filter to OOS windows only
oos_records = [r for r in bt_ecore.records if r["day"] >= oos_start_day]
print(f"  ECORE OOS records: {len(oos_records):,}  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Run Always-On Baseline (OOS)
# ===========================================================================
print("\n" + "=" * 70)
print("  [2] OOS — Always-On Baseline")
print("=" * 70)

t0 = time.perf_counter()
bt_always = Backtest(controller=ctrl_mod)  # same inventory control for fair comparison
run_always_on(bt_always, windows, state_seq, window_mids, window_day, oos_only=False)

ao_records = [r for r in bt_always.records if r["day"] >= oos_start_day]
print(f"  Always-On OOS records: {len(ao_records):,}  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] TASK B — OOS Metrics
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK B — OOS Metrics")
print("=" * 70)

for name, records in [("ECORE+IECORE", oos_records), ("Always-On", ao_records)]:
    if len(records) < 2: continue
    d = dna_decompose(records)

    daily_total = compute_daily_pnl(d["total_pnl"], d["days"])
    daily_exec = compute_daily_pnl(d["exec_pnl"], d["days"])

    cum_total = np.cumsum(daily_total)
    cum_exec = np.cumsum(daily_exec)
    dd_total = np.max(np.maximum.accumulate(cum_total) - cum_total)
    dd_exec = np.max(np.maximum.accumulate(cum_exec) - cum_exec)

    sh_total = daily_sharpe(daily_total)
    sh_exec = daily_sharpe(daily_exec)

    total_sum = float(np.sum(d["total_pnl"]))
    exec_sum = float(np.sum(d["exec_pnl"]))
    inv_sum = float(np.sum(d["inv_pnl"]))

    invs = d["inventories"]
    inv_std = float(np.std(invs))

    n_fills = sum(1 for r in records for _ in [0])  # not direct, use bt
    if name == "ECORE+IECORE":
        fills = bt_ecore.fills; quotes = bt_ecore.quotes
    else:
        fills = bt_always.fills; quotes = bt_always.quotes

    print(f"\n  {name}:")
    print(f"    Total PnL:           {total_sum:>+16,.0f}")
    print(f"    Execution PnL:       {exec_sum:>+16,.0f}")
    print(f"    Inventory PnL:       {inv_sum:>+16,.0f}")
    print(f"    Daily Sharpe (total): {sh_total:>16.2f}")
    print(f"    Daily Sharpe (exec):  {sh_exec:>16.2f}")
    print(f"    Max DD (total):      {dd_total:>16,.0f}")
    print(f"    Max DD (exec):       {dd_exec:>16,.0f}")
    print(f"    Inventory std:       {inv_std:>16,.0f}")
    print(f"    Total fills:         {fills:>16,d}")
    print(f"    Quotes placed:       {quotes:>16,d}")
    print(f"    PnL / fill:          {total_sum/max(fills,1):>+16.1f}")
    print(f"    PnL / quote:         {total_sum/max(quotes,1):>+16.3f}")
    print(f"    Fill efficiency:     {fills/max(quotes,1):>15.1%}")
    print(f"    OOS trading days:    {len(daily_total):>14,d}")


# ===========================================================================
# [4] TASK C — DNA on OOS
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK C — DNA Re-run on OOS")
print("=" * 70)

for name, records in [("ECORE+IECORE", oos_records), ("Always-On", ao_records)]:
    if len(records) < 2: continue
    d = dna_decompose(records)
    total_sum = float(np.sum(d["total_pnl"]))
    exec_sum = float(np.sum(d["exec_pnl"]))
    inv_sum = float(np.sum(d["inv_pnl"]))
    exec_pct = exec_sum / max(total_sum, 1) * 100
    inv_pct = inv_sum / max(total_sum, 1) * 100

    # Beta regression
    mid_rets = d["mid_returns"]
    total_pnl = d["total_pnl"]
    X = np.column_stack([np.ones(len(mid_rets)), mid_rets])
    beta_hat = np.linalg.lstsq(X, total_pnl, rcond=None)[0]
    residuals = total_pnl - (beta_hat[0] + beta_hat[1] * mid_rets)
    r2 = 1 - np.var(residuals) / max(np.var(total_pnl), 1e-8)

    daily_total = compute_daily_pnl(total_pnl, d["days"])
    daily_exec = compute_daily_pnl(d["exec_pnl"], d["days"])

    print(f"\n  {name}:")
    print(f"    Execution PnL:       {exec_sum:>+16,.0f}  ({exec_pct:+.0f}%)")
    print(f"    Inventory PnL:       {inv_sum:>+16,.0f}  ({inv_pct:+.0f}%)")
    print(f"    Beta (window):       {beta_hat[1]:>+16.1f}")
    print(f"    R^2 with mid:        {r2:>15.1%}")
    print(f"    Daily Sharpe (exec):  {daily_sharpe(daily_exec):>16.2f}")
    print(f"    Daily Sharpe (total): {daily_sharpe(daily_total):>16.2f}")


# ===========================================================================
# [5] TASK D — ECORE Stability on OOS
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK D — ECORE Stability on OOS")
print("=" * 70)

# Compare state distributions: train vs OOS
train_states = defaultdict(int)
oos_states = defaultdict(int)
for i, s in enumerate(state_seq):
    if window_day[i] < oos_start_day:
        train_states[s] += 1
    else:
        oos_states[s] += 1

# ECORE frequency
train_ecore_pct = sum(train_states[s] for s in ECORE) / max(sum(train_states.values()), 1)
oos_ecore_pct = sum(oos_states[s] for s in ECORE) / max(sum(oos_states.values()), 1)

print(f"\n  ECORE occupancy:")
print(f"    Train:               {train_ecore_pct:>14.1%}")
print(f"    OOS:                 {oos_ecore_pct:>14.1%}")
print(f"    Ratio OOS/Train:     {oos_ecore_pct/max(train_ecore_pct,1e-8):.2f}x")

# R6 attractor check
r6_states = [s for s in ECORE if s.startswith("R6")]
r6_train = sum(train_states[s] for s in r6_states) / max(sum(train_states[s] for s in ECORE), 1)
r6_oos = sum(oos_states[s] for s in r6_states) / max(sum(oos_states[s] for s in ECORE), 1)
print(f"\n  R6 fraction of ECORE:")
print(f"    Train:               {r6_train:>14.1%}")
print(f"    OOS:                 {r6_oos:>14.1%}")

# q2 width
train_q2 = sum(train_states[s] for s in train_states if "_q2_" in s) / max(sum(train_states.values()), 1)
oos_q2 = sum(oos_states[s] for s in oos_states if "_q2_" in s) / max(sum(oos_states.values()), 1)
print(f"\n  q2 (wide spread) fraction:")
print(f"    Train:               {train_q2:>14.1%}")
print(f"    OOS:                 {oos_q2:>14.1%}")

# Top ECORE states
print(f"\n  Top ECORE states (OOS):")
print(f"  {'State':<14s} {'Train':>8s} {'OOS':>8s} {'Ratio':>8s}")
print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8}")
sorted_ecore = sorted([s for s in ECORE if s in oos_states],
                      key=lambda s: oos_states[s], reverse=True)
for s in sorted_ecore[:10]:
    tr_n = train_states.get(s, 0); oos_n = oos_states.get(s, 0)
    tr_pct = tr_n / max(sum(train_states.values()), 1) * 100
    oos_pct = oos_n / max(sum(oos_states.values()), 1) * 100
    ratio = oos_pct / max(tr_pct, 1e-8)
    print(f"  {s:<14s} {tr_pct:>7.1f}% {oos_pct:>7.1f}% {ratio:>7.2f}x")


# ===========================================================================
# [6] TASK E — Unit Efficiency Comparison
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK E — Unit Efficiency (OOS)")
print("=" * 70)

eo_fills = bt_ecore.fills; eo_quotes = max(bt_ecore.quotes, 1)
ao_fills = bt_always.fills; ao_quotes = max(bt_always.quotes, 1)

# Only count OOS
eo_oos_records = len(oos_records)
ao_oos_records = len(ao_records)

# Use DNA results for OOS PnL
eo_d = dna_decompose(oos_records)
ao_d = dna_decompose(ao_records)
eo_pnl = float(np.sum(eo_d["total_pnl"]))
ao_pnl = float(np.sum(ao_d["total_pnl"]))

print(f"\n  {'Metric':<28s} {'ECORE+IECORE':>16s} {'Always-On':>16s} {'Ratio':>10s}")
print(f"  {'─'*28} {'─'*16} {'─'*16} {'─'*10}")

metrics = [
    ("Total PnL", eo_pnl, ao_pnl),
    ("Execution PnL", float(np.sum(eo_d["exec_pnl"])), float(np.sum(ao_d["exec_pnl"]))),
    ("Inventory PnL", float(np.sum(eo_d["inv_pnl"])), float(np.sum(ao_d["inv_pnl"]))),
    ("PnL / fill", eo_pnl/max(eo_fills,1), ao_pnl/max(ao_fills,1)),
    ("PnL / quote", eo_pnl/eo_quotes, ao_pnl/ao_quotes),
    ("Fill efficiency", eo_fills/eo_quotes, ao_fills/ao_quotes),
    ("Inv std", float(np.std(eo_d["inventories"])), float(np.std(ao_d["inventories"]))),
    ("Daily Sharpe (total)", daily_sharpe(compute_daily_pnl(eo_d["total_pnl"], eo_d["days"])),
                            daily_sharpe(compute_daily_pnl(ao_d["total_pnl"], ao_d["days"]))),
    ("Daily Sharpe (exec)", daily_sharpe(compute_daily_pnl(eo_d["exec_pnl"], eo_d["days"])),
                           daily_sharpe(compute_daily_pnl(ao_d["exec_pnl"], ao_d["days"]))),
]

for label, ev, bv in metrics:
    if abs(bv) > 1e-8:
        print(f"  {label:<28s} {ev:>16,.1f} {bv:>16,.1f} {ev/bv:>9.2f}x")
    else:
        print(f"  {label:<28s} {ev:>16,.1f} {bv:>16,.1f} {'—':>10s}")


# ===========================================================================
# [7] TASK F — Failure/Degradation Analysis
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK F — Degradation Analysis")
print("=" * 70)

# Compare OOS vs in-sample metrics
# We need in-sample metrics for comparison. Use train-day records.
train_eo_records = [r for r in bt_ecore.records if r["day"] < oos_start_day]
train_ao_records = [r for r in bt_always.records if r["day"] < oos_start_day]

if len(train_eo_records) > 2:
    eo_train_d = dna_decompose(train_eo_records)
    eo_train_exec = float(np.sum(eo_train_d["exec_pnl"]))
    eo_train_total = float(np.sum(eo_train_d["total_pnl"]))
    eo_train_daily = compute_daily_pnl(eo_train_d["total_pnl"], eo_train_d["days"])
    eo_train_sharpe = daily_sharpe(eo_train_daily)

    eo_oos_exec = float(np.sum(eo_d["exec_pnl"]))
    eo_oos_total = float(np.sum(eo_d["total_pnl"]))
    eo_oos_daily = compute_daily_pnl(eo_d["total_pnl"], eo_d["days"])
    eo_oos_sharpe = daily_sharpe(eo_oos_daily)

    degradation = {
        "total_pnl": (eo_oos_total - eo_train_total) / max(abs(eo_train_total), 1),
        "exec_pnl": (eo_oos_exec - eo_train_exec) / max(abs(eo_train_exec), 1),
        "sharpe": (eo_oos_sharpe - eo_train_sharpe) / max(abs(eo_train_sharpe), 1),
    }

    print(f"\n  ECORE+IECORE — In-sample vs OOS:")
    print(f"  {'Metric':<25s} {'In-Sample':>16s} {'OOS':>16s} {'Change':>10s}")
    print(f"  {'─'*25} {'─'*16} {'─'*16} {'─'*10}")
    print(f"  {'Total PnL':<25s} {eo_train_total:>+16,.0f} {eo_oos_total:>+16,.0f} "
          f"{degradation['total_pnl']:>+9.0%}")
    print(f"  {'Execution PnL':<25s} {eo_train_exec:>+16,.0f} {eo_oos_exec:>+16,.0f} "
          f"{degradation['exec_pnl']:>+9.0%}")
    print(f"  {'Daily Sharpe (total)':<25s} {eo_train_sharpe:>16.2f} {eo_oos_sharpe:>16.2f} "
          f"{degradation['sharpe']:>+9.0%}")

    # Spread compression check
    train_mid_chg = np.std(eo_train_d["mid_returns"])
    oos_mid_chg = np.std(eo_d["mid_returns"])
    print(f"  {'Mid vol (window)':<25s} {train_mid_chg:>16.1f} {oos_mid_chg:>16.1f} "
          f"{(oos_mid_chg/max(train_mid_chg,1)-1):>+9.0%}")

    # State decay: which ECORE states changed most?
    print(f"\n  State Frequency Shift (biggest changes):")
    for s in sorted_ecore[:8]:
        tr_pct = train_states.get(s, 0) / max(sum(train_states.values()), 1) * 100
        oos_pct = oos_states.get(s, 0) / max(sum(oos_states.values()), 1) * 100
        change = oos_pct - tr_pct
        direction = "UP" if change > 0 else "DOWN"
        if abs(change) > 0.1:
            print(f"    {s:<14s} {tr_pct:>6.2f}% -> {oos_pct:>6.2f}%  ({change:>+.2f}pp) {direction}")


# ===========================================================================
# [8] FINAL OOS VERDICT
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL OOS VERDICT")
print("=" * 70)

eo_daily_t = compute_daily_pnl(eo_d["total_pnl"], eo_d["days"])
eo_daily_e = compute_daily_pnl(eo_d["exec_pnl"], eo_d["days"])
sh_t = daily_sharpe(eo_daily_t)
sh_e = daily_sharpe(eo_daily_e)
exec_pct = float(np.sum(eo_d["exec_pnl"])) / max(float(np.sum(eo_d["total_pnl"])), 1) * 100

print(f"\n  OOS Summary (ECORE+IECORE, {len(eo_daily_t)} days):")
print(f"    Total PnL:           {float(np.sum(eo_d['total_pnl'])):>+16,.0f}")
print(f"    Execution PnL:       {float(np.sum(eo_d['exec_pnl'])):>+16,.0f}  ({exec_pct:+.0f}%)")
print(f"    Daily Sharpe (total): {sh_t:>16.2f}")
print(f"    Daily Sharpe (exec):  {sh_e:>16.2f}")
print(f"    Hit ratio:           {np.mean(eo_daily_t > 0):>15.1%}")

# Verdict
neg_days = int(np.sum(eo_daily_t < 0))
total_days = len(eo_daily_t)

if sh_e > 2.0 and exec_pct > 70:
    verdict = "CASE_A — Execution edge survives frozen OOS. Strong delta-neutral performance."
elif sh_e > 1.0 and exec_pct > 50:
    verdict = "CASE_A — Edge survives OOS. Solid risk-adjusted returns."
elif sh_e > 0.5:
    verdict = "CASE_B — Edge weakens OOS but remains positive. Further validation needed."
elif sh_e > 0:
    verdict = "CASE_B — Marginal edge OOS. May not survive transaction costs."
else:
    verdict = "CASE_C — Edge collapses OOS. In-sample results were overfit."

print(f"\n  {verdict}")

print(f"\n  Key OOS Facts:")
print(f"    Days:               {total_days}")
print(f"    Negative days:      {neg_days}  ({neg_days/total_days*100:.1f}%)")
print(f"    ECORE occupancy:    {oos_ecore_pct:.1%}  (train: {train_ecore_pct:.1%})")
print(f"    Execution fraction: {exec_pct:.0f}%")
print(f"    Max DD (exec):      {float(np.max(np.maximum.accumulate(np.cumsum(eo_daily_e)) - np.cumsum(eo_daily_e))):,.0f}")

print(f"\n{'═'*70}")
print(f"  FOOS v1 complete. Frozen OOS evaluation finished.")
print(f"{'═'*70}")

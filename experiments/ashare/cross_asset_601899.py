"""
X601899 v1 — Cross-Asset Friction Transfer

Frozen Hephaestus pipeline transferred to 601899 (Zijin Mining).
Compares edge thickness: spread capture / friction cost.

All parameters frozen from 000333 calibration.
"""

import sys, time, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

# ── 601899 data ──
TRAIN_DIR_601899 = r"c:\Users\ZaneLaw\Desktop\601899\RawTrainData"
TEST_DIR_601899 = r"c:\Users\ZaneLaw\Desktop\601899\RawTestData"
WINDOW_SIZE = 100; N_REGIMES = 8
TRAIN_DAYS = 40

ECORE = {f"R{r}_q2_T{t}" for r in range(8) for t in range(3) if r != 2}

# FROZEN from 000333
FILL_BY_STATE = {
    "R0_q2_T0":0.840,"R0_q2_T1":0.850,"R0_q2_T2":0.854,
    "R1_q2_T0":0.497,"R1_q2_T1":0.493,"R1_q2_T2":0.719,
    "R3_q2_T0":0.780,"R3_q2_T1":0.806,"R3_q2_T2":0.780,
    "R4_q2_T0":0.812,"R4_q2_T1":0.868,"R4_q2_T2":0.862,
    "R5_q2_T0":0.796,"R5_q2_T1":0.819,"R5_q2_T2":0.819,
    "R6_q2_T0":0.807,"R6_q2_T1":0.837,"R6_q2_T2":0.836,
    "R7_q2_T0":0.782,"R7_q2_T1":0.801,"R7_q2_T2":0.887,
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
MARKOUT_BY_STATE = {
    "R0_q2_T0":-0.4,"R0_q2_T1":-0.3,"R0_q2_T2":-0.2,
    "R1_q2_T0":-1.4,"R1_q2_T1":-2.5,"R1_q2_T2":-0.5,
    "R3_q2_T0":-0.8,"R3_q2_T1":-0.5,"R3_q2_T2":-0.6,
    "R4_q2_T0":-0.6,"R4_q2_T1":-0.3,"R4_q2_T2":-0.1,
    "R5_q2_T0":-0.9,"R5_q2_T1":-0.3,"R5_q2_T2":-0.4,
    "R6_q2_T0":-0.6,"R6_q2_T1":-0.4,"R6_q2_T2":-0.3,
    "R7_q2_T0":-0.4,"R7_q2_T1":+0.5,"R7_q2_T2":-0.4,
}

MAX_INVENTORY = 50000; SIZE_PER_FILL = 100

print("=" * 70)
print("  X601899 v1 — Cross-Asset Friction Transfer")
print("=" * 70)


# ===========================================================================
# [0] Build state sequence for 601899
# ===========================================================================
print("\n[0] Building 601899 state sequence ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)

# Load all data (train + test)
def load_files(data_dir):
    msg_files = sorted(glob.glob(str(Path(data_dir) / "message_*.parquet")))
    ob_files = sorted(glob.glob(str(Path(data_dir) / "orderbook_*.parquet")))
    return msg_files, ob_files

tr_msg, tr_ob = load_files(TRAIN_DIR_601899)
te_msg, te_ob = load_files(TEST_DIR_601899)
all_msg = tr_msg + te_msg; all_ob = tr_ob + te_ob
n_days = len(all_msg)
n_train_days = len(tr_msg)

all_features = []; all_raw = []; day_bounds = []
for day_idx, (mf, of) in enumerate(zip(all_msg, all_ob)):
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

# Train regime clustering on first TRAIN_DAYS of train data
train_day_end = min(TRAIN_DAYS, len(day_bounds))
tr_e = day_bounds[train_day_end - 1]
X_tr = X_all[:tr_e]; X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

# Tox thresholds from train only
tox_train = []
for d in range(train_day_end):
    raw = all_raw[d]; v = raw["valid"]
    if v.sum() < 100: continue
    tox_train.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
sorted_tox = np.sort(np.array(tox_train))
p30, p70 = np.percentile(sorted_tox, [30, 70])

# Build state sequence
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
        window_mids.append(float(w_mid[w_valid][-1]) if w_valid.sum() > 0 else 17000.0)
        window_day.append(d)

N_seq = len(state_seq)
window_day = np.array(window_day, dtype=int)
oos_start_day = n_train_days  # OOS = RawTestData
print(f"  {N_seq:,} windows  |  Train: {n_train_days} days  |  OOS: {n_days - n_train_days} days")
print(f"  Tox p30={p30:.6f} p70={p70:.6f}  |  Mid range: {min(window_mids):.0f}-{max(window_mids):.0f}")
print(f"  time={time.perf_counter()-t0:.1f}s")

# ── Spread statistics (Task B) ──
all_spreads = []
q2_spreads = []
for d in range(n_days):
    raw = all_raw[d]
    v = raw["valid"]
    if v.sum() < 100: continue
    all_spreads.extend(raw["sp"][v].tolist())
all_spreads = np.array(all_spreads)

print(f"\n  Spread Stats (601899):")
print(f"    Mean spread:         {np.mean(all_spreads):>12.1f} raw units")
print(f"    Median spread:       {np.median(all_spreads):>12.1f}")
print(f"    P90 spread:          {np.percentile(all_spreads, 90):>12.1f}")
print(f"    Spread/price (bps):  {np.mean(all_spreads)/np.mean([m for m in window_mids])*10000:.2f} bps")


# ===========================================================================
# Inventory Controller (FROZEN)
# ===========================================================================

class InvCtrl:
    def __init__(self, max_inv=MAX_INVENTORY, base_skew=0.5, suppress_thresh=0.7, hard_stop_thresh=0.9):
        self.max_inv = max_inv; self.base_skew = base_skew
        self.suppress_thresh = suppress_thresh; self.hard_stop_thresh = hard_stop_thresh

    def skew(self, inv, state):
        f = np.clip(inv / self.max_inv, -1.0, 1.0)
        ae = STATE_AE.get(state, 0.5)
        sk = self.base_skew * np.clip(1.0 - ae, 0.2, 0.9)
        if f > 0:
            bm = np.clip(1.0 - sk * f, 0.0, 1.0); am = np.clip(1.0 + sk * f, 1.0, 2.0)
        else:
            bm = np.clip(1.0 + sk * abs(f), 1.0, 2.0); am = np.clip(1.0 - sk * abs(f), 0.0, 1.0)
        if f > self.hard_stop_thresh: bm = 0.0
        elif f > self.suppress_thresh: bm *= 0.3
        if f < -self.hard_stop_thresh: am = 0.0
        elif f < -self.suppress_thresh: am *= 0.3
        s = sk * abs(f) * 2.0
        return (bm, am, +s, -s) if f > 0 else (bm, am, -s, +s)


# ===========================================================================
# Friction Model (same as EFL-SRV)
# ===========================================================================

class FrictionModel:
    def __init__(self, fee_mult=0.0, slippage_bps=0.0):
        self.fee_mult = fee_mult; self.slippage_bps = slippage_bps
        self.commission_bps = 2.5; self.stamp_duty_bps = 5.0
        self.exchange_fee_bps = 0.5; self.transfer_fee_bps = 0.2

    def fee_cost(self, side, notional):
        bps = (self.commission_bps + self.exchange_fee_bps + self.transfer_fee_bps) * self.fee_mult
        if side == 'ask': bps += self.stamp_duty_bps * self.fee_mult
        return notional * bps / 10000.0

    def apply_slippage(self, px, side):
        if self.slippage_bps == 0: return px
        f = 1.0 + self.slippage_bps / 10000.0
        return px * f if side == 'bid' else px / f


# ===========================================================================
# Backtest
# ===========================================================================

class Backtest:
    def __init__(self, friction, controller=None):
        self.fric = friction; self.ctrl = controller
        self.cash = 0.0; self.inventory = 0
        self.records = []; self.quotes = 0; self.fills = 0
        self.total_fees = 0.0; self.total_slippage = 0.0

    def equity(self, mid): return self.cash + self.inventory * mid

    def record_fill(self, side, px, size, state):
        spx = self.fric.apply_slippage(px, side)
        notional = spx * size
        fee = self.fric.fee_cost(side, notional)
        if side == 'bid': self.cash -= spx * size + fee; self.inventory += size
        else: self.cash += spx * size - fee; self.inventory -= size
        self.total_fees += fee; self.total_slippage += abs(spx - px) * size
        self.fills += 1

    def simulate_fills(self, state, mid):
        base_p = FILL_BY_STATE.get(state, 0.0)
        spread = STATE_SPREAD.get(state, 50.0)
        if self.ctrl:
            bm, am, bs, as_ = self.ctrl.skew(self.inventory, state)
            p_bid = base_p * bm; p_ask = base_p * am
            bid_px = mid - spread/2.0 + bs/10000.0 * mid
            ask_px = mid + spread/2.0 + as_/10000.0 * mid
        else:
            p_bid = p_ask = base_p
            bid_px = mid - spread/2.0; ask_px = mid + spread/2.0
        fills = []
        for _ in range(WINDOW_SIZE):
            if np.random.random() < p_bid: fills.append(('bid', bid_px))
            if np.random.random() < p_ask: fills.append(('ask', ask_px))
        return fills

    def snapshot(self, w_idx, mid, day):
        self.records.append({"w_idx": w_idx, "equity": self.equity(mid),
                             "cash": self.cash, "inventory": self.inventory,
                             "mid": mid, "day": day})


def dna_decompose(records):
    n = len(records); total = np.zeros(n-1); inv = np.zeros(n-1); exec_ = np.zeros(n-1)
    invs = np.array([r["inventory"] for r in records])
    mids = np.array([r["mid"] for r in records])
    days = np.array([r["day"] for r in records])
    for t in range(1, n):
        total[t-1] = records[t]["equity"] - records[t-1]["equity"]
        inv[t-1] = invs[t-1] * (mids[t] - mids[t-1])
        exec_[t-1] = total[t-1] - inv[t-1]
    return {"total": total, "inv": inv, "exec": exec_, "invs": invs, "days": days}


def daily_pnl(wpnl, days):
    d = defaultdict(float)
    for i, p in enumerate(wpnl): d[days[i+1]] += p
    return np.array(list(d.values()))


def sharpe(daily): return np.mean(daily) / max(np.std(daily), 1e-8) * np.sqrt(252)


def run_strategy(bt, windows, oos_only=False):
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        t_seq = state_seq[te_s:te_e]; t_mid = window_mids[te_s:te_e]
        t_day = window_day[te_s:te_e]
        if len(t_seq) < 10: continue
        test_days = np.unique(t_day)
        is_oos = np.all(test_days >= oos_start_day)
        if oos_only and not is_oos: continue
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
                bt.quotes += WINDOW_SIZE
            bt.snapshot(te_s + t, cm, cd)


# ===========================================================================
# Windows
# ===========================================================================
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

ctrl = InvCtrl()

# ===========================================================================
# [1] Run — Baseline (no friction)
# ===========================================================================
print("\n" + "=" * 70)
print("  [1] Baseline (no friction) — Full run")
print("=" * 70)

t0 = time.perf_counter()
bt_base = Backtest(FrictionModel(fee_mult=0, slippage_bps=0), controller=ctrl)
run_strategy(bt_base, windows, oos_only=False)
d_base = dna_decompose(bt_base.records)
base_total = float(np.sum(d_base["total"]))
base_exec = float(np.sum(d_base["exec"]))
base_inv = float(np.sum(d_base["inv"]))
base_daily = daily_pnl(d_base["total"], d_base["days"])
base_sh = sharpe(base_daily)
base_inv_std = float(np.std(d_base["invs"]))
print(f"  Total: {base_total:>+14,.0f}  Exec: {base_exec:>+14,.0f}  "
      f"Sh: {base_sh:.1f}  InvStd: {base_inv_std:,.0f}  "
      f"Fills: {bt_base.fills:,}  Quotes: {bt_base.quotes:,}  "
      f"time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [2] Run — Realistic A-Share Fees
# ===========================================================================
print("\n" + "=" * 70)
print("  [2] Realistic A-Share Fees")
print("=" * 70)

t0 = time.perf_counter()
bt_fee = Backtest(FrictionModel(fee_mult=1.0, slippage_bps=1.0), controller=ctrl)
run_strategy(bt_fee, windows, oos_only=False)
d_fee = dna_decompose(bt_fee.records)
fee_total = float(np.sum(d_fee["total"]))
fee_exec = float(np.sum(d_fee["exec"]))
fee_inv = float(np.sum(d_fee["inv"]))
fee_daily = daily_pnl(d_fee["total"], d_fee["days"])
fee_sh = sharpe(fee_daily)

# Edge thickness metrics
avg_mid = np.mean([r["mid"] for r in bt_fee.records])
avg_spread_raw = np.mean([STATE_SPREAD.get(s, 50) for s in STATE_SPREAD])
per_fill_spread_capture = avg_spread_raw * SIZE_PER_FILL / 2  # half spread
per_fill_notional = avg_mid * SIZE_PER_FILL
per_fill_fee_buy = per_fill_notional * 3.2 / 10000.0
per_fill_fee_sell = per_fill_notional * 8.2 / 10000.0
avg_fee_per_fill = (per_fill_fee_buy + per_fill_fee_sell) / 2

edge_thickness_ratio = per_fill_spread_capture / max(avg_fee_per_fill, 1)

print(f"  Total: {fee_total:>+14,.0f}  Exec: {fee_exec:>+14,.0f}  Sh: {fee_sh:.1f}")
print(f"  Fees paid: {bt_fee.total_fees:>14,.0f}  Slippage: {bt_fee.total_slippage:>14,.0f}")
print(f"  Fills: {bt_fee.fills:,}  Quotes: {bt_fee.quotes:,}  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [3] OOS Only
# ===========================================================================
print("\n" + "=" * 70)
print("  [3] OOS Only (last 22 days, frozen) — Realistic Fees")
print("=" * 70)

t0 = time.perf_counter()
bt_oos = Backtest(FrictionModel(fee_mult=1.0, slippage_bps=1.0), controller=ctrl)
run_strategy(bt_oos, windows, oos_only=True)
d_oos = dna_decompose(bt_oos.records)
oos_total = float(np.sum(d_oos["total"]))
oos_exec = float(np.sum(d_oos["exec"]))
oos_daily = daily_pnl(d_oos["total"], d_oos["days"])
oos_sh = sharpe(oos_daily)
oos_neg = int(np.sum(oos_daily < 0))
print(f"  OOS Total: {oos_total:>+14,.0f}  Exec: {oos_exec:>+14,.0f}  "
      f"Sh: {oos_sh:.1f}  Days: {len(oos_daily)}  NegDays: {oos_neg}  "
      f"time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [4] COMPARISON: 000333 vs 601899
# ===========================================================================
print("\n" + "=" * 70)
print("  EDGE THICKNESS COMPARISON: 000333 vs 601899")
print("=" * 70)

# 000333 numbers (from EFL-SRV)
# avg_mid ~798,500 (raw), spread ~154 raw (R6_q2_T0)
mid_333 = 798500
spread_333 = np.mean(list(STATE_SPREAD.values()))
per_fill_cap_333 = spread_333 * 100 / 2  # half-spread * 100 shares
per_fill_notional_333 = mid_333 * 100
fee_buy_333 = per_fill_notional_333 * 3.2 / 10000
fee_sell_333 = per_fill_notional_333 * 8.2 / 10000
avg_fee_333 = (fee_buy_333 + fee_sell_333) / 2
ratio_333 = per_fill_cap_333 / avg_fee_333
spread_bps_333 = spread_333 / mid_333 * 10000

# 601899 numbers
mid_899 = avg_mid
spread_899 = avg_spread_raw
per_fill_cap_899 = per_fill_spread_capture
per_fill_notional_899 = per_fill_notional
fee_buy_899 = per_fill_fee_buy
fee_sell_899 = per_fill_fee_sell
avg_fee_899 = avg_fee_per_fill
spread_bps_899 = spread_899 / mid_899 * 10000

print(f"\n  {'Metric':<35s} {'000333':>16s} {'601899':>16s} {'Ratio':>10s}")
print(f"  {'─'*35} {'─'*16} {'─'*16} {'─'*10}")
print(f"  {'Avg Mid Price':<35s} {mid_333:>16,.0f} {mid_899:>16,.0f} {mid_899/mid_333:>9.2f}x")
print(f"  {'Avg Spread (raw)':<35s} {spread_333:>16.0f} {spread_899:>16.0f} {spread_899/max(spread_333,1):>9.2f}x")
print(f"  {'Spread (bps of mid)':<35s} {spread_bps_333:>15.2f} {spread_bps_899:>15.2f} {spread_bps_899/max(spread_bps_333,1e-8):>9.2f}x")
print(f"  {'Per-fill Spread Capture':<35s} {per_fill_cap_333:>16,.0f} {per_fill_cap_899:>16,.0f} {per_fill_cap_899/max(per_fill_cap_333,1):>9.2f}x")
print(f"  {'Per-fill Fee (buy)':<35s} {fee_buy_333:>16,.0f} {fee_buy_899:>16,.0f} {fee_buy_899/max(fee_buy_333,1):>9.2f}x")
print(f"  {'Per-fill Fee (sell)':<35s} {fee_sell_333:>16,.0f} {fee_sell_899:>16,.0f} {fee_sell_899/max(fee_sell_333,1):>9.2f}x")
print(f"  {'Avg Fee per Fill':<35s} {avg_fee_333:>16,.0f} {avg_fee_899:>16,.0f} {avg_fee_899/max(avg_fee_333,1):>9.2f}x")
print(f"  {'EDGE THICKNESS RATIO':<35s} {ratio_333:>15.3f} {edge_thickness_ratio:>15.3f} {edge_thickness_ratio/max(ratio_333,1e-8):>9.2f}x")
print(f"  {'Ratio > 1.0 = Viable':<35s} {str(ratio_333 > 1.0):>16s} {str(edge_thickness_ratio > 1.0):>16s}")

# ===========================================================================
# [5] ECORE Stability (Task F)
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK F — ECORE Stability on 601899")
print("=" * 70)

ecore_count = sum(1 for s in state_seq if s in ECORE)
q2_count = sum(1 for s in state_seq if "_q2_" in s)
ecore_pct = ecore_count / N_seq * 100
q2_pct = q2_count / N_seq * 100

r6_ecore = sum(1 for s in state_seq if s in ECORE and s.startswith("R6"))
r6_pct = r6_ecore / max(ecore_count, 1) * 100

print(f"\n  ECORE occupancy:      {ecore_pct:.1f}%  ({ecore_count:,} / {N_seq:,})")
print(f"  q2 (wide spread):     {q2_pct:.1f}%  ({q2_count:,})")
print(f"  R6 / ECORE:           {r6_pct:.1f}%  ({r6_ecore:,})")
print(f"  000333 comparison:    ECORE=7.8% train, 12.3% OOS")

# State frequency
print(f"\n  Top ECORE states (601899):")
state_counts = defaultdict(int)
for s in state_seq:
    if s in ECORE: state_counts[s] += 1
for s, n in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
    pct = n / N_seq * 100
    print(f"    {s:<14s} {n:>8,d}  ({pct:.2f}%)")


# ===========================================================================
# [6] FINAL VERDICT
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL VERDICT")
print("=" * 70)

print(f"\n  Edge Thickness Ratio (601899): {edge_thickness_ratio:.3f}")
print(f"  Edge Thickness Ratio (000333): {ratio_333:.3f}")
improvement = edge_thickness_ratio / max(ratio_333, 1e-8)

print(f"\n  Baseline (no friction):    {base_total:>+16,.0f}  Sh={base_sh:.1f}")
print(f"  Realistic Fees:            {fee_total:>+16,.0f}  Sh={fee_sh:.1f}")
print(f"  OOS Only (fees):           {oos_total:>+16,.0f}  Sh={oos_sh:.1f}")

if edge_thickness_ratio > 1.0:
    if improvement > 3:
        verdict = "CASE_A — 601899 edge is DRAMATICALLY thicker. Viable under retail fees."
    elif improvement > 1.5:
        verdict = "CASE_A — 601899 edge significantly thicker. Close to viability."
    else:
        verdict = "CASE_B — 601899 modestly better than 000333 but still marginal."
elif edge_thickness_ratio > 0.5:
    if improvement > 3:
        verdict = "CASE_B — Major improvement from 000333, but still not breakeven at retail fees."
    else:
        verdict = "CASE_B — Improvement over 000333, but friction still dominates."
else:
    verdict = "CASE_C — 601899 edge thickness similar to 000333. Friction dominates both."

print(f"\n  {verdict}")
print(f"\n  Improvement factor vs 000333: {improvement:.1f}x")
print(f"  Break-even ratio needed:    1.00")
print(f"  601899 current ratio:       {edge_thickness_ratio:.3f}")

print(f"\n{'═'*70}")
print(f"  X601899 v1 complete.")
print(f"{'═'*70}")

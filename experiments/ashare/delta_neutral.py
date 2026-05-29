"""
DNA v1 — Delta-Neutral Audit

Decomposes IECORE PnL into:
  Execution PnL — spread capture + fill-level edge
  Inventory PnL — directional drift on open inventory

Tasks:
  A. Inventory Bias Audit
  B. Beta Regression (PnL on mid returns)
  C. Delta-Neutral PnL construction
  D. Recompute metrics on execution-only PnL
  E. PnL decomposition
  F. Trend sensitivity by subperiod
  G. Long/short symmetry test
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
print("  DNA v1 — Delta-Neutral Audit")
print("=" * 70)


# ===========================================================================
# [0] Build state sequence + window mid prices + day mapping
# ===========================================================================
print("\n[0] Building state sequence ...")
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
TRAIN_WIN = 20
tr_e = day_bounds[min(TRAIN_WIN-1, len(day_bounds)-1)]
X_tr = X_all[:tr_e]; X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

tox_vals = []
for d in range(TRAIN_WIN):
    raw = all_raw[d]; v = raw["valid"]
    if v.sum() < 100: continue
    tox_vals.extend((raw["sp"][v] / np.maximum(raw["dp"][v],1e-8))[:30000].tolist())
sorted_tox = np.sort(np.array(tox_vals))
p30, p70 = np.percentile(sorted_tox, [30, 70])

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
print(f"  {N_seq:,} windows across {n_days} days  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Inventory Controller (Moderate — same as IECORE)
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
# Backtest with per-window tracking for DNA decomposition
# ===========================================================================

class DNABacktest:
    def __init__(self, controller=None):
        self.ctrl = controller
        self.cash = 0.0; self.inventory = 0
        self.records = []  # per-window: {w_idx, equity, cash, inventory, mid, day}

    def equity(self, mid):
        return self.cash + self.inventory * mid

    def record_fill(self, side, price, size):
        if side == 'bid':
            self.cash -= price * size; self.inventory += size
        else:
            self.cash += price * size; self.inventory -= size

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
        self.records.append({
            "w_idx": w_idx, "equity": self.equity(mid),
            "cash": self.cash, "inventory": self.inventory,
            "mid": mid, "day": day,
        })


# ===========================================================================
# [1] Run backtest — both Symmetric and Moderate
# ===========================================================================
print("\n[1] Running backtests for DNA decomposition ...")
t0 = time.perf_counter()

TRAIN_W, TEST_W = 20, 5
n_windows = (n_days - TRAIN_W) // TEST_W

windows = []
for wi in range(n_windows):
    tsd = wi * TEST_W; ted = tsd + TRAIN_W
    tesd = ted; teed = min(tesd + TEST_W, n_days)
    tsw = 0 if tsd == 0 else day_bounds[tsd - 1]
    tew = day_bounds[min(ted - 1, len(day_bounds) - 1)]
    tesw = tew; teew = day_bounds[min(teed - 1, len(day_bounds) - 1)] if teed <= n_days else N_seq
    windows.append((tsw, tew, tesw, min(teew, N_seq)))

controllers = {
    "Symmetric": None,
    "Moderate": InventoryController(max_inv=MAX_INVENTORY, base_skew=0.5,
                                    suppress_thresh=0.7, hard_stop_thresh=0.9),
}
all_data = {}

for ctrl_name, ctrl in controllers.items():
    bt = DNABacktest(controller=ctrl)
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        t_seq = state_seq[te_s:te_e]; t_mid = window_mids[te_s:te_e]
        t_day = window_day[te_s:te_e]
        if len(t_seq) < 10: continue
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
                    bt.record_fill(side, px, SIZE_PER_FILL)
            bt.snapshot(te_s + t, cm, cd)
    all_data[ctrl_name] = bt.records
    print(f"  {ctrl_name}: {len(bt.records)} windows")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# DNA Decomposition Engine
# ===========================================================================

def dna_decompose(records):
    """
    Decompose total PnL into execution and inventory components.

    Total PnL_t = (cash_t + inv_t * mid_t) - (cash_{t-1} + inv_{t-1} * mid_{t-1})
    Inventory PnL_t = inv_{t-1} * (mid_t - mid_{t-1})
    Execution PnL_t = Total PnL_t - Inventory PnL_t
    """
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

    return {
        "total_pnl": total_pnl,
        "inv_pnl": inv_pnl,
        "exec_pnl": exec_pnl,
        "mid_returns": mid_returns,
        "inventories": inventories,
        "mids": mids,
        "days": days,
    }


def compute_sharpe(pnl_series, periods_per_year=252):
    """Annualized Sharpe from daily PnL."""
    return np.mean(pnl_series) / max(np.std(pnl_series), 1e-8) * np.sqrt(periods_per_year)


def compute_daily_pnl(window_pnl, days, day_offset=0):
    """Aggregate window-level PnL to daily."""
    daily = defaultdict(float)
    for i, pnl in enumerate(window_pnl):
        d = days[i + day_offset]
        daily[d] += pnl
    return np.array(list(daily.values()))


# ===========================================================================
# [2] TASK A — Inventory Bias Audit
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK A — Inventory Bias Audit")
print("=" * 70)

for name, records in all_data.items():
    inv = np.array([r["inventory"] for r in records])
    print(f"\n  {name}:")
    print(f"    Mean inventory:      {np.mean(inv):>+14,.0f}")
    print(f"    Median inventory:    {np.median(inv):>+14,.0f}")
    print(f"    Std inventory:       {np.std(inv):>14,.0f}")
    print(f"    Skewness:            {float(np.mean(((inv-np.mean(inv))/max(np.std(inv),1))**3)):>14.2f}")
    print(f"    Max long:            {np.max(inv):>14,.0f}")
    print(f"    Max short:           {np.min(inv):>14,.0f}")
    pct_long = np.mean(inv > 0)
    pct_short = np.mean(inv < 0)
    print(f"    Pct time long:       {pct_long:>13.1%}")
    print(f"    Pct time short:      {pct_short:>13.1%}")
    print(f"    Pct time flat:       {np.mean(inv == 0):>13.1%}")

    # Cumulative inventory path trend
    cum_inv = np.cumsum(inv) / len(inv)
    print(f"    Cumulative trend:    {cum_inv[-1]:>+14.3f} / window")

    # Directional bias flag
    mean_abs = np.mean(np.abs(inv))
    if abs(np.mean(inv)) > mean_abs * 0.3:
        direction = "NET LONG" if np.mean(inv) > 0 else "NET SHORT"
        print(f"    ** BIAS DETECTED: {direction} **")
    else:
        print(f"    No significant directional bias")


# ===========================================================================
# [3] TASK B — Beta Regression
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK B — Beta Regression (PnL on mid returns)")
print("=" * 70)

for name, records in all_data.items():
    d = dna_decompose(records)

    # Window-level regression: PnL_t = alpha + beta * mid_return_t + eps
    # The beta should equal mean inventory if PnL is purely directional
    mid_rets = d["mid_returns"]
    total_pnl = d["total_pnl"]
    mean_inv = np.mean(d["inventories"][:-1])  # lagged inventory

    # OLS: PnL = alpha + beta * delta_mid
    X = np.column_stack([np.ones(len(mid_rets)), mid_rets])
    beta_hat = np.linalg.lstsq(X, total_pnl, rcond=None)[0]
    alpha_win, beta_win = beta_hat[0], beta_hat[1]

    residuals = total_pnl - (alpha_win + beta_win * mid_rets)
    r2_win = 1 - np.var(residuals) / max(np.var(total_pnl), 1e-8)
    tstat_beta = beta_win / max(np.std(residuals) / np.sqrt(len(mid_rets)) / np.std(mid_rets), 1e-8)

    # Daily-level regression
    daily_total = compute_daily_pnl(total_pnl, d["days"])
    daily_mid = compute_daily_pnl(mid_rets, d["days"])
    X_daily = np.column_stack([np.ones(len(daily_total)), daily_mid])
    beta_daily = np.linalg.lstsq(X_daily, daily_total, rcond=None)[0]
    alpha_day, beta_day = beta_daily[0], beta_daily[1]
    residuals_d = daily_total - (alpha_day + beta_day * daily_mid)
    r2_day = 1 - np.var(residuals_d) / max(np.var(daily_total), 1e-8)

    # Execution-only regression (PnL stripped of inventory drift)
    exec_pnl = d["exec_pnl"]
    Xe = np.column_stack([np.ones(len(mid_rets)), mid_rets])
    beta_exec = np.linalg.lstsq(Xe, exec_pnl, rcond=None)[0]
    alpha_exec, beta_exec = beta_exec[0], beta_exec[1]
    residuals_e = exec_pnl - (alpha_exec + beta_exec * mid_rets)
    r2_exec = 1 - np.var(residuals_e) / max(np.var(exec_pnl), 1e-8)

    print(f"\n  {name} — Window-level ({len(total_pnl):,} obs):")
    print(f"    alpha (execution PnL):  {alpha_win:>+14,.0f}")
    print(f"    beta (mid sensitivity): {beta_win:>+14,.0f}")
    print(f"    Mean inventory:      {mean_inv:>14,.0f}")
    print(f"    R^2:                  {r2_win:>14.1%}")
    print(f"    t-stat (beta):          {tstat_beta:>14.2f}")
    if abs(beta_win / max(mean_inv, 1)) > 0.5:
        print(f"    beta ≈ mean_inventory → PnL is mostly directional")
    else:
        print(f"    beta ≠ mean_inventory → execution edge present")

    print(f"\n  {name} — Daily-level ({len(daily_total)} days):")
    print(f"    alpha (execution PnL):  {alpha_day:>+14,.0f}")
    print(f"    beta (mid sensitivity): {beta_day:>+14,.0f}")
    print(f"    R^2:                  {r2_day:>14.1%}")

    print(f"\n  {name} — Execution-only (delta-neutral) window-level:")
    print(f"    alpha (pure exec):       {alpha_exec:>+14,.0f}")
    print(f"    beta (residual):        {beta_exec:>+14,.0f}")
    print(f"    R^2 (with mid):       {r2_exec:>14.1%}")
    if r2_exec < 0.05:
        print(f"    Delta-neutralized: execution PnL uncorrelated with mid")


# ===========================================================================
# [4] TASK C+D — Delta-Neutral PnL + Recompute Metrics
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK C+D — Delta-Neutral Metrics")
print("=" * 70)

for name, records in all_data.items():
    d = dna_decompose(records)
    total = d["total_pnl"]
    exec_pnl = d["exec_pnl"]
    inv_pnl = d["inv_pnl"]
    days = d["days"]

    # Daily aggregation
    daily_total = compute_daily_pnl(total, days)
    daily_exec = compute_daily_pnl(exec_pnl, days)
    daily_inv = compute_daily_pnl(inv_pnl, days)

    # Compute metrics
    sharpe_total = compute_sharpe(daily_total)
    sharpe_exec = compute_sharpe(daily_exec)

    # Drawdown on total vs exec
    cum_total = np.cumsum(daily_total)
    cum_exec = np.cumsum(daily_exec)
    dd_total = np.max(np.maximum.accumulate(cum_total) - cum_total)
    dd_exec = np.max(np.maximum.accumulate(cum_exec) - cum_exec)

    # Hit ratios
    hit_total = np.mean(daily_total > 0)
    hit_exec = np.mean(daily_exec > 0)

    # Serial correlation
    ar1_total = np.corrcoef(daily_total[:-1], daily_total[1:])[0,1] if len(daily_total) > 2 else 0
    ar1_exec = np.corrcoef(daily_exec[:-1], daily_exec[1:])[0,1] if len(daily_exec) > 2 else 0

    print(f"\n  {name}:")
    print(f"  {'Metric':<30s} {'Total PnL':>16s} {'Execution PnL':>16s} {'Inventory PnL':>16s}")
    print(f"  {'─'*30} {'─'*16} {'─'*16} {'─'*16}")

    total_sum = float(np.sum(daily_total))
    exec_sum = float(np.sum(daily_exec))
    inv_sum = float(np.sum(daily_inv))
    print(f"  {'Cumulative PnL':<30s} {total_sum:>+16,.0f} {exec_sum:>+16,.0f} {inv_sum:>+16,.0f}")
    print(f"  {'Daily Sharpe (ann)':<30s} {sharpe_total:>16.2f} {sharpe_exec:>16.2f} {'—':>16s}")
    print(f"  {'Max Drawdown':<30s} {dd_total:>16,.0f} {dd_exec:>16,.0f} {'—':>16s}")
    print(f"  {'Hit ratio':<30s} {hit_total:>15.1%} {hit_exec:>15.1%} {'—':>16s}")
    print(f"  {'Daily AR(1)':<30s} {ar1_total:>16.3f} {ar1_exec:>16.3f} {'—':>16s}")

    # Skew & kurtosis
    skew_exec = float(np.mean(((daily_exec-np.mean(daily_exec))/max(np.std(daily_exec),1e-8))**3))
    kurt_exec = float(np.mean(((daily_exec-np.mean(daily_exec))/max(np.std(daily_exec),1e-8))**4) - 3)
    print(f"  {'Skewness':<30s} {'—':>16s} {skew_exec:>16.2f} {'—':>16s}")
    print(f"  {'Excess kurtosis':<30s} {'—':>16s} {kurt_exec:>16.2f} {'—':>16s}")

    # Daily std
    std_total = np.std(daily_total)
    std_exec = np.std(daily_exec)
    print(f"  {'Daily Std':<30s} {std_total:>16,.0f} {std_exec:>16,.0f} {'—':>16s}")

    # Pct of variance from inventory
    var_total = np.var(daily_total)
    var_exec = np.var(daily_exec)
    var_inv = np.var(daily_inv)
    cov_exec_inv = np.cov(daily_exec, daily_inv)[0,1] if len(daily_exec) > 1 else 0
    print(f"\n  Variance decomposition (daily):")
    print(f"    Var(total)     = {var_total:>14,.0f}")
    print(f"    Var(execution) = {var_exec:>14,.0f}  ({var_exec/max(var_total,1)*100:.1f}%)")
    print(f"    Var(inventory) = {var_inv:>14,.0f}  ({var_inv/max(var_total,1)*100:.1f}%)")
    print(f"    2*Cov(exec,inv)= {2*cov_exec_inv:>14,.0f}  ({2*cov_exec_inv/max(var_total,1)*100:.1f}%)")


# ===========================================================================
# [5] TASK E — PnL Decomposition
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK E — PnL Attribution")
print("=" * 70)

for name, records in all_data.items():
    d = dna_decompose(records)
    total_sum = float(np.sum(d["total_pnl"]))
    exec_sum = float(np.sum(d["exec_pnl"]))
    inv_sum = float(np.sum(d["inv_pnl"]))

    print(f"\n  {name}:")
    print(f"    Total PnL:           {total_sum:>+16,.0f}  (100%)")
    print(f"    Execution PnL:       {exec_sum:>+16,.0f}  ({exec_sum/total_sum*100:+.0f}%)")
    print(f"    Inventory PnL:       {inv_sum:>+16,.0f}  ({inv_sum/total_sum*100:+.0f}%)")

    # Further decomposition of execution PnL
    # Execution PnL ≈ spread capture - adverse + fill timing
    # Spread capture estimate: total fills * avg spread/2
    # We don't have exact fill counts, but can approximate
    print(f"\n    Interpretation:")
    if abs(inv_sum / max(total_sum, 1)) < 0.1:
        print(f"      Edge is PURE EXECUTION — inventory drift negligible")
    elif abs(inv_sum / max(total_sum, 1)) < 0.3:
        print(f"      Edge is MOSTLY execution — inventory adds modestly")
    elif abs(inv_sum / max(total_sum, 1)) < 0.5:
        print(f"      Edge is MIXED — inventory contribution material")
    else:
        print(f"      Edge is MOSTLY DIRECTIONAL — execution contribution secondary")

    # What fraction of inventory PnL is from trend vs random fluctuation?
    mid_rets = d["mid_returns"]
    cum_mid = np.cumsum(mid_rets)
    mid_trend = cum_mid[-1] / len(mid_rets)
    inv_mean = np.mean(d["inventories"][:-1])
    trend_contrib = inv_mean * mid_trend * len(mid_rets)
    print(f"\n    Trend contribution:   {trend_contrib:>+16,.0f}")
    print(f"    (mean_inv × mean_delta_mid × N_windows)")
    print(f"    Residual inv PnL:     {inv_sum - trend_contrib:>+16,.0f}")


# ===========================================================================
# [6] TASK F — Trend Sensitivity
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK F — Trend Sensitivity by Subperiod")
print("=" * 70)

for name, records in all_data.items():
    d = dna_decompose(records)
    days_arr = d["days"]
    mid_rets = d["mid_returns"]
    total_pnl = d["total_pnl"]
    exec_pnl = d["exec_pnl"]

    unique_days = np.unique(days_arr)
    n_days_test = len(unique_days)

    if n_days_test < 9:
        print(f"\n  {name}: only {n_days_test} days — splitting into thirds")
        split_n = max(n_days_test // 3, 3)
    else:
        split_n = n_days_test // 3

    # Split into early, mid, late
    day_splits = [
        ("Early", unique_days[:split_n]),
        ("Mid", unique_days[split_n:2*split_n]),
        ("Late", unique_days[2*split_n:]),
    ]

    print(f"\n  {name} — Subperiod Analysis:")
    print(f"  {'Period':<10s} {'Days':>6s} {'CumMid':>12s} {'TotalPnL':>16s} {'ExecPnL':>16s} "
          f"{'Sharpe(T)':>10s} {'Sharpe(E)':>10s} {'InvContrib':>10s}")
    print(f"  {'─'*10} {'─'*6} {'─'*12} {'─'*16} {'─'*16} {'─'*10} {'─'*10} {'─'*10}")

    for period_name, day_set in day_splits:
        mask = np.isin(days_arr, day_set)
        # PnL for windows in this period (use mask on window-level data)
        mask_win = mask[1:]  # offset by 1 since PnL is between windows

        period_total = np.sum(total_pnl[mask_win]) if np.any(mask_win) else 0
        period_exec = np.sum(exec_pnl[mask_win]) if np.any(mask_win) else 0
        period_mid_chg = np.sum(mid_rets[mask_win]) if np.any(mask_win) else 0

        # Daily aggregation within period
        daily_t = []
        daily_e = []
        for d in day_set:
            dmask = days_arr[1:] == d
            if np.any(dmask):
                daily_t.append(np.sum(total_pnl[dmask]))
                daily_e.append(np.sum(exec_pnl[dmask]))

        sharpe_t = compute_sharpe(np.array(daily_t)) if len(daily_t) > 2 else 0
        sharpe_e = compute_sharpe(np.array(daily_e)) if len(daily_e) > 2 else 0

        inv_contrib = (period_total - period_exec) / max(abs(period_total), 1) if abs(period_total) > 0 else 0

        print(f"  {period_name:<10s} {len(day_set):>6d} {period_mid_chg:>+12,.0f} "
              f"{period_total:>+16,.0f} {period_exec:>+16,.0f} "
              f"{sharpe_t:>9.1f} {sharpe_e:>9.1f} {inv_contrib:>+9.0%}")

    # Correlation: execution Sharpe vs mid trend
    print(f"\n    Stability check:")
    for period_name, day_set in day_splits:
        dmask = np.isin(days_arr, day_set)[1:]
        if np.any(dmask):
            period_mid = np.sum(mid_rets[dmask])
            period_exec = np.sum(exec_pnl[dmask])
            print(f"      {period_name}: cum_delta_mid={period_mid:+,.0f}  exec_PnL={period_exec:+,.0f}")


# ===========================================================================
# [7] TASK G — Long/Short Symmetry Test
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK G — Long/Short Symmetry")
print("=" * 70)

for name, records in all_data.items():
    d = dna_decompose(records)
    invs = d["inventories"]
    total_pnl = d["total_pnl"]
    exec_pnl = d["exec_pnl"]
    mid_rets = d["mid_returns"]

    # Separate windows by prior inventory sign
    prior_inv = invs[:-1]  # inventory BEFORE the window
    long_mask = prior_inv > 0
    short_mask = prior_inv < 0
    flat_mask = prior_inv == 0

    long_total = np.sum(total_pnl[long_mask]) if np.any(long_mask) else 0
    short_total = np.sum(total_pnl[short_mask]) if np.any(short_mask) else 0
    long_exec = np.sum(exec_pnl[long_mask]) if np.any(long_mask) else 0
    short_exec = np.sum(exec_pnl[short_mask]) if np.any(short_mask) else 0
    long_n = int(np.sum(long_mask))
    short_n = int(np.sum(short_mask))

    # Mean PnL per window conditional on inventory sign
    long_mean = np.mean(total_pnl[long_mask]) if np.any(long_mask) else 0
    short_mean = np.mean(total_pnl[short_mask]) if np.any(short_mask) else 0
    long_exec_mean = np.mean(exec_pnl[long_mask]) if np.any(long_mask) else 0
    short_exec_mean = np.mean(exec_pnl[short_mask]) if np.any(short_mask) else 0

    # Mid return during long vs short periods
    long_mid_mean = np.mean(mid_rets[long_mask]) if np.any(long_mask) else 0
    short_mid_mean = np.mean(mid_rets[short_mask]) if np.any(short_mask) else 0

    print(f"\n  {name}:")
    print(f"  {'Condition':<20s} {'Windows':>10s} {'TotalPnL':>14s} {'ExecPnL':>14s} "
          f"{'Meandelta_Mid':>12s} {'P/window':>12s}")
    print(f"  {'─'*20} {'─'*10} {'─'*14} {'─'*14} {'─'*12} {'─'*12}")
    print(f"  {'Long (inv > 0)':<20s} {long_n:>10,d} {long_total:>+14,.0f} {long_exec:>+14,.0f} "
          f"{long_mid_mean:>+12.2f} {long_mean:>+12,.0f}")
    print(f"  {'Short (inv < 0)':<20s} {short_n:>10,d} {short_total:>+14,.0f} {short_exec:>+14,.0f} "
          f"{short_mid_mean:>+12.2f} {short_mean:>+12,.0f}")

    # Symmetry test: is long PnL significantly different from short PnL?
    long_total_std = np.std(total_pnl[long_mask]) if np.any(long_mask) else 1
    short_total_std = np.std(total_pnl[short_mask]) if np.any(short_mask) else 1
    long_exec_std = np.std(exec_pnl[long_mask]) if np.any(long_mask) else 1
    short_exec_std = np.std(exec_pnl[short_mask]) if np.any(short_mask) else 1

    # Welch's t-test (approximate)
    se_total = np.sqrt(long_total_std**2/long_n + short_total_std**2/short_n) if long_n > 0 and short_n > 0 else 1
    t_total = (long_mean - short_mean) / max(se_total, 1e-8)
    se_exec = np.sqrt(long_exec_std**2/long_n + short_exec_std**2/short_n) if long_n > 0 and short_n > 0 else 1
    t_exec = (long_exec_mean - short_exec_mean) / max(se_exec, 1e-8)

    print(f"\n  Symmetry test (Welch's t):")
    print(f"    Total PnL:  long_mean={long_mean:+,.0f}  short_mean={short_mean:+,.0f}  t={t_total:.2f}")
    print(f"    Exec PnL:   long_mean={long_exec_mean:+,.0f}  short_mean={short_exec_mean:+,.0f}  t={t_exec:.2f}")

    if abs(t_total) > 2:
        print(f"    ** ASYMMETRIC: Total PnL differs significantly by inventory sign **")
    elif abs(t_exec) > 2:
        print(f"    ** ASYMMETRIC: Execution PnL differs by inventory sign **")
    else:
        print(f"    Symmetric: no significant difference between long/short PnL")

    # Directional advantage check
    if long_mid_mean > 0 and np.mean(invs) > 0:
        print(f"    ** STRUCTURAL ADVANTAGE: Long bias + rising mid = directional tailwind **")
    elif short_mid_mean < 0 and np.mean(invs) < 0:
        print(f"    ** STRUCTURAL ADVANTAGE: Short bias + falling mid = directional tailwind **")


# ===========================================================================
# [8] FINAL DNA VERDICT
# ===========================================================================
print("\n" + "=" * 70)
print("  DNA VERDICT")
print("=" * 70)

# Use Moderate controller for final verdict
d = dna_decompose(all_data["Moderate"])
total_sum = float(np.sum(d["total_pnl"]))
exec_sum = float(np.sum(d["exec_pnl"]))
inv_sum = float(np.sum(d["inv_pnl"]))

daily_total = compute_daily_pnl(d["total_pnl"], d["days"])
daily_exec = compute_daily_pnl(d["exec_pnl"], d["days"])
sharpe_total = compute_sharpe(daily_total)
sharpe_exec = compute_sharpe(daily_exec)

exec_pct = exec_sum / total_sum * 100
inv_pct = inv_sum / total_sum * 100

print(f"\n  PnL Attribution (Moderate controller):")
print(f"    Total PnL:           {total_sum:>+16,.0f}")
print(f"    Execution PnL:       {exec_sum:>+16,.0f}  ({exec_pct:+.0f}%)")
print(f"    Inventory PnL:       {inv_sum:>+16,.0f}  ({inv_pct:+.0f}%)")

print(f"\n  Sharpe Comparison:")
print(f"    Total (daily ann):   {sharpe_total:>16.2f}")
print(f"    Delta-neutral:       {sharpe_exec:>16.2f}")
sharpe_retention = sharpe_exec / max(sharpe_total, 0.01)
print(f"    Retention:           {sharpe_retention:>15.0%}")

# Verdict
if sharpe_retention > 0.8:
    verdict = "CASE_A — Execution edge dominates. Delta-neutral Sharpe remains strong."
elif sharpe_retention > 0.5:
    verdict = "CASE_B — Directional beta contributes, but execution edge survives delta-neutral."
elif exec_sum > 0:
    verdict = "CASE_B — Execution edge still positive, but directional drift is primary driver."
else:
    verdict = "CASE_C — Delta-neutral PnL is negative. Edge is entirely directional."

print(f"\n  {verdict}")

print(f"\n  Key Takeaway:")
if abs(inv_pct) < 20:
    print(f"    The system's PnL is genuinely execution-driven.")
    print(f"    Inventory drift accounts for only {abs(inv_pct):.0f}% of total PnL.")
elif inv_sum > 0:
    print(f"    The mid price drifted favorably during the test period,")
    print(f"    contributing {inv_pct:.0f}% of total PnL through inventory exposure.")
    print(f"    But execution edge alone is still profitable.")
else:
    print(f"    Inventory PnL is negative — the execution edge is fighting")
    print(f"    against directional headwinds and still winning.")

print(f"\n{'═'*70}")
print(f"  DNA v1 complete.")
print(f"{'═'*70}")

"""
EFL-SRV v1 — Execution Friction & Stress Validation

Systematically degrades execution conditions to find the breaking point
of Hephaestus execution edge. Frozen pipeline, no parameter changes.

Tasks:
  A. Explicit Fee Layer (commission, stamp duty, exchange fees)
  B. Slippage Layer (stale quotes, placement delay, cancel delay)
  C. Queue Fade (fill probability degradation)
  D. Impact Cost (linear + sqrt market impact)
  E. Stress Regimes (worst market conditions)
  F. Combined Severe Stress
  G. Delta-Neutral Revalidation under stress
  H. Capacity under friction
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
MARKOUT_BY_STATE = {
    "R0_q2_T0":-0.4,"R0_q2_T1":-0.3,"R0_q2_T2":-0.2,
    "R1_q2_T0":-1.4,"R1_q2_T1":-2.5,"R1_q2_T2":-0.5,
    "R3_q2_T0":-0.8,"R3_q2_T1":-0.5,"R3_q2_T2":-0.6,
    "R4_q2_T0":-0.6,"R4_q2_T1":-0.3,"R4_q2_T2":-0.1,
    "R5_q2_T0":-0.9,"R5_q2_T1":-0.3,"R5_q2_T2":-0.4,
    "R6_q2_T0":-0.6,"R6_q2_T1":-0.4,"R6_q2_T2":-0.3,
    "R7_q2_T0":-0.4,"R7_q2_T1":+0.5,"R7_q2_T2":-0.4,
}

MAX_INVENTORY = 50000; SIZE_PER_FILL = 100; MID_PRICE = 75000.0

print("=" * 70)
print("  EFL-SRV v1 — Execution Friction & Stress Validation")
print("=" * 70)


# ===========================================================================
# [0] Build state sequence
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
TRAIN_DAYS = 40
tr_e = day_bounds[min(TRAIN_DAYS-1, len(day_bounds)-1)]
X_tr = X_all[:tr_e]; X_tr_m = X_tr.mean(0); X_tr_s = np.maximum(X_tr.std(0), 1e-8)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr-X_tr_m)/X_tr_s, -10, 10))

tox_vals = []
for d in range(min(TRAIN_DAYS, len(all_raw))):
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
oos_start_day = TRAIN_DAYS
print(f"  {N_seq:,} windows  time={time.perf_counter()-t0:.1f}s")


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
# Friction Model
# ===========================================================================

class FrictionModel:
    """
    Configurable execution friction.

    A-share costs (per fill, as fraction of notional):
      - Commission: ~0.025% (2.5 bps)
      - Stamp duty: 0.05% on SELLS only (5 bps)
      - Exchange fee: ~0.005% (0.5 bps)
      - Transfer fee: ~0.002% (0.2 bps)
    Total: ~0.032% buys, ~0.082% sells
    """
    def __init__(self, fee_mult=1.0, slippage_bps=0.0, fill_decay=1.0,
                 impact_mult=0.0, adverse_mult=1.0, spread_decay=1.0,
                 cancel_lag_extra=0):
        self.fee_mult = fee_mult
        self.slippage_bps = slippage_bps   # per-fill slippage in bps
        self.fill_decay = fill_decay       # multiplier on p_fill (1.0 = no decay)
        self.impact_mult = impact_mult     # 0 = no impact, >0 = linear+sqrt model
        self.adverse_mult = adverse_mult   # multiply markout adverse
        self.spread_decay = spread_decay   # multiply spread capture
        self.cancel_lag_extra = cancel_lag_extra

        # Base A-share fees (in bps of notional)
        self.commission_bps = 2.5
        self.stamp_duty_bps = 5.0   # sells only
        self.exchange_fee_bps = 0.5
        self.transfer_fee_bps = 0.2

    def fee_cost(self, side, notional):
        """Cost in raw units (same as PnL). Notional is price * size."""
        total_bps = self.commission_bps + self.exchange_fee_bps + self.transfer_fee_bps
        if side == 'ask':  # sell
            total_bps += self.stamp_duty_bps
        bps_cost = total_bps * self.fee_mult
        return notional * bps_cost / 10000.0

    def apply_slippage(self, fill_price, side):
        """Worsen fill price by slippage (bps)."""
        slip_factor = 1.0 + self.slippage_bps / 10000.0
        if side == 'bid':  # we buy — slippage makes price worse (higher)
            return fill_price * slip_factor
        else:  # we sell — slippage makes price worse (lower)
            return fill_price / slip_factor

    def apply_impact(self, fill_price, size, side, cum_daily_volume):
        """Market impact: linear + sqrt model."""
        if self.impact_mult <= 0 or cum_daily_volume <= 0:
            return fill_price
        participation = size / max(cum_daily_volume, 1)
        # Almgren-Chriss style: impact = sigma * (participation)^0.5 + eta * participation
        impact_bps = (0.1 * np.sqrt(participation) + 0.05 * participation) * self.impact_mult * 10000
        if side == 'bid':
            return fill_price * (1.0 + impact_bps / 10000.0)
        else:
            return fill_price * (1.0 - impact_bps / 10000.0)

    def adjusted_fill_prob(self, base_p):
        return base_p * self.fill_decay

    def adjusted_spread(self, base_spread):
        return base_spread * self.spread_decay


# ===========================================================================
# Stress Backtest
# ===========================================================================

class StressBacktest:
    def __init__(self, friction, controller=None, capacity_mult=1.0):
        self.fric = friction; self.ctrl = controller
        self.capacity_mult = capacity_mult  # size multiplier
        self.cash = 0.0; self.inventory = 0
        self.records = []
        self.total_fees = 0.0; self.total_slippage = 0.0; self.total_impact = 0.0
        self.fills_count = 0; self.quotes_count = 0

    def equity(self, mid): return self.cash + self.inventory * mid

    def record_fill(self, side, px, size, state, mid):
        size_adj = size * self.capacity_mult
        # Apply slippage
        px_slipped = self.fric.apply_slippage(px, side)
        # Apply impact (approximate daily volume)
        px_final = self.fric.apply_impact(px_slipped, size_adj, side, mid * 10)

        # Fee
        notional = px_final * size_adj
        fee = self.fric.fee_cost(side, notional)

        if side == 'bid':
            self.cash -= px_final * size_adj + fee
            self.inventory += size_adj
        else:
            self.cash += px_final * size_adj - fee
            self.inventory -= size_adj

        self.total_fees += fee
        self.total_slippage += abs(px_slipped - px) * size_adj
        self.total_impact += abs(px_final - px_slipped) * size_adj
        self.fills_count += 1

    def simulate_fills(self, state, mid):
        base_p = self.fric.adjusted_fill_prob(FILL_BY_STATE.get(state, 0.0))
        spread = self.fric.adjusted_spread(STATE_SPREAD.get(state, 50.0))

        # Apply adverse multiplier to effective fill price
        markout = MARKOUT_BY_STATE.get(state, -0.5) * self.fric.adverse_mult
        adverse_adj = abs(markout) / 10000.0 * mid * self.fric.adverse_mult

        if self.ctrl:
            bm, am, bs, as_ = self.ctrl.skew(self.inventory, state)
            p_bid = base_p * bm; p_ask = base_p * am
            bid_px = mid - spread/2.0 + bs/10000.0 * mid + adverse_adj
            ask_px = mid + spread/2.0 + as_/10000.0 * mid - adverse_adj
        else:
            p_bid = base_p; p_ask = base_p
            bid_px = mid - spread/2.0 + adverse_adj
            ask_px = mid + spread/2.0 - adverse_adj

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


# ===========================================================================
# DNA decomposition
# ===========================================================================

def dna_decompose(records):
    n = len(records)
    total_pnl = np.zeros(n - 1); inv_pnl = np.zeros(n - 1)
    exec_pnl = np.zeros(n - 1); mid_rets = np.zeros(n - 1)
    invs = np.array([r["inventory"] for r in records])
    mids = np.array([r["mid"] for r in records])
    days = np.array([r["day"] for r in records])
    for t in range(1, n):
        total_pnl[t-1] = records[t]["equity"] - records[t-1]["equity"]
        inv_pnl[t-1] = invs[t-1] * (mids[t] - mids[t-1])
        exec_pnl[t-1] = total_pnl[t-1] - inv_pnl[t-1]
        mid_rets[t-1] = mids[t] - mids[t-1]
    return {"total": total_pnl, "inv": inv_pnl, "exec": exec_pnl,
            "mid_rets": mid_rets, "invs": invs, "mids": mids, "days": days}


def daily_pnl(window_pnl, days):
    d = defaultdict(float)
    for i, p in enumerate(window_pnl): d[days[i+1]] += p
    return np.array(list(d.values()))


def sharpe(daily): return np.mean(daily) / max(np.std(daily), 1e-8) * np.sqrt(252)


def run_backtest(friction, windows, state_seq, window_mids, window_day, ctrl, cap=1.0):
    bt = StressBacktest(friction, controller=ctrl, capacity_mult=cap)
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
                    bt.record_fill(side, px, SIZE_PER_FILL, pos_st, cm)
                bt.quotes_count += WINDOW_SIZE
            bt.snapshot(te_s + t, cm, cd)
    return bt


# ===========================================================================
# Windows setup
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

ctrl = InvCtrl(max_inv=MAX_INVENTORY, base_skew=0.5, suppress_thresh=0.7, hard_stop_thresh=0.9)


# ===========================================================================
# [1] TASK A+B+C+D — Layered Friction Sweep
# ===========================================================================
print("\n" + "=" * 70)
print("  FRICTION LAYER SWEEP")
print("=" * 70)

# Define scenarios
scenarios = {
    "Baseline (no friction)": FrictionModel(fee_mult=0, slippage_bps=0, fill_decay=1.0,
                                             impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Realistic A-Share Fees": FrictionModel(fee_mult=1.0, slippage_bps=0, fill_decay=1.0,
                                             impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fees + 1bp Slippage": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                          impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fees + 2bp Slippage": FrictionModel(fee_mult=1.0, slippage_bps=2.0, fill_decay=1.0,
                                          impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fees + 3bp Slippage": FrictionModel(fee_mult=1.0, slippage_bps=3.0, fill_decay=1.0,
                                          impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fees + 5bp Slippage": FrictionModel(fee_mult=1.0, slippage_bps=5.0, fill_decay=1.0,
                                          impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fill Decay 70%": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=0.7,
                                     impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fill Decay 50%": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=0.5,
                                     impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Fill Decay 30%": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=0.3,
                                     impact_mult=0, adverse_mult=1.0, spread_decay=1.0),
    "Adverse ×2": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                 impact_mult=0, adverse_mult=2.0, spread_decay=1.0),
    "Adverse ×3": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                 impact_mult=0, adverse_mult=3.0, spread_decay=1.0),
    "Spread Capture ×0.7": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                          impact_mult=0, adverse_mult=1.0, spread_decay=0.7),
    "Spread Capture ×0.5": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                          impact_mult=0, adverse_mult=1.0, spread_decay=0.5),
    "Light Impact (1x)": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                        impact_mult=1.0, adverse_mult=1.0, spread_decay=1.0),
    "Medium Impact (3x)": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                         impact_mult=3.0, adverse_mult=1.0, spread_decay=1.0),
    "Heavy Impact (5x)": FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                                        impact_mult=5.0, adverse_mult=1.0, spread_decay=1.0),
}

all_scenario_results = {}

print(f"\n  Running {len(scenarios)} scenarios ...")
t0 = time.perf_counter()

for sc_name, friction in scenarios.items():
    bt = run_backtest(friction, windows, state_seq, window_mids, window_day, ctrl)
    d = dna_decompose(bt.records)
    total_pnl = float(np.sum(d["total"]))
    exec_pnl = float(np.sum(d["exec"]))
    inv_pnl = float(np.sum(d["inv"]))
    dp = daily_pnl(d["total"], d["days"])
    de = daily_pnl(d["exec"], d["days"])
    sh_total = sharpe(dp)
    sh_exec = sharpe(de)
    dd_total = float(np.max(np.maximum.accumulate(np.cumsum(dp)) - np.cumsum(dp)))

    all_scenario_results[sc_name] = {
        "total_pnl": total_pnl, "exec_pnl": exec_pnl, "inv_pnl": inv_pnl,
        "sharpe_total": sh_total, "sharpe_exec": sh_exec,
        "max_dd": dd_total, "fees": bt.total_fees,
        "slippage": bt.total_slippage, "impact": bt.total_impact,
        "fills": bt.fills_count, "quotes": bt.quotes_count,
        "neg_days": int(np.sum(dp < 0)), "n_days": len(dp),
    }

    status = "SURVIVES" if exec_pnl > 0 else "DIES"
    print(f"  {sc_name:<30s} PnL={total_pnl:>+14,.0f}  Exec={exec_pnl:>+14,.0f}  "
          f"Sh={sh_total:>6.1f}  DD={dd_total:>12,.0f}  {status}")

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] TASK E — Stress Regime Selection
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK E — Stress Regime Validation")
print("=" * 70)

# Identify worst days by various criteria
day_volatility = defaultdict(list)
day_spread = defaultdict(list)
day_q2_pct = defaultdict(float)
day_total = defaultdict(int)
day_mid_chg = defaultdict(float)

for i, s in enumerate(state_seq):
    d = window_day[i]
    day_total[d] += 1
    if "_q2_" in s: day_q2_pct[d] += 1

for i in range(len(window_mids) - 1):
    d = window_day[i]
    day_volatility[d].append(abs(window_mids[i+1] - window_mids[i]))

for d in day_total:
    day_q2_pct[d] /= max(day_total[d], 1)
    day_volatility[d] = np.mean(day_volatility.get(d, [0]))

# Rank days
days_by_vol = sorted(day_volatility.keys(), key=lambda d: day_volatility[d], reverse=True)
days_by_q2 = sorted(day_q2_pct.keys(), key=lambda d: day_q2_pct[d])

high_vol_days = set(days_by_vol[:max(len(days_by_vol)//5, 3)])
low_q2_days = set(days_by_q2[:max(len(days_by_q2)//5, 3)])
# Combined: worst 30% of days by either metric
stress_day_set = high_vol_days | low_q2_days

print(f"\n  High volatility days:    {len(high_vol_days)}  (vol > {day_volatility[sorted(high_vol_days)[-1]]:.0f})")
print(f"  Low q2 occupancy days:   {len(low_q2_days)}  (q2% < {day_q2_pct[sorted(low_q2_days)[-1]]:.1%})")
print(f"  Combined stress days:    {len(stress_day_set)}")

# Run baseline friction on stress days only
base_fric = FrictionModel(fee_mult=1.0, slippage_bps=1.0, fill_decay=1.0,
                          impact_mult=0, adverse_mult=1.0, spread_decay=1.0)

bt_stress = StressBacktest(base_fric, controller=ctrl)
for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
    t_seq = state_seq[te_s:te_e]; t_mid = window_mids[te_s:te_e]
    t_day = window_day[te_s:te_e]
    if len(t_seq) < 10: continue
    test_days = set(np.unique(t_day))
    if not test_days.intersection(stress_day_set): continue  # skip non-stress windows
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
            fills = bt_stress.simulate_fills(pos_st, cm)
            for side, px in fills:
                bt_stress.record_fill(side, px, SIZE_PER_FILL, pos_st, cm)
            bt_stress.quotes_count += WINDOW_SIZE
        bt_stress.snapshot(te_s + t, cm, cd)

d_stress = dna_decompose(bt_stress.records)
stress_exec = float(np.sum(d_stress["exec"]))
stress_total = float(np.sum(d_stress["total"]))
stress_daily = daily_pnl(d_stress["total"], d_stress["days"])
stress_sh = sharpe(stress_daily)

print(f"\n  Stress Regime Results (baseline friction):")
print(f"    Total PnL:           {stress_total:>+14,.0f}")
print(f"    Execution PnL:       {stress_exec:>+14,.0f}")
print(f"    Sharpe:              {stress_sh:>14.2f}")
print(f"    Days:                {len(stress_daily)}")
print(f"    Negative days:       {int(np.sum(stress_daily < 0))}  ({np.mean(stress_daily < 0)*100:.1f}%)")


# ===========================================================================
# [3] TASK F — Combined Severe Stress
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK F — Combined Severe Stress")
print("=" * 70)

severe = FrictionModel(
    fee_mult=2.0,           # double fees
    slippage_bps=3.0,       # 3bp slippage
    fill_decay=0.3,         # 70% fill reduction
    impact_mult=3.0,        # medium impact
    adverse_mult=2.0,       # double adverse
    spread_decay=0.5,       # half spread capture
    cancel_lag_extra=5,     # +5 tick cancel delay
)

bt_severe = run_backtest(severe, windows, state_seq, window_mids, window_day, ctrl)
d_sev = dna_decompose(bt_severe.records)
sev_total = float(np.sum(d_sev["total"]))
sev_exec = float(np.sum(d_sev["exec"]))
sev_inv = float(np.sum(d_sev["inv"]))
sev_dp = daily_pnl(d_sev["total"], d_sev["days"])
sev_de = daily_pnl(d_sev["exec"], d_sev["days"])

print(f"\n  Severe Stress (Fee×2, Slip×3bp, Fill×0.3, Impact×3, Adverse×2, Spread×0.5):")
print(f"    Total PnL:           {sev_total:>+16,.0f}")
print(f"    Execution PnL:       {sev_exec:>+16,.0f}  ({sev_exec/max(sev_total,1)*100:+.0f}%)")
print(f"    Inventory PnL:       {sev_inv:>+16,.0f}")
print(f"    Daily Sharpe:        {sharpe(sev_dp):>16.2f}")
print(f"    Exec Sharpe:         {sharpe(sev_de):>16.2f}")
dd_sev = float(np.max(np.maximum.accumulate(np.cumsum(sev_dp)) - np.cumsum(sev_dp)))
print(f"    Max DD:              {dd_sev:>16,.0f}")
print(f"    Negative days:       {int(np.sum(sev_dp < 0))} / {len(sev_dp)}")
print(f"    Fees paid:           {bt_severe.total_fees:>16,.0f}")
print(f"    Slippage cost:       {bt_severe.total_slippage:>16,.0f}")
print(f"    Impact cost:         {bt_severe.total_impact:>16,.0f}")
print(f"    Total fills:         {bt_severe.fills_count:>14,d}")

# Friction breakdown
total_friction = bt_severe.total_fees + bt_severe.total_slippage + bt_severe.total_impact
gross_pnl = sev_total + total_friction
print(f"\n  Friction Breakdown:")
print(f"    Gross PnL (no friction):  {gross_pnl:>+16,.0f}")
print(f"    Fees:                     {bt_severe.total_fees:>+16,.0f}  ({bt_severe.total_fees/max(total_friction,1)*100:.0f}%)")
print(f"    Slippage:                 {bt_severe.total_slippage:>+16,.0f}  ({bt_severe.total_slippage/max(total_friction,1)*100:.0f}%)")
print(f"    Impact:                   {bt_severe.total_impact:>+16,.0f}  ({bt_severe.total_impact/max(total_friction,1)*100:.0f}%)")
print(f"    Total friction:           {total_friction:>+16,.0f}")
print(f"    Net PnL:                  {sev_total:>+16,.0f}")
print(f"    Friction / Gross:         {total_friction/max(gross_pnl,1)*100:.0f}%")


# ===========================================================================
# [4] TASK G — DNA Under Stress
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK G — Delta-Neutral Under Stress")
print("=" * 70)

# Re-run DNA on severe stress
X_sev = np.column_stack([np.ones(len(d_sev["mid_rets"])), d_sev["mid_rets"]])
beta_sev = np.linalg.lstsq(X_sev, d_sev["total"], rcond=None)[0]
resid_sev = d_sev["total"] - (beta_sev[0] + beta_sev[1] * d_sev["mid_rets"])
r2_sev = 1 - np.var(resid_sev) / max(np.var(d_sev["total"]), 1e-8)

print(f"\n  Severe Stress DNA:")
print(f"    Beta (window):        {beta_sev[1]:>+16.1f}")
print(f"    Alpha (exec PnL):     {beta_sev[0]:>+16,.0f}")
print(f"    R^2 with mid:         {r2_sev:>15.1%}")
print(f"    Exec fraction:        {sev_exec/max(sev_total,1)*100:.0f}%")

# Compare DNA across baseline, realistic, severe
for label, result_key in [("Baseline", "Baseline (no friction)"),
                            ("Realistic", "Realistic A-Share Fees"),
                            ("Severe", None)]:
    if result_key:
        r = all_scenario_results[result_key]
        exec_pct = r["exec_pnl"] / max(r["total_pnl"], 1) * 100
        print(f"\n  {label}:")
        print(f"    Exec PnL:  {r['exec_pnl']:>+16,.0f}  ({exec_pct:+.0f}%)")
        print(f"    Sharpe:    {r['sharpe_exec']:>16.2f}")
        print(f"    Max DD:    {r['max_dd']:>16,.0f}")
    else:
        exec_pct = sev_exec / max(sev_total, 1) * 100
        print(f"\n  Severe:")
        print(f"    Exec PnL:  {sev_exec:>+16,.0f}  ({exec_pct:+.0f}%)")
        print(f"    Sharpe:    {sharpe(sev_de):>16.2f}")
        print(f"    Max DD:    {dd_sev:>16,.0f}")


# ===========================================================================
# [5] TASK H — Capacity Test
# ===========================================================================
print("\n" + "=" * 70)
print("  TASK H — Capacity Under Friction")
print("=" * 70)

capacity_multipliers = [0.5, 1.0, 2.0, 3.0, 5.0]
cap_results = []

for cap in capacity_multipliers:
    bt_cap = StressBacktest(base_fric, controller=ctrl, capacity_mult=cap)
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
                fills = bt_cap.simulate_fills(pos_st, cm)
                for side, px in fills:
                    bt_cap.record_fill(side, px, SIZE_PER_FILL, pos_st, cm)
                bt_cap.quotes_count += WINDOW_SIZE
            bt_cap.snapshot(te_s + t, cm, cd)

    d_cap = dna_decompose(bt_cap.records)
    total_cap = float(np.sum(d_cap["total"]))
    exec_cap = float(np.sum(d_cap["exec"]))
    inv_std = float(np.std(d_cap["invs"]))
    dp_cap = daily_pnl(d_cap["total"], d_cap["days"])
    sh_cap = sharpe(dp_cap)
    dd_cap = float(np.max(np.maximum.accumulate(np.cumsum(dp_cap)) - np.cumsum(dp_cap)))
    max_inv = float(max(abs(np.min(d_cap["invs"])), abs(np.max(d_cap["invs"]))))

    cap_results.append({
        "cap": cap, "total_pnl": total_cap, "exec_pnl": exec_cap,
        "sharpe": sh_cap, "max_dd": dd_cap, "inv_std": inv_std,
        "max_inv": max_inv, "impact_cost": bt_cap.total_impact,
        "slippage_cost": bt_cap.total_slippage, "fees": bt_cap.total_fees,
    })

print(f"\n  {'Cap':>8s} {'PnL':>14s} {'Sharpe':>8s} {'MaxDD':>12s} "
      f"{'InvStd':>10s} {'MaxInv':>10s} {'Impact':>12s} {'Fees':>12s}")
print(f"  {'─'*8} {'─'*14} {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

for r in cap_results:
    print(f"  {r['cap']:>7.1f}x {r['total_pnl']:>+14,.0f} {r['sharpe']:>7.1f} {r['max_dd']:>12,.0f} "
          f"{r['inv_std']:>10,.0f} {r['max_inv']:>10,.0f} "
          f"{r['impact_cost']:>12,.0f} {r['fees']:>12,.0f}")

# Capacity collapse point
for r in cap_results:
    if r["total_pnl"] < 0:
        print(f"\n  ** CAPACITY LIMIT: Edge turns negative at {r['cap']:.1f}x **")
        break
else:
    print(f"\n  Edge survives up to {capacity_multipliers[-1]}x capacity")


# ===========================================================================
# [6] FINAL VERDICT
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL STRESS VERDICT")
print("=" * 70)

# Count failures across all scenarios (excluding baseline)
failures = []
for name, r in all_scenario_results.items():
    if "Baseline" in name: continue
    if r["exec_pnl"] < 0:
        failures.append((name, r["exec_pnl"], r["sharpe_exec"]))

print(f"\n  Scenarios tested:     {len(scenarios)}")
print(f"  Edge failures:        {len(failures)}")
for name, pnl, sh in failures:
    print(f"    FAIL: {name:<35s} ExecPnL={pnl:>+14,.0f}  Sh={sh:.1f}")

# Determine what kills the edge first
if failures:
    # Analyze which friction parameter causes failure
    fail_names = [f[0] for f in failures]
    if any("Fill Decay" in n for n in fail_names):
        primary_killer = "QUEUE COMPETITION (fill decay)"
    elif any("Adverse" in n for n in fail_names):
        primary_killer = "ADVERSE SELECTION"
    elif any("Slippage" in n for n in fail_names):
        primary_killer = "SLIPPAGE"
    elif any("Impact" in n for n in fail_names):
        primary_killer = "MARKET IMPACT"
    elif any("Spread" in n for n in fail_names):
        primary_killer = "SPREAD COMPRESSION"
    else:
        primary_killer = "COMBINED FRICTION"
    print(f"\n  Primary edge killer:  {primary_killer}")
else:
    print(f"\n  No single friction layer kills the edge.")

# Severe stress verdict
if sev_exec > 0:
    severe_verdict = "SURVIVES — Execution edge withstands combined severe stress"
else:
    severe_verdict = "DIES — Combined severe stress overwhelms the edge"

print(f"\n  Severe stress:        {severe_verdict}")

# Final verdict
if len(failures) == 0 and sev_exec > 0:
    verdict = "CASE_A — Edge survives all friction layers and severe combined stress."
elif len(failures) <= 3 and sev_exec > 0:
    verdict = "CASE_A — Edge robust to realistic friction. Fails only under extreme scenarios."
elif sev_exec > 0:
    verdict = "CASE_B — Edge survives severe stress but vulnerable to specific friction types."
elif len(failures) <= 5:
    verdict = "CASE_B — Edge fragile under moderate friction. Requires low-cost execution."
else:
    verdict = "CASE_C — Friction destroys edge. Not viable under realistic costs."

print(f"\n  {verdict}")

# Key numbers
base_r = all_scenario_results["Baseline (no friction)"]
real_r = all_scenario_results["Realistic A-Share Fees"]
print(f"\n  PnL Evolution:")
print(f"    Baseline (no friction):  {base_r['total_pnl']:>+14,.0f}  (100%)")
print(f"    Realistic A-Share Fees:  {real_r['total_pnl']:>+14,.0f}  ({real_r['total_pnl']/max(base_r['total_pnl'],1)*100:.0f}%)")
print(f"    Severe Combined Stress:  {sev_total:>+14,.0f}  ({sev_total/max(base_r['total_pnl'],1)*100:.0f}%)")
print(f"    Friction cost (realistic): {base_r['total_pnl'] - real_r['total_pnl']:>,.0f}")

print(f"\n{'═'*70}")
print(f"  EFL-SRV v1 complete.")
print(f"{'═'*70}")

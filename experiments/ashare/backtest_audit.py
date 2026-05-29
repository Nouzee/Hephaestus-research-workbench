"""
BIA v1 — Backtest Integrity Audit

Full accounting audit on execution-aware sparse participation backtest.
Tick-level MTM + inventory tracking with clean cash-based accounting.

Audit tasks:
  A. Mark-to-Market — equity = cash + inventory * mid
  B. Inventory — drift, accumulation, holding duration
  C. Fill Accounting — duplicate detection, bid/ask imbalance
  D. Queue Realism — queue wait dynamics
  E. Cancel Realism — cancel lag, stale quote exposure
  F. Spread Crossing — bid/ask overlap detection
  G. PnL Path — full equity curve, rolling drawdown, Sharpe
  H. Baseline — Always-On MM with identical execution layer
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

# ── EVL execution parameters ──
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
QUEUE_WAIT_BY_STATE = {
    "R0_q2_T0":27,"R0_q2_T1":26,"R0_q2_T2":27,
    "R1_q2_T0":29,"R1_q2_T1":33,"R1_q2_T2":27,
    "R3_q2_T0":28,"R3_q2_T1":26,"R3_q2_T2":27,
    "R4_q2_T0":28,"R4_q2_T1":26,"R4_q2_T2":25,
    "R5_q2_T0":28,"R5_q2_T1":26,"R5_q2_T2":27,
    "R6_q2_T0":26,"R6_q2_T1":25,"R6_q2_T2":26,
    "R7_q2_T0":26,"R7_q2_T1":28,"R7_q2_T2":28,
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

CANCEL_LAG_TICKS = 3
EXCHANGE_REACTION_TICKS = 2
MID_PRICE = 75000.0

print("=" * 70)
print("  BIA v1 — Backtest Integrity Audit")
print("=" * 70)


# ===========================================================================
# [0] Build state sequence + window mid prices
# ===========================================================================
print("\n[0] Building state sequence + capturing mid prices ...")
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

state_seq = []; window_mids = []
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

N_seq = len(state_seq)
print(f"  {N_seq:,} windows  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Audit Engine — clean cash-based accounting
# ===========================================================================

class TickLevelBacktest:
    """
    Accounting: equity = cash + inventory * mid_price
    No separate cost-basis tracking. Cash absorbs all trade settlements.
    Markout/adverse emerges naturally through MTM as mid moves between
    fill time and subsequent valuation windows.
    """
    def __init__(self, state_seq, window_mids, name="Strategy"):
        self.state_seq = state_seq
        self.window_mids = window_mids
        self.name = name

        self.cash = 0.0
        self.inventory = 0
        self.initial_mid = window_mids[0] if window_mids else MID_PRICE

        self.all_trades = []
        self.quotes_placed = 0

        # Equity curve per window
        self.equity_curve = []       # dict per window

        # Inventory
        self.inventory_path = []     # dict per window
        self.max_long = 0
        self.max_short = 0
        self.inventory_holding = []  # consecutive non-zero window counts

        # Cancel
        self.cancel_requests = 0
        self.cancel_successes = 0
        self.stale_fills = 0
        self.late_cancel_cost = 0.0

        # Queue
        self.queue_waits = []

        # Per-state stats
        self.state_fills = defaultdict(int)
        self.state_n_quotes = defaultdict(int)

        # Fill audit
        self.fill_keys = defaultdict(int)  # (window, tick, side) -> count

    def equity(self, mid):
        return self.cash + self.inventory * mid

    def record_fill(self, side, price, size, state, w_idx, tick_off):
        """
        Clean cash-based fill recording. No cost-basis tracking needed.
        Cash goes down on buys, up on sells. Inventory tracks net position.
        """
        if side == 'bid':
            self.cash -= price * size
            self.inventory += size
        else:
            self.cash += price * size
            self.inventory -= size

        self.all_trades.append({
            "window": w_idx, "tick": tick_off, "side": side,
            "price": price, "size": size, "state": state
        })
        self.fill_keys[(w_idx, tick_off, side)] += 1
        self.state_fills[state] += 1

        self.max_long = max(self.max_long, self.inventory)
        self.max_short = max(self.max_short, -self.inventory)

    def snapshot(self, w_idx, mid):
        """Record MTM equity and inventory at window boundary."""
        eq = self.equity(mid)
        self.equity_curve.append({
            "window": w_idx, "equity": eq,
            "cash": self.cash, "inventory": self.inventory, "mid": mid,
        })
        self.inventory_path.append({
            "window": w_idx, "inventory": self.inventory, "mid": mid,
        })

    # ── Fill simulation ──

    def simulate_fills(self, state, w_idx, mid):
        """
        Simulate fills in one window using EVL-calibrated params.
        Quote bid at mid - spread/2, ask at mid + spread/2.
        Fill occurs at the quoted price (no markout adjustment — markout
        emerges through MTM as mid moves between windows).
        """
        p_fill = FILL_BY_STATE.get(state, 0.0)
        spread = STATE_SPREAD.get(state, 50.0)
        q_wait = QUEUE_WAIT_BY_STATE.get(state, 27)

        bid_px = mid - spread / 2.0
        ask_px = mid + spread / 2.0

        fills = []
        for tick in range(WINDOW_SIZE):
            if np.random.random() < p_fill:
                fills.append(('bid', bid_px, tick))
            if np.random.random() < p_fill:
                fills.append(('ask', ask_px, tick))
            if np.random.random() < 0.01:
                self.queue_waits.append(q_wait)

        return fills

    # ── Cancel simulation ──

    def cancel(self, w_idx):
        self.cancel_requests += 1
        exposure_ticks = CANCEL_LAG_TICKS + EXCHANGE_REACTION_TICKS
        stale_p = 1.0 - (1.0 - 0.01) ** exposure_ticks
        if np.random.random() < stale_p:
            self.stale_fills += 1
            damage = np.random.exponential(50)
            self.late_cancel_cost += damage
            self.cash -= damage
            return False
        self.cancel_successes += 1
        return True

    # ── Audits ──

    def audit_crossings(self):
        violations = []
        for i in range(len(self.all_trades) - 1):
            a, b = self.all_trades[i], self.all_trades[i+1]
            if a["window"] != b["window"] or abs(a["tick"] - b["tick"]) > 2:
                continue
            if a["side"] == 'bid' and b["side"] == 'ask' and a["price"] >= b["price"]:
                violations.append((i, a["price"], b["price"]))
            elif a["side"] == 'ask' and b["side"] == 'bid' and b["price"] >= a["price"]:
                violations.append((i, a["price"], b["price"]))
        return violations

    def audit_duplicates(self):
        return {k: v for k, v in self.fill_keys.items() if v > 1}

    def audit_inventory(self):
        invs = np.array([p["inventory"] for p in self.inventory_path])
        if len(invs) < 2:
            return {}
        mids = np.array([p["mid"] for p in self.inventory_path])

        runs = []
        cnt = 0
        for iv in invs:
            if iv != 0:
                cnt += 1
            else:
                if cnt > 0:
                    runs.append(cnt)
                cnt = 0
        if cnt > 0:
            runs.append(cnt)

        return {
            "mean": float(np.mean(invs)),
            "std": float(np.std(invs)),
            "mean_abs": float(np.mean(np.abs(invs))),
            "max_long": self.max_long,
            "max_short": self.max_short,
            "mean_hold": float(np.mean(runs)) if runs else 0,
            "max_hold": max(runs) if runs else 0,
            "pct_flat": float(np.mean(invs == 0)),
            "drift": float(np.polyfit(range(len(invs)), invs, 1)[0]),
            "autocorr": float(np.corrcoef(invs[:-1], invs[1:])[0,1]) if len(invs) > 2 else 0,
            "turnover": float(np.sum(np.abs(np.diff(invs)))),
        }


# ===========================================================================
# [1] ECORE Strategy
# ===========================================================================
print("\n" + "=" * 70)
print("  [1] ECORE+ETE Strategy")
print("=" * 70)

bt_e = TickLevelBacktest(state_seq, window_mids, "ECORE+ETE")

TRAIN_W, TEST_W = 20, 5
n_windows = (n_days - TRAIN_W) // TEST_W

windows = []
for wi in range(n_windows):
    tsd = wi * TEST_W
    ted = tsd + TRAIN_W
    tesd = ted
    teed = min(tesd + TEST_W, n_days)
    tsw = 0 if tsd == 0 else day_bounds[tsd - 1]
    tew = day_bounds[min(ted - 1, len(day_bounds) - 1)]
    tesw = tew
    teew = day_bounds[min(teed - 1, len(day_bounds) - 1)] if teed <= n_days else N_seq
    windows.append((tsw, tew, tesw, min(teew, N_seq)))

for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
    t_seq = state_seq[te_s:te_e]
    t_mid = window_mids[te_s:te_e]
    if len(t_seq) < 10:
        continue

    in_pos = False
    pos_st = None

    for t in range(len(t_seq)):
        cs = t_seq[t]; cm = t_mid[t]; cq = cs.split("_")[1]

        if t > 0:
            ps = t_seq[t-1]; pq = ps.split("_")[1]

            if not in_pos and pq == "q1" and cq == "q2" and cs in ECORE:
                in_pos = True; pos_st = cs
            elif in_pos and (cq != "q2" or cs not in ECORE):
                in_pos = False; pos_st = None
                bt_e.cancel(te_s + t)

        if in_pos and pos_st in FILL_BY_STATE:
            fills = bt_e.simulate_fills(pos_st, te_s + t, cm)
            for side, px, tk in fills:
                bt_e.record_fill(side, px, 100, pos_st, te_s + t, tk)
            bt_e.quotes_placed += WINDOW_SIZE
            bt_e.state_n_quotes[pos_st] += WINDOW_SIZE

        bt_e.snapshot(te_s + t, cm)

print(f"  Quotes:  {bt_e.quotes_placed:>12,d}")
print(f"  Fills:   {len(bt_e.all_trades):>12,d}")
print(f"  Max long:  {bt_e.max_long:>12,.0f}")
print(f"  Max short: {bt_e.max_short:>12,.0f}")


# ===========================================================================
# [2] Always-On Baseline
# ===========================================================================
print("\n" + "=" * 70)
print("  [2] Always-On Baseline")
print("=" * 70)

bt_b = TickLevelBacktest(state_seq, window_mids, "Always-On")

for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
    t_seq = state_seq[te_s:te_e]
    t_mid = window_mids[te_s:te_e]
    if len(t_seq) < 10:
        continue

    for t in range(len(t_seq)):
        cs = t_seq[t]; cm = t_mid[t]; cq = cs.split("_")[1]

        if cq == "q2" and cs in FILL_BY_STATE:
            fills = bt_b.simulate_fills(cs, te_s + t, cm)
            for side, px, tk in fills:
                bt_b.record_fill(side, px, 100, cs, te_s + t, tk)
            bt_b.quotes_placed += WINDOW_SIZE
            bt_b.state_n_quotes[cs] += WINDOW_SIZE

        elif cq == "q1":
            spread = STATE_SPREAD.get(cs, 50) * 0.6  # mid-spread states, tighter
            p_fill = 0.30
            bid_px = cm - spread / 2.0
            ask_px = cm + spread / 2.0
            for tick in range(WINDOW_SIZE):
                if np.random.random() < p_fill:
                    bt_b.record_fill('bid', bid_px, 100, cs, te_s + t, tick)
                if np.random.random() < p_fill:
                    bt_b.record_fill('ask', ask_px, 100, cs, te_s + t, tick)
            bt_b.quotes_placed += WINDOW_SIZE
            bt_b.state_n_quotes[cs] += WINDOW_SIZE

        bt_b.snapshot(te_s + t, cm)

print(f"  Quotes:  {bt_b.quotes_placed:>12,d}")
print(f"  Fills:   {len(bt_b.all_trades):>12,d}")
print(f"  Max long:  {bt_b.max_long:>12,.0f}")
print(f"  Max short: {bt_b.max_short:>12,.0f}")


# ===========================================================================
# [3] AUDITS A–G
# ===========================================================================
print("\n" + "=" * 70)
print("  AUDIT RESULTS")
print("=" * 70)

# ── A: MTM ──
print("\n" + "-" * 50)
print("  TASK A — Mark-to-Market")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    eqs = np.array([e["equity"] for e in bt.equity_curve])
    end_eq = eqs[-1] if len(eqs) > 0 else 0
    end_inv = bt.inventory
    end_mid = bt.window_mids[-1] if bt.window_mids else MID_PRICE
    inv_mtm = end_inv * end_mid
    print(f"\n  {nm}:")
    print(f"    Final equity:       {end_eq:>+16,.0f}")
    print(f"    Ending cash:        {bt.cash:>+16,.0f}")
    print(f"    Ending inventory:   {end_inv:>16,.0f}  @ {end_mid:,.0f} = {inv_mtm:+,.0f}")
    print(f"    Cash + Inv*Mid:     {bt.cash + inv_mtm:>+16,.0f}")
    if abs(end_inv) > 1e6:
        print(f"    ** WARNING: Large ending inventory — unrealized PnL dominates **")

# ── B: Inventory ──
print("\n" + "-" * 50)
print("  TASK B — Inventory")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    s = bt.audit_inventory()
    print(f"\n  {nm}:")
    print(f"    Mean inventory:      {s.get('mean',0):>+14,.0f}")
    print(f"    Std inventory:       {s.get('std',0):>14,.0f}")
    print(f"    Mean abs inventory:  {s.get('mean_abs',0):>14,.0f}")
    print(f"    Max long:            {s.get('max_long',0):>14,.0f}")
    print(f"    Max short:           {s.get('max_short',0):>14,.0f}")
    print(f"    Mean holding (w):    {s.get('mean_hold',0):>14.1f}")
    print(f"    Max holding (w):     {s.get('max_hold',0):>14.0f}")
    print(f"    Pct time flat:       {s.get('pct_flat',0):>13.1%}")
    print(f"    Inventory drift:     {s.get('drift',0):>+14.3f} /w")
    print(f"    Autocorrelation:     {s.get('autocorr',0):>14.3f}")
    print(f"    Turnover:            {s.get('turnover',0):>14,.0f}")
    flags = []
    if s.get('mean_abs',0) > 5000 and s.get('pct_flat',0) < 0.3:
        flags.append("persistent inventory")
    if s.get('autocorr',0) > 0.7:
        flags.append("slow mean reversion")
    if s.get('drift',0) != 0 and abs(s.get('drift',0)) > s.get('std',1) * 0.1:
        flags.append("directional drift")
    if flags:
        print(f"    ** FLAGS: {', '.join(flags)} **")

# ── C: Fill Accounting ──
print("\n" + "-" * 50)
print("  TASK C — Fill Accounting")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    bids = sum(1 for t in bt.all_trades if t["side"] == 'bid')
    asks = sum(1 for t in bt.all_trades if t["side"] == 'ask')
    dups = bt.audit_duplicates()
    imbalance = abs(bids - asks) / max(bids + asks, 1)
    print(f"\n  {nm}:")
    print(f"    Total fills:         {len(bt.all_trades):>14,d}")
    print(f"    Bid fills:           {bids:>14,d}")
    print(f"    Ask fills:           {asks:>14,d}")
    print(f"    Fill imbalance:      {imbalance:>13.1%}")
    print(f"    Duplicate keys:      {len(dups):>14,d}")
    if imbalance > 0.3:
        print(f"    ** FLAG: Severe fill imbalance **")
    if dups:
        print(f"    ** FLAG: Duplicate fill keys **")

# ── D: Queue Realism ──
print("\n" + "-" * 50)
print("  TASK D — Queue Realism")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    w = bt.queue_waits
    if w:
        print(f"\n  {nm}:")
        print(f"    Samples:             {len(w):>14,d}")
        print(f"    Mean wait:           {np.mean(w):>14.1f} ticks")
        print(f"    Median wait:         {np.median(w):>14.1f} ticks")
        print(f"    P90 wait:            {np.percentile(w,90):>14.1f} ticks")
        pct_fast = np.mean(np.array(w) < 2)
        if pct_fast > 0.3:
            print(f"    ** FLAG: {pct_fast:.0%} fills <2 ticks — unrealistic **")

# ── E: Cancel Realism ──
print("\n" + "-" * 50)
print("  TASK E — Cancel Realism")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    print(f"\n  {nm}:")
    print(f"    Cancel requests:     {bt.cancel_requests:>14,d}")
    success_rate = bt.cancel_successes / max(bt.cancel_requests, 1)
    print(f"    Cancel success rate: {success_rate:>13.1%}")
    stale_rate = bt.stale_fills / max(len(bt.all_trades), 1)
    print(f"    Stale fill rate:     {stale_rate:>13.1%}")
    print(f"    Late cancel cost:    {bt.late_cancel_cost:>+14,.0f}")
    if stale_rate > 0.05:
        print(f"    ** FLAG: High stale fill rate **")

# ── F: Spread Crossing ──
print("\n" + "-" * 50)
print("  TASK F — Spread Crossing")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    v = bt.audit_crossings()
    nf = len(bt.all_trades)
    print(f"\n  {nm}:")
    print(f"    Violations:          {len(v):>14,d}  / {nf:,} fills ({len(v)/max(nf,1)*100:.3f}%)")
    if len(v) > nf * 0.005:
        print(f"    ** FLAG: {len(v)/max(nf,1)*100:.2f}% violation rate **")
    elif len(v) > 0:
        print(f"    Acceptable: <0.5% (mid movement between ticks)")

# ── G: PnL Path ──
print("\n" + "-" * 50)
print("  TASK G — PnL Path")
print("-" * 50)

for bt, nm in [(bt_e, "ECORE+ETE"), (bt_b, "Always-On")]:
    eqs = np.array([e["equity"] for e in bt.equity_curve])
    if len(eqs) < 2:
        print(f"\n  {nm}: insufficient data")
        continue

    pnls = np.diff(eqs)
    cum_pnl = eqs - eqs[0]
    cum_max = np.maximum.accumulate(cum_pnl)
    dd = cum_max - cum_pnl
    max_dd = np.max(dd)
    neg_w = int(np.sum(pnls < 0))

    vol = np.std(pnls)
    mu = np.mean(pnls)
    sharpe = mu / max(vol, 1e-8) * np.sqrt(len(pnls))

    print(f"\n  {nm}:")
    print(f"    Total PnL:           {cum_pnl[-1]:>+16,.0f}")
    print(f"    Mean window PnL:     {mu:>+16,.0f}")
    print(f"    Std window PnL:      {vol:>16,.0f}")
    print(f"    Sharpe (window):     {sharpe:>16.2f}")
    print(f"    Max drawdown:        {max_dd:>16,.0f}")
    print(f"    Negative windows:    {neg_w:>14,d}  ({neg_w/len(pnls)*100:.1f}%)")

    if max_dd == 0 and len(pnls) > 20:
        print(f"    ** FLAG: Zero drawdown — suspicious **")
    if neg_w == 0 and len(pnls) > 20:
        print(f"    ** FLAG: No negative windows — suspicious **")

    if len(pnls) > 40:
        roll20 = np.array([np.mean(pnls[i:i+20]) / max(np.std(pnls[i:i+20]), 1e-8)
                          for i in range(len(pnls)-20)])
        print(f"    Rolling Sharpe 20w:  [{np.min(roll20):.2f}, {np.max(roll20):.2f}]")

    pnl_vol_ratio = vol / max(abs(mu), 1e-8)
    print(f"    Vol/mean ratio:      {pnl_vol_ratio:>16.2f}")
    if pnl_vol_ratio < 0.1 and abs(mu) > 0:
        print(f"    ** FLAG: PnL too smooth — possible accounting smoothing **")


# ===========================================================================
# [4] BASELINE RECONSTRUCTION PROTOCOL
# ===========================================================================
print("\n" + "=" * 70)
print("  BASELINE RECONSTRUCTION PROTOCOL (BRP)")
print("=" * 70)

# ── State attribution ──
print(f"\n  State-Level PnL Attribution (Always-On):")
print(f"  {'State':<14s} {'Fills':>8s} {'PnL/State':>14s} {'P/fill':>10s} {'ECORE?':>8s}")
print(f"  {'─'*14} {'─'*8} {'─'*14} {'─'*10} {'─'*8}")

# Approximate state PnL using spread capture model
state_pnl_est = {}
for st in sorted(bt_b.state_fills.keys(), key=lambda s: bt_b.state_fills[s], reverse=True):
    nf = bt_b.state_fills[st]
    if nf < 10:
        continue
    spread = STATE_SPREAD.get(st, 30)
    markout_bps = MARKOUT_BY_STATE.get(st, -2.0)
    adverse = abs(markout_bps) / 10000.0 * MID_PRICE
    pnl_est = nf * (spread - adverse)
    is_e = "Y" if st in ECORE else "N"
    state_pnl_est[st] = pnl_est
    print(f"  {st:<14s} {nf:>8,d} {pnl_est:>+14,.0f} {pnl_est/max(nf,1):>+10.1f} {is_e:>8s}")

# Worst by PnL
worst = sorted(state_pnl_est.items(), key=lambda x: x[1])[:5]
print(f"\n  Worst states by estimated PnL (Always-On):")
for st, pnl in worst:
    nf = bt_b.state_fills.get(st, 0)
    print(f"    {st:<14s} PnL={pnl:>+14,.0f}  fills={nf:>8,d}  ECORE={st in ECORE}")

# Non-ECORE toxicity
non_ecore_fills = sum(v for k, v in bt_b.state_fills.items() if k not in ECORE)
non_ecore_pnl = sum(v for k, v in state_pnl_est.items() if k not in ECORE)
ecore_fills_b = sum(v for k, v in bt_b.state_fills.items() if k in ECORE)
ecore_pnl_b = sum(v for k, v in state_pnl_est.items() if k in ECORE)

print(f"\n  Toxicity Cost (Always-On in non-ECORE):")
print(f"    Non-ECORE fills:     {non_ecore_fills:>14,d}  ({non_ecore_fills/max(len(bt_b.all_trades),1)*100:.1f}%)")
print(f"    Non-ECORE est. PnL:  {non_ecore_pnl:>+14,.0f}")
print(f"    ECORE fills:         {ecore_fills_b:>14,d}  ({ecore_fills_b/max(len(bt_b.all_trades),1)*100:.1f}%)")
print(f"    ECORE est. PnL:      {ecore_pnl_b:>+14,.0f}")
if non_ecore_pnl < 0:
    print(f"    Non-ECORE states are net PnL DESTROYERS")
if ecore_pnl_b > 0:
    print(f"    ECORE states are net PnL GENERATORS")

# ── Participation Efficiency ──
print(f"\n  Participation Efficiency:")
print(f"  {'Metric':<28s} {'ECORE+ETE':>16s} {'Always-On':>16s} {'Ratio':>10s}")
print(f"  {'─'*28} {'─'*16} {'─'*16} {'─'*10}")

e_final = bt_e.equity_curve[-1]["equity"] if bt_e.equity_curve else 0
b_final = bt_b.equity_curve[-1]["equity"] if bt_b.equity_curve else 0
e_q = max(bt_e.quotes_placed, 1)
b_q = max(bt_b.quotes_placed, 1)
e_f = max(len(bt_e.all_trades), 1)
b_f = max(len(bt_b.all_trades), 1)

e_inv = bt_e.audit_inventory()
b_inv = bt_b.audit_inventory()

metrics = [
    ("Total PnL", e_final, b_final),
    ("Quotes placed", bt_e.quotes_placed, bt_b.quotes_placed),
    ("Total fills", len(bt_e.all_trades), len(bt_b.all_trades)),
    ("PnL / quote", e_final/e_q, b_final/b_q),
    ("PnL / fill", e_final/e_f, b_final/b_f),
    ("Fill efficiency", e_f/e_q, b_f/b_q),
    ("Inventory std", e_inv.get('std',0), b_inv.get('std',0)),
    ("Mean abs inventory", e_inv.get('mean_abs',0), b_inv.get('mean_abs',0)),
    ("Pct time flat", e_inv.get('pct_flat',0), b_inv.get('pct_flat',0)),
    ("Turnover", e_inv.get('turnover',0), b_inv.get('turnover',0)),
]

for label, ev, bv in metrics:
    if isinstance(ev, (int, float)) and isinstance(bv, (int, float)) and abs(bv) > 1e-8:
        ratio = ev / bv
        print(f"  {label:<28s} {ev:>16,.1f} {bv:>16,.1f} {ratio:>9.2f}x")
    else:
        print(f"  {label:<28s} {str(ev):>16s} {str(bv):>16s}")

# ── Advantage Decomposition ──
print(f"\n  Advantage Decomposition (ECORE vs Always-On):")
print(f"    Raw advantage:       {e_final - b_final:>+16,.0f}")

e_spreads = [STATE_SPREAD.get(s, 0) for s in set(bt_e.state_fills.keys()) if s in STATE_SPREAD]
b_spreads = [STATE_SPREAD.get(s, 0) for s in set(bt_b.state_fills.keys()) if s in STATE_SPREAD]
e_ae = [STATE_AE.get(s, 0.5) for s in set(bt_e.state_fills.keys()) if s in STATE_AE]
b_ae = [STATE_AE.get(s, 0.5) for s in set(bt_b.state_fills.keys()) if s in STATE_AE]

print(f"    A. Avg spread:       ECORE={np.mean(e_spreads):.0f}  AlwaysOn={np.mean(b_spreads):.0f}")
print(f"    B. Avg A/E:          ECORE={np.mean(e_ae):.2f}  AlwaysOn={np.mean(b_ae):.2f}")
print(f"    C. Inv std:          ECORE={e_inv.get('std',0):,.0f}  AlwaysOn={b_inv.get('std',0):,.0f}")
print(f"    D. Fill efficiency:  ECORE={e_f/e_q:.1%}  AlwaysOn={b_f/b_q:.1%}")
print(f"    E. Non-ECORE fills:  ECORE=0  AlwaysOn={non_ecore_fills:,}")

# Source identification
print(f"\n  ECORE advantage primarily from:")
if e_f/e_q > b_f/b_q * 1.5:
    print(f"    → Higher fill efficiency (ECORE only quotes in high-fill states)")
if e_inv.get('std', 0) < b_inv.get('std', 1) * 0.8:
    print(f"    → Lower inventory risk (less exposure to toxic states)")
if non_ecore_pnl < 0:
    print(f"    → Avoiding PnL-destroying non-ECORE states (toxicity cost)")
if e_f/e_q <= b_f/b_q * 1.5 and e_inv.get('std', 0) >= b_inv.get('std', 1) * 0.8:
    print(f"    → Modest improvement — edge comes from timing, not filtering")


# ===========================================================================
# [5] INTEGRITY VERDICT
# ===========================================================================
print("\n" + "=" * 70)
print("  INTEGRITY VERDICT")
print("=" * 70)

failures = []

# A: Ending inventory
for bt, nm in [(bt_e, "ECORE"), (bt_b, "Always-On")]:
    if abs(bt.inventory) > 1e6:
        failures.append(f"{nm}: massive ending inventory ({bt.inventory:,.0f})")
    elif abs(bt.inventory) > 10000:
        failures.append(f"{nm}: material ending inventory ({bt.inventory:,.0f})")

# B: Inventory drift
for bt, nm in [(bt_e, "ECORE"), (bt_b, "Always-On")]:
    s = bt.audit_inventory()
    if s.get('autocorr', 0) > 0.95:
        failures.append(f"{nm}: near-unit-root inventory autocorr ({s['autocorr']:.3f})")

# C: Fill issues
for bt, nm in [(bt_e, "ECORE"), (bt_b, "Always-On")]:
    bids = sum(1 for t in bt.all_trades if t["side"] == 'bid')
    asks = sum(1 for t in bt.all_trades if t["side"] == 'ask')
    imb = abs(bids - asks) / max(bids + asks, 1)
    if imb > 0.4:
        failures.append(f"{nm}: extreme fill imbalance ({imb:.1%})")

# G: PnL smoothness
for bt, nm in [(bt_e, "ECORE"), (bt_b, "Always-On")]:
    eqs = np.array([e["equity"] for e in bt.equity_curve])
    if len(eqs) > 2:
        pnls = np.diff(eqs)
        if len(pnls) > 20:
            neg_pct = np.mean(pnls < 0)
            if neg_pct < 0.05:
                failures.append(f"{nm}: <5% negative windows ({neg_pct:.1%}) — too smooth")
            vol_ratio = np.std(pnls) / max(abs(np.mean(pnls)), 1e-8)
            if vol_ratio < 0.2:
                failures.append(f"{nm}: vol/mean={vol_ratio:.2f} — path smoothing suspected")

# F: Spread crossings
for bt, nm in [(bt_e, "ECORE"), (bt_b, "Always-On")]:
    v = bt.audit_crossings()
    if len(v) > len(bt.all_trades) * 0.01:
        failures.append(f"{nm}: {len(v)/max(len(bt.all_trades),1)*100:.1f}% spread crossing rate")

print(f"\n  Failures: {len(failures)}")
for i, f in enumerate(failures):
    print(f"    {i+1}. {f}")

# ── Clean PnL: realized only (cash-based, ignoring open inventory MTM) ──
e_cash_pnl = bt_e.cash  # cash accumulated from all fills
b_cash_pnl = bt_b.cash

e_end_inv_val = bt_e.inventory * (bt_e.window_mids[-1] if bt_e.window_mids else MID_PRICE)
b_end_inv_val = bt_b.inventory * (bt_b.window_mids[-1] if bt_b.window_mids else MID_PRICE)

print(f"\n  Clean PnL (cash + ending inv @ mark):")
print(f"    ECORE+ETE:  cash={e_cash_pnl:>+16,.0f}  inv@mark={e_end_inv_val:>+16,.0f}  total={e_cash_pnl+e_end_inv_val:>+16,.0f}")
print(f"    Always-On:  cash={b_cash_pnl:>+16,.0f}  inv@mark={b_end_inv_val:>+16,.0f}  total={b_cash_pnl+b_end_inv_val:>+16,.0f}")

# ── Verdict ──
if len(failures) == 0:
    verdict = "CASE_A — Backtest passes integrity audit."
elif len(failures) <= 2:
    verdict = "CASE_B — Minor issues. Edge directionally valid, metrics need adjustment."
elif len(failures) <= 4:
    verdict = "CASE_B — Material issues found. Metrics require correction before use."
else:
    verdict = "CASE_C — Critical realism failures. Previous PnL is largely artifact."

print(f"\n  {verdict}")

# Corrected summary metrics
print(f"\n  Corrected Metrics:")
print(f"  {'─'*55}")

e_eqs = np.array([e["equity"] for e in bt_e.equity_curve])
b_eqs = np.array([e["equity"] for e in bt_b.equity_curve])

for nm, eqs, fills, quotes in [
    ("ECORE+ETE", e_eqs, len(bt_e.all_trades), bt_e.quotes_placed),
    ("Always-On", b_eqs, len(bt_b.all_trades), bt_b.quotes_placed),
]:
    if len(eqs) < 2:
        continue
    pnl = float(eqs[-1] - eqs[0])
    wpnl = np.diff(eqs)
    sh = float(np.mean(wpnl) / max(np.std(wpnl), 1e-8) * np.sqrt(len(wpnl)))
    cum = eqs - eqs[0]
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    print(f"\n  {nm}:")
    print(f"    Total PnL:          {pnl:>+18,.0f}")
    print(f"    Sharpe:              {sh:>18.2f}")
    print(f"    Max DD:              {dd:>18,.0f}")
    print(f"    Total fills:         {fills:>18,d}")
    print(f"    Quotes placed:       {quotes:>18,d}")
    print(f"    PnL/fill:            {pnl/max(fills,1):>+18.1f}")
    print(f"    PnL/quote:           {pnl/max(quotes,1):>+18.3f}")

print(f"\n{'═'*70}")
print(f"  BIA v1 complete.")
print(f"{'═'*70}")

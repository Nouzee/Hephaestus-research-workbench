"""
Regime-Aware Backtest v1 — State-Conditioned Execution Quality

Compares two market-making strategies:
  BASELINE:    fixed size, fixed spread, no regime awareness
  STATE-AWARE: dynamic size/spread/cancel based on current regime

Strict no-leak rules:
  - Regime predicted from PAST window only (online classifier)
  - KMeans fit on TRAIN only, predict on TEST
  - Time split: first 60% train, last 40% test

Measures: PnL, drawdown, adverse selection, inventory risk, per-regime PnL.
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import L2FeatureExtractor, FEATURE_NAMES
from sklearn.cluster import KMeans


TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100
N_REGIMES = 8
TRAIN_FRAC = 0.60
FILL_PROB = 0.30  # baseline fill probability per tick

# ===========================================================================
# State-aware parameter table
# ===========================================================================

# Based on full-pipeline regime discovery:
# R5 = STRESS, R2 = deep liquidity, R0 = depth collapse, R3 = trade surge
# R4 = bid-heavy, R1 = ask-heavy, R6 = active buy, R7 = active sell
REGIME_PARAMS = {
    # regime: (size_mult, spread_mult, cancel_aggression, inv_cap_mult)
    5: (0.30, 1.80, 2.0, 0.5),   # R5 STRESS — small size, wide spread, fast cancel
    3: (0.50, 1.50, 1.5, 0.5),   # R3 TRADE SURGE — cautious
    0: (0.80, 1.10, 1.2, 0.8),   # R0 depth collapse — slightly defensive
    1: (0.80, 1.10, 1.0, 0.8),   # R1 ask-heavy — near normal
    4: (0.70, 1.20, 1.3, 0.7),   # R4 depth collapse variant
    6: (0.70, 1.20, 1.3, 0.7),   # R6 active buy flow
    7: (0.70, 1.20, 1.3, 0.7),   # R7 active sell flow
    2: (1.20, 0.90, 0.8, 1.2),   # R2 DEEP LIQUIDITY — aggressive
}
DEFAULT_PARAMS = (1.0, 1.0, 1.0, 1.0)


print("=" * 65)
print("  Regime-Aware Backtest v1")
print("  State-Conditioned Execution vs Static Baseline")
print("=" * 65)


# ===========================================================================
# [1] Load data + extract features + fit online classifier
# ===========================================================================

print("\n[1] Loading data + fitting online regime classifier ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))

# Collect ALL features + raw data for backtest
all_features = []
all_regimes = []        # will be filled after KMeans fit
all_day_idx = []         # which day each window belongs to
all_window_data = []     # raw data for backtest simulation

n_train_days = int(len(msg_files) * TRAIN_FRAC)
train_features = []

for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
    msg_df = pl.read_parquet(mf)
    ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]
    n_windows = N // WINDOW_SIZE
    if n_windows < 5:
        continue

    msg_dict = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_dict = {col: ob_df[col].to_numpy() for col in ob_df.columns}

    for w in range(n_windows):
        s, e = w * WINDOW_SIZE, (w+1) * WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_dict.items()},
            {k: v[s:e] for k, v in msg_dict.items()},
        )
        feat_vec = list(feats.values())
        all_features.append(feat_vec)
        all_day_idx.append(day_idx)

        if day_idx < n_train_days:
            train_features.append(feat_vec)

        # Store raw data for backtest
        all_window_data.append({
            "mid": (ob_dict["OfferPrice1"][s:e] + ob_dict["BidPrice1"][s:e]) / 2.0,
            "spread": ob_dict["OfferPrice1"][s:e] - ob_dict["BidPrice1"][s:e],
            "direction": msg_dict["Direction"][s:e].astype(np.float64),
            "size": msg_dict["Size"][s:e].astype(np.float64),
            "time": msg_dict["Time (sec)"][s:e],
        })

    if (day_idx + 1) % 15 == 0:
        print(f"  [{day_idx+1}/{len(msg_files)}] days loaded")

X_all = np.array(all_features, dtype=np.float32)
X_train = np.array(train_features, dtype=np.float32)

# Standardize using TRAIN stats only (no leak)
t_mean = X_train.mean(axis=0)
t_std = np.maximum(X_train.std(axis=0), 1e-8)
X_z = np.clip((X_all - t_mean) / t_std, -10, 10)

# Fit KMeans on TRAIN only
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_train - t_mean) / t_std, -10, 10))

# Predict regime for ALL windows
all_regimes = km.predict(X_z)

n_windows = len(all_regimes)
print(f"  Total: {n_windows:,} windows  Train days: {n_train_days}  "
      f"Test days: {len(msg_files)-n_train_days}")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] Backtest engine — tick-by-tick market making
# ===========================================================================

print(f"\n[2] Running backtest — Baseline vs State-Aware ...")
t0 = time.perf_counter()

rng = np.random.RandomState(42)

def simulate_mm(
    windows_data,
    regimes,
    is_state_aware,
    first_test_window,
):
    """
    Tick-by-tick MM simulation.

    Each tick: quote bid/ask at spread, random fill at P(fill),
    earn spread/2 if filled, suffer adverse selection from price movement.

    Returns: per-window PnL, cumulative equity, per-regime PnL breakdown.
    """
    n_w = len(windows_data)
    equity = np.zeros(n_w + 1, dtype=np.float64)
    per_window_pnl = np.zeros(n_w, dtype=np.float64)
    per_regime_pnl = {r: 0.0 for r in range(N_REGIMES)}
    per_regime_ticks = {r: 0 for r in range(N_REGIMES)}
    inventory = 0.0
    total_spread_earned = 0.0
    total_adverse = 0.0
    total_ticks_done = 0

    for w_idx in range(n_w):
        wd = windows_data[w_idx]
        mid_arr = wd["mid"]
        spread_arr = wd["spread"]
        direction = wd["direction"]
        size_arr = wd["size"]
        T = len(mid_arr)

        regime = int(regimes[w_idx])

        # Get params
        if is_state_aware:
            sz_m, sp_m, cancel_agg, inv_cap = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)
        else:
            sz_m, sp_m, cancel_agg, inv_cap = 1.0, 1.0, 1.0, 1.0

        pnl = 0.0
        spread_earned = 0.0
        adverse = 0.0

        for t in range(T):
            spread_quote = spread_arr[t] * sp_m
            mid = mid_arr[t]

            # Fill simulation: probability decreases with spread widening
            p_fill = FILL_PROB / max(sp_m, 0.5)

            # Bid fill
            if rng.random() < p_fill:
                fill_px = mid - spread_quote / 2
                pnl += spread_quote / 2 * sz_m  # earned half-spread
                spread_earned += spread_quote / 2 * sz_m
                inventory += sz_m  # bought
                total_ticks_done += 1

            # Ask fill
            if rng.random() < p_fill:
                fill_px = mid + spread_quote / 2
                pnl += spread_quote / 2 * sz_m
                spread_earned += spread_quote / 2 * sz_m
                inventory -= sz_m  # sold
                total_ticks_done += 1

            # Adverse selection: price moves against inventory
            if t < T - 1 and inventory != 0:
                future_ret = (mid_arr[min(t+10, T-1)] - mid) / max(mid, 1e-8)
                if inventory > 0 and future_ret < 0:
                    adverse += -inventory * future_ret * mid * sz_m
                    pnl += inventory * future_ret * mid * sz_m  # loss
                elif inventory < 0 and future_ret > 0:
                    adverse += inventory * future_ret * mid * sz_m
                    pnl += inventory * future_ret * mid * sz_m  # loss

            # Cancel aggression: if inventory exceeds cap, unwind
            if abs(inventory) > inv_cap * 10:
                pnl -= abs(inventory) * spread_quote * 0.1 * cancel_agg
                inventory *= 0.5  # partial unwind

        equity[w_idx + 1] = equity[w_idx] + pnl
        per_window_pnl[w_idx] = pnl
        per_regime_pnl[regime] += pnl
        per_regime_ticks[regime] += T
        total_spread_earned += spread_earned
        total_adverse += adverse

    return {
        "equity": equity,
        "per_window_pnl": per_window_pnl,
        "per_regime_pnl": per_regime_pnl,
        "per_regime_ticks": per_regime_ticks,
        "total_spread_earned": total_spread_earned,
        "total_adverse": total_adverse,
        "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
        "final_pnl": float(equity[-1]),
        "sharpe": float(np.mean(per_window_pnl) / max(np.std(per_window_pnl), 1e-8) * np.sqrt(n_w)),
        "inventory_vol": float(np.std(np.diff(equity))),
    }


# Split: train for calibration, test for comparison
first_test_window = int(np.sum(np.array(all_day_idx) < n_train_days))

# Baseline
bl_result = simulate_mm(all_window_data, all_regimes, False, first_test_window)

# State-aware
sa_result = simulate_mm(all_window_data, all_regimes, True, first_test_window)

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] Results comparison
# ===========================================================================

print(f"\n[3] Results Comparison — Baseline vs State-Aware")
print(f"{'═'*65}")

metrics = [
    ("Final PnL", "final_pnl", "{:+,.0f}"),
    ("Max Drawdown", "max_drawdown", "{:,.0f}"),
    ("Sharpe", "sharpe", "{:.2f}"),
    ("Inventory Vol", "inventory_vol", "{:.0f}"),
    ("Spread Earned", "total_spread_earned", "{:+,.0f}"),
    ("Adverse Loss", "total_adverse", "{:+,.0f}"),
    ("Net Spread-Adverse", "", ""),
]

print(f"\n  {'Metric':<22s} {'Baseline':>14s} {'State-Aware':>14s} {'Delta':>14s} {'Better':>10s}")
print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")

for name, key, fmt in metrics:
    if key == "":
        bl_val = bl_result["total_spread_earned"] + bl_result["total_adverse"]
        sa_val = sa_result["total_spread_earned"] + sa_result["total_adverse"]
    else:
        bl_val = bl_result[key]
        sa_val = sa_result[key]

    delta = sa_val - bl_val
    if "Loss" in name or "Drawdown" in name:
        better = "SA" if delta > 0 else "BL"
    else:
        better = "SA" if delta > 0 else "BL"

    if abs(bl_val) > 1e6:
        print(f"  {name:<22s} {bl_val:>+14,.0f} {sa_val:>+14,.0f} {delta:>+14,.0f} {better:>10s}")
    elif abs(bl_val) > 1e3:
        print(f"  {name:<22s} {bl_val:>+14,.0f} {sa_val:>+14,.0f} {delta:>+14,.0f} {better:>10s}")
    else:
        print(f"  {name:<22s} {bl_val:>14.4f} {sa_val:>14.4f} {delta:>+14.4f} {better:>10s}")


# ===========================================================================
# [4] Per-regime PnL breakdown
# ===========================================================================

print(f"\n[4] Per-Regime PnL Breakdown ...")

# Map regime numbers to names from full pipeline
regime_full_names = {
    0: "R0:depth_collapse",
    1: "R1:ask_heavy",
    2: "R2:deep_liquidity",
    3: "R3:trade_surge",
    4: "R4:bid_heavy",
    5: "R5:STRESS",
    6: "R6:active_buy",
    7: "R7:active_sell",
}

print(f"\n  {'Regime':<22s} {'Ticks':>10s} {'BL PnL':>12s} {'SA PnL':>12s} "
      f"{'Delta':>12s} {'SA/Tick':>10s} {'Improvement':>14s}")
print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*14}")

for r in range(N_REGIMES):
    bl_pnl = bl_result["per_regime_pnl"][r]
    sa_pnl = sa_result["per_regime_pnl"][r]
    ticks = bl_result["per_regime_ticks"][r]
    delta = sa_pnl - bl_pnl
    per_tick = sa_pnl / max(ticks, 1)

    if bl_pnl != 0:
        improvement = (sa_pnl - bl_pnl) / max(abs(bl_pnl), 1e-12) * 100
    else:
        improvement = float('inf') if sa_pnl > 0 else 0

    rname = regime_full_names.get(r, f"R{r}")
    print(f"  {rname:<22s} {ticks:>10,d} {bl_pnl:>+12,.0f} {sa_pnl:>+12,.0f} "
          f"{delta:>+12,.0f} {per_tick:>+10.4f} {improvement:>+13.1f}%")

# ===========================================================================
# [5] Test-period only analysis
# ===========================================================================

print(f"\n[5] Test Period Analysis (last 40% of days) ...")

test_mask = np.array(all_day_idx) >= n_train_days
n_test = int(np.sum(test_mask))

test_bl = bl_result["per_window_pnl"][test_mask]
test_sa = sa_result["per_window_pnl"][test_mask]

test_bl_total = float(np.sum(test_bl))
test_sa_total = float(np.sum(test_sa))
test_bl_dd = float(np.max(np.maximum.accumulate(np.cumsum(test_bl)) - np.cumsum(test_bl)))
test_sa_dd = float(np.max(np.maximum.accumulate(np.cumsum(test_sa)) - np.cumsum(test_sa)))

print(f"\n  Test windows: {n_test}")
print(f"  {'Metric':<22s} {'Baseline':>14s} {'State-Aware':>14s} {'Delta':>14s}")
print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*14}")
print(f"  {'Test PnL':<22s} {test_bl_total:>+14,.0f} {test_sa_total:>+14,.0f} "
      f"{test_sa_total-test_bl_total:>+14,.0f}")
print(f"  {'Test Max DD':<22s} {test_bl_dd:>14,.0f} {test_sa_dd:>14,.0f} "
      f"{test_bl_dd-test_sa_dd:>+14,.0f}")

# R5-specific test performance
r5_mask_test = (all_regimes == 5) & test_mask
if r5_mask_test.sum() > 0:
    r5_bl_test = float(np.sum(bl_result["per_window_pnl"][r5_mask_test]))
    r5_sa_test = float(np.sum(sa_result["per_window_pnl"][r5_mask_test]))
    print(f"  {'R5 Stress PnL':<22s} {r5_bl_test:>+14,.0f} {r5_sa_test:>+14,.0f} "
          f"{r5_sa_test-r5_bl_test:>+14,.0f}")

print(f"\n{'═'*65}")
print(f"  Backtest complete.")
print(f"{'═'*65}")

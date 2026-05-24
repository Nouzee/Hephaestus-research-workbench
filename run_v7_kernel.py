"""
V7 Minimal Trading Kernel — one alpha, one throttle, one PnL.

Collapses the entire system into:
  1. PressureMemory → inventory skew (direct, continuous, no FSM)
  2. Structure volatility → risk throttle (on/off filter only)
  3. FillModel → stochastic execution PnL

Walk-forward validation across 2 folds.
Reports: total PnL, Sharpe, win rate, avg skew, throttle rate.
"""

import gc, sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.pressure_memory import PressureMemory, PressureMemoryConfig
from modules.execution.fill_model import FillModel
from sklearn.decomposition import sparse_encode

# ===========================================================================
# Config
# ===========================================================================

BATCH_SIZE, FWD_TICKS = 2048, 50
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

# Trading kernel parameters (the ONLY tunable knobs)
K_SKEW = 0.5            # position = -K * pressure_memory  (fade the pressure)
STRUCTURE_VOL_THRESHOLD = 2.0  # z-score: above this → reduce K

FOLDS = [
    {"name": "Fold 1", "train": (0.00, 0.60), "test": (0.60, 0.80)},
    {"name": "Fold 2", "train": (0.20, 0.80), "test": (0.80, 1.00)},
]

print("=" * 62)
print("  V7 Minimal Trading Kernel")
print("  Pressure → Position  |  Structure → Throttle  |  Fill → PnL")
print("=" * 62)


# ===========================================================================
# [1] Load data
# ===========================================================================

print("\n[1] Loading ...")
t0 = time.perf_counter()
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

raw = pl.read_parquet(SOURCE, columns=[
    "mid_px", "spread", "total_depth", "signed_imbalance", "duration_ms"])
offset = raw.shape[0] - N
mid_px = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
spread_arr = raw["spread"].to_numpy().astype(np.float64)[offset:]
depth = raw["total_depth"].to_numpy().astype(np.float64)[offset:]
signed_imb = raw["signed_imbalance"].to_numpy().astype(np.float64)[offset:]
duration = raw["duration_ms"].to_numpy().astype(np.float64)[offset:]
del raw

D0 = np.load(str(DICT_PATH))
n_batches = N // BATCH_SIZE

alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)
del X; gc.collect()

mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = np.abs(
    (mid_px[FWD_TICKS:] - mid_px[:-FWD_TICKS]) / (np.abs(mid_px[:-FWD_TICKS]) + 1e-12))
print(f"  {N:,} ticks, {n_batches} batches, time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Signal extractors
# ===========================================================================

def _ext(d,s,e):
    if e-s<10: return 0.0
    d0,d1=np.median(d[s:s+10]),np.median(d[e-10:e])
    return float(max(-(d1-d0)/max(d0,1e-12),0.0))
def _spr(sp,s,e):
    return float(np.log1p(np.max(sp[s:e])/max(np.mean(sp[s:e]),1e-12)))


# ===========================================================================
# [2] Walk-forward
# ===========================================================================

print(f"\n[2] Walk-forward ...")
print(f"{'='*62}")

all_results = []

for fold_idx, fold in enumerate(FOLDS):
    tr_s = int(fold["train"][0] * n_batches)
    tr_e = int(fold["train"][1] * n_batches)
    te_s = int(fold["test"][0] * n_batches)
    te_e = int(fold["test"][1] * n_batches)

    # ── Init kernel ──
    pm = PressureMemory(PressureMemoryConfig(
        decay_same=0.995, decay_flip=0.70, baseline_window=100))
    fm = FillModel()

    # Rolling stats for structure throttle
    spread_hist = np.zeros(100, dtype=np.float32)
    spread_hist_ptr = 0
    spread_hist_full = False

    # Train pass: warm up pressure memory + structure baseline
    for b in range(tr_s, tr_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        signed_obi_mean = float(np.mean(signed_imb[s:e]))
        pm.update(signed_obi_mean)
        # Structure baseline
        spread_shock = _spr(spread_arr, s, e)
        spread_hist[spread_hist_ptr] = spread_shock
        spread_hist_ptr = (spread_hist_ptr + 1) % 100
        if spread_hist_ptr == 0:
            spread_hist_full = True
    spread_hist_full = True

    # Test pass: trade
    pnl_history = []
    skew_history = []
    throttle_history = []

    for b in range(te_s, te_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        batch_mid = mid_px[s:e]
        batch_spread = spread_arr[s:e]
        batch_ret = mid_ret[s:e]
        batch_imb = float(np.mean(signed_imb[s:e]))

        # ── 1. Pressure → alpha ──
        _, p_sig = pm.update(batch_imb)
        pressure_z = p_sig["z_score"]
        pressure_dir = p_sig["direction"]

        # Position target: fade the pressure
        # If buy pressure → go short (negative position)
        position_target = -K_SKEW * pressure_dir * min(pressure_z, 3.0)

        # ── 2. Structure → throttle ──
        spread_shock = _spr(spread_arr, s, e)
        spread_z = ((spread_shock - spread_hist.mean())
                     / max(spread_hist.std(), 1e-8))
        throttle = 1.0 if spread_z < STRUCTURE_VOL_THRESHOLD else 0.5

        # Update structure history
        spread_hist[spread_hist_ptr] = spread_shock
        spread_hist_ptr = (spread_hist_ptr + 1) % 100

        # ── 3. Execute with fills ──
        # Skew controls which side we quote more aggressively
        skew = position_target * throttle

        # Bid/ask quoting decision based on skew
        # Positive skew → more aggressive on ask (sell into buy pressure)
        bid_size = max(1.0 - skew, 0.1)
        ask_size = max(1.0 + skew, 0.1)
        spread_mult = 1.0 + 0.1 * abs(skew)  # slightly wider when skewed

        # Fill probabilities
        fp = fm.probability(
            spread_mult=spread_mult,
            queue_ratio=0.3,
            imbalance=batch_imb,
            vol_z=spread_z,
            pressure_dir=pressure_dir,
            pressure_z=pressure_z,
        )

        # Simulate fills stochastically (size affects PnL, not fill probability)
        rng = np.random.RandomState(42 + b)
        fills_bid = rng.binomial(1, fp["p_fill_bid"], BATCH_SIZE).astype(bool)
        fills_ask = rng.binomial(1, fp["p_fill_ask"], BATCH_SIZE).astype(bool)

        # PnL: spread earned on fills minus adverse selection (scaled by quote size)
        spread_half = batch_spread * 0.5 * spread_mult
        adverse = np.abs(batch_ret) * batch_mid

        pnl = 0.0
        for t in range(BATCH_SIZE):
            if fills_bid[t]:
                pnl += (spread_half[t] - adverse[t]) * bid_size
            if fills_ask[t]:
                pnl += (spread_half[t] - adverse[t]) * ask_size

        pnl_history.append(pnl)
        skew_history.append(skew)
        throttle_history.append(throttle)

    # ── Metrics ──
    pnl_arr = np.array(pnl_history)
    total_pnl = float(np.sum(pnl_arr))
    mean_pnl = float(np.mean(pnl_arr))
    std_pnl = float(np.std(pnl_arr))
    sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(len(pnl_arr))
    win_rate = float(np.mean(pnl_arr > 0))
    avg_skew = float(np.mean(np.abs(skew_history)))
    throttle_rate = float(np.mean(np.array(throttle_history) < 1.0))

    # Pressure vs future PnL correlation (validation)
    p_hist = np.array(pm.P_trace[-len(pnl_arr):])
    corr_p_pnl = np.corrcoef(np.abs(p_hist), pnl_arr)[0, 1]

    print(f"\n  {fold['name']}:")
    print(f"    PnL: {total_pnl:+,.0f}  Mean: {mean_pnl:+,.1f}/batch  "
          f"Sharpe: {sharpe:.2f}  Win: {win_rate:.1%}")
    print(f"    Avg |skew|: {avg_skew:.3f}  Throttle rate: {throttle_rate:.1%}")
    print(f"    Corr(|P|, PnL): {corr_p_pnl:+.4f}")

    all_results.append({
        "name": fold["name"], "total_pnl": total_pnl, "mean_pnl": mean_pnl,
        "sharpe": sharpe, "win_rate": win_rate, "avg_skew": avg_skew,
        "throttle_rate": throttle_rate, "corr_p_pnl": corr_p_pnl,
    })


# ===========================================================================
# [3] Parameter sweep (lightweight)
# ===========================================================================

print(f"\n[3] Parameter sweep (K_skew × struct_threshold) ...")
print(f"  {'K_skew':>7s} {'thresh':>7s} {'Fold1 PnL':>12s} {'Fold2 PnL':>12s} "
      f"{'Total':>12s} {'Sharpe':>8s}")

best_total = -float('inf')
best_params = (K_SKEW, STRUCTURE_VOL_THRESHOLD)

for K in [0.3, 0.5, 0.7, 1.0]:
    for thresh in [1.5, 2.0, 2.5, 3.0]:
        fold_pnls = []
        fold_sharpes = []
        for fold_idx, fold in enumerate(FOLDS):
            tr_s = int(fold["train"][0] * n_batches)
            tr_e = int(fold["train"][1] * n_batches)
            te_s = int(fold["test"][0] * n_batches)
            te_e = int(fold["test"][1] * n_batches)

            pm = PressureMemory(PressureMemoryConfig(decay_same=0.995, decay_flip=0.70))
            fm = FillModel()
            spread_hist = np.zeros(100, dtype=np.float32)
            spread_hist_ptr = 0

            for b in range(tr_s, tr_e):
                pm.update(float(np.mean(signed_imb[b*BATCH_SIZE:(b+1)*BATCH_SIZE])))
                s_sp = _spr(spread_arr, b*BATCH_SIZE, (b+1)*BATCH_SIZE)
                spread_hist[spread_hist_ptr] = s_sp
                spread_hist_ptr = (spread_hist_ptr + 1) % 100

            batch_pnls = []
            for b in range(te_s, te_e):
                s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
                imb = float(np.mean(signed_imb[s:e]))
                _, p_sig = pm.update(imb)
                pz, pd = p_sig["z_score"], p_sig["direction"]
                pos = -K * pd * min(pz, 3.0)

                s_sp = _spr(spread_arr, s, e)
                s_z = (s_sp - spread_hist.mean()) / max(spread_hist.std(), 1e-8)
                thrott = 1.0 if s_z < thresh else 0.5
                skew = pos * thrott

                bid_sz = max(1.0 - skew, 0.1)
                ask_sz = max(1.0 + skew, 0.1)
                sp_mult = 1.0 + 0.1 * abs(skew)

                fp = fm.probability(spread_mult=sp_mult, queue_ratio=0.3,
                                    imbalance=imb, vol_z=s_z,
                                    pressure_dir=pd, pressure_z=pz)
                rng = np.random.RandomState(42 + b)
                f_bid = rng.binomial(1, fp["p_fill_bid"], BATCH_SIZE)
                f_ask = rng.binomial(1, fp["p_fill_ask"], BATCH_SIZE)

                sh = spread_arr[s:e] * 0.5 * sp_mult
                adv = np.abs(mid_ret[s:e]) * mid_px[s:e]
                pnl_b = 0.0
                for t in range(BATCH_SIZE):
                    if f_bid[t]: pnl_b += (sh[t] - adv[t]) * bid_sz
                    if f_ask[t]: pnl_b += (sh[t] - adv[t]) * ask_sz
                batch_pnls.append(pnl_b)

                spread_hist[spread_hist_ptr] = s_sp
                spread_hist_ptr = (spread_hist_ptr + 1) % 100

            pnl_arr = np.array(batch_pnls)
            fold_pnls.append(float(np.sum(pnl_arr)))
            fold_sharpes.append(float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-8)
                                      * np.sqrt(len(pnl_arr))))

        total = sum(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        marker = " < BEST" if total > best_total else ""
        if total > best_total:
            best_total = total
            best_params = (K, thresh)

        print(f"  {K:>7.2f} {thresh:>7.1f} {fold_pnls[0]:>+12,.0f} "
              f"{fold_pnls[1]:>+12,.0f} {total:>+12,.0f} {avg_sharpe:>8.2f}{marker}")

print(f"\n  Best: K_skew={best_params[0]}, thresh={best_params[1]} "
      f"(total PnL={best_total:+,.0f})")

print(f"\n{'='*62}")
print(f"  V7 Kernel complete.")
print(f"{'='*62}")

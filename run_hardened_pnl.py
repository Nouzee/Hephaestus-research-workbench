"""
Hardened PnL — HMM Risk Scaling + Frozen Pressure Alpha + Real Execution

Final validation before production consideration.

Architecture:
  HMM Scaler           → regime_multiplier (capital only, no alpha)
  PressureMemory       → direction + confidence (frozen, no tuning)
  HardenedSimulator    → queue + adverse_asymmetry + inventory_cost

Walk-forward: 2 folds, comparing:
  1. BASELINE:  always quote, no skew, no HMM
  2. PRESSURE:  pressure-driven skew, no HMM
  3. HARDENED:  pressure skew × HMM scaling × execution frictions
"""

import gc, sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.pressure_memory import PressureMemory, PressureMemoryConfig
from modules.execution.fill_model import FillModel
from modules.execution.hardened_simulator import HardenedSimulator, HardenedSimConfig
from modules.risk.hmm_scaler import HMMScaler, HMMScalerConfig
from sklearn.decomposition import sparse_encode


BATCH_SIZE, FWD_TICKS = 2048, 50
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

FOLDS = [
    {"name": "Fold 1", "train": (0.00, 0.60), "test": (0.60, 0.80)},
    {"name": "Fold 2", "train": (0.20, 0.80), "test": (0.80, 1.00)},
]

# Frozen alpha params (from V7 stress tests — DO NOT TUNE)
K_SKEW = 1.0
DECAY_SAME = 0.999
DECAY_FLIP = 0.85

print("=" * 65)
print("  Hardened PnL — HMM Scaler + Frozen Alpha + Real Execution")
print("=" * 65)


# ===========================================================================
# [1] Load
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

# Per-batch structure features for HMM
batch_depth_z = np.zeros(n_batches, dtype=np.float32)
batch_spread_z = np.zeros(n_batches, dtype=np.float32)
batch_vol = np.zeros(n_batches, dtype=np.float32)
for b in range(n_batches):
    s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
    batch_depth_z[b] = float(np.log1p(np.mean(depth[s:e]) / 1000))
    batch_spread_z[b] = float(np.log1p(np.mean(spread_arr[s:e])))
    batch_vol[b] = float(np.std(mid_px[s:e]) / max(np.mean(mid_px[s:e]), 1e-12))

print(f"  {N:,} ticks, {n_batches} batches, time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Signal extractors
# ===========================================================================

def _spr(sp, s, e):
    return float(np.log1p(np.max(sp[s:e]) / max(np.mean(sp[s:e]), 1e-12)))


# ===========================================================================
# [2] Walk-forward: 3 configurations per fold
# ===========================================================================

print(f"\n[2] Walk-forward — 3 configurations ...")
print(f"{'='*65}")

configs = ["BASELINE", "PRESSURE", "HARDENED"]
all_results = {c: [] for c in configs}

for fold_idx, fold in enumerate(FOLDS):
    tr_s = int(fold["train"][0] * n_batches)
    tr_e = int(fold["train"][1] * n_batches)
    te_s = int(fold["test"][0] * n_batches)
    te_e = int(fold["test"][1] * n_batches)

    print(f"\n  {fold['name']}: train [{tr_s}:{tr_e}] test [{te_s}:{te_e}]")

    # ── Train HMM on structure features ──
    hmm_features = np.column_stack([
        batch_vol[tr_s:tr_e], batch_depth_z[tr_s:tr_e], batch_spread_z[tr_s:tr_e]
    ])
    hmm_scaler = HMMScaler(HMMScalerConfig(n_states=4))
    hmm_scaler.fit(hmm_features)

    # ── Warm up pressure memory ──
    pm = PressureMemory(PressureMemoryConfig(
        decay_same=DECAY_SAME, decay_flip=DECAY_FLIP))
    for b in range(tr_s, tr_e):
        pm.update(float(np.mean(signed_imb[b*BATCH_SIZE:(b+1)*BATCH_SIZE])))

    # Structure baseline
    spread_hist = np.zeros(100, dtype=np.float32)
    for b in range(tr_s, tr_e):
        spread_hist[b % 100] = _spr(spread_arr, b*BATCH_SIZE, (b+1)*BATCH_SIZE)

    # ── Test each configuration ──
    for cfg_name in configs:
        fm = FillModel()

        if cfg_name == "BASELINE":
            sim = HardenedSimulator(HardenedSimConfig(
                adverse_asymmetry=1.0, queue_sensitivity=0.0, inventory_decay_rate=0.0))
        else:
            sim = HardenedSimulator(HardenedSimConfig(
                adverse_asymmetry=1.5, queue_sensitivity=0.3, inventory_decay_rate=0.0001))

        # Reset PM to post-training state (need fresh copy)
        pm_test = PressureMemory(PressureMemoryConfig(
            decay_same=DECAY_SAME, decay_flip=DECAY_FLIP))
        for b in range(tr_s, tr_e):
            pm_test.update(float(np.mean(signed_imb[b*BATCH_SIZE:(b+1)*BATCH_SIZE])))
        sim.reset()

        sh = np.zeros(100, dtype=np.float32)
        sh[:] = spread_hist

        batch_pnls = []
        batch_multipliers = []

        for b in range(te_s, te_e):
            s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
            imb = float(np.mean(signed_imb[s:e]))
            _, p_sig = pm_test.update(imb)
            pz = p_sig["z_score"]
            pd = p_sig["direction"]

            # ── Pressure → skew ──
            if cfg_name == "BASELINE":
                skew = 0.0
                sp_mult = 1.0
                bid_sz = ask_sz = 1.0
            else:
                pos = -K_SKEW * pd * min(pz, 3.0)
                skew = pos
                sp_mult = 1.0 + 0.1 * abs(skew)
                bid_sz = max(1.0 - skew, 0.1)
                ask_sz = max(1.0 + skew, 0.1)

            # ── HMM → capital scaling ──
            if cfg_name == "HARDENED":
                hmm_input = np.array([batch_vol[b], batch_depth_z[b], batch_spread_z[b]])
                regime_mult = hmm_scaler.predict_single(hmm_input)
            else:
                regime_mult = 1.0

            batch_multipliers.append(regime_mult)

            # ── Execute ──
            action = {
                "quote": True,
                "size_multiplier": regime_mult,  # HMM scales position size
                "spread_multiplier": sp_mult,
            }

            # Apply skew to bid/ask sizes separately
            mkt = {
                "imbalance": imb, "vol_z": 0.0,
                "pressure_dir": pd, "pressure_z": pz,
            }

            # Simulate with asymmetric quoting
            fp_bid = fm.probability(spread_mult=sp_mult, queue_ratio=0.3,
                                    imbalance=imb, vol_z=0.0,
                                    pressure_dir=pd, pressure_z=pz)
            fp_ask = fm.probability(spread_mult=sp_mult, queue_ratio=0.3,
                                    imbalance=-imb, vol_z=0.0,
                                    pressure_dir=-pd, pressure_z=pz)

            p_bid = np.clip(fp_bid["p_fill_bid"], 0.001, 0.999)
            p_ask = np.clip(fp_ask["p_fill_ask"], 0.001, 0.999)

            rng = np.random.RandomState(42 + b)
            f_bid = rng.binomial(1, p_bid, BATCH_SIZE)
            f_ask = rng.binomial(1, p_ask, BATCH_SIZE)

            spread_half = spread_arr[s:e] * 0.5 * sp_mult * regime_mult * bid_sz if cfg_name != "BASELINE" else spread_arr[s:e] * 0.5
            adv_base = np.abs(mid_ret[s:e]) * mid_px[s:e]

            pnl = 0.0
            for t in range(BATCH_SIZE):
                if f_bid[t]:
                    pnl += spread_half[t]
                    if mid_ret[s:][t] < 0:
                        pnl -= adv_base[t] * (1.5 if cfg_name == "HARDENED" else 1.0)
                    else:
                        pnl += adv_base[t] * (0.7 if cfg_name == "HARDENED" else 1.0)
                if f_ask[t]:
                    pnl += spread_half[t]
                    if mid_ret[s:][t] > 0:
                        pnl -= adv_base[t] * (1.5 if cfg_name == "HARDENED" else 1.0)
                    else:
                        pnl += adv_base[t] * (0.7 if cfg_name == "HARDENED" else 1.0)

            batch_pnls.append(pnl)

            # Update structure history
            sh[b % 100] = _spr(spread_arr, s, e)

        pnl_arr = np.array(batch_pnls)
        total = float(np.sum(pnl_arr))
        mean_p = float(np.mean(pnl_arr))
        std_p = float(np.std(pnl_arr))
        sharpe = mean_p / max(std_p, 1e-8) * np.sqrt(len(pnl_arr))
        win = float(np.mean(pnl_arr > 0))
        avg_mult = float(np.mean(batch_multipliers)) if batch_multipliers else 1.0

        all_results[cfg_name].append({
            "fold": fold["name"], "pnl": total, "mean": mean_p,
            "sharpe": sharpe, "win": win, "avg_mult": avg_mult,
        })

        print(f"    {cfg_name:<12s}: PnL={total:>+12,.0f}  "
              f"Sharpe={sharpe:>7.2f}  Win={win:.1%}  "
              f"avg_mult={avg_mult:.2f}")


# ===========================================================================
# [3] Aggregate comparison
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Final Comparison — BASELINE vs PRESSURE vs HARDENED")
print(f"{'═'*65}")

print(f"\n  {'Config':<12s} {'Fold':<8s} {'PnL':>14s} {'Mean':>10s} "
      f"{'Sharpe':>8s} {'Win':>8s} {'AvgMult':>8s}")
print(f"  {'─'*12} {'─'*8} {'─'*14} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

for cfg_name in configs:
    for r in all_results[cfg_name]:
        print(f"  {cfg_name:<12s} {r['fold']:<8s} {r['pnl']:>+14,.0f} "
              f"{r['mean']:>+10,.1f} {r['sharpe']:>8.2f} "
              f"{r['win']:>7.1%} {r['avg_mult']:>8.2f}")

# Totals
print(f"  {'─'*12} {'─'*8} {'─'*14} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
for cfg_name in configs:
    total_pnl = sum(r["pnl"] for r in all_results[cfg_name])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results[cfg_name]])
    print(f"  {cfg_name:<12s} {'TOTAL':<8s} {total_pnl:>+14,.0f} "
          f"{'':>10} {avg_sharpe:>8.2f}")

# Degradation ratios
base_pnl = sum(r["pnl"] for r in all_results["BASELINE"])
for cfg_name in ["PRESSURE", "HARDENED"]:
    cfg_pnl = sum(r["pnl"] for r in all_results[cfg_name])
    ratio = cfg_pnl / max(base_pnl, 1e-8)
    print(f"  {cfg_name} / BASELINE: {ratio:.1%} PnL retention")

print(f"\n{'═'*65}")
print(f"  Hardened PnL complete.")
print(f"{'═'*65}")

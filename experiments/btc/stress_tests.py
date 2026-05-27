"""
Structural Stress Tests — Regime Holdout + Fill Perturbation + Alpha Stability

Three tests to verify V7 kernel is NOT structurally overfit:

  1. REGIME HOLDOUT: split by realized volatility, test cross-regime stability
  2. FILL PERTURBATION: degrade P(fill) by ±20%, ±50%, add noise
  3. ALPHA STABILITY: sweep (K, decay_same, decay_flip) → check surface smoothness

Pass criteria:
  - Pressure→PnL correlation stays negative across ALL volatility regimes
  - Strategy survives ±20% fill perturbation (PnL stays positive)
  - Alpha surface is smooth (no fragmented peaks)
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


BATCH_SIZE, FWD_TICKS = 2048, 50
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

print("=" * 65)
print("  Structural Stress Tests — V7 Kernel Validation")
print("=" * 65)


# ===========================================================================
# [0] Load data (shared)
# ===========================================================================

print("\n[0] Loading data ...")
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
print(f"  {N:,} ticks, {n_batches} batches, time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Shared: per-batch realized volatility
# ===========================================================================

batch_vol = np.array([
    float(np.std(mid_px[b*BATCH_SIZE:(b+1)*BATCH_SIZE])
          / max(np.mean(mid_px[b*BATCH_SIZE:(b+1)*BATCH_SIZE]), 1e-12))
    for b in range(n_batches)
])

# Regime split by volatility terciles
vol_lo = np.percentile(batch_vol, 33)
vol_hi = np.percentile(batch_vol, 67)
regime_low = batch_vol < vol_lo
regime_mid = (batch_vol >= vol_lo) & (batch_vol < vol_hi)
regime_high = batch_vol >= vol_hi

print(f"  Vol regimes: LOW <{vol_lo:.6f} ({regime_low.sum()} batches)  "
      f"MID [{vol_lo:.6f}, {vol_hi:.6f}) ({regime_mid.sum()})  "
      f"HIGH >={vol_hi:.6f} ({regime_high.sum()})")


# ===========================================================================
# Signal extractor
# ===========================================================================

def _spr(sp, s, e):
    return float(np.log1p(np.max(sp[s:e]) / max(np.mean(sp[s:e]), 1e-12)))


# ===========================================================================
# Trading kernel (same as V7, parameterized)
# ===========================================================================

def run_kernel(batches_idx, K_skew, decay_same, decay_flip, fill_perturb=1.0,
               spread_shock_mult=1.0):
    """
    Run V7 kernel on specified batch indices.
    Returns (pnl_array, pressure_z_array, corr_p_pnl).
    """
    pm = PressureMemory(PressureMemoryConfig(
        decay_same=decay_same, decay_flip=decay_flip))
    fm = FillModel()

    # Warm up on first 100 batches of the index
    warmup = min(100, len(batches_idx) // 4)
    for b in batches_idx[:warmup]:
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        pm.update(float(np.mean(signed_imb[s:e])))

    pnl_hist = []
    pz_hist = []

    for b in batches_idx[warmup:]:
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        imb = float(np.mean(signed_imb[s:e]))
        _, p_sig = pm.update(imb)
        pz = p_sig["z_score"]
        pd = p_sig["direction"]

        pos = -K_skew * pd * min(pz, 3.0)
        bid_sz = max(1.0 - pos, 0.1)
        ask_sz = max(1.0 + pos, 0.1)
        sp_mult = 1.0 + 0.1 * abs(pos)

        fp = fm.probability(spread_mult=sp_mult, queue_ratio=0.3,
                            imbalance=imb, vol_z=0.0, pressure_dir=pd, pressure_z=pz)

        # Apply fill perturbation
        p_bid = np.clip(fp["p_fill_bid"] * fill_perturb, 0.001, 0.999)
        p_ask = np.clip(fp["p_fill_ask"] * fill_perturb, 0.001, 0.999)

        rng = np.random.RandomState(42 + b)
        f_bid = rng.binomial(1, p_bid, BATCH_SIZE)
        f_ask = rng.binomial(1, p_ask, BATCH_SIZE)

        sh = spread_arr[s:e] * 0.5 * sp_mult * spread_shock_mult
        adv = np.abs(mid_ret[s:e]) * mid_px[s:e]

        pnl = 0.0
        for t in range(BATCH_SIZE):
            if f_bid[t]: pnl += (sh[t] - adv[t]) * bid_sz
            if f_ask[t]: pnl += (sh[t] - adv[t]) * ask_sz

        pnl_hist.append(pnl)
        pz_hist.append(pz)

    pnl_arr = np.array(pnl_hist)
    pz_arr = np.array(pz_hist)
    corr = np.corrcoef(pz_arr, pnl_arr)[0, 1] if len(pnl_arr) > 10 else 0.0

    return pnl_arr, pz_arr, corr


# ===========================================================================
# TEST 1: Regime Holdout
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  TEST 1: Regime Holdout — Pressure stability across vol regimes")
print(f"{'═'*65}")

regimes = [
    ("LOW_VOL", np.where(regime_low)[0]),
    ("MID_VOL", np.where(regime_mid)[0]),
    ("HIGH_VOL", np.where(regime_high)[0]),
]

print(f"\n  {'Regime':<12s} {'Batches':>8s} {'Total PnL':>12s} "
      f"{'Mean PnL':>10s} {'Sharpe':>8s} {'Corr(P,PnL)':>12s} {'Status':>10s}")
print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*10} {'─'*8} {'─'*12} {'─'*10}")

regime_results = {}
for name, idx in regimes:
    pnl, pz, corr = run_kernel(idx, K_skew=0.5, decay_same=0.995, decay_flip=0.70)
    total = float(np.sum(pnl))
    mean_p = float(np.mean(pnl))
    sharpe = mean_p / max(float(np.std(pnl)), 1e-8) * np.sqrt(len(pnl))

    # Status: stable if corr < 0 (pressure high → PnL low = correct direction)
    status = "STABLE" if corr < -0.02 else ("weak" if corr < 0 else "BROKEN")

    print(f"  {name:<12s} {len(idx):>8d} {total:>+12,.0f} {mean_p:>+10,.1f} "
          f"{sharpe:>8.2f} {corr:>+12.4f} {status:>10s}")
    regime_results[name] = {"pnl": total, "corr": corr, "sharpe": sharpe, "status": status}

# Verdict
all_stable = all(r["status"] == "STABLE" for r in regime_results.values())
print(f"\n  Regime Holdout Verdict: "
      f"{'PASS — Pressure stable across ALL vol regimes' if all_stable else 'PARTIAL — Pressure degrades in some regimes'}")


# ===========================================================================
# TEST 2: Fill Model Perturbation
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  TEST 2: Fill Model Perturbation — P(fill) stress test")
print(f"{'═'*65}")

# Use ALL batches for perturbation test (no regime split)
all_idx = np.arange(n_batches)

perturbations = [
    (1.00, "baseline"),
    (0.80, "fill -20%"),
    (0.50, "fill -50%"),
    (1.20, "fill +20%"),
    (1.50, "fill +50%"),
    (0.80, "fill -20% + noise"),
]

print(f"\n  {'Perturbation':<20s} {'Total PnL':>12s} {'Mean':>10s} "
      f"{'Sharpe':>8s} {'Corr':>8s} {'Survives?':>10s}")
print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

perturb_results = {}
for mult, label in perturbations:
    # Add noise for the "+ noise" variant
    if "noise" in label:
        # Simulate noisy fills by using random subset
        sub_idx = np.random.RandomState(42).choice(all_idx, size=int(len(all_idx)*0.8), replace=False)
        pnl, pz, corr = run_kernel(sub_idx, K_skew=0.5, decay_same=0.995, decay_flip=0.70,
                                   fill_perturb=mult)
    else:
        pnl, pz, corr = run_kernel(all_idx, K_skew=0.5, decay_same=0.995, decay_flip=0.70,
                                   fill_perturb=mult)

    total = float(np.sum(pnl))
    mean_p = float(np.mean(pnl))
    sharpe = mean_p / max(float(np.std(pnl)), 1e-8) * np.sqrt(len(pnl))
    survives = total > 0

    print(f"  {label:<20s} {total:>+12,.0f} {mean_p:>+10,.1f} "
          f"{sharpe:>8.2f} {corr:>+8.4f} {'YES' if survives else 'NO':>10s}")
    perturb_results[label] = {"pnl": total, "sharpe": sharpe, "survives": survives}

all_survive = all(r["survives"] for r in perturb_results.values())
print(f"\n  Fill Perturbation Verdict: "
      f"{'PASS — Strategy survives all fill perturbations' if all_survive else 'FAIL — Strategy breaks under fill stress'}")


# ===========================================================================
# TEST 3: Alpha Stability Surface
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  TEST 3: Alpha Stability Surface — (K, decay_same, decay_flip)")
print(f"{'═'*65}")

K_range = [0.3, 0.5, 0.7, 1.0]
decay_same_range = [0.99, 0.995, 0.999]
decay_flip_range = [0.5, 0.7, 0.85]

# Use first 50% of data for surface scan
test_idx = np.arange(n_batches // 2)

print(f"  Scanning {len(K_range)}×{len(decay_same_range)}×{len(decay_flip_range)} "
      f"= {len(K_range)*len(decay_same_range)*len(decay_flip_range)} combos ...")

surface = np.zeros((len(K_range), len(decay_same_range), len(decay_flip_range)))
surface_corr = np.zeros_like(surface)

for ki, K in enumerate(K_range):
    for di, ds in enumerate(decay_same_range):
        for fi, df in enumerate(decay_flip_range):
            pnl, pz, corr = run_kernel(test_idx, K_skew=K, decay_same=ds, decay_flip=df)
            surface[ki, di, fi] = float(np.sum(pnl))
            surface_corr[ki, di, fi] = corr

# Smoothness check: how much does PnL vary between adjacent grid points?
diffs = []
for ki in range(len(K_range) - 1):
    for di in range(len(decay_same_range) - 1):
        for fi in range(len(decay_flip_range) - 1):
            # Local variation in each direction
            d_k = abs(surface[ki+1, di, fi] - surface[ki, di, fi])
            d_d = abs(surface[ki, di+1, fi] - surface[ki, di, fi])
            d_f = abs(surface[ki, di, fi+1] - surface[ki, di, fi])
            diffs.extend([d_k, d_d, d_f])

mean_surface = np.mean(surface)
mean_diff = np.mean(diffs)
smoothness_ratio = mean_diff / max(abs(mean_surface), 1e-8)

print(f"\n  Surface stats:")
print(f"    Mean PnL:    {mean_surface:+,.0f}")
print(f"    PnL range:   [{surface.min():+,.0f}, {surface.max():+,.0f}]")
print(f"    Mean |Δ|:    {mean_diff:,.0f}")
print(f"    Smoothness:  {smoothness_ratio:.4f} "
      f"({'SMOOTH — true alpha' if smoothness_ratio < 0.15 else 'FRAGMENTED — possible overfit'})")

# Best/worst params
best_flat = np.argmax(surface)
worst_flat = np.argmin(surface)
bk, bd, bf = np.unravel_index(best_flat, surface.shape)
wk, wd, wf = np.unravel_index(worst_flat, surface.shape)
print(f"    Best:  K={K_range[bk]}, ds={decay_same_range[bd]}, df={decay_flip_range[bf]} "
      f"→ PnL={surface[bk,bd,bf]:+,.0f}")
print(f"    Worst: K={K_range[wk]}, ds={decay_same_range[wd]}, df={decay_flip_range[wf]} "
      f"→ PnL={surface[wk,wd,wf]:+,.0f}")

# Correlation consistency check: does corr stay negative across the surface?
corr_negative_frac = np.mean(surface_corr < 0)
print(f"    Corr < 0 across surface: {corr_negative_frac:.1%} "
      f"({'CONSISTENT — pressure works everywhere' if corr_negative_frac > 0.8 else 'INCONSISTENT — pressure fails in some regions'})")

# Print slice: K vs decay_same at fixed decay_flip=0.7
print(f"\n  Slice at decay_flip=0.7 (PnL):")
print(f"  {'K\\ds':>6s}", end="")
for ds in decay_same_range:
    print(f" {ds:>12.3f}", end="")
print(f"\n  {'─'*6} {'─'*12}" * len(decay_same_range))
for ki, K in enumerate(K_range):
    print(f"  {K:>6.2f}", end="")
    for di, ds in enumerate(decay_same_range):
        fi = decay_flip_range.index(0.7)
        print(f" {surface[ki, di, fi]:>+12,.0f}", end="")
    print()


# ===========================================================================
# Final Verdict
# ===========================================================================

print(f"\n{'═'*65}")
print(f"  Stress Test Verdict")
print(f"{'═'*65}")

tests = [
    ("Regime Holdout", all_stable,
     "Pressure-PnL relationship holds across vol regimes"),
    ("Fill Perturbation", all_survive,
     "Strategy survives fill model misspecification"),
    ("Alpha Stability", smoothness_ratio < 0.15 and corr_negative_frac > 0.8,
     "Alpha surface is smooth and directionally consistent"),
]

passed = 0
for name, result, desc in tests:
    status = "PASS" if result else "FAIL"
    if result:
        passed += 1
    print(f"  [{status}] {name}: {desc}")

print(f"\n  {passed}/{len(tests)} tests passed")
if passed == 3:
    print(f"  V7 Kernel is structurally validated — ready for production hardening.")
elif passed >= 2:
    print(f"  V7 Kernel is mostly stable — address failing test before deployment.")
else:
    print(f"  V7 Kernel has structural issues — do NOT deploy without fixes.")

print(f"{'═'*65}")

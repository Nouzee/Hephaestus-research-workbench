"""
State Engine Pipeline — Unified MarketState in action.

Demonstrates the V6 architecture: all modules read/write one MarketState,
FillModel reads state-only, ConsistencyScanner checks alignment.

Walk-forward: 2 folds, state-update → action → fill PnL → consistency scan.
"""

import gc, sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.state.market_state import MarketState
from modules.state.state_updater import StateUpdater, StateUpdaterConfig
from modules.state.consistency_scanner import ConsistencyScanner
from modules.execution.fill_model import FillModel
from modules.execution.execution_simulator import ExecutionSimulator
from modules.execution.pnl_attribution import PnLAttributor
from sklearn.decomposition import sparse_encode

# ===========================================================================
# Config
# ===========================================================================

BATCH_SIZE = 2048
FWD_TICKS = 50
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

FOLDS = [
    {"name": "Fold 1", "train": (0.00, 0.60), "test": (0.60, 0.80)},
    {"name": "Fold 2", "train": (0.20, 0.80), "test": (0.80, 1.00)},
]

print("=" * 62)
print("  State Engine — Unified MarketState + Fill PnL + Consistency")
print("=" * 62)

# ===========================================================================
# [1] Load data
# ===========================================================================

print("\n[1] Loading data ...")
t0 = time.perf_counter()
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

raw = pl.read_parquet(SOURCE, columns=[
    "mid_px", "spread", "total_depth", "signed_imbalance", "duration_ms"])
offset = raw.shape[0] - N
mid_px = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
spread = raw["spread"].to_numpy().astype(np.float64)[offset:]
depth = raw["total_depth"].to_numpy().astype(np.float64)[offset:]
signed_imb = raw["signed_imbalance"].to_numpy().astype(np.float64)[offset:]
duration = raw["duration_ms"].to_numpy().astype(np.float64)[offset:]
del raw

D0 = np.load(str(DICT_PATH))
n_batches = N // BATCH_SIZE
print(f"  {N:,} ticks, {n_batches} batches, time={time.perf_counter()-t0:.1f}s")

# Sparse encode
alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)
gram_trace = np.zeros(n_batches, dtype=np.float32)
for b in range(n_batches):
    s, e = b * BATCH_SIZE, (b+1) * BATCH_SIZE
    G = (alpha_full[s:e].T @ alpha_full[s:e]) / BATCH_SIZE
    gram_trace[b] = float(np.trace(G))
del alpha_full, X; gc.collect()

mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = np.abs(
    (mid_px[FWD_TICKS:] - mid_px[:-FWD_TICKS]) / (np.abs(mid_px[:-FWD_TICKS]) + 1e-12))

# Signal extractors
def _ext(d,s,e):
    if e-s<10: return 0.0
    d0,d1=np.median(d[s:s+10]),np.median(d[e-10:e])
    return float(max(-(d1-d0)/max(d0,1e-12),0.0))
def _obi(si,s,e):
    a=np.abs(si[s:e]); return float(np.log1p(np.percentile(a,95)/max(np.mean(a),1e-12)))
def _spr(sp,s,e):
    return float(np.log1p(np.max(sp[s:e])/max(np.mean(sp[s:e]),1e-12)))
def _burst(dur,s,e):
    if e-s<10: return 0.0
    return float(np.log1p(np.median(dur[s:e])/max(np.percentile(dur[s:e],5),1e-12)))


# ===========================================================================
# [2] Walk-forward with State Engine
# ===========================================================================

print(f"\n[2] Walk-forward with State Engine ...")
print(f"{'='*62}")

all_fold_results = []

for fold_idx, fold in enumerate(FOLDS):
    print(f"\n{'─'*62}")
    print(f"  {fold['name']}")
    print(f"{'─'*62}")

    tr_s = int(fold["train"][0] * n_batches)
    tr_e = int(fold["train"][1] * n_batches)
    te_s = int(fold["test"][0] * n_batches)
    te_e = int(fold["test"][1] * n_batches)

    # ── Init state engine ──
    state = MarketState()
    updater = StateUpdater()
    fill_model = FillModel()
    sim = ExecutionSimulator()

    # Train pass: warm up state
    for b in range(tr_s, tr_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        raw_signals = {
            "depth_evap": _ext(depth, s, e),
            "obi_impulse": _obi(signed_imb, s, e),
            "spread_shock": _spr(spread, s, e),
            "cancel_burst": _burst(duration, s, e),
        }
        signed_obi_mean = float(np.mean(signed_imb[s:e]))
        updater.update(state,
            raw_signals["depth_evap"], raw_signals["obi_impulse"],
            raw_signals["spread_shock"], raw_signals["cancel_burst"],
            signed_obi_mean)

    # Reset execution state (keep structure/pressure context)
    sim.reset()

    # Test pass: state → action → fill → PnL
    test_structure = []
    test_pressure = []
    test_execution = []
    test_pnl = []
    test_actions = []

    for b in range(te_s, te_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        raw_signals = {
            "depth_evap": _ext(depth, s, e),
            "obi_impulse": _obi(signed_imb, s, e),
            "spread_shock": _spr(spread, s, e),
            "cancel_burst": _burst(duration, s, e),
        }
        signed_obi_mean = float(np.mean(signed_imb[s:e]))

        # Single state update
        updater.update(state,
            raw_signals["depth_evap"], raw_signals["obi_impulse"],
            raw_signals["spread_shock"], raw_signals["cancel_burst"],
            signed_obi_mean)

        # Action from state (single point of decision)
        action = state.to_action()

        # Fill probability from state ONLY (not raw features)
        fp = fill_model.probability(
            spread_mult=action["spread_multiplier"],
            queue_ratio=0.3,
            imbalance=float(np.mean(signed_imb[s:e])),
            vol_z=state.spread_z,
            pressure_dir=int(np.sign(state.latent_direction)),
            pressure_z=state.pressure_z,
        )

        # Simulate fills
        batch_mid = mid_px[s:e]
        batch_spread = spread[s:e]
        batch_ret = mid_ret[s:e]
        market_state = {
            "imbalance": float(np.mean(signed_imb[s:e])),
            "vol_z": state.spread_z,
            "pressure_dir": int(np.sign(state.latent_direction)),
            "pressure_z": state.pressure_z,
        }

        pnl_result = sim.simulate_batch(
            batch_mid, batch_spread, batch_ret, action, market_state,
            queue_ratio=0.3,
        )

        test_structure.append(state.structure_regime)
        test_pressure.append(state.pressure_regime)
        test_execution.append(state.execution_regime)
        test_pnl.append(pnl_result["pnl_total"])
        test_actions.append(action)

    # ── Consistency scan ──
    cs = ConsistencyScanner()
    cs.scan(
        np.array(test_structure), np.array(test_pressure),
        np.array(test_execution), np.array(test_pnl),
    )
    cs.report()

    # ── Summary ──
    total_pnl = sum(test_pnl)
    n_test = te_e - te_s
    mean_pnl = np.mean(test_pnl)
    std_pnl = np.std(test_pnl)
    sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(n_test)

    action_types = {}
    for a in test_actions:
        desc = a["description"]
        action_types[desc] = action_types.get(desc, 0) + 1

    print(f"\n  State Engine Summary:")
    print(f"    Total PnL: {total_pnl:+,.0f}  Mean: {mean_pnl:+.2f}/batch  "
          f"Sharpe: {sharpe:.2f}")
    print(f"    Avg structure: {np.mean(test_structure):.3f}  "
          f"pressure: {np.mean(test_pressure):.3f}  "
          f"execution: {np.mean(test_execution):.3f}")
    print(f"    Action distribution: {action_types}")

    all_fold_results.append({
        "name": fold["name"],
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "sharpe": sharpe,
        "avg_structure": float(np.mean(test_structure)),
        "avg_pressure": float(np.mean(test_pressure)),
        "consistency": cs.overall_score,
    })

    # Reset for next fold
    updater.reset()


# ===========================================================================
# [3] Aggregate
# ===========================================================================

print(f"\n{'═'*62}")
print(f"  State Engine — Walk-Forward Aggregate")
print(f"{'═'*62}")
print(f"  {'Fold':<10s} {'PnL':>12s} {'Sharpe':>8s} "
      f"{'Avg S':>6s} {'Avg P':>6s} {'Consistency':>12s}")
print(f"  {'─'*10} {'─'*12} {'─'*8} {'─'*6} {'─'*6} {'─'*12}")

for r in all_fold_results:
    print(f"  {r['name']:<10s} {r['total_pnl']:>+12,.0f} {r['sharpe']:>8.2f} "
          f"{r['avg_structure']:>6.3f} {r['avg_pressure']:>6.3f} "
          f"{r['consistency']:>+12.4f}")

print(f"{'═'*62}")
print(f"  State Engine complete.")
print(f"{'═'*62}")

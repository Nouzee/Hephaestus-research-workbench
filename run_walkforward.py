"""
Walk-Forward Cross-Validation — Time-Split PnL Validation

3-fold walk-forward:
  Fold 1: train [0:60%],  test [60%:80%]
  Fold 2: train [20%:80%], test [80%:100%]

For each fold:
  1. Causal alignment (lag sweep) on TRAIN only
  2. Build ToxicityScorer with fold-specific lags
  3. Score train (build z-score baseline) + test
  4. Run StateMachine on test
  5. PnL backtest on test only

Aggregates: total PnL, Sharpe, adverse selection by FSM state, per-fold stability.
"""

import gc, sys, time, json
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.online_dict import OnlineDictLearner, OnlineDictConfig
from modules.dictionary.toxicity_scorer import ToxicityScorer, ToxicityScorerConfig
from modules.risk.risk_controller import RiskController, RiskControllerConfig
from modules.dictionary.pnl_backtest import PnLBacktest, PnLBacktestConfig
from sklearn.decomposition import sparse_encode

# ===========================================================================
# Config
# ===========================================================================

BATCH_SIZE, WINDOW_SIZE = 2048, 100
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

# Walk-forward fold definitions (as fraction of total batches)
FOLDS = [
    {"name": "Fold 1", "train": (0.00, 0.60), "test": (0.60, 0.80)},
    {"name": "Fold 2", "train": (0.20, 0.80), "test": (0.80, 1.00)},
]

FWD_TICKS = 50  # future return horizon for impact

print("=" * 62)
print("  Walk-Forward Cross-Validation")
print("  3-fold time-split: train-on-past, test-on-future")
print("=" * 62)


# ===========================================================================
# [1] Load shared data (once)
# ===========================================================================

print("\n[1] Loading shared data ...")
t0 = time.perf_counter()

builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

raw = pl.read_parquet(SOURCE, columns=["mid_px", "spread", "total_depth",
    "signed_imbalance", "duration_ms"])
offset = raw.shape[0] - N
mid_px = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
spread = raw["spread"].to_numpy().astype(np.float64)[offset:]
depth = raw["total_depth"].to_numpy().astype(np.float64)[offset:]
signed_imb = raw["signed_imbalance"].to_numpy().astype(np.float64)[offset:]
duration = raw["duration_ms"].to_numpy().astype(np.float64)[offset:]
del raw

D0 = np.load(str(DICT_PATH))
K = D0.shape[0]
n_batches = N // BATCH_SIZE
n_windows = N // WINDOW_SIZE

print(f"  Ticks: {N:,}  Batches: {n_batches}  Windows: {n_windows}  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [2] Pre-compute sparse encoding + Gram trace (shared across folds)
# ===========================================================================

print("\n[2] Sparse encoding + Gram trace ...")
t0 = time.perf_counter()
alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)

gram_trace_full = np.zeros(n_batches, dtype=np.float32)
for b in range(n_batches):
    s, e = b * BATCH_SIZE, (b+1) * BATCH_SIZE
    G = (alpha_full[s:e].T @ alpha_full[s:e]) / BATCH_SIZE
    gram_trace_full[b] = float(np.trace(G))

del alpha_full, X
gc.collect()
print(f"  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [3] Impact labels (shared)
# ===========================================================================

print("\n[3] Computing impact labels ...")
mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = np.abs(
    (mid_px[FWD_TICKS:] - mid_px[:-FWD_TICKS])
    / (np.abs(mid_px[:-FWD_TICKS]) + 1e-12)
)
batch_impact = np.array([
    np.mean(mid_ret[b*BATCH_SIZE:(b+1)*BATCH_SIZE]) for b in range(n_batches)
])
print(f"  Impact mean={batch_impact.mean():.6e}  std={batch_impact.std():.6e}")


# ===========================================================================
# Signal extractors (same as causal_alignment.py)
# ===========================================================================

def _extract_depth(d, s, e):
    if e - s < 10: return 0.0
    d0, d1 = np.median(d[s:s+10]), np.median(d[e-10:e])
    return float(max(-(d1 - d0) / max(d0, 1e-12), 0.0))

def _extract_obi(si, s, e):
    a = np.abs(si[s:e]); mu, p95 = np.mean(a), np.percentile(a, 95)
    return float(np.log1p(p95 / max(mu, 1e-12)))

def _extract_spread(sp, s, e):
    mu, mx = np.mean(sp[s:e]), np.max(sp[s:e])
    return float(np.log1p(mx / max(mu, 1e-12)))

def _extract_burst(dur, s, e):
    if e - s < 10: return 0.0
    med, p5 = np.median(dur[s:e]), np.percentile(dur[s:e], 5)
    return float(np.log1p(med / max(p5, 1e-12)))


# ===========================================================================
# Per-fold: causal alignment on train only
# ===========================================================================

def causal_alignment_on_train(train_start_batch, train_end_batch):
    """Sweep lags using ONLY training data. Returns dict of {signal: optimal_lead}."""
    n_train = train_end_batch - train_start_batch
    if n_train < 50:
        return {"depth_evap": 0, "obi_impulse": 0, "spread_shock": 0,
                "cancel_burst": 0, "gram_aux": 0}

    # Extract signals for train batches
    sigs = {}
    names = ["depth_evap", "obi_impulse", "spread_shock", "cancel_burst"]
    extractors = [_extract_depth, _extract_obi, _extract_spread, _extract_burst]
    arrays = [depth, signed_imb, spread, duration]

    for name, fn, arr in zip(names, extractors, arrays):
        sigs[name] = np.array([fn(arr, b*BATCH_SIZE, (b+1)*BATCH_SIZE)
                               for b in range(train_start_batch, train_end_batch)],
                              dtype=np.float32)

    # Gram aux
    gram_train = gram_trace_full[train_start_batch:train_end_batch]
    sigs["gram_aux"] = np.zeros(n_train, dtype=np.float32)
    for i in range(1, n_train):
        sigs["gram_aux"][i] = float(max(gram_train[i] - gram_train[i-1], 0.0))

    # Impact
    imp_train = batch_impact[train_start_batch:train_end_batch]

    # Sweep per signal
    leads = {}
    for name, sig in sigs.items():
        best_lead, best_corr = 0, -999.0
        for lag in range(0, min(20, n_train // 2)):
            if lag == 0:
                c = np.corrcoef(sig, imp_train)[0, 1]
            else:
                c = np.corrcoef(sig[:-lag], imp_train[lag:])[0, 1]
            if c > best_corr:
                best_corr, best_lead = c, lag
        leads[name] = int(best_lead) if best_corr > 0.01 else 0
    return leads


# ===========================================================================
# [4] Walk-forward loop
# ===========================================================================

print(f"\n[4] Walk-forward validation ({len(FOLDS)} folds) ...")
print(f"{'='*62}")

all_test_results = []
aggregate_pnl_baseline = []
aggregate_pnl_fsm = []
aggregate_impacts = {"NORMAL": [], "WATCH": [], "HARD_ALERT": [], "RECOVERY": []}

for fold_idx, fold in enumerate(FOLDS):
    print(f"\n{'─'*62}")
    print(f"  {fold['name']}")
    print(f"{'─'*62}")

    train_start = int(fold["train"][0] * n_batches)
    train_end   = int(fold["train"][1] * n_batches)
    test_start  = int(fold["test"][0] * n_batches)
    test_end    = int(fold["test"][1] * n_batches)

    n_train = train_end - train_start
    n_test = test_end - test_start
    print(f"  Train: batches [{train_start}:{train_end}] ({n_train})")
    print(f"  Test:  batches [{test_start}:{test_end}] ({n_test})")

    # ── 4a: Causal alignment on train only ──
    print(f"  [a] Causal alignment (train only) ...")
    t_a = time.perf_counter()
    leads = causal_alignment_on_train(train_start, train_end)
    print(f"    depth_evap={leads['depth_evap']}  obi={leads['obi_impulse']}  "
          f"spread={leads['spread_shock']}  burst={leads['cancel_burst']}  "
          f"gram={leads['gram_aux']}  ({time.perf_counter()-t_a:.1f}s)")

    # ── 4b: Build scorer with fold-specific lags ──
    # Write temp lag config
    temp_config = {
        "alignment": {},
        "weights": {"depth_evap": 1.0, "obi_impulse": 1.0, "spread_shock": 0.5,
                    "cancel_burst": 0.3, "gram_aux": 0.3},
    }
    for name, lead in leads.items():
        temp_config["alignment"][name] = {
            "optimal_lead": lead, "correlation": 0.0,
            "is_predictive": lead > 0, "role": "predictive" if lead > 0 else "diagnostic"
        }
    temp_path = CACHE / f"_temp_lags_fold{fold_idx}.json"
    with open(temp_path, "w") as f:
        json.dump(temp_config, f)

    scorer = ToxicityScorer(ToxicityScorerConfig(
        lag_config_path=str(temp_path),
        baseline_window=50,
        weights=temp_config["weights"],
        warn_sigma=1.5, hard_sigma=2.5, hard_persistence=2,
    ))
    rc = RiskController(scorer, RiskControllerConfig(recovery_cooldown=10, watch_max_consecutive=5))

    # ── 4c: Score on train (build baseline) + test ──
    print(f"  [b] Scoring train+test ...")
    t_b = time.perf_counter()

    # Train pass: build z-score baseline
    for b in range(train_start, train_end):
        s, e = b * BATCH_SIZE, (b+1) * BATCH_SIZE
        signals = {
            "depth_evap": _extract_depth(depth, s, e),
            "obi_impulse": _extract_obi(signed_imb, s, e),
            "spread_shock": _extract_spread(spread, s, e),
            "cancel_burst": _extract_burst(duration, s, e),
        }
        gram_v = gram_trace_full[b] - gram_trace_full[b-1] if b > train_start else 0.0
        scorer.score(signals, gram_v)  # build history only, discard result

    # Test pass: score + FSM + collect actions
    test_results = []
    for b in range(test_start, test_end):
        s, e = b * BATCH_SIZE, (b+1) * BATCH_SIZE
        signals = {
            "depth_evap": _extract_depth(depth, s, e),
            "obi_impulse": _extract_obi(signed_imb, s, e),
            "spread_shock": _extract_spread(spread, s, e),
            "cancel_burst": _extract_burst(duration, s, e),
        }
        gram_v = gram_trace_full[b] - gram_trace_full[b-1] if b > test_start else 0.0
        result = rc.update(signals, gram_v)
        test_results.append(result)

    print(f"    time={time.perf_counter()-t_b:.1f}s")

    # ── 4d: PnL backtest on test only ──
    print(f"  [c] PnL backtest (test only) ...")
    t_c = time.perf_counter()

    test_actions = [r["action"] for r in test_results]
    test_mid = mid_px[test_start*BATCH_SIZE:test_end*BATCH_SIZE]
    test_spread = spread[test_start*BATCH_SIZE:test_end*BATCH_SIZE]
    test_ret = mid_ret[test_start*BATCH_SIZE:test_end*BATCH_SIZE]

    bt = PnLBacktest(PnLBacktestConfig(spread_capture_frac=0.5))
    bt.run_with_actions(
        mid_px=test_mid, spread=test_spread,
        actions=test_actions, future_ret=test_ret,
        batch_size=BATCH_SIZE,
    )
    metrics = bt.report()

    # ── 4e: Per-fold impact by FSM state ──
    fsm_states = np.array([
        {"NORMAL": 0, "WATCH": 1, "HARD_ALERT": 2, "RECOVERY": 3}[r["fsm_state"]]
        for r in test_results
    ])
    test_impact = batch_impact[test_start:test_end]

    print(f"\n  Impact by FSM state (test only):")
    for si, sname in enumerate(["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]):
        mask = fsm_states == si
        if mask.sum() > 3:
            imp = test_impact[mask].mean()
            aggregate_impacts[sname].append(float(imp))
            print(f"    {sname:12s}: {imp:.6e}  ({mask.sum()} batches)")

    # Correlations
    test_scores = np.array([r["score"] for r in test_results])
    corr_score = np.corrcoef(test_scores, test_impact)[0, 1]
    corr_fsm = np.corrcoef(fsm_states.astype(float), test_impact)[0, 1]
    print(f"  Corr(score, impact)={corr_score:+.4f}  "
          f"Corr(FSM, impact)={corr_fsm:+.4f}")

    # Store
    aggregate_pnl_baseline.append(float(bt.baseline_equity[-1]))
    aggregate_pnl_fsm.append(float(bt.toxicity_equity[-1]))
    all_test_results.append({
        "name": fold["name"],
        "leads": leads,
        "metrics": metrics,
        "corr_score": corr_score,
        "corr_fsm": corr_fsm,
    })

    # Cleanup
    scorer = None; rc = None; gc.collect()
    temp_path.unlink(missing_ok=True)


# ===========================================================================
# [5] Aggregate results
# ===========================================================================

print(f"\n{'═'*62}")
print(f"  Aggregate Walk-Forward Results")
print(f"{'═'*62}")

print(f"\n  Per-Fold Summary:")
print(f"  {'Fold':<10s} {'Corr(score)':>12s} {'Corr(FSM)':>12s} "
      f"{'Baseline PnL':>14s} {'FSM PnL':>14s} {'Δ':>10s}")
print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*14} {'─'*14} {'─'*10}")

total_base = 0.0
total_fsm = 0.0
for r in all_test_results:
    m = r["metrics"]
    base = m.get("baseline_total_pnl", 0)
    fsm = m.get("toxicity_total_pnl", 0)
    delta = fsm - base
    total_base += base
    total_fsm += fsm
    print(f"  {r['name']:<10s} {r['corr_score']:>+12.4f} {r['corr_fsm']:>+12.4f} "
          f"{base:>+14,.1f} {fsm:>+14,.1f} {delta:>+10,.1f}")

print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*14} {'─'*14} {'─'*10}")
total_delta = total_fsm - total_base
print(f"  {'TOTAL':<10s} {'':>12} {'':>12} "
      f"{total_base:>+14,.1f} {total_fsm:>+14,.1f} {total_delta:>+10,.1f}")

print(f"\n  Average Impact by FSM State (across folds):")
for sname in ["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]:
    vals = aggregate_impacts[sname]
    if vals:
        print(f"    {sname:12s}: {np.mean(vals):.6e}  "
              f"(N={len(vals)} folds, range=[{np.min(vals):.6e}, {np.max(vals):.6e}])")

print(f"\n  Causal Alignment Stability (lead variation across folds):")
all_lead_names = set()
for r in all_test_results:
    all_lead_names.update(r["leads"].keys())
for name in sorted(all_lead_names):
    leads_across = [r["leads"].get(name, -1) for r in all_test_results]
    print(f"    {name:20s}: {leads_across}  "
          f"{'STABLE' if max(leads_across) - min(leads_across) <= 3 else 'UNSTABLE'}")

print(f"\n{'═'*62}")
print(f"  Walk-forward complete.")
print(f"{'═'*62}")

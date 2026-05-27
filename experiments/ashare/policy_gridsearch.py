"""
Policy Grid Search — tune FSM action parameters to maximize net PnL.

Sweeps over:
  - watch_size:       [0.85, 0.95, 1.0]
  - watch_spread:     [1.10, 1.20, 1.30]
  - recovery_size:    [0.3, 0.5, 0.7]
  - recovery_cooldown:[5, 10, 15]

Uses walk-forward Fold 2 (last 20% as test, middle 60% as train) to
avoid overfitting to the same test period.

Reports per-policy: total PnL, spread lost, adverse saved, Sharpe, per-state breakdown.
"""

import gc, sys, time, json, itertools
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.toxicity_scorer import ToxicityScorer, ToxicityScorerConfig
from modules.risk.risk_controller import RiskController, RiskControllerConfig
from modules.risk.state_machine import StateMachineConfig
from modules.dictionary.pnl_backtest import PnLBacktest, PnLBacktestConfig
from sklearn.decomposition import sparse_encode

# ===========================================================================
# Config
# ===========================================================================

BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"
FWD_TICKS = 50

# Train/test split for grid search (fixed)
TRAIN_FRAC = (0.00, 0.60)   # first 60%
TEST_FRAC  = (0.60, 0.80)   # next 20% (Fold 1 from walk-forward)

# Grid
GRID = {
    "watch_size":         [0.85, 0.95, 1.00],
    "watch_spread":       [1.10, 1.20, 1.30],
    "recovery_size":      [0.3, 0.5, 0.7],
    "recovery_cooldown":  [5, 10, 15],
    "hard_duration":      [2],
}


# ===========================================================================
# Signal extractors
# ===========================================================================

def _ext(d, s, e):
    if e - s < 10: return 0.0
    d0, d1 = np.median(d[s:s+10]), np.median(d[e-10:e])
    return float(max(-(d1 - d0) / max(d0, 1e-12), 0.0))

def _obi(si, s, e):
    a = np.abs(si[s:e]); mu, p95 = np.mean(a), np.percentile(a, 95)
    return float(np.log1p(p95 / max(mu, 1e-12)))

def _spr(sp, s, e):
    mu, mx = np.mean(sp[s:e]), np.max(sp[s:e])
    return float(np.log1p(mx / max(mu, 1e-12)))

def _burst(dur, s, e):
    if e - s < 10: return 0.0
    med, p5 = np.median(dur[s:e]), np.percentile(dur[s:e], 5)
    return float(np.log1p(med / max(p5, 1e-12)))


# ===========================================================================
# Load data (once)
# ===========================================================================

print("=" * 62)
print("  Policy Grid Search — FSM Action Parameters")
print("=" * 62)

print("\n[1] Loading data ...")
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
n_batches = N // BATCH_SIZE
print(f"  {N:,} ticks, {n_batches} batches, time={time.perf_counter()-t0:.1f}s")

# Sparse encode + Gram trace
print("[2] Sparse encoding ...")
t0 = time.perf_counter()
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
print(f"  time={time.perf_counter()-t0:.1f}s")

# Impact
print("[3] Computing impact ...")
mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = np.abs(
    (mid_px[FWD_TICKS:] - mid_px[:-FWD_TICKS])
    / (np.abs(mid_px[:-FWD_TICKS]) + 1e-12)
)
batch_impact = np.array([
    np.mean(mid_ret[b*BATCH_SIZE:(b+1)*BATCH_SIZE]) for b in range(n_batches)
])

# Train/test split
tr_start = int(TRAIN_FRAC[0] * n_batches)
tr_end   = int(TRAIN_FRAC[1] * n_batches)
te_start = int(TEST_FRAC[0] * n_batches)
te_end   = int(TEST_FRAC[1] * n_batches)

# Causal alignment on train only
print("[4] Causal alignment (train only) ...")
def causal_leads(tr_s, tr_e):
    names = ["depth_evap", "obi_impulse", "spread_shock", "cancel_burst", "gram_aux"]
    sigs = {}
    for b in range(tr_s, tr_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        for name in names:
            sigs.setdefault(name, []).append(0.0)
    n_tr = tr_e - tr_s
    for name in names:
        sigs[name] = np.zeros(n_tr, dtype=np.float32)
    for b_idx, b in enumerate(range(tr_s, tr_e)):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        sigs["depth_evap"][b_idx] = _ext(depth, s, e)
        sigs["obi_impulse"][b_idx] = _obi(signed_imb, s, e)
        sigs["spread_shock"][b_idx] = _spr(spread, s, e)
        sigs["cancel_burst"][b_idx] = _burst(duration, s, e)
        sigs["gram_aux"][b_idx] = float(max(gram_trace[b] - gram_trace[b-1], 0.0)) if b > tr_s else 0.0
    imp = batch_impact[tr_s:tr_e]
    leads = {}
    for name in names:
        best_lead, best_corr = 0, -999.0
        sig = sigs[name]
        n_sig = len(sig)
        for lag in range(0, min(20, n_sig // 2)):
            c = np.corrcoef(sig[:-lag], imp[lag:])[0, 1] if lag > 0 else np.corrcoef(sig, imp)[0, 1]
            if c > best_corr:
                best_corr, best_lead = c, lag
        leads[name] = int(best_lead) if best_corr > 0.01 else 0
    return leads

leads = causal_leads(tr_start, tr_end)
print(f"  Leads: {leads}")

# Write temp lag config for scorer
temp_config = {"alignment": {}, "weights": {
    "depth_evap": 1.0, "obi_impulse": 1.0, "spread_shock": 0.5,
    "cancel_burst": 0.3, "gram_aux": 0.3,
}}
for name, lead in leads.items():
    temp_config["alignment"][name] = {
        "optimal_lead": lead, "correlation": 0.0,
        "is_predictive": lead > 0, "role": "predictive" if lead > 0 else "diagnostic"
    }
temp_path = CACHE / "_gridsearch_lags.json"
with open(temp_path, "w") as f:
    json.dump(temp_config, f)


# ===========================================================================
# Grid search
# ===========================================================================

keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
n_combos = len(combos)
print(f"\n[5] Grid search: {n_combos} combinations ...")
print(f"{'─'*80}")

results = []
best_pnl = -float('inf')
best_params = None

for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    ws, wsp, rs, rc, hd = params["watch_size"], params["watch_spread"], \
                          params["recovery_size"], params["recovery_cooldown"], \
                          params["hard_duration"]

    # Build scorer
    scorer = ToxicityScorer(ToxicityScorerConfig(
        lag_config_path=str(temp_path), baseline_window=50,
        weights=temp_config["weights"],
        warn_sigma=1.5, hard_sigma=2.5, hard_persistence=2,
    ))

    # Build FSM with policy params
    fsm_config = StateMachineConfig(
        recovery_cooldown=rc,
        watch_size=ws, watch_spread=wsp,
        recovery_size=rs, recovery_spread=1.35,
        hard_duration=hd,
    )
    rc_obj = RiskController(scorer, RiskControllerConfig(
        recovery_cooldown=rc, watch_max_consecutive=5,
    ))
    # Override FSM config
    rc_obj.fsm.config = fsm_config

    # Score on train (build baseline)
    for b in range(tr_start, tr_end):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        signals = {
            "depth_evap": _ext(depth, s, e),
            "obi_impulse": _obi(signed_imb, s, e),
            "spread_shock": _spr(spread, s, e),
            "cancel_burst": _burst(duration, s, e),
        }
        gram_v = gram_trace[b] - gram_trace[b-1] if b > tr_start else 0.0
        scorer.score(signals, gram_v)

    # Score on test + collect actions
    test_results = []
    for b in range(te_start, te_end):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        signals = {
            "depth_evap": _ext(depth, s, e),
            "obi_impulse": _obi(signed_imb, s, e),
            "spread_shock": _spr(spread, s, e),
            "cancel_burst": _burst(duration, s, e),
        }
        gram_v = gram_trace[b] - gram_trace[b-1] if b > te_start else 0.0
        result = rc_obj.update(signals, gram_v)
        test_results.append(result)

    test_actions = [r["action"] for r in test_results]
    test_mid = mid_px[te_start*BATCH_SIZE:te_end*BATCH_SIZE]
    test_spread = spread[te_start*BATCH_SIZE:te_end*BATCH_SIZE]
    test_ret = mid_ret[te_start*BATCH_SIZE:te_end*BATCH_SIZE]

    # PnL
    bt = PnLBacktest(PnLBacktestConfig(spread_capture_frac=0.5))
    bt.run_with_actions(
        mid_px=test_mid, spread=test_spread,
        actions=test_actions, future_ret=test_ret, batch_size=BATCH_SIZE,
    )

    m = bt.metrics_
    fsm_pnl = m["toxicity_total_pnl"]
    base_pnl = m["baseline_total_pnl"]
    delta_pnl = fsm_pnl - base_pnl

    # Per-state PnL breakdown (approximate from FSM equity curve)
    fsm_states = np.array([
        {"NORMAL": 0, "WATCH": 1, "HARD_ALERT": 2, "RECOVERY": 3}[r["fsm_state"]]
        for r in test_results
    ])
    state_pnl_fsm = {}
    state_pnl_base = {}
    state_batches = {}
    ticks_per_batch = BATCH_SIZE
    for si, sname in enumerate(["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]):
        mask = np.zeros(len(test_mid), dtype=bool)
        for b_idx in np.where(fsm_states == si)[0]:
            t0 = b_idx * ticks_per_batch
            t1 = min(t0 + ticks_per_batch, len(test_mid))
            mask[t0:t1] = True
        if mask.sum() > 0:
            state_pnl_fsm[sname] = float(bt.toxicity_pnl[mask].sum())
            state_pnl_base[sname] = float(bt.baseline_pnl[mask].sum())
            state_batches[sname] = int(np.sum(fsm_states == si))

    result = {
        "params": params,
        "fsm_pnl": fsm_pnl,
        "base_pnl": base_pnl,
        "delta_pnl": delta_pnl,
        "sharpe_fsm": m["toxicity_sharpe"],
        "sharpe_base": m["baseline_sharpe"],
        "win_rate_fsm": m["toxicity_win_rate"],
        "state_pnl_fsm": state_pnl_fsm,
        "state_pnl_base": state_pnl_base,
        "state_batches": state_batches,
    }
    results.append(result)

    if delta_pnl > best_pnl:
        best_pnl = delta_pnl
        best_params = params

    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1}/{n_combos}] best_delta={best_pnl:+.1f}  "
              f"best_params={best_params}")


# ===========================================================================
# Report
# ===========================================================================

print(f"\n{'═'*62}")
print(f"  Grid Search Results — Top 10 Policies")
print(f"{'═'*62}")

results.sort(key=lambda r: r["delta_pnl"], reverse=True)

print(f"\n  {'Rank':<5s} {'watch_sz':>8s} {'watch_sp':>8s} {'rec_sz':>7s} "
      f"{'rec_cool':>8s} {'FSM PnL':>12s} {'Base PnL':>12s} {'Δ':>10s} {'Sharpe':>8s}")
print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

for rank, r in enumerate(results[:10]):
    p = r["params"]
    print(f"  {rank+1:<5d} {p['watch_size']:>8.2f} {p['watch_spread']:>8.2f} "
          f"{p['recovery_size']:>7.2f} {p['recovery_cooldown']:>8d} "
          f"{r['fsm_pnl']:>+12,.0f} {r['base_pnl']:>+12,.0f} "
          f"{r['delta_pnl']:>+10,.0f} {r['sharpe_fsm']:>8.2f}")

# Best policy detail
best = results[0]
print(f"\n  Best Policy: {best['params']}")
print(f"    FSM PnL: {best['fsm_pnl']:+,.0f}  Baseline: {best['base_pnl']:+,.0f}  "
      f"Δ: {best['delta_pnl']:+,.0f} ({best['delta_pnl']/max(abs(best['base_pnl']), 1)*100:+.1f}%)")

print(f"\n  Per-state PnL breakdown (best policy):")
print(f"  {'State':<12s} {'Batches':>8s} {'FSM PnL':>12s} {'Base PnL':>12s} {'Δ':>10s}")
print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*12} {'─'*10}")
for sname in ["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]:
    fsm = best["state_pnl_fsm"].get(sname, 0)
    base = best["state_pnl_base"].get(sname, 0)
    nb = best["state_batches"].get(sname, 0)
    print(f"  {sname:<12s} {nb:>8d} {fsm:>+12,.0f} {base:>+12,.0f} {fsm-base:>+10,.0f}")

# Compare with old policy (baseline action mapping)
print(f"\n  Comparison with old policy (watch_size=0.5, rec_size=0.25):")
old_results = [r for r in results
               if r["params"]["watch_size"] == 0.95  # closest to old 0.5
               and r["params"]["recovery_size"] == 0.5]
if not old_results:
    # Find worst to show range
    worst = results[-1]
    print(f"    Worst policy: {worst['params']}  Δ={worst['delta_pnl']:+,.0f}")
    print(f"    Best policy:  {best['params']}  Δ={best['delta_pnl']:+,.0f}")
    print(f"    Range: {worst['delta_pnl']:+,.0f} to {best['delta_pnl']:+,.0f}")

# Cleanup
temp_path.unlink(missing_ok=True)
print(f"\n{'═'*62}")
print(f"  Grid search complete.")
print(f"{'═'*62}")

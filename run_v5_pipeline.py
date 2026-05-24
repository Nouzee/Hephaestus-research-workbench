"""
V5 Pipeline — Causal-Aligned Multi-Signal Toxicity + State Machine + PnL

Full stack:
  1. Load X + raw signals
  2. Load D_0
  3. Sparse encode -> alpha (for Gram aux)
  4. Windowing + HMM (for Gram aux context)
  5. Streaming: online dict + precursor scorer + risk controller + state machine
  6. Impact labels
  7. PnL backtest with state-machine actions
"""

import gc, sys, time, json
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.online_dict import OnlineDictLearner, OnlineDictConfig
from modules.dictionary.gram_tracker import GramTracker, GramTrackerConfig
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
LAG_PATH = CACHE / "causal_alignment.json"

print("=" * 62)
print("  V5 Pipeline — Causal-Aligned Toxicity + State Machine + PnL")
print("=" * 62)

# ===========================================================================
# [1] Load data
# ===========================================================================
print("\n[1/7] Loading data ...")
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

print(f"  X: {X.shape}  Raw: {N:,} ticks  time={time.perf_counter()-t0:.1f}s")

# ===========================================================================
# [2] Load D_0
# ===========================================================================
print("\n[2/7] Loading D_0 ...")
D0 = np.load(str(DICT_PATH))
K = D0.shape[0]
print(f"  D_0: ({K}, {M})")

# ===========================================================================
# [3] Sparse encode + Gram trace (for gram_aux)
# ===========================================================================
print(f"\n[3/7] Sparse encoding + Gram trace ...")
t0 = time.perf_counter()
alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)
print(f"  alpha: {alpha_full.shape}  time={time.perf_counter()-t0:.1f}s")

# Per-batch Gram trace for gram_aux
n_batches = N // BATCH_SIZE
gram_trace = np.zeros(n_batches, dtype=np.float32)
for b in range(n_batches):
    s, e = b * BATCH_SIZE, (b+1) * BATCH_SIZE
    a = alpha_full[s:e]
    G = (a.T @ a) / BATCH_SIZE
    gram_trace[b] = float(np.trace(G))

del alpha_full
gc.collect()

# ===========================================================================
# [4] Streaming: precursor scoring + state machine
# ===========================================================================
print(f"\n[4/7] Streaming: Precursor Scorer + State Machine ...")
t0 = time.perf_counter()

# Online dict
odl = OnlineDictLearner(D0, OnlineDictConfig(n_components=K, alpha=1.0, gamma=0.995, dead_atom_window=500))

# Toxicity scorer (loads causal alignment config)
scorer = ToxicityScorer(ToxicityScorerConfig(
    baseline_window=50,
    weights={"depth_evap": 1.0, "obi_impulse": 1.0, "spread_shock": 0.5,
             "cancel_burst": 0.3, "gram_aux": 0.3},
    warn_sigma=1.5, hard_sigma=2.5, hard_persistence=2,
))

# Risk controller
rc = RiskController(scorer, RiskControllerConfig(recovery_cooldown=10, watch_max_consecutive=5))

# Per-batch signal extraction helpers
def _extract(d, s, e):
    if e - s < 10: return 0.0
    d0, d1 = np.median(d[s:s+10]), np.median(d[e-10:e])
    return float(max(-(d1 - d0) / max(d0, 1e-12), 0.0))

def _obi(si, s, e):
    a = np.abs(si[s:e]); mu, p95 = np.mean(a), np.percentile(a, 95)
    return float(np.log1p(p95 / max(mu, 1e-12)))

def _spread(sp, s, e):
    mu, mx = np.mean(sp[s:e]), np.max(sp[s:e])
    return float(np.log1p(mx / max(mu, 1e-12)))

def _burst(dur, s, e):
    if e - s < 10: return 0.0
    med, p5 = np.median(dur[s:e]), np.percentile(dur[s:e], 5)
    return float(np.log1p(med / max(p5, 1e-12)))

results = []
for b in range(n_batches):
    s, e = b * BATCH_SIZE, (b+1) * BATCH_SIZE

    # Online dict
    odl.partial_fit(X[s:e])

    # Gram velocity (aux)
    gram_v = gram_trace[b] - gram_trace[b-1] if b > 0 else 0.0

    # Extract raw signals
    signals = {
        "depth_evap": _extract(depth, s, e),
        "obi_impulse": _obi(signed_imb, s, e),
        "spread_shock": _spread(spread, s, e),
        "cancel_burst": _burst(duration, s, e),
    }

    # Score + state machine
    result = rc.update(signals, gram_v)
    results.append(result)

    if (b+1) % 500 == 0:
        fsm_states = [r["fsm_state"] for r in results[-500:]]
        n_warn = sum(1 for s in fsm_states if s == "WATCH")
        n_hard = sum(1 for s in fsm_states if s == "HARD_ALERT")
        n_rec = sum(1 for s in fsm_states if s == "RECOVERY")
        print(f"  Batch {b+1}/{n_batches}  "
              f"score={result['score']:.2f}  "
              f"fsm={result['fsm_state']:12s}  "
              f"states: N{500-n_warn-n_hard-n_rec} W{n_warn} H{n_hard} R{n_rec}")

del X
gc.collect()
elapsed = time.perf_counter() - t0
print(f"  Streaming complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ===========================================================================
# [5] Summary
# ===========================================================================
print(f"\n[5/7] Toxicity + Risk Summary ...")
sc_summary = scorer.summary()
fsm_summary = rc.fsm.summary()

print(f"  Scorer: mean={sc_summary['score_mean']:.3f}  std={sc_summary['score_std']:.3f}  "
      f"p95={sc_summary['score_p95']:.3f}")
print(f"  WARN={sc_summary['warn_rate']:.1%}  HARD={sc_summary['hard_rate']:.1%}")
print(f"  FSM states: {fsm_summary['state_distribution']}")
print(f"  FSM transitions: {fsm_summary['transitions']}")

# Signal contributions
print(f"  Signal lead alignment:")
lag_cfg = json.load(open(LAG_PATH)) if LAG_PATH.exists() else {}
for name, cfg in lag_cfg.get("alignment", {}).items():
    print(f"    {name:20s}  lead={cfg['optimal_lead']:>3d}  "
          f"corr={cfg['correlation']:+.4f}  role={cfg.get('role','?')}")

# ===========================================================================
# [6] Impact labels
# ===========================================================================
print(f"\n[6/7] Computing impact labels ...")
mid_ret = np.zeros(N, dtype=np.float64)
FWD = 50
mid_ret[:-FWD] = np.abs(
    (mid_px[FWD:] - mid_px[:-FWD]) / (np.abs(mid_px[:-FWD]) + 1e-12)
)
batch_impact = np.array([
    np.mean(mid_ret[b*BATCH_SIZE:(b+1)*BATCH_SIZE]) for b in range(n_batches)
])

# Correlation
scores = np.array(scorer.scores, dtype=np.float32)
fsm_numeric = np.array([
    {"NORMAL": 0, "WATCH": 1, "HARD_ALERT": 2, "RECOVERY": 3}[r["fsm_state"]]
    for r in results
])
hard_flag = (fsm_numeric == 2).astype(float)

corr_score = np.corrcoef(scores, batch_impact)[0, 1]
corr_fsm = np.corrcoef(fsm_numeric, batch_impact)[0, 1]
corr_hard = np.corrcoef(hard_flag, batch_impact)[0, 1]

print(f"  Corr(score, impact):     {corr_score:+.4f}")
print(f"  Corr(FSM_state, impact): {corr_fsm:+.4f}")
print(f"  Corr(HARD_flag, impact): {corr_hard:+.4f}")

# Impact during states
for si, sname in enumerate(["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]):
    mask = fsm_numeric == si
    if mask.sum() > 5:
        imp = batch_impact[mask].mean()
        print(f"  Impact during {sname:12s}: {imp:.6e}  ({mask.sum()} batches)")

# ===========================================================================
# [7] PnL Backtest with state machine actions
# ===========================================================================
print(f"\n[7/7] PnL Backtest — State-Machine-Aware MM ...")

actions_list = [r["action"] for r in results]

bt = PnLBacktest(PnLBacktestConfig(spread_capture_frac=0.5))
bt.run_with_actions(
    mid_px=mid_px,
    spread=spread,
    actions=actions_list,
    future_ret=mid_ret,
    batch_size=BATCH_SIZE,
)
metrics = bt.report()

# ===========================================================================
# Exports
# ===========================================================================
np.save(str(CACHE / "v5_scores.npy"), scores)
np.save(str(CACHE / "v5_fsm_states.npy"), fsm_numeric)
np.save(str(CACHE / "v5_actions.npy"),
        np.array([a.get("size_multiplier", 1.0) for a in actions_list], dtype=np.float32))
print(f"\n  Exports -> {CACHE}/")

print(f"\n{'='*62}")
print(f"  V5 Pipeline complete.")
print(f"{'='*62}")

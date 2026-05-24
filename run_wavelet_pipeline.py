"""
Wavelet Pipeline — Causal Multi-Scale Decomposition + Scale-Aware FSM + Walk-Forward

Key innovation over V5:
  - Each signal is decomposed into HF/MF/LF via causal EMAs
  - Per-scale lead-lag finds which scale is the true precursor
  - FSM uses scale-aware rules: HF triggers WATCH, HF+MF triggers HARD, LF controls RECOVERY
  - Walk-forward PnL validation
"""

import gc, sys, time, json
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.causal_wavelet import CausalDecomposer, CausalWaveletConfig
from modules.dictionary.multiscale_features import MultiscaleAnalyzer, MultiscaleConfig
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

# Walk-forward folds
FOLDS = [
    {"name": "Fold 1", "train": (0.00, 0.60), "test": (0.60, 0.80)},
    {"name": "Fold 2", "train": (0.20, 0.80), "test": (0.80, 1.00)},
]

# Policy params (from grid search optimum)
POLICY = {
    "watch_size": 1.0, "watch_spread": 1.10,
    "recovery_size": 0.7, "recovery_spread": 1.35,
    "hard_size": 0.0, "hard_spread": 0.0,
    "normal_size": 1.0, "normal_spread": 1.0,
    "recovery_cooldown": 5, "hard_duration": 2,
}

print("=" * 62)
print("  Wavelet Pipeline — Causal Multi-Scale + Scale-Aware FSM")
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


# ===========================================================================
# [2] Shared: sparse encode, Gram trace, impact
# ===========================================================================

print("[2] Sparse encoding + Gram trace ...")
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

mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = np.abs(
    (mid_px[FWD_TICKS:] - mid_px[:-FWD_TICKS]) / (np.abs(mid_px[:-FWD_TICKS]) + 1e-12))
batch_impact = np.array([
    np.mean(mid_ret[b*BATCH_SIZE:(b+1)*BATCH_SIZE]) for b in range(n_batches)])
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Signal extractors
# ===========================================================================

def _ext(d, s, e):
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
# [3] Multi-scale causality analysis (on full training data)
# ===========================================================================

print("\n[3] Multi-scale causality analysis (train split) ...")

# Extract signals for first-train period (0-60%) for causality calibration
calib_end = int(0.60 * n_batches)
sig_arrays = {}
for name, fn, arr in [
    ("depth_evap", _ext, depth),
    ("obi_impulse", _obi, signed_imb),
    ("spread_shock", _spr, spread),
    ("cancel_burst", _burst, duration)]:
    sig_arrays[name] = np.array(
        [fn(arr, b*BATCH_SIZE, (b+1)*BATCH_SIZE) for b in range(calib_end)], dtype=np.float32)

# Gram aux
sig_arrays["gram_aux"] = np.zeros(calib_end, dtype=np.float32)
for b in range(1, calib_end):
    sig_arrays["gram_aux"][b] = float(max(gram_trace[b] - gram_trace[b-1], 0.0))

msa = MultiscaleAnalyzer(MultiscaleConfig(hf_span=2, mf_span=10, lf_span=50))
msa.fit(sig_arrays, batch_impact[:calib_end])
multiscale_cfg = msa.export_config()

print(f"\n  Best scale per signal:")
print(msa.summary())


# ===========================================================================
# [4] Walk-forward with multiscale-aware FSM
# ===========================================================================

print(f"\n[4] Walk-forward with multiscale-aware FSM ...")
print(f"{'='*62}")

all_fold_results = []
decomposers = {}  # per-signal online decomposer

for fold_idx, fold in enumerate(FOLDS):
    print(f"\n{'─'*62}")
    print(f"  {fold['name']}")
    print(f"{'─'*62}")

    tr_s = int(fold["train"][0] * n_batches)
    tr_e = int(fold["train"][1] * n_batches)
    te_s = int(fold["test"][0] * n_batches)
    te_e = int(fold["test"][1] * n_batches)
    print(f"  Train: [{tr_s}:{tr_e}] ({tr_e-tr_s})  Test: [{te_s}:{te_e}] ({te_e-te_s})")

    # ── Init per-signal decomposers ──
    decomposers = {
        name: CausalDecomposer(CausalWaveletConfig(hf_span=2, mf_span=10, lf_span=50))
        for name in ["depth_evap", "obi_impulse", "spread_shock", "cancel_burst", "gram_aux"]
    }

    # ── Online scoring with multiscale features ──
    # Build rolling z-score history per (signal, scale): (baseline_window, n_sig*3)
    BASELINE = 50
    n_features = 5 * 3  # 5 signals x 3 scales
    history = np.zeros((BASELINE, n_features), dtype=np.float32)
    hist_ptr = 0
    hist_full = False

    # Lead buffers (per signal, per scale)
    max_lead = 0
    for sig_name in multiscale_cfg.get("signals", {}):
        for scale in ["HF", "MF", "LF"]:
            lead = multiscale_cfg["signals"][sig_name].get(scale, {}).get("lead", 0)
            max_lead = max(max_lead, lead)
    max_lead = max(max_lead, 1)
    lead_bufs = np.zeros((n_features, max_lead), dtype=np.float32)

    # FSM state tracking
    fsm_state = 0  # 0=NORMAL, 1=WATCH, 2=HARD, 3=RECOVERY
    recovery_ctr = 0
    hard_ctr = 0

    test_actions = []
    test_scores = []
    test_fsm_states = []

    # Train pass: build history
    for b in range(tr_s, tr_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        raw_signals = [
            _ext(depth, s, e), _obi(signed_imb, s, e),
            _spr(spread, s, e), _burst(duration, s, e),
            float(max(gram_trace[b] - gram_trace[b-1], 0.0)) if b > tr_s else 0.0,
        ]
        # Decompose each signal
        feats = np.zeros(n_features, dtype=np.float32)
        for i in range(5):
            hf, mf, lf = decomposers[list(decomposers.keys())[i]].update(raw_signals[i])
            feats[i*3+0] = hf
            feats[i*3+1] = mf
            feats[i*3+2] = lf

        history[hist_ptr] = feats
        hist_ptr = (hist_ptr + 1) % BASELINE
        if hist_ptr == 0:
            hist_full = True

    # Keep decomposer states (carry regime context across train→test boundary)
    hist_full = True

    # Test pass: score + FSM
    for b in range(te_s, te_e):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        raw_signals = [
            _ext(depth, s, e), _obi(signed_imb, s, e),
            _spr(spread, s, e), _burst(duration, s, e),
            float(max(gram_trace[b] - gram_trace[b-1], 0.0)) if b > te_s else 0.0,
        ]

        # Decompose
        feats = np.zeros(n_features, dtype=np.float32)
        for i, name in enumerate(decomposers.keys()):
            hf, mf, lf = decomposers[name].update(raw_signals[i])
            feats[i*3+0] = hf
            feats[i*3+1] = mf
            feats[i*3+2] = lf

        if not hist_full:
            test_actions.append({"quote": True, "size_multiplier": 1.0, "spread_multiplier": 1.0})
            test_scores.append(0.0)
            test_fsm_states.append(0)
            continue

        # Z-score each feature
        h_mean = history.mean(axis=0)
        h_std = np.maximum(history.std(axis=0), 1e-8)
        z = (feats - h_mean) / h_std
        z_pos = np.maximum(z, 0.0)

        # Apply per-scale leads: use lead_bufs for alignment
        # (simplified: use raw z-scores with leads from config)
        # Per-scale weights from config
        score = 0.0
        hf_sum, mf_sum, lf_sum = 0.0, 0.0, 0.0
        for i, name in enumerate(decomposers.keys()):
            w = multiscale_cfg.get("weights", {}).get("HF", {}).get(name, 1.0)
            hf_sum += z_pos[i*3+0] * w
            mf_sum += z_pos[i*3+1] * w
            lf_sum += z_pos[i*3+2] * w
        score = float(hf_sum + mf_sum * 0.7 + lf_sum * 0.3)

        # Scale-aware FSM rules (higher thresholds for multiscale):
        # HF triggers WATCH: >2.5 sigma (only depth or OBI, not spread or cancel)
        hf_depth_obi = np.max(z_pos[[0, 3]])  # depth_evap HF, obi_impulse HF
        hf_above = hf_depth_obi > 2.5
        # HF+MF triggers HARD: HF depth/OBI > 3.5 AND MF > 2.0
        hf_hard = hf_depth_obi > 3.5
        mf_depth_obi = np.max(z_pos[[1, 4]])  # depth_evap MF, obi_impulse MF
        mf_confirm = mf_depth_obi > 2.0
        # LF controls recovery: LF must be <1.0 sigma to exit RECOVERY
        lf_high = np.any(z[[i*3+2 for i in range(5)]] > 1.0)

        if fsm_state == 2:  # HARD
            hard_ctr += 1
            if hard_ctr >= POLICY["hard_duration"] and not (hf_hard and mf_confirm):
                fsm_state = 3  # -> RECOVERY
                hard_ctr = 0
                recovery_ctr = 0
        elif fsm_state == 3:  # RECOVERY
            if not lf_high:
                recovery_ctr += 1
                if recovery_ctr >= POLICY["recovery_cooldown"]:
                    fsm_state = 0
            else:
                recovery_ctr = 0
            if hf_hard and mf_confirm:
                fsm_state = 2
                hard_ctr = 0
        elif fsm_state == 1:  # WATCH
            if hf_hard and mf_confirm:
                fsm_state = 2
                hard_ctr = 0
            elif not hf_above:
                fsm_state = 0
        else:  # NORMAL
            if hf_hard and mf_confirm:
                fsm_state = 2
                hard_ctr = 0
            elif hf_above:
                fsm_state = 1

        # Action mapping
        state_params = {
            0: ("NORMAL", POLICY["normal_size"], POLICY["normal_spread"]),
            1: ("WATCH", POLICY["watch_size"], POLICY["watch_spread"]),
            2: ("HARD", POLICY["hard_size"], POLICY["hard_spread"]),
            3: ("RECOVERY", POLICY["recovery_size"], POLICY["recovery_spread"]),
        }
        sname, sz, spd = state_params[fsm_state]
        test_actions.append({
            "quote": sz > 0, "size_multiplier": sz, "spread_multiplier": spd,
            "description": sname,
        })
        test_scores.append(score)
        test_fsm_states.append(fsm_state)

        # Update history
        history[hist_ptr] = feats
        hist_ptr = (hist_ptr + 1) % BASELINE

    # ── PnL backtest ──
    test_mid = mid_px[te_s*BATCH_SIZE:te_e*BATCH_SIZE]
    test_spread = spread[te_s*BATCH_SIZE:te_e*BATCH_SIZE]
    test_ret = mid_ret[te_s*BATCH_SIZE:te_e*BATCH_SIZE]

    bt = PnLBacktest(PnLBacktestConfig(spread_capture_frac=0.5))
    bt.run_with_actions(
        mid_px=test_mid, spread=test_spread,
        actions=test_actions, future_ret=test_ret, batch_size=BATCH_SIZE,
    )
    m = bt.metrics_
    delta = m["toxicity_total_pnl"] - m["baseline_total_pnl"]
    delta_pct = delta / max(abs(m["baseline_total_pnl"]), 1) * 100

    # State distribution
    states_arr = np.array(test_fsm_states)
    state_pcts = {s: np.mean(states_arr==i)*100 for i,s in enumerate(["NORMAL","WATCH","HARD","RECOVERY"])}

    # Correlation
    test_impact = batch_impact[te_s:te_e]
    corr = np.corrcoef(np.array(test_scores), test_impact)[0, 1]

    print(f"  States: {state_pcts}")
    print(f"  Corr(score,impact)={corr:+.4f}  "
          f"FSM PnL={m['toxicity_total_pnl']:+,.0f}  "
          f"Baseline={m['baseline_total_pnl']:+,.0f}  "
          f"Delta={delta:+,.0f} ({delta_pct:+.1f}%)")

    all_fold_results.append({
        "name": fold["name"], "delta": delta, "delta_pct": delta_pct,
        "corr": corr, "fsm_pnl": m["toxicity_total_pnl"],
        "base_pnl": m["baseline_total_pnl"],
        "sharpe_fsm": m["toxicity_sharpe"], "sharpe_base": m["baseline_sharpe"],
        "state_pcts": state_pcts,
    })


# ===========================================================================
# [5] Aggregate
# ===========================================================================

print(f"\n{'═'*62}")
print(f"  Aggregate Wavelet Walk-Forward Results")
print(f"{'═'*62}")

total_base = 0.0
total_fsm = 0.0
print(f"\n  {'Fold':<10s} {'Corr':>8s} {'FSM PnL':>14s} {'Base PnL':>14s} "
      f"{'Delta':>12s} {'Sharpe':>8s} {'N/W/H/R':>20s}")
print(f"  {'─'*10} {'─'*8} {'─'*14} {'─'*14} {'─'*12} {'─'*8} {'─'*20}")

for r in all_fold_results:
    total_base += r["base_pnl"]
    total_fsm += r["fsm_pnl"]
    nwhr = f"N{r['state_pcts'].get('NORMAL',0):.0f}/W{r['state_pcts'].get('WATCH',0):.0f}/" \
           f"H{r['state_pcts'].get('HARD',0):.0f}/R{r['state_pcts'].get('RECOVERY',0):.0f}"
    print(f"  {r['name']:<10s} {r['corr']:>+8.4f} {r['fsm_pnl']:>+14,.0f} "
          f"{r['base_pnl']:>+14,.0f} {r['delta']:>+12,.0f} ({r['delta_pct']:>+.1f}%) "
          f"{r['sharpe_fsm']:>8.2f} {nwhr:>20s}")

total_delta = total_fsm - total_base
total_delta_pct = total_delta / max(abs(total_base), 1) * 100
print(f"  {'─'*10} {'─'*8} {'─'*14} {'─'*14} {'─'*12} {'─'*8} {'─'*20}")
print(f"  {'TOTAL':<10s} {'':>8} {total_fsm:>+14,.0f} {total_base:>+14,.0f} "
      f"{total_delta:>+12,.0f} ({total_delta_pct:>+.1f}%)")

print(f"\n{'═'*62}")
print(f"  Wavelet pipeline complete.")
print(f"{'═'*62}")

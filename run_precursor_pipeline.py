"""
Precursor Pipeline v4 — Multi-Signal Shock Precursor + PnL Backtest

Replaces single-signal Gram toxicity with fused multi-signal precursor:
  S1 Depth Evaporation  (liquidity vanishing velocity)
  S2 OBI Impulse        (one-sided order flow spike)
  S3 Spread Shock       (sudden spread widening)
  S4 Event Burst        (tick frequency surge)
  S5 Gram Acceleration  (auxiliary topology signal)

Each signal is z-scored over rolling baseline. Score = weighted sum of positive z's.
Two-tier circuit breaker: WARN (2+ signals > 1.5s) | HARD (> 2.5s AND persistent).

Pipeline:
  1. Load X + raw precursor data
  2. Load D_0
  3. Sparse encode -> alpha
  4. Window HMM -> regimes
  5. Per-regime Gram baselines
  6. Streaming: online dict + Gram + PrecursorScorer
  7. Impact labels
  8. PnL backtest with precursor score
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.online_dict import OnlineDictLearner, OnlineDictConfig
from modules.dictionary.gram_tracker import GramTracker, GramTrackerConfig
from modules.dictionary.hmm_regime import HMMRegime, HMMRegimeConfig
from modules.dictionary.precursor_scorer import PrecursorScorer, PrecursorConfig
from modules.dictionary.pnl_backtest import PnLBacktest, PnLBacktestConfig
from sklearn.decomposition import sparse_encode


# ===========================================================================
# Config
# ===========================================================================

BATCH_SIZE = 2048
WINDOW_SIZE = 100
SOURCE_PARQUET = (
    r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
)
CACHE_DIR = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE_DIR / "dict_atoms_3.npy"

print("=" * 62)
print("  Precursor Pipeline v4")
print("  Multi-Signal Shock Precursor + Two-Tier Circuit Breaker + PnL")
print("=" * 62)


# ===========================================================================
# Step 1: Load data
# ===========================================================================

print("\n[1/8] Loading data ...")

# Standardized X
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

# Raw data for precursor signals + PnL
raw_cols = ["mid_px", "spread", "total_depth", "signed_imbalance", "duration_ms"]
_raw = pl.read_parquet(SOURCE_PARQUET, columns=raw_cols)
_offset = _raw.shape[0] - N
mid_px_raw = _raw["mid_px"].to_numpy().astype(np.float64)[_offset:]
spread_raw = _raw["spread"].to_numpy().astype(np.float64)[_offset:]
depth_raw = _raw["total_depth"].to_numpy().astype(np.float64)[_offset:]
signed_imb_raw = _raw["signed_imbalance"].to_numpy().astype(np.float64)[_offset:]
duration_raw = _raw["duration_ms"].to_numpy().astype(np.float64)[_offset:]
del _raw, _offset

print(f"  X: {X.shape}  ({X.nbytes/1024**2:.1f} MB)")
print(f"  Raw signals: depth [{depth_raw.min():.0f},{depth_raw.max():.0f}]  "
      f"spread mean={spread_raw.mean():.2f}  "
      f"imb std={signed_imb_raw.std():.2f}")


# ===========================================================================
# Step 2: Load D_0
# ===========================================================================

print("\n[2/8] Loading pre-trained dictionary ...")
if not DICT_PATH.exists():
    print(f"  ERROR: {DICT_PATH} not found.")
    sys.exit(1)
D0 = np.load(str(DICT_PATH))
K = D0.shape[0]
print(f"  D_0: ({K} atoms x {M} features)")


# ===========================================================================
# Step 3: Sparse encode
# ===========================================================================

print(f"\n[3/8] Sparse encoding {N:,} samples ...")
t0 = time.perf_counter()
alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)
print(f"  alpha: {alpha_full.shape}  sparsity={np.mean(np.abs(alpha_full)>1e-8)*100:.1f}%  "
      f"time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Step 4: Window HMM
# ===========================================================================

print(f"\n[4/8] Windowing alpha ({WINDOW_SIZE}-tick) + HMM ...")
t0 = time.perf_counter()
W = WINDOW_SIZE
n_windows = N // W

# Per-window Gram features
def window_gram_features(alpha, W):
    n_w = len(alpha) // W
    K = alpha.shape[1]
    nf = K * (K + 1) // 2
    feats = np.zeros((n_w, nf), dtype=np.float32)
    grams = np.zeros((n_w, K, K), dtype=np.float32)
    for w_idx in range(n_w):
        a = alpha[w_idx*W:(w_idx+1)*W]
        G = (a.T @ a) / W
        idx = 0
        for i in range(K):
            for j in range(i, K):
                feats[w_idx, idx] = G[i, j]
                idx += 1
        grams[w_idx] = G
    return feats, grams

hmm_feats, window_grams = window_gram_features(alpha_full, W)

# HMM
hmm_train_n = min(20_000, n_windows)
rng = np.random.RandomState(42)
hmm_idx = rng.choice(n_windows, hmm_train_n, replace=False)
hmm_idx.sort()
hmm = HMMRegime(HMMRegimeConfig(n_states=3, covariance_type="full", n_iter=200, random_state=42))
hmm.fit(hmm_feats[hmm_idx])
all_regimes, _ = hmm.predict(hmm_feats)
all_regimes = all_regimes.astype(np.int32)

for s in range(3):
    print(f"  Regime {s}: {np.mean(all_regimes==s)*100:.1f}%")
print(f"  Persistence: {np.mean(np.diff(all_regimes)==0)*100:.1f}%  "
      f"time={time.perf_counter()-t0:.1f}s")

# Free
del alpha_full, hmm_feats
gc.collect()


# ===========================================================================
# Step 5: Regime-conditional Gram baselines
# ===========================================================================

print(f"\n[5/8] Building per-regime Gram baselines ...")
regime_gram_mean = {}
regime_gram_std = {}
for s in range(3):
    mask = all_regimes == s
    grams_s = window_grams[mask]
    if len(grams_s) > 5:
        regime_gram_mean[s] = grams_s.mean(axis=0)
        gflat = grams_s.reshape(len(grams_s), -1)
        regime_gram_std[s] = np.maximum(np.std(gflat, axis=0).reshape(K, K), 1e-8)
        print(f"  State {s}: {len(grams_s)} windows  ||G||_F={np.linalg.norm(regime_gram_mean[s],'fro'):.3f}")

del window_grams
gc.collect()


# ===========================================================================
# Step 6: Streaming — Precursor Scoring
# ===========================================================================

print(f"\n[6/8] Streaming: Online Dict + Gram + Precursor Scoring ...")
t0 = time.perf_counter()

# Online dict
odl = OnlineDictLearner(D0, OnlineDictConfig(n_components=K, alpha=1.0, gamma=0.995, dead_atom_window=500))

# Gram tracker (for Gram velocity as aux signal)
gt = GramTracker(GramTrackerConfig(window_size=200))

# Precursor scorer
ps = PrecursorScorer(PrecursorConfig(
    baseline_window=50,
    w_depth=1.0, w_obi=1.0, w_spread=1.0, w_burst=0.5, w_gram_accel=0.3,
    warn_sigma=1.5, hard_sigma=2.5, hard_persistence=2,
))

n_batches = N // BATCH_SIZE
batch_regimes = np.zeros(n_batches, dtype=np.int32)

# For storing Gram velocity (computed post-hoc from Gram deviation trace)
gram_v_history = np.zeros(n_batches, dtype=np.float32)
gram_trace = np.zeros(n_batches, dtype=np.float32)

for b in range(n_batches):
    start = b * BATCH_SIZE
    end = start + BATCH_SIZE

    # Online dict update
    X_batch = X[start:end]
    alpha_batch = odl.partial_fit(X_batch)

    # Gram tracking
    gt.update(alpha_batch, D=odl.D_km)
    gram_trace[b] = gt.G_alpha_trace if hasattr(gt, 'G_alpha_trace') else float(np.trace(gt.G_alpha)) if gt.G_alpha is not None else 0.0

    # Extract raw precursor signals for this batch
    signals = ps.extract_signals(
        depth_raw=depth_raw[start:end],
        signed_imb_raw=signed_imb_raw[start:end],
        spread_raw=spread_raw[start:end],
        duration_raw=duration_raw[start:end],
        gram_v=gram_v_history[b-1] if b > 0 else 0.0,  # will update below
    )

    # Score
    score, tier = ps.score_batch(signals)

    # Map batch to regime
    w0 = start // W
    w1 = end // W
    batch_regimes[b] = int(all_regimes[min(w0, n_windows-1)])

    if (b + 1) % 500 == 0:
        contribs = ps.summary().get("signal_contributions", {})
        print(f"  Batch {b+1}/{n_batches}  "
              f"score={score:.2f}  tier={tier:6s}  "
              f"regime={batch_regimes[b]}  "
              f"D={contribs.get('depth_evap',0):.2f}  "
              f"O={contribs.get('obi_impulse',0):.2f}  "
              f"S={contribs.get('spread_shock',0):.2f}")

# Compute Gram velocity post-hoc from trace
gram_v_history[1:] = np.diff(gram_trace)
gram_v_history[0] = 0.0

elapsed = time.perf_counter() - t0
print(f"  Streaming complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Free X
del X
gc.collect()

# Summary
ps_summary = ps.summary()
print(f"\n  Precursor Score Distribution:")
print(f"    Mean={ps_summary['score_mean']:.3f}  Std={ps_summary['score_std']:.3f}  "
      f"P95={ps_summary['score_p95']:.3f}")
print(f"    WARN rate: {ps_summary['warn_rate']:.1%}  HARD rate: {ps_summary['hard_rate']:.1%}")
print(f"  Signal contributions (avg z_pos x weight):")
for k, v in ps_summary['signal_contributions'].items():
    print(f"    {k:20s}: {v:.4f}")

# Toxicity per batch = precursor score (for PnL)
batch_toxicity = np.array(ps.scores, dtype=np.float32)
tier_flags = np.array([{"NORMAL": 0, "WARN": 1, "HARD": 2}[t] for t in ps.tiers], dtype=np.int32)


# ===========================================================================
# Step 7: Impact labels
# ===========================================================================

print(f"\n[7/8] Generating impact labels ...")

FWD_TICKS = 50
mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = (
    (mid_px_raw[FWD_TICKS:] - mid_px_raw[:-FWD_TICKS])
    / (np.abs(mid_px_raw[:-FWD_TICKS]) + 1e-12)
)

batch_impact = np.array([
    np.mean(np.abs(mid_ret[b*BATCH_SIZE:(b+1)*BATCH_SIZE]))
    for b in range(n_batches)
], dtype=np.float64)

# Correlation
corr_score_impact = np.corrcoef(batch_toxicity, batch_impact)[0, 1]
corr_hard_impact = np.corrcoef((tier_flags == 2).astype(float), batch_impact)[0, 1]

print(f"  Corr(precursor_score, future_impact):  {corr_score_impact:+.4f}")
print(f"  Corr(HARD_flag, future_impact):        {corr_hard_impact:+.4f}")

# Impact during HARD tier
hard_mask = tier_flags == 2
if hard_mask.sum() > 5:
    impact_hard = batch_impact[hard_mask].mean()
    impact_normal = batch_impact[~hard_mask].mean()
    ratio = impact_hard / max(impact_normal, 1e-16)
    print(f"  Impact during HARD:   {impact_hard:.6e}")
    print(f"  Impact during NORMAL: {impact_normal:.6e}")
    print(f"  Ratio: {ratio:.2f}x"
          f"{' [OK] PRECURSOR PREDICTS IMPACT!' if ratio > 1.15 else ''}")


# ===========================================================================
# Step 8: PnL Backtest
# ===========================================================================

print(f"\n[8/8] PnL Backtest — MM with multi-signal precursor breaker ...")

bt = PnLBacktest(PnLBacktestConfig(
    spread_capture_frac=0.5,
    tox_percentile_threshold=90.0,
))
bt.run(
    mid_px=mid_px_raw,
    spread=spread_raw,
    toxicity_scores=batch_toxicity,
    future_ret=mid_ret,
)
bt.report()


# ===========================================================================
# Exports
# ===========================================================================

np.save(str(CACHE_DIR / "precursor_scores_v4.npy"), batch_toxicity)
np.save(str(CACHE_DIR / "precursor_tiers_v4.npy"), tier_flags)
np.save(str(CACHE_DIR / "precursor_contributions_v4.npy"),
        np.array(list(ps_summary['signal_contributions'].values()), dtype=np.float32))
print(f"\n  Exports -> {CACHE_DIR}/")
print(f"    precursor_scores_v4.npy  ({len(batch_toxicity)} batches)")
print(f"    precursor_tiers_v4.npy   ({len(tier_flags)} batches)")
print(f"\n{'='*62}")
print(f"  Pipeline complete.")
print(f"{'='*62}")

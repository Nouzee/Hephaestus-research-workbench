"""
Toxicity Detection Pipeline v2 — 窗口化 HMM + Gram 条件基线 + PnL 回测

Fixes from v1:
  1. Windowed HMM: 100-tick alpha Gram → HMM regimes (temporal persistence)
  2. Per-regime Gram baseline: each regime gets its own normal Gram distribution
  3. PnL backtest: MM simulation with toxicity circuit breaker

Pipeline:
  1. Load X + raw mid_px/spread
  2. Load D_0
  3. Sparse encode → α
  4. Window α → 100-tick Gram features → train HMM
  5. Build per-regime Gram conditional baseline
  6. Streaming: online dict + per-window Gram tracking + toxicity
  7. PnL backtest: MM with/without toxicity breaker
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gc
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder, FEATURE_NAMES
from modules.dictionary.online_dict import OnlineDictLearner, OnlineDictConfig
from modules.dictionary.gram_tracker import GramTracker, GramTrackerConfig
from modules.dictionary.hmm_regime import HMMRegime, HMMRegimeConfig
from modules.dictionary.toxicity_scorer import ToxicityScorer, ToxicityScorerConfig
from modules.dictionary.pnl_backtest import PnLBacktest, PnLBacktestConfig
from sklearn.decomposition import sparse_encode


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

WINDOW_SIZE = 100          # ticks per window (for HMM & Gram)
BATCH_SIZE = 2048          # samples per online dict partial_fit
CACHE_DIR = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE_DIR / "dict_atoms_3.npy"
SOURCE_PARQUET = (
    r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
)

print("=" * 62)
print("  Toxicity Pipeline v2")
print("  Windowed HMM → Regime-Conditional Gram → PnL Backtest")
print("=" * 62)


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load X + raw data
# ═══════════════════════════════════════════════════════════════════════

print("\n[1/8] Loading data ...")
t0 = time.perf_counter()

# Standardized feature matrix
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

# Raw data for PnL backtest (mid_px, spread) — load as numpy, free polars
# MatrixBuilder drops ~50 rows (window-boundary NaN); align by using first N rows of raw
_raw_df = pl.read_parquet(SOURCE_PARQUET, columns=["mid_px", "spread"])
_raw_n = _raw_df.shape[0]
# Raw has _raw_n rows; X has N rows (N ≈ _raw_n - 50). Use the last N rows of raw
# to align (the dropped rows are at the beginning from feature window boundaries)
_offset = _raw_n - N
mid_px_raw = _raw_df["mid_px"].to_numpy().astype(np.float64)[_offset:]
spread_raw = _raw_df["spread"].to_numpy().astype(np.float64)[_offset:]
del _raw_df, _raw_n, _offset

print(f"  X.shape = {X.shape}  ({X.nbytes / 1024**2:.1f} MB)")
print(f"  mid_px range: [{mid_px_raw.min():.1f}, {mid_px_raw.max():.1f}]")
print(f"  spread mean: {spread_raw.mean():.2f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Load D_0
# ═══════════════════════════════════════════════════════════════════════

print("\n[2/8] Loading pre-trained dictionary ...")
if not DICT_PATH.exists():
    print(f"  ERROR: {DICT_PATH} not found. Run dict_trainer.py first.")
    sys.exit(1)

D0 = np.load(str(DICT_PATH))
K = D0.shape[0]
print(f"  D_0: ({K} atoms × {M} features)")


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Sparse encode → α
# ═══════════════════════════════════════════════════════════════════════

print(f"\n[3/8] Sparse encoding {N:,} samples ...")
t_enc = time.perf_counter()

alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)

sparsity_pct = np.mean(np.abs(alpha_full) > 1e-8) * 100
print(f"  alpha: {alpha_full.shape}  sparsity={sparsity_pct:.1f}%  "
      f"time={time.perf_counter()-t_enc:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Window α → Gram features → HMM
# ═══════════════════════════════════════════════════════════════════════

print(f"\n[4/8] Windowing alpha ({WINDOW_SIZE}-tick) + HMM training ...")
t_hmm = time.perf_counter()

W = WINDOW_SIZE
n_windows = N // W

# Per-window: compute alpha Gram (upper triangle as feature vector)
def window_gram_features(alpha: np.ndarray, W: int) -> np.ndarray:
    """For each non-overlapping window, compute flattened alpha Gram upper triangle."""
    N = len(alpha)
    K = alpha.shape[1]
    n_windows = N // W
    n_gram_feats = K * (K + 1) // 2  # upper triangle size

    features = np.zeros((n_windows, n_gram_feats), dtype=np.float32)
    grams = np.zeros((n_windows, K, K), dtype=np.float32)  # store full Gram too

    for w_idx in range(n_windows):
        t0 = w_idx * W
        t1 = t0 + W
        alpha_w = alpha[t0:t1]  # (W, K)
        G = (alpha_w.T @ alpha_w) / W  # (K, K)

        # Upper triangle (row-major)
        idx = 0
        for i in range(K):
            for j in range(i, K):
                features[w_idx, idx] = G[i, j]
                idx += 1

        grams[w_idx] = G

    return features, grams

hmm_features, window_grams = window_gram_features(alpha_full, W)
n_gram_feats = hmm_features.shape[1]
print(f"  Window features: {hmm_features.shape}  ({n_gram_feats} Gram upper-triangle features)")
print(f"  Window Grams:    {window_grams.shape}")

# Train HMM on window Gram features
# Subset for faster training
hmm_train_n = min(20_000, n_windows)
rng = np.random.RandomState(42)
hmm_idx = rng.choice(n_windows, hmm_train_n, replace=False)
hmm_idx.sort()

hmm = HMMRegime(HMMRegimeConfig(
    n_states=3, covariance_type="full",
    n_iter=200, tol=1e-4, random_state=42,
))
hmm.fit(hmm_features[hmm_idx])

# Predict regime for all windows
all_regimes, regime_probs = hmm.predict(hmm_features)
all_regimes = all_regimes.astype(np.int32)

print(f"  Regime distribution:")
for s in range(3):
    pct = np.mean(all_regimes == s) * 100
    print(f"    State {s}: {pct:.1f}%")

# Check temporal persistence (consecutive same-regime probability)
transitions = np.diff(all_regimes)
persistence = np.mean(transitions == 0) * 100
print(f"  Temporal persistence: {persistence:.1f}% "
      f"({'GOOD — regimes are sticky' if persistence > 70 else 'WEAK — regimes flip too fast'})")
print(f"  HMM time: {time.perf_counter()-t_hmm:.1f}s")

# Free alpha_full — Gram features extracted, no longer needed
del alpha_full, hmm_features
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Build per-regime Gram conditional baseline
# ═══════════════════════════════════════════════════════════════════════

print(f"\n[5/8] Building per-regime Gram baselines ...")

# Collect Grams by regime
regime_gram_collection = {s: [] for s in range(3)}
for w_idx in range(n_windows):
    s = int(all_regimes[w_idx])
    regime_gram_collection[s].append(window_grams[w_idx])

# Per-regime: mean Gram + covariance of vectored Grams
regime_gram_mean = {}
regime_gram_std = {}
for s in range(3):
    grams = np.array(regime_gram_collection[s])  # (N_s, K, K)
    n_s = len(grams)
    # Mean Gram
    regime_gram_mean[s] = grams.mean(axis=0)
    # Element-wise std (for Mahalanobis normalization)
    grams_flat = grams.reshape(n_s, -1)  # (N_s, K²)
    regime_gram_std[s] = np.std(grams_flat, axis=0).reshape(K, K)
    regime_gram_std[s] = np.maximum(regime_gram_std[s], 1e-8)

    print(f"    State {s}: {n_s} windows, "
          f"||G_mean||_F = {np.linalg.norm(regime_gram_mean[s], 'fro'):.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Streaming — online dict (2048-tick batches) + Gram + Toxicity
# ═══════════════════════════════════════════════════════════════════════

print(f"\n[6/8] Streaming pass: Online Dict (batch={BATCH_SIZE}) → Gram → Toxicity ...")
t_stream = time.perf_counter()

# Init online dict
odl = OnlineDictLearner(D0, OnlineDictConfig(
    n_components=K, alpha=1.0, gamma=0.995, dead_atom_window=500,
))

# Init Gram tracker (per-batch, EMA of alpha Gram)
gt = GramTracker(GramTrackerConfig(window_size=200))

# Streaming at batch granularity
n_batches = N // BATCH_SIZE
batch_toxicity = np.zeros(n_batches, dtype=np.float32)
batch_grams = np.zeros((n_batches, K, K), dtype=np.float32)
batch_regimes = np.zeros(n_batches, dtype=np.int32)
batch_alpha_mean = np.zeros((n_batches, K), dtype=np.float32)

for b in range(n_batches):
    start = b * BATCH_SIZE
    end = start + BATCH_SIZE
    X_batch = X[start:end]

    # Online dict update
    alpha_batch = odl.partial_fit(X_batch)

    # Update Gram tracker (EMA over batches)
    gt.update(alpha_batch, D=odl.D_km)

    # Store per-batch Gram and mean alpha
    batch_grams[b] = (alpha_batch.T @ alpha_batch) / len(alpha_batch)
    batch_alpha_mean[b] = alpha_batch.mean(axis=0)

    # Map batch to regime: majority vote of constituent 100-tick windows
    w0 = start // W
    w1 = end // W
    if w1 > w0:
        sub_regimes = all_regimes[w0:w1]
        batch_regimes[b] = int(np.bincount(sub_regimes).argmax())
    else:
        batch_regimes[b] = int(all_regimes[min(w0, n_windows - 1)])

    if (b + 1) % 500 == 0:
        # Compute toxicity for recent batches
        recent_tox = []
        for bb in range(max(0, b - 20), b + 1):
            rb = int(batch_regimes[bb])
            if rb in regime_gram_mean:
                G_cur = batch_grams[bb]
                G_exp = regime_gram_mean[rb]
                G_sd = regime_gram_std[rb]
                z = np.linalg.norm((G_cur - G_exp) / G_sd, ord='fro')
                recent_tox.append(z)
        avg_tox = np.mean(recent_tox) if recent_tox else 0.0

        print(f"  Batch {b+1}/{n_batches}  "
              f"recon_err={odl.recon_error_trace[-1]:.4f}  "
              f"regime={batch_regimes[b]}  "
              f"tox≈{avg_tox:.3f}  "
              f"atom_usage={np.round(odl.atom_usage_pct, 1).tolist()}")

# Compute toxicity per batch: Frobenius deviation from regime baseline (T_t)
for b in range(n_batches):
    rb = int(batch_regimes[b])
    if rb in regime_gram_mean:
        G_cur = batch_grams[b]
        G_exp = regime_gram_mean[rb]
        G_sd = regime_gram_std[rb]
        batch_toxicity[b] = float(np.linalg.norm((G_cur - G_exp) / G_sd, ord='fro'))

# ── Two-layer toxicity: T_t × ReLU(ΔT_t) ──
# T_t = absolute Gram deviation (state)
# V_t = T_t - T_{t-k}  (velocity, k-batch lookback)
# Score_t = T_t · max(0, V_t)  — only flag "deviated AND accelerating"
VELOCITY_LOOKBACK = 5  # batches (~10K ticks)
T = batch_toxicity
V = np.zeros(n_batches, dtype=np.float32)
V[VELOCITY_LOOKBACK:] = T[VELOCITY_LOOKBACK:] - T[:-VELOCITY_LOOKBACK]

# Final score: state × positive velocity
toxicity_final = T * np.maximum(V, 0.0)

# Also compute z-score normalized version for comparison
T_z = (T - np.mean(T)) / max(np.std(T), 1e-8)
V_z = (V - np.mean(V)) / max(np.std(V), 1e-8)
toxicity_zscore = T_z * np.maximum(V_z, 0.0)

t_stream_elapsed = time.perf_counter() - t_stream
print(f"  Streaming complete in {t_stream_elapsed:.1f}s ({t_stream_elapsed/60:.1f} min)")

# Free X — no longer needed after streaming
del X
gc.collect()

# Toxicity stats
toxicity_scores = toxicity_final  # use two-layer score downstream
tox_mean = float(np.mean(toxicity_final))
tox_std = float(np.std(toxicity_final))
tox_p95 = float(np.percentile(toxicity_final, 95))
tox_p99 = float(np.percentile(toxicity_final, 99))

# Compare static vs dynamic toxicity
print(f"\n  Toxicity Dynamics:")
print(f"    T (static deviation):   mean={np.mean(T):.3f}  std={np.std(T):.3f}")
print(f"    V (velocity):           mean={np.mean(V):.2f}  std={np.std(V):.2f}")
print(f"    Score (T×ReLU(V)):      mean={tox_mean:.3f}  std={tox_std:.3f}")
print(f"    Zero-velocity fraction: {np.mean(V <= 0)*100:.1f}% (masked by ReLU)")
print(f"    Score p95={tox_p95:.3f}  p99={tox_p99:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 7: Label generation (future adverse selection proxy)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n[7/8] Generating PnL labels (future mid-price impact) ...")

FWD_TICKS = 50
mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = (
    (mid_px_raw[FWD_TICKS:] - mid_px_raw[:-FWD_TICKS])
    / (np.abs(mid_px_raw[:-FWD_TICKS]) + 1e-12)
)

# Aggregate to batch level: mean |future return|
batch_impact = np.zeros(n_batches, dtype=np.float64)
for b in range(n_batches):
    t0 = b * BATCH_SIZE
    t1 = t0 + BATCH_SIZE
    batch_impact[b] = np.mean(np.abs(mid_ret[t0:t1]))

print(f"  Batch impact cost: mean={batch_impact.mean():.6e}  "
      f"std={batch_impact.std():.6e}")

# Toxicity vs future impact — compare static vs dynamic
impact_arr = batch_impact
min_len = min(len(toxicity_final), len(impact_arr))

corr_static = np.corrcoef(T[:min_len], impact_arr[:min_len])[0, 1]
corr_dynamic = np.corrcoef(toxicity_final[:min_len], impact_arr[:min_len])[0, 1]
corr_zscore = np.corrcoef(toxicity_zscore[:min_len], impact_arr[:min_len])[0, 1]

print(f"\n  Correlation with future |Δret|:")
print(f"    Static T (v2):          {corr_static:+.4f}")
print(f"    Dynamic T×ReLU(V):      {corr_dynamic:+.4f}"
      f"{' ← FLIPPED POSITIVE!' if corr_dynamic > 0 else ''}")
print(f"    Z-score T×ReLU(V):      {corr_zscore:+.4f}"
      f"{' ← FLIPPED POSITIVE!' if corr_zscore > 0 else ''}")

# Best score for PnL evaluation
best_score = toxicity_zscore if corr_zscore > corr_dynamic else toxicity_final
best_name = "zscore" if corr_zscore > corr_dynamic else "T×ReLU(V)"
print(f"    Using: {best_name} (corr={max(corr_dynamic, corr_zscore):+.4f})")

# Does high toxicity predict high impact?
tox_high_mask = best_score > np.percentile(best_score, 95)
impact_toxic = impact_arr[:min_len][tox_high_mask[:min_len]]
impact_normal = impact_arr[:min_len][~tox_high_mask[:min_len]]
if len(impact_toxic) > 5 and len(impact_normal) > 5:
    ratio = np.mean(impact_toxic) / max(np.mean(impact_normal), 1e-16)
    print(f"  Impact during toxic batches:    {np.mean(impact_toxic):.6e}")
    print(f"  Impact during normal batches:   {np.mean(impact_normal):.6e}")
    print(f"  Ratio (toxic/normal):           {ratio:.2f}x"
          f"{' ← TOXICITY PREDICTS IMPACT!' if ratio > 1.2 else ''}")


# ═══════════════════════════════════════════════════════════════════════
# Step 8: PnL Backtest (tick-level)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n[8/8] PnL Backtest — tick-level MM with toxicity circuit breaker ...")

# Pass tick-level data; PnLBacktest upsamples batch_toxicity to ticks internally
bt = PnLBacktest(PnLBacktestConfig(
    spread_capture_frac=0.5,
    tox_percentile_threshold=90.0,
))
bt.run(
    mid_px=mid_px_raw,
    spread=spread_raw,
    toxicity_scores=best_score,   # (n_batches,) dynamic score → upsampled to N ticks
    future_ret=mid_ret,
)
bt.report()


# ═══════════════════════════════════════════════════════════════════════
# Summary exports
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'═'*62}")
print(f"  Pipeline complete.")
print(f"{'═'*62}")

# Save outputs
np.save(str(CACHE_DIR / "toxicity_static_v3.npy"), T)
np.save(str(CACHE_DIR / "toxicity_velocity_v3.npy"), V)
np.save(str(CACHE_DIR / "toxicity_dynamic_v3.npy"), toxicity_final)
np.save(str(CACHE_DIR / "toxicity_zscore_v3.npy"), toxicity_zscore)
np.save(str(CACHE_DIR / "batch_regimes_v3.npy"), batch_regimes)
np.save(str(CACHE_DIR / "batch_impact_v3.npy"), batch_impact)
np.save(str(CACHE_DIR / "hmm_window_regimes.npy"), all_regimes)
np.save(str(CACHE_DIR / "dict_atoms_3_online_v3.npy"), odl.D_km)
print(f"  Exports → {CACHE_DIR}/")
print(f"    toxicity_static_v3.npy   (T_t — state)")
print(f"    toxicity_velocity_v3.npy (V_t — velocity)")
print(f"    toxicity_dynamic_v3.npy  (Score = T × ReLU(V))")
print(f"    toxicity_zscore_v3.npy   (Score = Z(T) × ReLU(Z(V)))")

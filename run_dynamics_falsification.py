"""
Market Dynamics Falsification Report V1

Three tests to determine if the A-matrix is real dynamics or statistical artifact:

  TEST 1: Null Model — time-shuffle, block-shuffle, phase-randomize
           Compare spectral radius: real vs null.
           real >> null = true system. real ≈ null = artifact.

  TEST 2: Mode Knockout — clamp M1, remove M3, amplify M1
           Observe ΔA, Δλ, ΔPnL after intervention.
           System collapses = causal core. Minor change = descriptive.

  TEST 3: Regime Stability — compare A across NORMAL / FRAGILE / HIGH_VOL
           Measure matrix distance and eigenvalue shift.
           Stable structure = true system. Large drift = regime artifact.

  FORCED VERDICT (exactly one):
    CONFIRMED — dynamical system with causal structure
    WEAK      — structured noise, some real patterns
    ARTIFACT  — statistical correlation, no dynamics
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from projects.compressibility_frontier.metrics.state_segmenter import segment_all
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor
from projects.compressibility_frontier.experiments.mode_dynamics import ModeDynamics

BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Market Dynamics Falsification Report V1")
print("  Null Models + Knockout + Regime Stability")
print("=" * 65)


# ===========================================================================
# [0] Build mode time series (shared)
# ===========================================================================

print("\n[0] Building 8D mode time series ...")
t0 = time.perf_counter()

builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

raw = pl.read_parquet(SOURCE,
    columns=["mid_px", "spread", "total_depth", "signed_imbalance",
             "duration_ms", "trade_px", "trade_sz", "bid_px", "bid_sz",
             "ask_px", "ask_sz", "trade_side"])
offset = raw.shape[0] - N
n_batches = N // BATCH_SIZE

dec = LayerDecomposer()
all_features = []
realized_vol = np.zeros(n_batches, dtype=np.float32)
spread_bps_arr = np.zeros(n_batches, dtype=np.float32)
total_depth_arr = np.zeros(n_batches, dtype=np.float32)
pnl_spread = np.zeros(n_batches, dtype=np.float64)
pnl_adverse = np.zeros(n_batches, dtype=np.float64)

mid_ret = np.zeros(N, dtype=np.float64)
FWD = 50
raw_mid = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
mid_ret[:-FWD] = np.abs((raw_mid[FWD:] - raw_mid[:-FWD]) / (np.abs(raw_mid[:-FWD]) + 1e-12))
rng = np.random.RandomState(42)

for b in range(n_batches):
    s, e = offset + b*BATCH_SIZE, offset + (b+1)*BATCH_SIZE
    batch = raw.slice(s, e-s)
    layers = dec.decompose_batch(
        batch["trade_px"].to_numpy().astype(np.float64),
        batch["trade_sz"].to_numpy().astype(np.float64),
        batch["trade_side"].to_numpy().astype(np.float64),
        batch["bid_px"].to_numpy().astype(np.float64),
        batch["bid_sz"].to_numpy().astype(np.float64),
        batch["ask_px"].to_numpy().astype(np.float64),
        batch["ask_sz"].to_numpy().astype(np.float64),
        batch["duration_ms"].to_numpy().astype(np.float64),
    )
    feat = np.array([
        layers["order_flow"]["trade_arrival_rate"],
        layers["order_flow"]["signed_imbalance"],
        layers["order_flow"]["flow_persistence"],
        layers["liquidity"]["spread_bps"],
        layers["liquidity"]["total_depth"],
        layers["liquidity"]["queue_pressure"],
        layers["liquidity"]["liquidity_tension"],
        layers["price_impact"]["realized_volatility"],
        layers["price_impact"]["nonlinear_response"],
    ], dtype=np.float32)
    all_features.append(feat)
    realized_vol[b] = layers["price_impact"]["realized_volatility"]
    spread_bps_arr[b] = layers["liquidity"]["spread_bps"]
    total_depth_arr[b] = layers["liquidity"]["total_depth"]

    batch_spread = batch["spread"].to_numpy().astype(np.float64)
    sh = batch_spread * 0.5
    adv = np.abs(mid_ret[s-offset:e-offset]) * raw_mid[s-offset:e-offset]
    fb = rng.binomial(1, 0.85, BATCH_SIZE).astype(bool)
    fa = rng.binomial(1, 0.85, BATCH_SIZE).astype(bool)
    pnl_spread[b] = np.sum(sh[fb]) + np.sum(sh[fa])
    pnl_adverse[b] = -np.sum(adv[fb]) - np.sum(adv[fa])

del raw

feat_matrix = np.array(all_features, dtype=np.float32)
f_mean, f_std = feat_matrix.mean(axis=0), np.maximum(feat_matrix.std(axis=0), 1e-8)
feat_z = np.clip((feat_matrix - f_mean) / f_std, -10, 10)

me = ModeExtractor(n_modes=8)
me.fit(feat_z, ["v"+str(i) for i in range(9)])
z_raw = me.project(feat_z)
z_mean, z_std = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-8)
z_series = (z_raw - z_mean) / z_std

regimes = segment_all(realized_vol, spread_bps_arr, total_depth_arr)
pnl_net = pnl_spread + pnl_adverse

# Real A-matrix
md_real = ModeDynamics()
md_real.fit(z_series, [f"M{i}" for i in range(8)])
rho_real = max(abs(np.linalg.eigvals(md_real.A)))

print(f"  z_series: {z_series.shape}  real ρ(A)={rho_real:.4f}  time={time.perf_counter()-t0:.1f}s")


# ═════════════════════════════════════════════════════════════════════
# TEST 1: Null Models
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  TEST 1: Null Model Spectral Radius Comparison")
print(f"{'═'*65}")

N_series, K = z_series.shape
rng_null = np.random.RandomState(42)

# Null-1: Time shuffle — destroy temporal order
z_shuffle = z_series.copy()
for k in range(K):
    rng_null.shuffle(z_shuffle[:, k])

md_null1 = ModeDynamics()
md_null1.fit(z_shuffle, [f"M{i}" for i in range(K)])
rho_null1 = max(abs(np.linalg.eigvals(md_null1.A)))

# Null-2: Block shuffle — preserve local autocorrelation
block_size = 100
n_blocks = N_series // block_size
block_order = np.arange(n_blocks)
rng_null.shuffle(block_order)
z_block = np.zeros_like(z_series)
for i, blk in enumerate(block_order):
    s, e = blk*block_size, min((blk+1)*block_size, N_series)
    s2, e2 = i*block_size, i*block_size + (e-s)
    z_block[s2:e2] = z_series[s:e]

md_null2 = ModeDynamics()
md_null2.fit(z_block, [f"M{i}" for i in range(K)])
rho_null2 = max(abs(np.linalg.eigvals(md_null2.A)))

# Null-3: Phase randomization — preserve spectrum, destroy causality
z_phase = np.zeros_like(z_series)
for k in range(K):
    fft = np.fft.rfft(z_series[:, k])
    phase = np.angle(fft)
    mag = np.abs(fft)
    rng_null.shuffle(phase)
    fft_rand = mag * np.exp(1j * phase)
    z_phase[:, k] = np.fft.irfft(fft_rand, n=N_series)

md_null3 = ModeDynamics()
md_null3.fit(z_phase, [f"M{i}" for i in range(K)])
rho_null3 = max(abs(np.linalg.eigvals(md_null3.A)))

print(f"\n  Spectral radius ρ(A):")
print(f"    REAL (full dynamics):     ρ = {rho_real:.4f}")
print(f"    Null-1 (time shuffle):    ρ = {rho_null1:.4f}  "
      f"ratio = {rho_real/max(rho_null1,1e-12):.2f}x")
print(f"    Null-2 (block shuffle):   ρ = {rho_null2:.4f}  "
      f"ratio = {rho_real/max(rho_null2,1e-12):.2f}x")
print(f"    Null-3 (phase rand):      ρ = {rho_null3:.4f}  "
      f"ratio = {rho_real/max(rho_null3,1e-12):.2f}x")

# Null model verdict
ratios = [
    rho_real / max(rho_null1, 1e-12),
    rho_real / max(rho_null2, 1e-12),
    rho_real / max(rho_null3, 1e-12),
]
avg_ratio = np.mean(ratios)
if avg_ratio > 1.3:
    null_verdict = "REAL DYNAMICS — spectral radius significantly exceeds all nulls"
elif avg_ratio > 1.05:
    null_verdict = "WEAK STRUCTURE — real > null but margin is thin"
else:
    null_verdict = "STATISTICAL ARTIFACT — real ≈ null, no dynamical structure"

print(f"\n  Avg ratio (real/null): {avg_ratio:.2f}x")
print(f"  Null Test Verdict: {null_verdict}")


# ═════════════════════════════════════════════════════════════════════
# TEST 2: Mode Knockout
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  TEST 2: Mode Knockout — Causal Intervention")
print(f"{'═'*65}")

# Identify M1 and M3 indices (from previous run: flow_persistence and vol modes)
# M1 = flow_persistence mode (index 1), M3 = vol mode (index 3)
idx_M1 = 1  # flow_persistence
idx_M3 = 3  # real_vol

print(f"\n  Baseline PnL: mean={pnl_net.mean():.1f}  std={pnl_net.std():.1f}")

# ── Experiment A: Clamp M1 to zero ──
z_clamp = z_series.copy()
z_clamp[:, idx_M1] = 0.0
md_clamp = ModeDynamics()
md_clamp.fit(z_clamp, [f"M{i}" for i in range(K)])
rho_clamp = max(abs(np.linalg.eigvals(md_clamp.A)))
delta_rho_A = rho_real - rho_clamp

print(f"\n  [A] Clamp M1 → 0:")
print(f"      ρ(A) from {rho_real:.4f} → {rho_clamp:.4f}  (Δ={delta_rho_A:+.4f})")
print(f"      ΔA_Frobenius = {np.linalg.norm(md_real.A - md_clamp.A, 'fro'):.4f}")

# ── Experiment B: Remove M3 (set to zero) ──
z_remove = z_series.copy()
z_remove[:, idx_M3] = 0.0
md_remove = ModeDynamics()
md_remove.fit(z_remove, [f"M{i}" for i in range(K)])
rho_remove = max(abs(np.linalg.eigvals(md_remove.A)))
delta_rho_B = rho_real - rho_remove

print(f"\n  [B] Remove M3 → 0:")
print(f"      ρ(A) from {rho_real:.4f} → {rho_remove:.4f}  (Δ={delta_rho_B:+.4f})")
print(f"      ΔA_Frobenius = {np.linalg.norm(md_real.A - md_remove.A, 'fro'):.4f}")

# ── Experiment C: Amplify M1 × 1.5 ──
z_amp = z_series.copy()
z_amp[:, idx_M1] *= 1.5
md_amp = ModeDynamics()
md_amp.fit(z_amp, [f"M{i}" for i in range(K)])
rho_amp = max(abs(np.linalg.eigvals(md_amp.A)))
delta_rho_C = rho_real - rho_amp

print(f"\n  [C] Amplify M1 × 1.5:")
print(f"      ρ(A) from {rho_real:.4f} → {rho_amp:.4f}  (Δ={delta_rho_C:+.4f})")
print(f"      ΔA_Frobenius = {np.linalg.norm(md_real.A - md_amp.A, 'fro'):.4f}")

# Knockout verdict
max_delta = max(abs(delta_rho_A), abs(delta_rho_B), abs(delta_rho_C))
if max_delta > 0.1:
    knockout_verdict = "CAUSAL — interventions change system dynamics significantly"
elif max_delta > 0.03:
    knockout_verdict = "WEAK CAUSAL — some effect but structure is robust"
else:
    knockout_verdict = "DESCRIPTIVE — interventions have negligible effect"

print(f"\n  Max Δρ: {max_delta:.4f}")
print(f"  Knockout Verdict: {knockout_verdict}")


# ═════════════════════════════════════════════════════════════════════
# TEST 3: Regime Stability
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  TEST 3: Regime Stability — Cross-Regime Matrix Comparison")
print(f"{'═'*65}")

# Fit A per regime
regime_names = []
for rname, mask in regimes.items():
    if mask.sum() < 50 or rname.startswith("NON_"):
        continue
    indices = np.where(mask)[0]
    z_r = z_series[indices]
    if len(z_r) < 50:
        continue

    Y = z_r[1:]; X = z_r[:-1]
    XtX = X.T @ X; XtY = X.T @ Y
    A_r = np.linalg.solve(XtX + 1e-4*np.eye(K), XtY).T
    ev_r = np.linalg.eigvals(A_r)
    rho_r = max(abs(ev_r))

    regime_names.append((rname, A_r, rho_r))
    print(f"  {rname:<16s}: ρ={rho_r:.4f}  "
          f"stable={int(np.sum(np.real(ev_r)<0))}  "
          f"max_Re(λ)={float(np.max(np.real(ev_r))):+.4f}")

# Pairwise matrix distances
if len(regime_names) >= 2:
    print(f"\n  Cross-regime matrix distances ||A_i - A_j||_F:")
    for i in range(len(regime_names)):
        for j in range(i+1, len(regime_names)):
            rn_i, A_i, _ = regime_names[i]
            rn_j, A_j, _ = regime_names[j]
            dist = float(np.linalg.norm(A_i - A_j, 'fro'))
            norm_i = float(np.linalg.norm(A_i, 'fro'))
            rel_dist = dist / max(norm_i, 1e-12)
            print(f"    {rn_i:<16s} <-> {rn_j:<16s}: |D|={dist:.4f}  "
                  f"relative={rel_dist:.2%}")

# Regime stability verdict
if len(regime_names) >= 2:
    rel_dists = []
    for i in range(len(regime_names)):
        for j in range(i+1, len(regime_names)):
            _, A_i, _ = regime_names[i]
            _, A_j, _ = regime_names[j]
            dist = float(np.linalg.norm(A_i - A_j, 'fro'))
            norm_i = float(np.linalg.norm(A_i, 'fro'))
            rel_dists.append(dist / max(norm_i, 1e-12))

    avg_rel_dist = np.mean(rel_dists)
    if avg_rel_dist < 0.15:
        regime_verdict = "STABLE — A-matrix is regime-invariant (true structure)"
    elif avg_rel_dist < 0.30:
        regime_verdict = "MODERATE DRIFT — structure adapts to regime but core preserved"
    else:
        regime_verdict = "UNSTABLE — A-matrix is regime-dependent (artifact)"
else:
    avg_rel_dist = 0
    regime_verdict = "INSUFFICIENT DATA"

print(f"\n  Avg relative distance: {avg_rel_dist:.2%}")
print(f"  Regime Stability Verdict: {regime_verdict}")


# ═════════════════════════════════════════════════════════════════════
# FORCED FINAL VERDICT
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  F I N A L   V E R D I C T")
print(f"{'═'*65}")

# Scoring
score = 0
if avg_ratio > 1.3: score += 2
elif avg_ratio > 1.05: score += 1
if max_delta > 0.1: score += 2
elif max_delta > 0.03: score += 1
if avg_rel_dist < 0.30: score += 1

if score >= 4:
    FINAL = "CONFIRMED DYNAMICAL SYSTEM"
    detail = "Market mode dynamics are real, causal, and regime-stable."
elif score >= 2:
    FINAL = "WEAK STRUCTURED NOISE"
    detail = "Some real dynamics exist but statistical artifacts dominate."
else:
    FINAL = "STATISTICAL ARTIFACT"
    detail = "A-matrix is a correlation artifact, not a dynamical system."

print(f"\n  ┌─────────────────────────────────────────────────────────┐")
print(f"  │  VERDICT: {FINAL:<44s} │")
print(f"  │  {detail:<51s} │")
print(f"  └─────────────────────────────────────────────────────────┘")

print(f"\n  Evidence:")
print(f"    Null model ratio:     {avg_ratio:.2f}x  {'[PASS]' if avg_ratio > 1.05 else '[FAIL]'}")
print(f"    Knockout sensitivity: {max_delta:.4f}  {'[PASS]' if max_delta > 0.03 else '[FAIL]'}")
print(f"    Regime stability:     {avg_rel_dist:.2%}  {'[PASS]' if avg_rel_dist < 0.30 else '[FAIL]'}")

print(f"\n{'═'*65}")
print(f"  Falsification Report complete.")
print(f"{'═'*65}")

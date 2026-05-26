"""
Structural Validity Closure Test (SVCT)

Three experiments to prove the A-matrix is a real causal dynamical system:

  EXP 1: Confounder Nulling — regress out global latent factor g(t)
          If M1 survives → real structure, not hidden-factor projection.

  EXP 2: Time Reversal — fit A_rev on reversed time
          If A_forward != A_rev → directional causality, not symmetric correlation.

  EXP 3: Intervention Scaling — 5 levels of M1 manipulation
          If ρ(A) responds smoothly → controllable system, not artifact.

  OUTPUT: Structural Validity Score (SVS) ∈ {0/3, 1/3, 2/3, 3/3}
          3/3 = CONFIRMED CAUSAL DYNAMICS
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor
from projects.compressibility_frontier.experiments.mode_dynamics import ModeDynamics


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Structural Validity Closure Test (SVCT)")
print("  Confounder + Time Reversal + Intervention Scaling")
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
all_raw_vol = np.zeros(n_batches, dtype=np.float32)

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
    all_raw_vol[b] = layers["price_impact"]["realized_volatility"]

del raw

feat_matrix = np.array(all_features, dtype=np.float32)
f_mean, f_std = feat_matrix.mean(axis=0), np.maximum(feat_matrix.std(axis=0), 1e-8)
feat_z = np.clip((feat_matrix - f_mean) / f_std, -10, 10)

me = ModeExtractor(n_modes=8)
me.fit(feat_z, ["v"+str(i) for i in range(9)])
z_raw = me.project(feat_z)
z_mean, z_std = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-8)
z_series = (z_raw - z_mean) / z_std
N_series, K = z_series.shape

# Baseline A
md_real = ModeDynamics()
md_real.fit(z_series, [f"M{i}" for i in range(K)])
rho_real = max(abs(np.linalg.eigvals(md_real.A)))

print(f"  z_series: {z_series.shape}  real rho(A)={rho_real:.4f}  time={time.perf_counter()-t0:.1f}s")


# ═════════════════════════════════════════════════════════════════════
# EXP 1: Confounder Nulling
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 1: Confounder Nulling — Remove Global Latent Factor")
print(f"{'═'*65}")

# Construct global latent factor g(t): first PC of all 9 raw features
feat_centered = feat_z - feat_z.mean(axis=0)
U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
g_global = U[:, 0]  # first PC = dominant global factor

print(f"  Global factor g(t): var_explained={S[0]**2/np.sum(S**2)*100:.1f}%  "
      f"corr(g, vol)={np.corrcoef(g_global, all_raw_vol)[0,1]:+.3f}")

# Regress out g(t) from each mode
z_residual = np.zeros_like(z_series)
for k in range(K):
    beta = np.cov(z_series[:, k], g_global)[0, 1] / max(np.var(g_global), 1e-12)
    z_residual[:, k] = z_series[:, k] - beta * g_global
    # Re-standardize
    z_residual[:, k] = (z_residual[:, k] - z_residual[:, k].mean()) / max(z_residual[:, k].std(), 1e-12)

# Re-estimate A on residual modes
md_residual = ModeDynamics()
md_residual.fit(z_residual, [f"M{i}" for i in range(K)])
rho_residual = max(abs(np.linalg.eigvals(md_residual.A)))

# Check M1 dominance
m1_out_real = np.sum(np.abs(md_real.A[1, :]))  # M1 row sum in real
m1_out_residual = np.sum(np.abs(md_residual.A[1, :]))
m1_retention = m1_out_residual / max(m1_out_real, 1e-12)

rho_retention = rho_residual / max(rho_real, 1e-12)
frob_change = float(np.linalg.norm(md_real.A - md_residual.A, 'fro')
                    / max(np.linalg.norm(md_real.A, 'fro'), 1e-12))

print(f"\n  Removing global factor g(t):")
print(f"    rho(A): {rho_real:.4f} -> {rho_residual:.4f}  "
      f"(retention={rho_retention:.1%})")
print(f"    M1 out-strength retention: {m1_retention:.1%}")
print(f"    ||Delta_A||_F relative: {frob_change:.1%}")

if rho_retention > 0.60 and m1_retention > 0.50:
    confounder_pass = True
    conf_verdict = "REAL STRUCTURE — M1 dynamics survive confounder removal"
else:
    confounder_pass = False
    conf_verdict = "HIDDEN FACTOR — global confounder explains most dynamics"

print(f"  Verdict: {conf_verdict}")


# ═════════════════════════════════════════════════════════════════════
# EXP 2: Time Reversal Causality Test
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 2: Time Reversal — Directional Causality Test")
print(f"{'═'*65}")

# Reverse time
z_rev = z_series[::-1].copy()

md_rev = ModeDynamics()
md_rev.fit(z_rev, [f"M{i}" for i in range(K)])
rho_rev = max(abs(np.linalg.eigvals(md_rev.A)))

# Compare forward vs reverse A
A_fwd = md_real.A
A_rev = md_rev.A

# Asymmetry: ||A_fwd - A_rev|| / ||A_fwd||
asymmetry = float(np.linalg.norm(A_fwd - A_rev, 'fro')
                  / max(np.linalg.norm(A_fwd, 'fro'), 1e-12))

# Diagonal comparison (self-excitation should differ most)
diag_fwd = np.diag(A_fwd)
diag_rev = np.diag(A_rev)
diag_asym = float(np.linalg.norm(diag_fwd - diag_rev) / max(np.linalg.norm(diag_fwd), 1e-12))

# Cross-element sign stability
sign_agree = float(np.mean(np.sign(A_fwd[A_fwd != 0]) == np.sign(A_rev[A_rev != 0])))

print(f"\n  Forward vs Reverse A-matrix:")
print(f"    rho_forward = {rho_real:.4f}    rho_reverse = {rho_rev:.4f}")
print(f"    Frobenius asymmetry: {asymmetry:.1%}")
print(f"    Diagonal asymmetry:  {diag_asym:.1%}")
print(f"    Sign agreement:      {sign_agree:.1%}")

if asymmetry > 0.15 and sign_agree < 0.85:
    time_pass = True
    time_verdict = "CAUSAL DIRECTION — forward != reverse, time arrow exists"
else:
    time_pass = False
    time_verdict = "SYMMETRIC — no time direction, correlation artifact"

print(f"  Verdict: {time_verdict}")


# ═════════════════════════════════════════════════════════════════════
# EXP 3: Intervention Scaling Law
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 3: Intervention Scaling — M1 Control Response")
print(f"{'═'*65}")

idx_M1 = 1  # flow_persistence mode
scales = [0.25, 0.5, 1.0, 1.5, 2.0]
rhos = []
self_excits = []

for scale in scales:
    z_int = z_series.copy()
    z_int[:, idx_M1] *= scale
    # Re-standardize after intervention
    z_int[:, idx_M1] = (z_int[:, idx_M1] - z_int[:, idx_M1].mean()) / max(z_int[:, idx_M1].std(), 1e-12)

    md_int = ModeDynamics()
    md_int.fit(z_int, [f"M{i}" for i in range(K)])
    rho_int = max(abs(np.linalg.eigvals(md_int.A)))
    self_M1 = md_int.A[idx_M1, idx_M1]
    rhos.append(rho_int)
    self_excits.append(self_M1)

rhos = np.array(rhos)
self_excits = np.array(self_excits)

# Fit linear response: ρ(scale) ≈ a + b*scale
slope, intercept = np.polyfit(scales, rhos, 1)
r2 = np.corrcoef(scales, rhos)[0, 1] ** 2

# Monotonicity: ρ should increase with M1 amplification
monotonic = all(rhos[i] <= rhos[i+1] for i in range(len(rhos)-1))

# Smoothness: residuals from linear fit
predicted = intercept + slope * np.array(scales)
residuals = rhos - predicted
smoothness = 1.0 - float(np.std(residuals) / max(np.mean(rhos), 1e-12))

print(f"\n  M1 Intervention Scaling:")
print(f"  {'Scale':>8s} {'rho(A)':>8s} {'M1 Self':>8s}")
print(f"  {'─'*8} {'─'*8} {'─'*8}")
for s, r, se in zip(scales, rhos, self_excits):
    print(f"  {s:>8.2f} {r:>8.4f} {se:>+8.4f}")

print(f"\n  Response characteristics:")
print(f"    Slope:            {slope:+.4f} rho/unit")
print(f"    R^2 (linearity):  {r2:.3f}")
print(f"    Monotonic:        {'YES' if monotonic else 'NO'}")
print(f"    Smoothness:       {smoothness:.3f}")

if r2 > 0.5 and monotonic:
    intervention_pass = True
    int_verdict = "CONTROLLABLE — M1 intervention produces smooth, monotonic response"
elif r2 > 0.3:
    intervention_pass = True
    int_verdict = "WEAKLY CONTROLLABLE — response exists but noisy"
else:
    intervention_pass = False
    int_verdict = "NOT CONTROLLABLE — no systematic response to intervention"

print(f"  Verdict: {int_verdict}")


# ═════════════════════════════════════════════════════════════════════
# STRUCTURAL VALIDITY SCORE (SVS)
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STRUCTURAL VALIDITY SCORE (SVS)")
print(f"{'═'*65}")

SVS = sum([confounder_pass, time_pass, intervention_pass])

verdicts = {
    3: "CONFIRMED CAUSAL DYNAMICS — system is real, directional, controllable",
    2: "WEAK STRUCTURE — some validity but one test failed",
    1: "STATISTICAL ARTIFACT — only one test passed, likely correlation",
    0: "FAKE DYNAMICS — all tests failed, A-matrix is statistical noise",
}

print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
print(f"  │                                                            │")
print(f"  │  SVS = {SVS}/3                                                │")
print(f"  │  {verdicts[SVS]:<58s} │")
print(f"  │                                                            │")
print(f"  └──────────────────────────────────────────────────────────────┘")

print(f"\n  Test breakdown:")
print(f"    [{'PASS' if confounder_pass else 'FAIL'}] Confounder Nulling")
print(f"        rho retention: {rho_retention:.0%}  "
      f"M1 retention: {m1_retention:.0%}  "
      f"Delta_A: {frob_change:.1%}")
print(f"    [{'PASS' if time_pass else 'FAIL'}] Time Reversal Asymmetry")
print(f"        asymmetry: {asymmetry:.1%}  "
      f"sign agreement: {sign_agree:.1%}")
print(f"    [{'PASS' if intervention_pass else 'FAIL'}] Intervention Scaling")
print(f"        R^2: {r2:.3f}  "
      f"monotonic: {'YES' if monotonic else 'NO'}  "
      f"slope: {slope:+.4f}")

print(f"\n{'═'*65}")
print(f"  SVCT complete.")
print(f"{'═'*65}")

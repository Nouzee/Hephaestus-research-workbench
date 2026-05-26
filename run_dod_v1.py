"""
Drift Origin Decomposition (DOD v1)

Finds what generates the 1D dynamical backbone of the market mode system.

  STEP 1: Extract v1 = dominant eigenvector of A, interpret it
  STEP 2: Project raw microstructure features onto v1 backbone
  STEP 3: Causality test — lag correlation + Granger on raw space
  STEP 4: Remove v1 — does the system collapse?

Answers: "Why is the market system effectively 1-dimensional?"
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Drift Origin Decomposition (DOD v1)")
print("  What generates the 1D backbone?")
print("=" * 65)


# ===========================================================================
# [0] Build system
# ===========================================================================

print("\n[0] Building mode system + raw features ...")
t0 = time.perf_counter()

builder = MatrixBuilder()
X, _ = builder.assemble()
N_ticks, M_feat = X.shape

raw = pl.read_parquet(SOURCE)
offset = raw.shape[0] - N_ticks
n_batches = N_ticks // BATCH_SIZE

dec = LayerDecomposer()

# Collect BOTH mode features AND raw microstructure variables per batch
all_mode_features = []
raw_vars = {}  # {name: (n_batches,) array}

# Raw variable extractors (batch-level means)
for raw_col in ["signed_imbalance", "flow_persistence", "spread_bps",
                "total_depth", "queue_pressure", "liquidity_tension",
                "realized_volatility", "nonlinear_response"]:
    raw_vars[raw_col] = np.zeros(n_batches, dtype=np.float32)

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

    all_mode_features.append([
        layers["order_flow"]["trade_arrival_rate"],
        layers["order_flow"]["signed_imbalance"],
        layers["order_flow"]["flow_persistence"],
        layers["liquidity"]["spread_bps"],
        layers["liquidity"]["total_depth"],
        layers["liquidity"]["queue_pressure"],
        layers["liquidity"]["liquidity_tension"],
        layers["price_impact"]["realized_volatility"],
        layers["price_impact"]["nonlinear_response"],
    ])

    # Raw microstructure vars (different from mode features — these are
    # the DIRECT batch aggregates, not mode projections)
    raw_vars["signed_imbalance"][b] = layers["order_flow"]["signed_imbalance"]
    raw_vars["flow_persistence"][b] = layers["order_flow"]["flow_persistence"]
    raw_vars["spread_bps"][b] = layers["liquidity"]["spread_bps"]
    raw_vars["total_depth"][b] = layers["liquidity"]["total_depth"]
    raw_vars["queue_pressure"][b] = layers["liquidity"]["queue_pressure"]
    raw_vars["liquidity_tension"][b] = layers["liquidity"]["liquidity_tension"]
    raw_vars["realized_volatility"][b] = layers["price_impact"]["realized_volatility"]
    raw_vars["nonlinear_response"][b] = layers["price_impact"]["nonlinear_response"]

del raw

feat_matrix = np.array(all_mode_features, dtype=np.float32)
f_mean, f_std = feat_matrix.mean(axis=0), np.maximum(feat_matrix.std(axis=0), 1e-8)
feat_z = np.clip((feat_matrix - f_mean) / f_std, -10, 10)

me = ModeExtractor(n_modes=8)
me.fit(feat_z, ["v"+str(i) for i in range(9)])
z_raw = me.project(feat_z)
z_mean_g, z_std_g = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-8)
z_series = (z_raw - z_mean_g) / z_std_g
N_series, K_full = z_series.shape

# 6-mode basis A
BASIS = [1, 2, 3, 4, 5, 6]
z_basis = z_series[:, BASIS]
Y = z_basis[1:]; X_mat = z_basis[:-1]
A = np.linalg.solve(X_mat.T @ X_mat + 1e-4*np.eye(6), X_mat.T @ Y).T

print(f"  z_basis: {z_basis.shape}  rho(A)={max(abs(np.linalg.eigvals(A))):.4f}  "
      f"time={time.perf_counter()-t0:.1f}s")


# ═════════════════════════════════════════════════════════════════════
# STEP 1: Extract v1 — dominant drift direction
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 1: Dominant Backbone Direction v1")
print(f"{'═'*65}")

ev_A, evec_A = np.linalg.eig(A)
# Sort by real part descending
idx_sort = np.argsort(np.real(ev_A))[::-1]
ev_A = ev_A[idx_sort]
evec_A = evec_A[:, idx_sort]

v1 = np.real(evec_A[:, 0])  # dominant eigenvector
v1 = v1 / np.linalg.norm(v1)

print(f"\n  Eigenvalues of A:")
for k in range(6):
    lam = ev_A[k]
    bar_len = int(abs(lam) * 40 / max(abs(ev_A[0]), 1e-12))
    bar = "#" * bar_len + "." * (40 - bar_len)
    print(f"    λ{k}: {np.real(lam):>+8.4f} + {np.imag(lam):>+8.4f}i  {bar}")

print(f"\n  v1 (dominant drift direction) = ")
for i, basis_idx in enumerate(BASIS):
    marker = " < LARGEST" if abs(v1[i]) == max(abs(v1)) else ""
    print(f"    mode M{basis_idx}: {v1[i]:>+8.4f}{marker}")

# Interpret v1: what combination of modes is it?
print(f"\n  v1 interpretation:")
dom_idx = np.argmax(np.abs(v1))
dom_sign = "positive" if v1[dom_idx] > 0 else "negative"
print(f"    Dominant: M{BASIS[dom_idx]} ({dom_sign})")
print(f"    v1 pushes the system toward: "
      f"{'expansion' if np.real(ev_A[0]) > 0 else 'contraction'}")


# ═════════════════════════════════════════════════════════════════════
# STEP 2: Project raw features onto v1
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 2: Project Raw Microstructure onto v1 Backbone")
print(f"{'═'*65}")

# Standardize raw vars
raw_matrix = np.column_stack([raw_vars[name] for name in raw_vars.keys()])
raw_names = list(raw_vars.keys())
raw_std = (raw_matrix - raw_matrix.mean(axis=0)) / np.maximum(raw_matrix.std(axis=0), 1e-8)

# Backbone time series: b(t) = z_basis(t) @ v1
backbone = z_basis @ v1                             # (N,) projection onto v1

# Correlate each raw variable with the backbone
print(f"\n  Raw variable correlation with 1D backbone:")
print(f"  {'Variable':<25s} {'Corr with v1':>14s} {'|Corr|':>10s} {'Role':>20s}")
print(f"  {'─'*25} {'─'*14} {'─'*10} {'─'*20}")

raw_corrs = {}
for idx, name in enumerate(raw_names):
    c = np.corrcoef(raw_std[:, idx], backbone)[0, 1]
    raw_corrs[name] = c
    role = ("PRIMARY GENERATOR" if abs(c) > 0.4 else
            "strong driver" if abs(c) > 0.2 else
            "moderate" if abs(c) > 0.1 else "weak")
    print(f"  {name:<25s} {c:>+14.4f} {abs(c):>10.4f} {role:>20s}")

# Also: project raw features through the ORIGINAL feature→mode mapping
# The mode extractor's phi matrix maps raw features → modes
phi = me.phi  # (9, 8) — from original 9 features to 8 modes
phi_basis = phi[:, BASIS]  # (9, 6)
v1_raw_weights = phi_basis @ v1  # (9,) — raw feature contribution to backbone

print(f"\n  Raw feature contribution to v1 (through PCA mapping):")
print(f"  {'Raw Feature':<25s} {'Weight in v1':>14s}")
print(f"  {'─'*25} {'─'*14}")
for idx, name in enumerate(["arrival", "signed_imb", "flow_persist",
                              "spread", "depth", "queue_press",
                              "liq_tension", "real_vol", "nonlinear"]):
    print(f"  {name:<25s} {v1_raw_weights[idx]:>+14.4f}")


# ═════════════════════════════════════════════════════════════════════
# STEP 3: Backbone Causality — does v1 lead raw variables?
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 3: Backbone Causality — v1 lead-lag on raw variables")
print(f"{'═'*65}")

# For each raw variable: corr(v1[t], raw[t+k])
print(f"\n  Lead-lag: v1(t) → raw_var(t+k)")
print(f"  {'Variable':<25s} {'k=0':>8s} {'k=1':>8s} {'k=3':>8s} {'k=5':>8s} {'k=10':>8s} {'Direction':>14s}")
print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")

for name in raw_names:
    rv = raw_std[:, raw_names.index(name)]
    corrs = []
    for k in [0, 1, 3, 5, 10]:
        if k == 0:
            c = np.corrcoef(backbone, rv)[0, 1]
        else:
            c = np.corrcoef(backbone[:-k], rv[k:])[0, 1]
        corrs.append(c)

    # Direction: does v1 lead (corr increases with k) or follow (decreases)?
    if abs(corrs[1]) > abs(corrs[0]) * 1.1:
        direction = "LEADS"
    elif abs(corrs[0]) > abs(corrs[1]) * 1.1:
        direction = "COINCIDENT"
    else:
        direction = "neutral"

    print(f"  {name:<25s} {corrs[0]:>+8.4f} {corrs[1]:>+8.4f} "
          f"{corrs[2]:>+8.4f} {corrs[3]:>+8.4f} {corrs[4]:>+8.4f} {direction:>14s}")


# ═════════════════════════════════════════════════════════════════════
# STEP 4: Remove v1 — system collapse?
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 4: Remove v1 — System Collapse Test")
print(f"{'═'*65}")

# Project out the backbone: z_residual = z - (z @ v1) @ v1^T
z_removed = z_basis - np.outer(backbone, v1)

# Re-fit A
Y_r = z_removed[1:]; X_r = z_removed[:-1]
A_removed = np.linalg.solve(X_r.T @ X_r + 1e-4*np.eye(6), X_r.T @ Y_r).T
rho_removed = max(abs(np.linalg.eigvals(A_removed)))

# R² collapse
def r2_predict(z_data):
    Yf = z_data[1:]; Xf = z_data[:-1]
    XtXf = Xf.T @ Xf; XtYf = Xf.T @ Yf
    try:
        W = np.linalg.solve(XtXf + 1e-4*np.eye(6), XtYf)
    except np.linalg.LinAlgError:
        return 0.0
    Yp = Xf @ W
    ss_r = np.sum((Yf - Yp)**2)
    ss_t = np.sum((Yf - Yf.mean(axis=0))**2)
    return float(1.0 - ss_r / max(ss_t, 1e-12))

r2_full = r2_predict(z_basis)
r2_removed = r2_predict(z_removed)

print(f"\n  Before removing v1:")
print(f"    rho(A) = {max(abs(np.linalg.eigvals(A))):.4f}")
print(f"    R^2    = {r2_full:.4f}")

print(f"\n  After removing v1 (projecting out backbone):")
print(f"    rho(A) = {rho_removed:.4f}  "
      f"({'DROPS' if rho_removed < max(abs(np.linalg.eigvals(A)))*0.8 else 'stable'})")
print(f"    R^2    = {r2_removed:.4f}  "
      f"({'COLLAPSES' if r2_removed < r2_full*0.5 else 'survives'})")

rho_retention = rho_removed / max(max(abs(np.linalg.eigvals(A))), 1e-12)
r2_retention = r2_removed / max(r2_full, 1e-12)

if rho_retention < 0.5:
    collapse = "FULL COLLAPSE — v1 IS the system"
elif rho_retention < 0.8:
    collapse = "PARTIAL — v1 is major but not sole driver"
else:
    collapse = "NO COLLAPSE — v1 is descriptive, not generative"

print(f"\n  Collapse verdict: {collapse}")


# ═════════════════════════════════════════════════════════════════════
# Final: What generates the 1D backbone?
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  Drift Origin Verdict")
print(f"{'═'*65}")

# Rank raw variables by absolute correlation with backbone
ranked = sorted(raw_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Top 3 generators of the 1D backbone:")
for i, (name, c) in enumerate(ranked[:3]):
    print(f"    {i+1}. {name:<25s}  corr={c:+.4f}")

# Classify the origin
top_name, top_corr = ranked[0]

if abs(top_corr) > 0.4:
    origin = f"MICROSTRUCTURE — {top_name} directly generates the backbone"
elif abs(top_corr) > 0.15:
    origin = f"WEAK MICROSTRUCTURE — {top_name} partially drives it"
else:
    origin = "REPRESENTATION ARTIFACT — backbone is a PCA construction, not microstructure"

print(f"\n  Origin classification: {origin}")

# Eigenvalue interpretation
dom_lam = np.real(ev_A[0])
if dom_lam > 0.05:
    dynamics_nature = "EXPANSIVE — system drifts away from origin (no attractor)"
elif dom_lam < -0.05:
    dynamics_nature = "CONTRACTIVE — system has a stable attractor"
else:
    dynamics_nature = "NEUTRAL — system is at the edge of stability"

print(f"  Dynamics nature: {dynamics_nature}")
print(f"  λ1 = {dom_lam:+.4f}  (ρ(A) = {max(abs(ev_A)):.4f})")

print(f"\n  Final answer:")
print(f"    The 1D backbone is generated by: {top_name}")
print(f"    Its nature is:                   {dynamics_nature.lower()}")
print(f"    Removing it causes:              {collapse.lower()}")
print(f"{'═'*65}")

"""
Market Information Geometry (MIG v1)

Rebuilds the geometric structure of the 6-mode minimal basis.

  STEP 1: Three distance metrics (prediction / dynamical / information)
  STEP 2: Intrinsic geometry — MDS embedding + eigen spectrum decay
  STEP 3: Curvature — Ricci-like proxy on mode space
  STEP 4: Flow field — A·M vector field, attractor/repeller, divergence

Answers: "What is the shape of market mode space?"
"""

import sys, time
from pathlib import Path
import numpy as np
import polars as pl
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor
from projects.compressibility_frontier.experiments.mode_dynamics import ModeDynamics


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Market Information Geometry (MIG v1)")
print("  Mode Space Metric + Curvature + Flow Field")
print("=" * 65)


# ===========================================================================
# [0] Build 8D mode time series + A-matrix
# ===========================================================================

print("\n[0] Building mode system ...")
t0 = time.perf_counter()

builder = MatrixBuilder()
X, _ = builder.assemble()
N_ticks, M_feat = X.shape

raw = pl.read_parquet(SOURCE,
    columns=["mid_px", "spread", "total_depth", "signed_imbalance",
             "duration_ms", "trade_px", "trade_sz", "bid_px", "bid_sz",
             "ask_px", "ask_sz", "trade_side"])
offset = raw.shape[0] - N_ticks
n_batches = N_ticks // BATCH_SIZE

dec = LayerDecomposer()
all_features = []
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
    all_features.append([
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

del raw
feat_matrix = np.array(all_features, dtype=np.float32)
f_mean, f_std = feat_matrix.mean(axis=0), np.maximum(feat_matrix.std(axis=0), 1e-8)
feat_z = np.clip((feat_matrix - f_mean) / f_std, -10, 10)

me = ModeExtractor(n_modes=8)
me.fit(feat_z, ["v"+str(i) for i in range(9)])
z_raw = me.project(feat_z)
z_mean_g, z_std_g = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-8)
z_series = (z_raw - z_mean_g) / z_std_g
N_series, K_full = z_series.shape

# A-matrix
Y = z_series[1:]; X_mat = z_series[:-1]
XtX = X_mat.T @ X_mat; XtY = X_mat.T @ Y
A_full = np.linalg.solve(XtX + 1e-4*np.eye(K_full), XtY).T

# Minimal basis from OMB: B* = {1,2,3,4,5,6}  (0-indexed)
BASIS = [1, 2, 3, 4, 5, 6]
K = len(BASIS)
z_basis = z_series[:, BASIS]
A_basis = A_full[np.ix_(BASIS, BASIS)]

print(f"  Basis: {BASIS}  z_shape={z_basis.shape}  "
      f"rho(A)={max(abs(np.linalg.eigvals(A_basis))):.4f}  "
      f"time={time.perf_counter()-t0:.1f}s")


# ═════════════════════════════════════════════════════════════════════
# STEP 1: Three Distance Metrics (6x6)
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 1: Mode Space Metrics (6x6)")
print(f"{'═'*65}")

# ── Metric 1: Prediction distance ──
# d_pred(i,j) = R² drop when removing BOTH i and j, minus solo drops
def r2_with_subset(indices):
    idx = [BASIS[i] for i in indices]
    X_sub = z_series[:-1][:, idx]
    Y_full = z_series[1:][:, BASIS]
    XtX_s = X_sub.T @ X_sub; XtY_s = X_sub.T @ Y_full
    try:
        W = np.linalg.solve(XtX_s + 1e-4*np.eye(len(idx)), XtY_s)
    except np.linalg.LinAlgError:
        return 0.0
    Y_pred = X_sub @ W
    ss_res = np.sum((Y_full - Y_pred)**2)
    ss_tot = np.sum((Y_full - Y_full.mean(axis=0))**2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))

full_r2_basis = r2_with_subset(list(range(K)))
solo_r2 = {i: r2_with_subset([i]) for i in range(K)}

D_pred = np.zeros((K, K))
for i in range(K):
    for j in range(i+1, K):
        pair_r2 = r2_with_subset([i, j])
        # Joint information loss vs additive expectation
        d = (solo_r2[i] + solo_r2[j]) - pair_r2
        D_pred[i, j] = D_pred[j, i] = float(max(d, 0.0))

# ── Metric 2: Dynamical distance ──
D_dyn = np.zeros((K, K))
for i in range(K):
    for j in range(i+1, K):
        d = float(np.linalg.norm(A_basis[i] - A_basis[j]))
        D_dyn[i, j] = D_dyn[j, i] = d

# ── Metric 3: Information distance ──
# d_info(i,j) = 1 - |corr(z_i, z_j)|  (complement of linear dependence)
corr_matrix = np.corrcoef(z_basis.T)
D_info = 1.0 - np.abs(corr_matrix)

# Print all three
for name, D in [("Prediction", D_pred), ("Dynamical", D_dyn), ("Information", D_info)]:
    print(f"\n  {name} Distance Matrix:")
    header = "         " + "".join(f"  M{BASIS[j]:>3d} " for j in range(K))
    print(header)
    for i in range(K):
        row = f"  M{BASIS[i]:>3d}  " + "".join(f"{D[i,j]:>7.4f}" for j in range(K))
        print(row)

    # Eigen spectrum of distance matrix
    ev = np.linalg.eigvalsh(D)
    ev_pos = ev[ev > 1e-8]
    decay_90 = np.searchsorted(np.cumsum(ev_pos[::-1]) / ev_pos.sum(), 0.90) + 1
    print(f"    Eigen-decay: 90% in {decay_90} dims (of {K})  "
          f"ratio σ1/σK={ev_pos[-1]/max(ev_pos[0],1e-12):.1f}x")


# ═════════════════════════════════════════════════════════════════════
# STEP 2: Intrinsic Geometry — MDS + Spectral Embedding
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 2: Intrinsic Geometry — MDS Embedding")
print(f"{'═'*65}")

# Classical MDS on prediction distance
D = D_pred.copy()
n = K
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (D ** 2) @ H
ev_mds, evec_mds = np.linalg.eigh(B)
# Sort descending
idx_sort = np.argsort(ev_mds)[::-1]
ev_mds = ev_mds[idx_sort]
evec_mds = evec_mds[:, idx_sort]
ev_pos_mds = ev_mds[ev_mds > 1e-8]

print(f"\n  MDS Eigen spectrum (prediction distance):")
cum = 0
for d in range(min(5, len(ev_pos_mds))):
    v = ev_pos_mds[d]
    pct = v / ev_pos_mds.sum() * 100
    cum += pct
    bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
    print(f"    dim {d+1}: {v:.4f} ({pct:>5.1f}%)  cum={cum:.1f}%  {bar}")

intrinsic_dim = np.searchsorted(np.cumsum(ev_pos_mds) / ev_pos_mds.sum(), 0.90) + 1
print(f"\n  Intrinsic dimension: {intrinsic_dim}/{K}  "
      f"({'LOW — space is compressible' if intrinsic_dim <= 3 else 'HIGH — space is intrinsically high-D'})")

# Coordinates in first 2 dims
coords_2d = evec_mds[:, :2] * np.sqrt(np.maximum(ev_mds[:2], 0))[:, None]
print(f"\n  Mode coordinates (first 2 MDS dims):")
for i in range(K):
    print(f"    M{BASIS[i]}: ({coords_2d[0,i]:>+8.4f}, {coords_2d[1,i]:>+8.4f})")


# ═════════════════════════════════════════════════════════════════════
# STEP 3: Curvature — Ricci-like proxy
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 3: Curvature — Ricci-like proxy on mode space")
print(f"{'═'*65}")

# κ(i,j) = Δd_dyn(i,j) - Δd_info(i,j)
# Positive = dynamical distance exceeds information distance (expansion)
# Negative = information distance exceeds dynamical distance (contraction)
D_dyn_norm = D_dyn / max(D_dyn.max(), 1e-12)
D_info_norm = D_info / max(D_info.max(), 1e-12)
Kappa = D_dyn_norm - D_info_norm

print(f"\n  Curvature matrix (positive=expansion, negative=contraction):")
header = "         " + "".join(f"  M{BASIS[j]:>3d} " for j in range(K))
print(header)
for i in range(K):
    row = f"  M{BASIS[i]:>3d}  " + "".join(f"{Kappa[i,j]:>+7.4f}" for j in range(K))
    print(row)

n_pos = int(np.sum(Kappa > 0.05))
n_neg = int(np.sum(Kappa < -0.05))
n_flat = K*K - K - n_pos - n_neg  # exclude diagonal

print(f"\n  Curvature regions:")
print(f"    Positive (expansion):  {n_pos} edges  "
      f"{'dominant — space is EXPANDING' if n_pos > n_neg else ''}")
print(f"    Negative (contraction): {n_neg} edges  "
      f"{'dominant — space is CONTRACTING' if n_neg > n_pos else ''}")
print(f"    Flat (≈0):              {n_flat} edges")

if n_pos > n_neg * 1.5:
    curvature_type = "EXPANSIVE — dynamical forces push modes apart"
elif n_neg > n_pos * 1.5:
    curvature_type = "CONTRACTIVE — information structure pulls modes together"
else:
    curvature_type = "MIXED — both expansion and contraction regions"

print(f"  Type: {curvature_type}")


# ═════════════════════════════════════════════════════════════════════
# STEP 4: Flow Field — A·M vector field
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  STEP 4: Flow Field — A·M on the 6D mode space")
print(f"{'═'*65}")

# Compute flow at each point: F(z) = A @ z  (expected next state)
# Sample representative points: cluster centers of z_basis
from scipy.cluster.vq import kmeans2
n_clusters = 8
centroids, labels = kmeans2(z_basis.astype(np.float64), n_clusters, minit='points')

# Flow at each centroid
flows = np.array([A_basis @ c for c in centroids])

# Divergence: trace of Jacobian = trace(A) at each point (constant for linear)
divergence = float(np.trace(A_basis))

# Fixed points: solve (A - I) z = 0
M_identity = np.eye(K)
try:
    null_space = linalg.null_space(A_basis - M_identity)
    has_fixed_point = null_space.shape[1] > 0
except Exception:
    has_fixed_point = False

# Attractor/repeller classification from eigenvalues
ev_A = np.linalg.eigvals(A_basis)
n_attracting = int(np.sum(np.real(ev_A) < 0))
n_repelling = int(np.sum(np.real(ev_A) > 0.02))
n_neutral = K - n_attracting - n_repelling

print(f"\n  Flow field properties:")
print(f"    Divergence (trace A): {divergence:+.4f}  "
      f"({'EXPANDING' if divergence > 0.01 else 'CONTRACTING' if divergence < -0.01 else 'DIVERGENCE-FREE'})")
print(f"    Attracting directions:  {n_attracting}/{K}")
print(f"    Repelling directions:   {n_repelling}/{K}")
print(f"    Neutral directions:     {n_neutral}/{K}")
print(f"    Fixed point exists:     {'YES' if has_fixed_point else 'NO'}")

# Largest expansion direction
ev_idx = np.argmax(np.real(ev_A))
print(f"    Max Re(λ): {np.real(ev_A[ev_idx]):+.4f}  "
      f"(direction of fastest {'expansion' if np.real(ev_A[ev_idx]) > 0 else 'contraction'})")

print(f"\n  Flow magnitude at centroids:")
for c_idx in range(min(5, n_clusters)):
    flow_mag = float(np.linalg.norm(flows[c_idx]))
    point_mag = float(np.linalg.norm(centroids[c_idx]))
    ratio = flow_mag / max(point_mag, 1e-12)
    print(f"    centroid {c_idx}: |F|/|z| = {ratio:.4f}  "
          f"({'amplifying' if ratio > 0.5 else 'damping'})")


# ═════════════════════════════════════════════════════════════════════
# Final Summary
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  MIG v1 Complete — Market Mode Space Geometry")
print(f"{'═'*65}")

print(f"\n  Geometry Summary:")
print(f"    Intrinsic dimension:   {intrinsic_dim}/{K}")
print(f"    Curvature type:        {curvature_type}")
print(f"    Flow divergence:       {divergence:+.4f}")
print(f"    Attracting/Repelling:  {n_attracting}/{n_repelling}")

# Final classification
if intrinsic_dim <= 2 and abs(divergence) < 0.02:
    geometry_type = "EUCLIDEAN LOW-DIM — market is geometrically simple"
elif intrinsic_dim <= 3:
    geometry_type = "CURVED LOW-DIM MANIFOLD — has intrinsic geometry"
elif abs(divergence) < 0.01:
    geometry_type = "FLAT HIGH-DIM FIELD — pure high-dimensional random flow"
else:
    geometry_type = "EXPANSIVE HIGH-DIM SYSTEM — drifting high-dimensional dynamics"

print(f"\n  Geometry type: {geometry_type}")
print(f"{'═'*65}")

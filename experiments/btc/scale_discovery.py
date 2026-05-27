"""
MSDP Formal Execution — A-Share Minimal Scale Discovery

Strict protocol. No parameter tuning for desired outcome.
Uses only completed modules from modules/probability/.
"""

import sys, time, glob
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from projects.ashare.regime_segmentation import L2FeatureExtractor
from sklearn.cluster import KMeans

from modules.probability.msdp import MSDP
from modules.probability.msdp_guard import MSDPGuard


TRAIN_DIR = r"c:\Users\ZaneLaw\Desktop\000333\RawTrainData"
WINDOW_SIZE = 100
N_REGIMES = 8
N_PATHS = 80   # MC paths for MSDP

print("=" * 70)
print("  MSDP — Minimal Scale Discovery Phase")
print("  Formal CASE A/B/C Verdict for A-Share Market")
print("=" * 70)


# ===========================================================================
# [1] Build 5-scale hierarchy from A-share data
# ===========================================================================

print("\n[1] Building S0-S4 scale hierarchy from 000333 L2 data ...")
t0 = time.perf_counter()

extractor = L2FeatureExtractor(window_size=WINDOW_SIZE)
msg_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "message_*.parquet")))
ob_files = sorted(glob.glob(str(Path(TRAIN_DIR) / "orderbook_*.parquet")))

# Use 15 days for MSDP (enough for statistical significance, fast enough)
n_days = min(15, len(msg_files))
n_train_days = int(n_days * 0.60)

all_features = []; train_features = []
for day_idx in range(n_days):
    mf, of = msg_files[day_idx], ob_files[day_idx]
    msg_df = pl.read_parquet(mf); ob_df = pl.read_parquet(of)
    N = msg_df.shape[0]; n_w = N // WINDOW_SIZE
    if n_w < 5: continue
    msg_d = {col: msg_df[col].to_numpy() for col in msg_df.columns}
    ob_d = {col: ob_df[col].to_numpy() for col in ob_df.columns}
    for w in range(n_w):
        s, e = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
        feats = extractor.extract_window(
            {k: v[s:e] for k, v in ob_d.items()},
            {k: v[s:e] for k, v in msg_d.items()})
        all_features.append(list(feats.values()))
        if day_idx < n_train_days: train_features.append(list(feats.values()))

X_all = np.array(all_features, dtype=np.float32)
X_tr = np.array(train_features, dtype=np.float32)
tr_m = X_tr.mean(axis=0); tr_s = np.maximum(X_tr.std(axis=0), 1e-8)
X_z = np.clip((X_all - tr_m) / tr_s, -10, 10)
km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
km.fit(np.clip((X_tr - tr_m) / tr_s, -10, 10))
regimes = km.predict(X_z)

# ── S0: raw L2 features (16-dim standardized) ──
S0 = X_z.astype(np.float64)

# ── S1: PCA modes (first d_eff = 6 components) ──
X_c = S0 - S0.mean(axis=0)
_, S_svd, Vt = np.linalg.svd(X_c, full_matrices=False)
cum_var = np.cumsum(S_svd**2) / np.sum(S_svd**2)
d_eff = int(np.searchsorted(cum_var, 0.90)) + 1
d_eff = max(d_eff, 3)
S1 = (X_c @ Vt[:d_eff].T).astype(np.float64)

# ── S2: regime labels (8-state discrete, one-hot) ──
S2 = np.eye(N_REGIMES)[regimes].astype(np.float64)

# ── S3: hazard space (continuous tox proxy from spread/depth) ──
# Use first 2 features (spread, depth proxies) + regime for hazard
spread_proxy = S0[:, 3]  # spread_bps feature
depth_proxy = S0[:, 4]   # total_depth feature
hazard_continuous = np.clip(
    np.abs(spread_proxy) / (np.abs(depth_proxy) + 1e-8), 0, 1
).reshape(-1, 1).astype(np.float64)
S3 = np.column_stack([hazard_continuous, spread_proxy.reshape(-1, 1)])

# ── S4: backbone 1D projection ──
# Approximate backbone as first PC direction
backbone = (S0 @ Vt[0]).reshape(-1, 1).astype(np.float64)
S4 = backbone

# Trim to same length
min_N = min(len(S0), len(S1), len(S2), len(S3), len(S4))
features_by_scale = [
    S0[:min_N], S1[:min_N], S2[:min_N], S3[:min_N], S4[:min_N]
]
# Each scale needs its OWN regime discretization for kernel comparison
# S2: original KMeans on full features
regime_S2 = regimes[:min_N]

# S0: higher-resolution clustering (more clusters = finer granularity)
km_S0 = KMeans(n_clusters=12, random_state=42, n_init=5, max_iter=200)
regime_S0 = km_S0.fit_predict(S0[:min_N])

# S1: mode-space clustering
km_S1 = KMeans(n_clusters=8, random_state=42, n_init=5, max_iter=200)
regime_S1 = km_S1.fit_predict(S1[:min_N])

# S3: hazard-space discretization (fewer clusters)
km_S3 = KMeans(n_clusters=5, random_state=42, n_init=5, max_iter=200)
regime_S3 = km_S3.fit_predict(S3[:min_N])

# S4: backbone discretization (binary high/low drift)
regime_S4 = (S4[:min_N].ravel() > np.median(S4[:min_N])).astype(np.int32)

# Per-scale regime sequences for kernel comparison
regime_seqs = [regime_S0, regime_S1, regime_S2, regime_S3, regime_S4]

print(f"  Windows: {min_N}  d_eff={d_eff}")
print(f"  S0:{S0.shape} S1:{S1.shape} S2:{S2.shape} S3:{S3.shape} S4:{S4.shape}")
print(f"  Per-scale clusters: S0={km_S0.n_clusters} S1={km_S1.n_clusters} "
      f"S2={N_REGIMES} S3={km_S3.n_clusters} S4=2")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [2] GUARD CHECK (mandatory pre-flight)
# ===========================================================================

print(f"\n[2] MSDP Guard — Pre-Flight Check ...")
t0 = time.perf_counter()

# Quick guard: use entropy-based F(k) (valid across arbitrary dimensions)
def entropy_F(feats):
    """F(k) = |H(S_k) - H(S_{k+1})| — entropy gap between adjacent scales."""
    from modules.probability.scale_flow import ScaleFlow
    F = np.zeros(len(feats) - 1)
    for k in range(len(feats) - 1):
        F[k] = abs(
            ScaleFlow._gaussian_entropy(feats[k]) -
            ScaleFlow._gaussian_entropy(feats[k+1])
        )
    return F / max(F.max(), 1e-8)  # normalize

F_heph_q = entropy_F(features_by_scale)

# Quick null: shuffled (destroy structure, preserve marginals)
rng = np.random.RandomState(42)
shuffled = [X.copy() for X in features_by_scale]
for X_s in shuffled:
    for d in range(X_s.shape[1]):
        rng.shuffle(X_s[:, d])
F_null_q = entropy_F(shuffled)

# Also: iid null (Gaussian with same moments)
iid_features = []
for X in features_by_scale:
    mu, cov = np.mean(X, axis=0), np.cov(X.T)
    X_iid = rng.multivariate_normal(mu, cov + 1e-8*np.eye(X.shape[1]), size=len(X))
    iid_features.append(X_iid)
F_iid_q = entropy_F(iid_features)

guard = MSDPGuard(snr_threshold=2.0)  # entropy F(null)/F(heph) > 2 means heph is 2x more stable
guard_result = guard.check(F_heph_q, None, F_null_q)
guard.print_report(guard_result)

if not guard_result.passed:
    print(f"\n  GUARD BLOCKED. Aborting MSDP.")
    print(f"\n{'═'*70}")
    print(f"  FINAL VERDICT: CASE_C — NO SCALE")
    print(f"  Reason: {guard_result.block_reason}")
    print(f"{'═'*70}")
    sys.exit(0)

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [3] FULL MSDP EXECUTION
# ===========================================================================

print(f"\n[3] MSDP — Full Execution ({N_PATHS} MC paths) ...")
t0 = time.perf_counter()

msdp = MSDP(
    n_paths=N_PATHS,
    plateau_threshold=0.20,
    min_plateau_width=2,
    null_threshold=0.70,
    guard_enabled=True,
)
msdp.run(features_by_scale, regime_seqs)
msdp.print_report()

print(f"  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# [4] FORMAL MSDP VERDICT
# ===========================================================================

r = msdp.result
print(f"\n{'═'*70}")
print(f"  FORMAL MSDP VERDICT")
print(f"{'═'*70}")

print(f"\n  1. Overall Verdict:")
print(f"     {r.classification}")

if r.minimal_scale >= 0:
    print(f"\n  2. k* / Stable Interval:")
    print(f"     k* = {r.minimal_scale}")
    print(f"     Stable interval: [{r.stable_interval[0]}, {r.stable_interval[1]}]")
    print(f"     Plateau width: {r.interval_width}")
    print(f"     Plateau strength: {r.plateau_strength:.3f}")
else:
    print(f"\n  2. k* / Stable Interval: NONE")

print(f"\n  3. F(k) Summary:")
if r.F_curve is not None:
    print(f"     mean(F) = {np.mean(r.F_curve):.4f}")
    print(f"     std(F)  = {np.std(r.F_curve):.4f}")
    print(f"     F(k) per transition: {np.round(r.F_curve, 4).tolist()}")

print(f"\n  4. Null Comparison:")
for null_name, null_F in r.F_null_curves.items():
    ratio = np.mean(r.F_curve) / max(np.mean(null_F), 1e-8)
    significant = "SIGNIFICANT" if ratio < 0.70 else "INDISTINGUISHABLE"
    print(f"     {null_name:<15s}: ratio={ratio:.4f}  [{significant}]")

print(f"\n  5. Anti-Test Results:")
at = r.scale_flow_result
print(f"     Basis rotation:  {'PASS' if at.anti_basis_passed else 'FAIL'} (k* shift)")
print(f"     Noise injection: {'PASS' if at.anti_noise_passed else 'FAIL'} (10% noise)")
print(f"     Kernel permute:  {'PASS' if at.anti_permute_passed else 'FAIL'} (shuffled Z)")

print(f"\n  6. Cross-Time Stability:")
print(f"     {'STABLE' if at.cross_time_stable else 'UNSTABLE'} across train/val/test")

print(f"\n  7. Final Interpretation:")
if r.classification == "CASE_A":
    print(f"     This market HAS an intrinsic representation scale at k*={r.minimal_scale}.")
    print(f"     Adding more scales beyond k* does not increase information,")
    print(f"     improve dynamics closure, or enhance decision stability.")
elif r.classification == "CASE_B":
    print(f"     This market has a WEAK scale band, not a strict fixed point.")
    print(f"     Compression is regime-dependent and partially unstable under perturbation.")
else:
    print(f"     This market is a SCALE-FREE STOCHASTIC FIELD.")
    print(f"     No intrinsic representation scale exists — all compression is lossy.")
    print(f"     Structure is observation-dependent, not intrinsic.")

print(f"\n{'═'*70}")
print(f"  MSDP Protocol Complete.")
print(f"{'═'*70}")

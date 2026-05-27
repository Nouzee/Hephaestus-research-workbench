"""
Observability Minimal Basis (OMB v1)

Finds the minimal sufficient representation of the 8D market mode system.

Four experiments:
  EXP 1: Subset Prediction — which mode subset predicts future full state?
  EXP 2: Mutual Information — which modes carry the most future information?
  EXP 3: Redundancy Collapse — which modes are replaceable?
  EXP 4: Greedy Minimal Basis — B* with coverage >= 95%, minimal redundancy.

Answers: "What is the market's irreducible information vocabulary?"
"""

import sys, time
from pathlib import Path
from itertools import combinations
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve()))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.research.market_decon import LayerDecomposer
from projects.compressibility_frontier.experiments.mode_extractor import ModeExtractor


BATCH_SIZE = 2048
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

print("=" * 65)
print("  Observability Minimal Basis (OMB v1)")
print("  Information Geometry of the 8D Market System")
print("=" * 65)


# ===========================================================================
# [0] Build 8D mode time series
# ===========================================================================

print("\n[0] Building 8D mode time series ...")
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

N_series, K = z_series.shape  # K=8
print(f"  z_series: {z_series.shape}  time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Prediction target: M(t+1) the full 8D future state
# ===========================================================================

Y_full = z_series[1:]      # (N-1, 8) — future state
X_full = z_series[:-1]     # (N-1, 8) — current state


def predict_with_subset(subset_indices):
    """Predict full 8D future state using only modes in subset_indices."""
    X_sub = X_full[:, subset_indices]   # (N-1, |S|)
    # OLS: W = (X^T X)^{-1} X^T Y
    XtX = X_sub.T @ X_sub
    XtY = X_sub.T @ Y_full
    try:
        W = np.linalg.solve(XtX + 1e-4 * np.eye(len(subset_indices)), XtY)
    except np.linalg.LinAlgError:
        return {"R2": 0.0, "loss": float('inf'), "entropy": float('inf')}

    Y_pred = X_sub @ W
    # Per-dimension R²
    ss_res = np.sum((Y_full - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_full - Y_full.mean(axis=0)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2_mean = float(np.mean(r2_per_dim))

    # Total loss
    loss = float(np.mean((Y_full - Y_pred) ** 2))

    # Residual entropy: -sum(p_i * log(p_i)) on residual covariance eigenvalues
    residuals = Y_full - Y_pred
    cov_res = (residuals.T @ residuals) / len(residuals)
    eigvals = np.linalg.eigvalsh(cov_res)
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / eigvals.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    return {"R2": r2_mean, "loss": loss, "entropy": entropy,
            "r2_per_dim": r2_per_dim, "subset_size": len(subset_indices)}


# ═════════════════════════════════════════════════════════════════════
# EXP 1: Subset Prediction Test
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 1: Subset Prediction — R^2 of future state from mode subsets")
print(f"{'═'*65}")

# Full model baseline
full_result = predict_with_subset(list(range(K)))
print(f"\n  Full model (8 modes): R^2={full_result['R2']:.4f}  "
      f"loss={full_result['loss']:.4f}  entropy={full_result['entropy']:.4f}")

# Single mode baseline
print(f"\n  Single mode prediction power:")
print(f"  {'Mode':>6s} {'R^2':>8s} {'Loss':>8s} {'Entropy':>8s} {'Role':>14s}")
print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")
single_results = {}
for k in range(K):
    r = predict_with_subset([k])
    single_results[k] = r
    role = "PRIMARY" if r["R2"] > 0.3 else ("SECONDARY" if r["R2"] > 0.15 else "WEAK")
    print(f"  M{k:>5d} {r['R2']:>8.4f} {r['loss']:>8.4f} {r['entropy']:>8.4f} {role:>14s}")

# All subsets of size 2 and 3 (most informative)
print(f"\n  Top 10 subsets by R^2:")
subset_scan = []
for size in [2, 3, 4, 5, 6, 7]:
    for combo in combinations(range(K), size):
        r = predict_with_subset(list(combo))
        subset_scan.append({"subset": combo, **r})
    if len(subset_scan) > 200:
        break  # prune for speed

subset_scan.sort(key=lambda x: x["R2"], reverse=True)
for i, s in enumerate(subset_scan[:10]):
    print(f"  {i+1:>2d}. {list(s['subset'])}  R^2={s['R2']:.4f}  "
          f"loss={s['loss']:.4f}  size={s['subset_size']}")


# ═════════════════════════════════════════════════════════════════════
# EXP 2: Mutual Information Ranking
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 2: Mutual Information — I(M_i(t); M(t+1))")
print(f"{'═'*65}")

# MI approximation: I(X;Y) ≈ -0.5 * log(1 - R^2) for linear Gaussian
def mi_from_r2(r2):
    r2_clipped = np.clip(r2, 0.0, 0.9999)
    return float(-0.5 * np.log(1.0 - r2_clipped))

print(f"\n  Single-mode MI with future full state:")
print(f"  {'Mode':>6s} {'R^2':>8s} {'I(M;future)':>12s} {'% of total':>10s}")
print(f"  {'─'*6} {'─'*8} {'─'*12} {'─'*10}")

total_mi = sum(mi_from_r2(single_results[k]["R2"]) for k in range(K))
mi_values = {}
for k in range(K):
    mi = mi_from_r2(single_results[k]["R2"])
    mi_values[k] = mi
    pct = mi / max(total_mi, 1e-12) * 100
    print(f"  M{k:>5d} {single_results[k]['R2']:>8.4f} {mi:>12.4f} {pct:>9.1f}%")

# Pair synergy: I(M_i, M_j; future) - I(M_i; future) - I(M_j; future)
print(f"\n  Top pair synergies (positive = complementary information):")
pairs = []
for i in range(K):
    for j in range(i+1, K):
        r_pair = predict_with_subset([i, j])
        mi_pair = mi_from_r2(r_pair["R2"])
        mi_solo = mi_values[i] + mi_values[j]
        synergy = mi_pair - mi_solo
        pairs.append((i, j, synergy))
pairs.sort(key=lambda x: x[2], reverse=True)
for i, j, syn in pairs[:5]:
    print(f"    M{i}+M{j}: synergy={syn:+.4f}  "
          f"({'COMPLEMENTARY' if syn > 0.05 else 'redundant' if syn < -0.05 else 'additive'})")


# ═════════════════════════════════════════════════════════════════════
# EXP 3: Redundancy Collapse
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 3: Redundancy Collapse — performance drop when removing M_i")
print(f"{'═'*65}")

full_r2 = full_result["R2"]
all_modes = list(range(K))

ablation = {}
for k in range(K):
    subset = [m for m in all_modes if m != k]
    r = predict_with_subset(subset)
    delta = full_r2 - r["R2"]
    ablation[k] = {"R2": r["R2"], "delta": delta}

ablation_sorted = sorted(ablation.items(), key=lambda x: x[1]["delta"], reverse=True)

print(f"\n  {'Mode':>6s} {'R^2(-M_i)':>10s} {'Delta_R2':>10s} {'Interpretation':>20s}")
print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*20}")
for k, v in ablation_sorted:
    if v["delta"] > 0.01:
        interp = "IRREPLACEABLE core"
    elif v["delta"] > 0.003:
        interp = "contributing"
    elif v["delta"] > 0:
        interp = "marginal"
    else:
        interp = "REDUNDANT — can drop"
    print(f"  M{k:>5d} {v['R2']:>10.4f} {v['delta']:>+10.4f} {interp:>20s}")


# ═════════════════════════════════════════════════════════════════════
# EXP 4: Greedy Minimal Basis
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 4: Greedy Minimal Basis Construction")
print(f"{'═'*65}")

threshold = 0.95 * full_r2  # 95% of full model performance
available = set(range(K))
selected = []
current_r2 = 0.0

print(f"\n  Target: R^2 >= {threshold:.4f} (95% of full model {full_r2:.4f})")
print(f"\n  Greedy selection:")
print(f"  {'Step':>5s} {'Add':>5s} {'R^2':>8s} {'Gain':>8s} {'Coverage':>9s}")
print(f"  {'─'*5} {'─'*5} {'─'*8} {'─'*8} {'─'*9}")

step = 0
while current_r2 < threshold and available:
    best_mode = None
    best_r2 = current_r2

    for m in available:
        candidate = selected + [m]
        r = predict_with_subset(candidate)
        if r["R2"] > best_r2:
            best_r2 = r["R2"]
            best_mode = m

    if best_mode is None:
        break

    gain = best_r2 - current_r2
    coverage = best_r2 / max(full_r2, 1e-12) * 100
    selected.append(best_mode)
    available.remove(best_mode)
    current_r2 = best_r2
    step += 1

    marker = " < MINIMAL BASIS" if current_r2 >= threshold else ""
    print(f"  {step:>5d} M{best_mode:>4d} {current_r2:>8.4f} {gain:>+8.4f} "
          f"{coverage:>8.1f}%{marker}")


# Minimal basis report
print(f"\n  Minimal Basis B* = {sorted(selected)}")
print(f"    Modes in basis: {len(selected)}/{K}  ({len(selected)/K*100:.0f}%)")
print(f"    R^2:            {current_r2:.4f}")
print(f"    Coverage:        {current_r2/max(full_r2,1e-12)*100:.1f}% of full model")
print(f"    Compression:     {(K-len(selected))/K*100:.0f}% dimension reduction")

# What's excluded?
excluded = [m for m in range(K) if m not in selected]
print(f"    Excluded modes:  {excluded}")
if excluded:
    excl_deltas = {k: ablation[k]["delta"] for k in excluded}
    print(f"    Their Delta_R2:  {[f'M{k}={excl_deltas[k]:.4f}' for k in excluded]}")


# ═════════════════════════════════════════════════════════════════════
# Final Summary
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  OMB v1 Complete — Market Information Geometry")
print(f"{'═'*65}")

print(f"\n  Key findings:")
print(f"    Full model R^2:      {full_r2:.4f}")
print(f"    Minimal basis size:  {len(selected)}/{K}")
print(f"    Basis coverage:      {current_r2/max(full_r2,1e-12)*100:.1f}%")
print(f"    Strongest single:    M{max(single_results, key=lambda k: single_results[k]['R2'])}"
      f" (R^2={max(s['R2'] for s in single_results.values()):.4f})")
print(f"    Irreplaceable modes: {[f'M{k}' for k,v in ablation_sorted if v['delta']>0.01]}")
print(f"    Redundant modes:     {[f'M{k}' for k,v in ablation_sorted if v['delta']<=0]}")
print(f"{'═'*65}")

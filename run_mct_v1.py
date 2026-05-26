"""
Market Controllability Test (MCT v1)

Tests whether the 8D mode system is controllable via M1 intervention.

Three intervention types on M1 (flow_persistence, proven causal hub):
  TYPE A: Impulse shock — M1(t0) += epsilon
  TYPE B: Sustained bias — M1(t0:t0+T) += epsilon
  TYPE C: Clamp — M1(t) = mean(M1) for T batches

Four response variables:
  (1) Mode propagation: ΔM_i(t+k) for other modes
  (2) Stability shift: Δρ = ρ(A_post) - ρ(A_pre)
  (3) Interaction shift: ||A_post - A_pre||_F
  (4) Recovery time: τ until system returns to 90% baseline

Three controllability metrics:
  Gain G = ||ΔM||_post / |ε|
  Persistence P = ∫ ΔM(t) dt
  Structural shift S = ||A_post - A_pre||_F

CRITICAL: NO re-standardization after intervention. Use RAW mode values.
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


BATCH_SIZE, FWD_TICKS = 2048, 50
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

WINDOW = 200          # batches per analysis window
STRIDE = 50           # stride between windows
N_INTERVENTIONS = 10  # number of intervention windows

print("=" * 65)
print("  Market Controllability Test (MCT v1)")
print("  Impulse + Sustained + Clamp on M1")
print("=" * 65)


# ===========================================================================
# [0] Build mode time series (raw, NO re-standardization after fit)
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

# Mode extraction (one-time, global)
me = ModeExtractor(n_modes=8)
me.fit(feat_z, ["v"+str(i) for i in range(9)])
z_raw = me.project(feat_z)
z_mean_g, z_std_g = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-8)
z_series = (z_raw - z_mean_g) / z_std_g  # ONE-TIME standardization

N_series, K = z_series.shape
idx_M1 = 1

# Baseline A on full data
md_full = ModeDynamics()
md_full.fit(z_series, [f"M{i}" for i in range(K)])
rho_baseline = max(abs(np.linalg.eigvals(md_full.A)))
A_baseline = md_full.A.copy()

print(f"  z_series: {z_series.shape}  baseline rho={rho_baseline:.4f}  "
      f"time={time.perf_counter()-t0:.1f}s")


# ===========================================================================
# Helper: fit A on a window, return rho + A
# ===========================================================================

def fit_window(z_window):
    """Fit A on a window of z_series. Returns (A, rho)."""
    if len(z_window) < 20:
        return None, None
    Y = z_window[1:]
    X_mat = z_window[:-1]
    XtX = X_mat.T @ X_mat
    XtY = X_mat.T @ Y
    try:
        A = np.linalg.solve(XtX + 1e-4*np.eye(K), XtY).T
    except np.linalg.LinAlgError:
        return None, None
    rho = max(abs(np.linalg.eigvals(A)))
    return A, rho


# ===========================================================================
# Run intervention experiments
# ===========================================================================

print(f"\n[1] Running {N_INTERVENTIONS} intervention windows "
      f"(window={WINDOW}, stride={STRIDE}) ...")
print(f"{'='*65}")

results = {"impulse": [], "sustained": [], "clamp": []}

for i_start in range(0, N_series - WINDOW - 100, STRIDE):
    if len(results["impulse"]) >= N_INTERVENTIONS:
        break

    t0_w = i_start
    t1_w = i_start + WINDOW
    z_window = z_series[t0_w:t1_w].copy()  # RAW values, no re-standardize

    # Pre-intervention A
    A_pre, rho_pre = fit_window(z_window)
    if A_pre is None:
        continue

    sigma_M1 = float(np.std(z_window[:, idx_M1]))
    eps_1sigma = sigma_M1

    # ── TYPE A: Impulse shock ──
    z_impulse = z_window.copy()
    t_shock = WINDOW // 3
    z_impulse[t_shock, idx_M1] += eps_1sigma * 2  # +2σ shock
    A_post_i, rho_post_i = fit_window(z_impulse)
    if A_post_i is not None:
        # Mode propagation: ΔM for all modes at t_shock+1..t_shock+10
        delta_M = np.mean(np.abs(
            z_impulse[t_shock+1:t_shock+11] - z_window[t_shock+1:t_shock+11]
        ), axis=0)
        gain = float(np.mean(delta_M) / abs(eps_1sigma * 2)) if eps_1sigma > 0 else 0
        persistence = float(np.sum(np.abs(
            z_impulse[t_shock+1:t_shock+21] - z_window[t_shock+1:t_shock+21]
        )))
        struct_shift = float(np.linalg.norm(A_post_i - A_pre, 'fro'))
        delta_rho = rho_post_i - rho_pre

        # Recovery time: steps until ||z_impulse - z_window|| < 10% of peak deviation
        deviations = np.array([
            np.linalg.norm(z_impulse[t_shock+k] - z_window[t_shock+k])
            for k in range(1, min(100, WINDOW - t_shock - 1))
        ])
        peak = deviations.max()
        if peak > 1e-8:
            recovered = np.where(deviations < 0.1 * peak)[0]
            tau = int(recovered[0]) if len(recovered) > 0 else len(deviations)
        else:
            tau = 0

        results["impulse"].append({
            "gain": gain, "persistence": persistence,
            "struct_shift": struct_shift, "delta_rho": delta_rho,
            "recovery_time": tau,
        })

    # ── TYPE B: Sustained bias ──
    z_sustained = z_window.copy()
    t_bias_start = WINDOW // 3
    T_bias = 20
    z_sustained[t_bias_start:t_bias_start+T_bias, idx_M1] += eps_1sigma
    A_post_s, rho_post_s = fit_window(z_sustained)
    if A_post_s is not None:
        delta_M_s = np.mean(np.abs(
            z_sustained[t_bias_start+T_bias:t_bias_start+T_bias+10]
            - z_window[t_bias_start+T_bias:t_bias_start+T_bias+10]
        ), axis=0)
        gain_s = float(np.mean(delta_M_s) / abs(eps_1sigma)) if eps_1sigma > 0 else 0
        persistence_s = float(np.sum(np.abs(
            z_sustained[t_bias_start:t_bias_start+T_bias+20]
            - z_window[t_bias_start:t_bias_start+T_bias+20]
        )))
        struct_shift_s = float(np.linalg.norm(A_post_s - A_pre, 'fro'))
        delta_rho_s = rho_post_s - rho_pre

        results["sustained"].append({
            "gain": gain_s, "persistence": persistence_s,
            "struct_shift": struct_shift_s, "delta_rho": delta_rho_s,
        })

    # ── TYPE C: Clamp ──
    z_clamp = z_window.copy()
    t_clamp_start = WINDOW // 3
    T_clamp = 50
    mu_M1 = float(np.mean(z_window[:, idx_M1]))
    z_clamp[t_clamp_start:t_clamp_start+T_clamp, idx_M1] = mu_M1
    A_post_c, rho_post_c = fit_window(z_clamp)
    if A_post_c is not None:
        delta_M_c = np.mean(np.abs(
            z_clamp[t_clamp_start+T_clamp:t_clamp_start+T_clamp+10]
            - z_window[t_clamp_start+T_clamp:t_clamp_start+T_clamp+10]
        ), axis=0)
        gain_c = float(np.mean(delta_M_c) / max(sigma_M1, 1e-12))
        struct_shift_c = float(np.linalg.norm(A_post_c - A_pre, 'fro'))
        delta_rho_c = rho_post_c - rho_pre

        results["clamp"].append({
            "gain": gain_c, "struct_shift": struct_shift_c,
            "delta_rho": delta_rho_c,
        })


# ===========================================================================
# Aggregate + Classify
# ===========================================================================

print(f"\n[2] Aggregating results across {N_INTERVENTIONS} windows ...")

metrics = {}
for itype in ["impulse", "sustained", "clamp"]:
    if not results[itype]:
        continue
    arr = results[itype]
    metrics[itype] = {
        "gain_mean": float(np.mean([r["gain"] for r in arr])),
        "gain_std": float(np.std([r["gain"] for r in arr])),
        "struct_shift_mean": float(np.mean([r["struct_shift"] for r in arr])),
        "delta_rho_mean": float(np.mean([r["delta_rho"] for r in arr])),
    }
    if "persistence" in arr[0]:
        metrics[itype]["persistence_mean"] = float(np.mean([r["persistence"] for r in arr]))
    if "recovery_time" in arr[0]:
        metrics[itype]["recovery_time_mean"] = float(np.mean([r["recovery_time"] for r in arr]))

# Print
print(f"\n  {'Type':<12s} {'Gain':>8s} {'Persistence':>12s} "
      f"{'StructShift':>12s} {'DeltaRho':>10s} {'Recovery':>9s}")
print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*9}")

for itype in ["impulse", "sustained", "clamp"]:
    m = metrics.get(itype, {})
    if not m:
        continue
    print(f"  {itype:<12s} {m.get('gain_mean',0):>8.4f} "
          f"{m.get('persistence_mean',0):>12.2f} "
          f"{m.get('struct_shift_mean',0):>12.4f} "
          f"{m.get('delta_rho_mean',0):>+10.4f} "
          f"{m.get('recovery_time_mean',0):>9.1f}")


# ===========================================================================
# Classification
# ===========================================================================

print(f"\n[3] Classification ...")

# Use impulse as primary (cleanest signal)
imp = metrics.get("impulse", {})
G = imp.get("gain_mean", 0)
P = imp.get("persistence_mean", 0)
S = imp.get("struct_shift_mean", 0)
drho = imp.get("delta_rho_mean", 0)
tau = imp.get("recovery_time_mean", 0)

print(f"\n  Controllability metrics (impulse shock):")
print(f"    Gain G:           {G:.4f}  "
      f"({'STRONG' if G > 1 else 'weak' if G > 0.3 else 'NONE'})")
print(f"    Persistence P:    {P:.2f}  "
      f"({'LONG' if P > 10 else 'SHORT'})")
print(f"    Structural S:     {S:.4f}  "
      f"({'SIGNIFICANT' if S > 0.1 else 'minor'})")
print(f"    Delta rho:        {drho:+.4f}")
print(f"    Recovery tau:     {tau:.1f} batches")

# Classification rules
if G > 1.0 and drho > 0.02 and S > 0.1:
    classification = "CONTROLLABLE"
    detail = "M1 intervention produces strong, persistent structural change"
elif G > 0.3 and abs(drho) > 0.005:
    classification = "WEAKLY CONTROLLABLE"
    detail = "M1 intervention has measurable but transient effects"
else:
    classification = "UNCONTROLLABLE"
    detail = "System self-normalizes; M1 is descriptive, not causal"

print(f"\n  ┌──────────────────────────────────────────────────────────┐")
print(f"  │  MCT v1 Classification: {classification:<33s} │")
print(f"  │  {detail:<50s} │")
print(f"  └──────────────────────────────────────────────────────────┘")


# ===========================================================================
# Mode-specific propagation (which modes respond to M1 shock?)
# ===========================================================================

print(f"\n[4] Mode-specific propagation (M1 impulse -> other modes) ...")

# Re-run one clean impulse experiment on a representative window
rep_start = N_series // 3
rep_end = rep_start + WINDOW
z_rep = z_series[rep_start:rep_end].copy()
t_shock = WINDOW // 3

z_shocked = z_rep.copy()
sigma = float(np.std(z_rep[:, idx_M1]))
z_shocked[t_shock, idx_M1] += sigma * 2

# Per-mode response at k=1,3,5,10 after shock
print(f"\n  {'Mode':>4s} {'k=1':>8s} {'k=3':>8s} {'k=5':>8s} {'k=10':>8s} {'PeakResp':>10s}")
print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

for k_mode in range(K):
    responses = []
    for lag in [1, 3, 5, 10]:
        if t_shock + lag < WINDOW:
            resp = z_shocked[t_shock+lag, k_mode] - z_rep[t_shock+lag, k_mode]
            responses.append(resp)
        else:
            responses.append(0.0)
    peak = max(abs(r) for r in responses) if responses else 0
    marker = " < RESPONDS" if peak > 0.1 else ""
    print(f"  M{k_mode:>3d} {responses[0]:>+8.4f} {responses[1]:>+8.4f} "
          f"{responses[2]:>+8.4f} {responses[3]:>+8.4f} {peak:>10.4f}{marker}")

print(f"\n{'═'*65}")
print(f"  MCT v1 complete.")
print(f"{'═'*65}")

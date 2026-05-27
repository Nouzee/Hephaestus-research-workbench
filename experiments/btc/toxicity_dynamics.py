"""
Toxicity Dynamics Diagnostic ? lead-lag, phase decomposition, jerk analysis.

Loads v3 saved outputs and answers:
  1. Does T_t lead or lag impact? (lead-lag cross-correlation)
  2. Does velocity V_t have predictive power at different leads?
  3. Phase decomposition: T-high+V-pos vs T-high+V-neg vs T-low
  4. Does jerk (acceleration of deviation) predict impact?
  5. At what lead does correlation peak, and is it positive?
"""

import numpy as np
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"

# Load
T = np.load(str(CACHE / "toxicity_static_v3.npy"))       # static deviation
V = np.load(str(CACHE / "toxicity_velocity_v3.npy"))      # velocity
Score = np.load(str(CACHE / "toxicity_dynamic_v3.npy"))   # T ? ReLU(V)
Impact = np.load(str(CACHE / "batch_impact_v3.npy"))      # future |?ret|
Regimes = np.load(str(CACHE / "batch_regimes_v3.npy"))    # HMM regime per batch

N = len(T)
min_len = min(N, len(Impact))
T = T[:min_len]
V = V[:min_len]
Score = Score[:min_len]
Impact = Impact[:min_len]

# ???????????????????????????????????????????????????????????????????????
# 1. Lead-lag cross-correlation: corr(T[t], Impact[t+lag])
# ???????????????????????????????????????????????????????????????????????

print("=" * 70)
print("  Lead-Lag Analysis: corr(signal[t], impact[t+lag])")
print("  Positive lag = signal LEADS impact (predictive)")
print("=" * 70)

MAX_LAG = 20

print(f"\n{'Lag':>5}  {'T (static)':>12}  {'V (velocity)':>12}  {'T?ReLU(V)':>12}  {'z(T)?ReLU(z(V))':>16}")
print(f"{'?'*5}  {'?'*12}  {'?'*12}  {'?'*12}  {'?'*16}")

best_lag = 0
best_corr = -999

for lag in range(MAX_LAG + 1):
    if lag == 0:
        corr_T = np.corrcoef(T, Impact)[0, 1]
        corr_V = np.corrcoef(V, Impact)[0, 1]
        corr_S = np.corrcoef(Score, Impact)[0, 1]

        Tz = (T - T.mean()) / T.std()
        Vz = (V - V.mean()) / V.std()
        Sz = Tz * np.maximum(Vz, 0)
        corr_Sz = np.corrcoef(Sz, Impact)[0, 1]
    else:
        corr_T = np.corrcoef(T[:-lag], Impact[lag:])[0, 1]
        corr_V = np.corrcoef(V[:-lag], Impact[lag:])[0, 1]
        corr_S = np.corrcoef(Score[:-lag], Impact[lag:])[0, 1]
        corr_Sz = np.corrcoef(Sz[:-lag], Impact[lag:])[0, 1]

    marker = " ? PEAK" if corr_S > best_corr else ""
    if corr_S > best_corr:
        best_corr = corr_S
        best_lag = lag

    print(f"{lag:>5}  {corr_T:>+12.4f}  {corr_V:>+12.4f}  {corr_S:>+12.4f}  {corr_Sz:>+16.4f}{marker}")

print(f"\n  Best predictive lag: {best_lag} batches (corr={best_corr:+.4f})")
if best_corr > 0.02:
    print(f"  [OK] Signal IS predictive at lag={best_lag} (corr={best_corr:+.4f})")
else:
    print(f"  [LOW] Signal never becomes positively predictive (best={best_corr:+.4f})")
    print(f"  -> Gram deviation fundamentally measures POST-impact state, not PRE-impact")

# ???????????????????????????????????????????????????????????????????????
# 2. Phase decomposition
# ???????????????????????????????????????????????????????????????????????

print(f"\n{'?'*70}")
print(f"  Phase Decomposition: T (deviation) ? V (velocity direction)")
print(f"{'?'*70}")

T_median = np.median(T)
phase_T_high_V_pos = (T > T_median) & (V > 0)
phase_T_high_V_neg = (T > T_median) & (V <= 0)
phase_T_low_V_pos = (T <= T_median) & (V > 0)
phase_T_low_V_neg = (T <= T_median) & (V <= 0)

phases = {
    "T? V? (accelerating away)": phase_T_high_V_pos,
    "T? V? (returning to normal)": phase_T_high_V_neg,
    "T? V? (normal, speeding up)": phase_T_low_V_pos,
    "T? V? (normal, slowing down)": phase_T_low_V_neg,
}

for name, mask in phases.items():
    n = mask.sum()
    pct = n / len(T) * 100
    imp_mean = Impact[mask].mean() if n > 0 else 0
    imp_std = Impact[mask].std() if n > 1 else 0
    T_mean = T[mask].mean() if n > 0 else 0
    V_mean = V[mask].mean() if n > 0 else 0
    print(f"  {name:35s}  {pct:5.1f}%  impact={imp_mean:.4e}?{imp_std:.4e}  "
          f"T?={T_mean:.1f}  V?={V_mean:.1f}")

# The critical test: does T?V? have higher future impact than T?V??
if phase_T_high_V_pos.sum() > 10 and phase_T_high_V_neg.sum() > 10:
    imp_pos = Impact[phase_T_high_V_pos].mean()
    imp_neg = Impact[phase_T_high_V_neg].mean()
    ratio = imp_pos / max(imp_neg, 1e-16)
    print(f"\n  Critical test: T?V? impact / T?V? impact = {ratio:.2f}x")
    if ratio > 1.1:
        print(f"  [OK] Velocity direction matters ? accelerating deviation is worse")
    else:
        print(f"  [X] Velocity doesn't differentiate future impact")

# ???????????????????????????????????????????????????????????????????????
# 3. Jerk (acceleration) analysis
# ???????????????????????????????????????????????????????????????????????

print(f"\n{'?'*70}")
print(f"  Jerk Analysis: J_t = V_t - V_{t-3} (acceleration of deviation)")
print(f"{'?'*70}")

JERK_LAG = 3
J = np.zeros_like(V)
J[JERK_LAG:] = V[JERK_LAG:] - V[:-JERK_LAG]

# Lead-lag for jerk
for lag in range(MAX_LAG + 1):
    if lag == 0:
        corr_J = np.corrcoef(J, Impact)[0, 1]
    else:
        corr_J = np.corrcoef(J[:-lag], Impact[lag:])[0, 1]
    if lag <= 5 or corr_J > 0.02:
        print(f"  J[t] -> impact[t+{lag}]: corr={corr_J:+.4f}"
              f"{' ? JERK PREDICTS!' if corr_J > 0.03 else ''}")

# Combined: T ? ReLU(V) ? ReLU(J) ? state ? velocity ? acceleration
Score_J = Score * np.maximum(J, 0)
corr_SJ = np.corrcoef(Score_J, Impact)[0, 1]
for lag in range(1, 6):
    c = np.corrcoef(Score_J[:-lag], Impact[lag:])[0, 1]
    print(f"  T?ReLU(V)?ReLU(J)[t] -> impact[t+{lag}]: corr={c:+.4f}"
          f"{' ? TRIPLE LAYER WORKS!' if c > 0.03 else ''}")

# ???????????????????????????????????????????????????????????????????????
# 4. Regime-conditional analysis
# ???????????????????????????????????????????????????????????????????????

print(f"\n{'?'*70}")
print(f"  Regime-Conditional Lead-Lag")
print(f"{'?'*70}")

for s in range(3):
    mask = Regimes[:min_len] == s
    if mask.sum() < 50:
        continue
    T_s = T[mask]
    V_s = V[mask]
    Score_s = Score[mask]
    Impact_s = Impact[mask]

    n_s = mask.sum()
    corr_T = np.corrcoef(T_s, Impact_s)[0, 1]
    corr_S = np.corrcoef(Score_s, Impact_s)[0, 1]

    # Best lead correlation within this regime
    best_regime_corr = -999
    best_regime_lag = 0
    for lag in range(1, min(11, n_s // 2)):
        c = np.corrcoef(Score_s[:-lag], Impact_s[lag:])[0, 1]
        if c > best_regime_corr:
            best_regime_corr = c
            best_regime_lag = lag

    print(f"  Regime {s} ({n_s} batches): "
          f"corr(T,imp)={corr_T:+.4f}  "
          f"corr(Score,imp)={corr_S:+.4f}  "
          f"best_lead[{best_regime_lag}]={best_regime_corr:+.4f}"
          f"{' ? REGIME-SPECIFIC SIGNAL!' if best_regime_corr > 0.05 else ''}")

# ???????????????????????????????????????????????????????????????????????
# 5. Summary
# ???????????????????????????????????????????????????????????????????????

print(f"\n{'?'*70}")
print(f"  Summary")
print(f"{'?'*70}")

# Is there ANY positive correlation at any lead?
all_corrs = []
for lag in range(MAX_LAG + 1):
    if lag == 0:
        all_corrs.append(np.corrcoef(Score, Impact)[0, 1])
    else:
        all_corrs.append(np.corrcoef(Score[:-lag], Impact[lag:])[0, 1])

max_corr = max(all_corrs)
max_lag = np.argmax(all_corrs)

if max_corr > 0.02:
    print(f"  [OK] Best correlation: {max_corr:+.4f} at lead={max_lag} batches")
    print(f"  -> Gram dynamics DO have predictive power with correct temporal alignment")
else:
    print(f"  [X] Best correlation: {max_corr:+.4f} (all lags negative or near-zero)")
    print(f"  -> Gram deviation is NOT a leading indicator of impact")
    print(f"  -> Need alternative signals: depth collapse, OBI impulse, spread shock")

"""
Causal Validation — PressureMemory Alpha Proof

Four experiments to determine if pressure is causal or spurious:

  EXP 1: Lag Causality — cross-correlation Pressure_t -> PnL_(t+k), k=-50..+50
         Peak at k>0 = predictive. Peak at k<=0 = reactive/proxy.

  EXP 2: Shock Isolation — top 1% pressure events, decompose PnL before/during/after
         PnL worst AFTER shock = reactive. PnL worst BEFORE/DURING = predictive.

  EXP 3: Cross-Regime Robustness — by volatility, liquidity, spread quantiles
         Where does pressure-PnL correlation break?

  EXP 4: Null Model — shuffled, sign-flipped, time-reversed pressure
         PnL disappears under null = true alpha. PnL survives = spurious.

  DECOMP: PnL = spread_capture - adverse_selection - inventory_cost
          Which component does pressure actually affect?
"""

import gc, sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.dictionary.matrix_builder import MatrixBuilder
from modules.dictionary.pressure_memory import PressureMemory, PressureMemoryConfig
from sklearn.decomposition import sparse_encode


BATCH_SIZE, FWD_TICKS = 2048, 50
SOURCE = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
CACHE = Path(__file__).resolve().parent / "modules" / "dictionary" / "cache"
DICT_PATH = CACHE / "dict_atoms_3.npy"

# Frozen alpha
DECAY_SAME, DECAY_FLIP = 0.999, 0.85
K_SKEW = 1.0

print("=" * 65)
print("  Causal Validation — PressureMemory Alpha")
print("=" * 65)


# ===========================================================================
# [0] Load data + compute pressure trace + per-batch PnL
# ===========================================================================

print("\n[0] Loading + computing pressure trace ...")
t0 = time.perf_counter()
builder = MatrixBuilder()
X, _ = builder.assemble()
N, M = X.shape

raw = pl.read_parquet(SOURCE, columns=[
    "mid_px", "spread", "total_depth", "signed_imbalance", "duration_ms"])
offset = raw.shape[0] - N
mid_px = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
spread_arr = raw["spread"].to_numpy().astype(np.float64)[offset:]
depth = raw["total_depth"].to_numpy().astype(np.float64)[offset:]
signed_imb = raw["signed_imbalance"].to_numpy().astype(np.float64)[offset:]
duration = raw["duration_ms"].to_numpy().astype(np.float64)[offset:]
del raw

D0 = np.load(str(DICT_PATH))
n_batches = N // BATCH_SIZE

alpha_full = sparse_encode(
    X.astype(np.float64), D0.astype(np.float64),
    alpha=1.0, algorithm='lasso_lars', n_jobs=-1, max_iter=1000,
).astype(np.float32)
del X; gc.collect()

mid_ret = np.zeros(N, dtype=np.float64)
mid_ret[:-FWD_TICKS] = np.abs(
    (mid_px[FWD_TICKS:] - mid_px[:-FWD_TICKS]) / (np.abs(mid_px[:-FWD_TICKS]) + 1e-12))

# ── Compute pressure trace + per-batch PnL ──
pm = PressureMemory(PressureMemoryConfig(decay_same=DECAY_SAME, decay_flip=DECAY_FLIP))
# Warm up
for b in range(100):
    pm.update(float(np.mean(signed_imb[b*BATCH_SIZE:(b+1)*BATCH_SIZE])))

pressure_z = np.zeros(n_batches, dtype=np.float32)
pressure_dir = np.zeros(n_batches, dtype=np.int32)
pressure_raw = np.zeros(n_batches, dtype=np.float32)
batch_pnl_spread = np.zeros(n_batches, dtype=np.float64)
batch_pnl_adverse = np.zeros(n_batches, dtype=np.float64)

rng = np.random.RandomState(42)
for b in range(n_batches):
    s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
    imb = float(np.mean(signed_imb[s:e]))
    P, p_sig = pm.update(imb)
    pressure_z[b] = p_sig["z_score"]
    pressure_dir[b] = p_sig["direction"]
    pressure_raw[b] = P

    # Simple execution PnL (no skew, just measure baseline components)
    sh = spread_arr[s:e] * 0.5
    adv = np.abs(mid_ret[s:e]) * mid_px[s:e]

    # Bernoulli fills at baseline prob (~0.85 at 1x spread)
    f_bid = rng.binomial(1, 0.85, BATCH_SIZE)
    f_ask = rng.binomial(1, 0.85, BATCH_SIZE)

    spread_pnl = 0.0
    adverse_pnl = 0.0
    for t in range(BATCH_SIZE):
        if f_bid[t]:
            spread_pnl += sh[t]
            adverse_pnl -= adv[t]
        if f_ask[t]:
            spread_pnl += sh[t]
            adverse_pnl -= adv[t]

    batch_pnl_spread[b] = spread_pnl
    batch_pnl_adverse[b] = adverse_pnl

batch_pnl_net = batch_pnl_spread + batch_pnl_adverse
batch_vol = np.array([
    float(np.std(mid_px[b*BATCH_SIZE:(b+1)*BATCH_SIZE])
          / max(np.mean(mid_px[b*BATCH_SIZE:(b+1)*BATCH_SIZE]), 1e-12))
    for b in range(n_batches)
])
batch_spread_mean = np.array([
    float(np.mean(spread_arr[b*BATCH_SIZE:(b+1)*BATCH_SIZE])) for b in range(n_batches)
])

print(f"  {n_batches} batches, pressure range z=[{pressure_z.min():.1f}, {pressure_z.max():.1f}]")
print(f"  time={time.perf_counter()-t0:.1f}s")


# ═════════════════════════════════════════════════════════════════════
# EXP 1: Lag Causality Test
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 1: Lag Causality — Pressure_t -> PnL_(t+k)")
print(f"{'═'*65}")

MAX_LAG = 50
lags = np.arange(-MAX_LAG, MAX_LAG + 1)
corrs_pz = np.zeros(len(lags))
corrs_pr = np.zeros(len(lags))  # raw pressure (not z-scored)

pz = pressure_z[100:]  # skip warmup
pnl = batch_pnl_net[100:]
pr = pressure_raw[100:]
n_eff = min(len(pz), len(pnl))
pz, pnl, pr = pz[:n_eff], pnl[:n_eff], pr[:n_eff]

for i, k in enumerate(lags):
    if k > 0:
        corrs_pz[i] = np.corrcoef(pz[:-k], pnl[k:])[0, 1]
        corrs_pr[i] = np.corrcoef(pr[:-k], pnl[k:])[0, 1]
    elif k < 0:
        k_abs = -k
        corrs_pz[i] = np.corrcoef(pz[k_abs:], pnl[:-k_abs])[0, 1]
        corrs_pr[i] = np.corrcoef(pr[k_abs:], pnl[:-k_abs])[0, 1]
    else:
        corrs_pz[i] = np.corrcoef(pz, pnl)[0, 1]
        corrs_pr[i] = np.corrcoef(pr, pnl)[0, 1]

# Find peaks
peak_lead_idx = np.argmin(corrs_pz[MAX_LAG:])  # most negative at k>=0
peak_lead = lags[MAX_LAG:][peak_lead_idx]
peak_lead_corr = corrs_pz[MAX_LAG:][peak_lead_idx]

peak_lag_idx = np.argmin(corrs_pz[:MAX_LAG])  # most negative at k<0
peak_lag = lags[:MAX_LAG][peak_lag_idx]
peak_lag_corr = corrs_pz[:MAX_LAG][peak_lag_idx]

print(f"\n  Correlation Pressure_z → PnL:")
print(f"    Max LEAD:  k={peak_lead:+.0f}  corr={peak_lead_corr:+.4f}  "
      f"({'PREDICTIVE — pressure leads PnL' if peak_lead > 0 and peak_lead_corr < -0.02 else 'weak lead'})")
print(f"    Max LAG:   k={peak_lag:+.0f}  corr={peak_lag_corr:+.4f}  "
      f"({'REACTIVE — PnL leads pressure' if peak_lag < 0 and peak_lag_corr < -0.02 else 'weak lag'})")

# Causality direction verdict
lead_strength = abs(peak_lead_corr) if peak_lead > 0 else 0
lag_strength = abs(peak_lag_corr) if peak_lag < 0 else 0

if lead_strength > lag_strength * 1.5:
    verdict = "CAUSAL — pressure PRECEDES PnL changes"
elif lag_strength > lead_strength * 1.5:
    verdict = "REACTIVE — pressure FOLLOWS PnL changes"
else:
    verdict = "BIDIRECTIONAL — pressure both leads and lags PnL"

print(f"\n  Causality Verdict: {verdict}")

# Show key lags
print(f"\n  Key lags (Pressure_z → PnL):")
for k in [-20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20]:
    idx = np.where(lags == k)[0][0]
    marker = " < PEAK" if k == peak_lead else (" < LAG_PEAK" if k == peak_lag else "")
    print(f"    k={k:>+3d}: corr={corrs_pz[idx]:>+8.4f}{marker}")


# ═════════════════════════════════════════════════════════════════════
# EXP 2: Shock Isolation Test
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 2: Shock Isolation — PnL around extreme pressure events")
print(f"{'═'*65}")

# Find top 1% pressure_z events (must be separated by ≥20 batches)
threshold = np.percentile(pz, 99)
event_idx = np.where(pz > threshold)[0]

# Filter: keep events separated by at least 20 batches
filtered_events = []
last_event = -100
for idx in event_idx:
    if idx - last_event >= 20:
        filtered_events.append(idx)
        last_event = idx
event_idx = np.array(filtered_events[:50])  # top 50 isolated events

WINDOW = 30
before_pnl = []
during_pnl = []
after_pnl = []

for ev in event_idx:
    if ev >= WINDOW and ev + WINDOW < n_eff:
        before_pnl.append(np.mean(pnl[ev-WINDOW:ev-5]))    # before, skip transition
        during_pnl.append(np.mean(pnl[ev-3:ev+3]))          # during event
        after_pnl.append(np.mean(pnl[ev+5:ev+WINDOW]))      # after

before_mean = np.mean(before_pnl) if before_pnl else 0
during_mean = np.mean(during_pnl) if during_pnl else 0
after_mean = np.mean(after_pnl) if after_pnl else 0

print(f"\n  Extreme pressure events: {len(event_idx)} (top 1%, z > {threshold:.1f})")
print(f"    PnL BEFORE  event (t-30 to t-5):  {before_mean:>+12,.1f}")
print(f"    PnL DURING  event (t-3 to t+3):   {during_mean:>+12,.1f}")
print(f"    PnL AFTER   event (t+5 to t+30):  {after_mean:>+12,.1f}")

# Verdict: when is PnL worst?
periods = {"BEFORE": before_mean, "DURING": during_mean, "AFTER": after_mean}
worst_period = min(periods, key=periods.get)

if worst_period == "BEFORE":
    shock_verdict = "PREDICTIVE — PnL drops BEFORE extreme pressure (alpha works)"
elif worst_period == "DURING":
    shock_verdict = "COINCIDENT — PnL and pressure move together (real-time signal)"
else:
    shock_verdict = "REACTIVE — PnL drops AFTER pressure spike (lagging indicator)"

print(f"\n  Shock Verdict: {shock_verdict}")


# ═════════════════════════════════════════════════════════════════════
# EXP 3: Cross-Regime Robustness
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 3: Cross-Regime Robustness — by vol/liquidity/spread")
print(f"{'═'*65}")

# Split by quantiles
for regime_name, regime_feature in [
    ("Volatility", batch_vol),
    ("Spread", batch_spread_mean),
]:
    lo = np.percentile(regime_feature[100:100+n_eff], 33)
    hi = np.percentile(regime_feature[100:100+n_eff], 67)
    f = regime_feature[100:100+n_eff]

    regimes = {
        "LOW": f < lo,
        "MID": (f >= lo) & (f < hi),
        "HIGH": f >= hi,
    }

    print(f"\n  {regime_name} regimes:")
    print(f"    {'Regime':<8s} {'Batches':>8s} {'Corr(Pz,PnL)':>14s} {'Status':>12s}")
    print(f"    {'─'*8} {'─'*8} {'─'*14} {'─'*12}")

    all_stable = True
    for rname, mask in regimes.items():
        if mask.sum() < 20:
            continue
        c = np.corrcoef(pz[mask], pnl[mask])[0, 1]
        status = "STABLE" if c < -0.02 else ("weak" if c < 0 else "BROKEN")
        if c >= 0:
            all_stable = False
        print(f"    {rname:<8s} {mask.sum():>8d} {c:>+14.4f} {status:>12s}")

    print(f"    → {'ALL STABLE' if all_stable else 'DEGRADES in some regimes'}")


# ═════════════════════════════════════════════════════════════════════
# EXP 4: Null Model Test
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  EXP 4: Null Model — shuffled / sign-flipped / time-reversed")
print(f"{'═'*65}")

rng_null = np.random.RandomState(42)

# 1. Shuffled pressure
pz_shuffled = pz.copy()
rng_null.shuffle(pz_shuffled)

# 2. Sign-flipped pressure
pz_flipped = -pz

# 3. Time-reversed pressure
pz_reversed = pz[::-1]

# Run simple PnL simulation with each null pressure
def null_pnl(pz_null, pd_null):
    """Simple skewed MM PnL with null pressure signal."""
    pnl_hist = []
    for b in range(100, n_eff):
        s, e = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        pz_b = pz_null[b-100]
        pd_b = pd_null[b-100]
        pos = -K_SKEW * pd_b * min(pz_b, 3.0)
        bid_sz = max(1.0 - pos, 0.1)
        ask_sz = max(1.0 + pos, 0.1)

        sh = spread_arr[s:e] * 0.5
        adv = np.abs(mid_ret[s:e]) * mid_px[s:e]

        rng = np.random.RandomState(42 + b)
        f_bid = rng.binomial(1, 0.85, BATCH_SIZE)
        f_ask = rng.binomial(1, 0.85, BATCH_SIZE)

        pnl_b = 0.0
        for t in range(BATCH_SIZE):
            if f_bid[t]: pnl_b += (sh[t] - adv[t]) * bid_sz
            if f_ask[t]: pnl_b += (sh[t] - adv[t]) * ask_sz
        pnl_hist.append(pnl_b)
    return float(np.sum(pnl_hist))

# Real pressure PnL (baseline for comparison)
pnl_real = null_pnl(pz, pressure_dir[100:100+n_eff])
pnl_shuffled = null_pnl(pz_shuffled, pressure_dir[100:100+n_eff])
pnl_flipped = null_pnl(pz_flipped, pressure_dir[100:100+n_eff])
pnl_reversed = null_pnl(pz_reversed, pressure_dir[100:100+n_eff])

print(f"\n  PnL under different pressure signals:")
print(f"    Real pressure:     {pnl_real:>+14,.0f}")
print(f"    Shuffled pressure: {pnl_shuffled:>+14,.0f}  "
      f"({'alpha survives' if pnl_real > pnl_shuffled else 'SPURIOUS — shuffle wins'})")
print(f"    Sign-flipped:      {pnl_flipped:>+14,.0f}  "
      f"({'alpha survives' if pnl_real > pnl_flipped else 'SPURIOUS — flip wins'})")
print(f"    Time-reversed:     {pnl_reversed:>+14,.0f}  "
      f"({'alpha survives' if pnl_real > pnl_reversed else 'SPURIOUS — reverse wins'})")

# Null test verdict
null_wins = (pnl_shuffled > pnl_real * 0.9 or pnl_flipped > pnl_real * 0.9
             or pnl_reversed > pnl_real * 0.9)
null_verdict = ("SPURIOUS — null models match or beat real pressure"
                if null_wins else
                "CAUSAL — real pressure significantly outperforms all null models")
print(f"\n  Null Model Verdict: {null_verdict}")


# ═════════════════════════════════════════════════════════════════════
# DECOMP: PnL Attribution
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  PnL Decomposition — spread_capture - adverse_selection - inventory")
print(f"{'═'*65}")

total_spread = float(np.sum(batch_pnl_spread[100:]))
total_adverse = float(np.sum(batch_pnl_adverse[100:]))
total_net = float(np.sum(batch_pnl_net[100:]))

print(f"\n  Total PnL decomposition ({n_eff} batches):")
print(f"    Spread capture:     {total_spread:>+14,.0f}  ({total_spread/abs(total_net)*100:+.0f}%)")
print(f"    Adverse selection:  {total_adverse:>+14,.0f}  ({total_adverse/abs(total_net)*100:+.0f}%)")
print(f"    Net PnL:            {total_net:>+14,.0f}")

# Which component does pressure affect?
corr_p_spread = np.corrcoef(pz, batch_pnl_spread[100:100+n_eff])[0, 1]
corr_p_adverse = np.corrcoef(pz, batch_pnl_adverse[100:100+n_eff])[0, 1]

print(f"\n  Pressure_z correlation with components:")
print(f"    Corr(Pz, spread_capture):  {corr_p_spread:+.4f}")
print(f"    Corr(Pz, adverse):         {corr_p_adverse:+.4f}")

if abs(corr_p_adverse) > abs(corr_p_spread) * 1.5:
    print(f"    → Pressure primarily affects ADVERSE SELECTION (real risk signal)")
elif abs(corr_p_spread) > abs(corr_p_adverse) * 1.5:
    print(f"    → Pressure primarily affects SPREAD CAPTURE (liquidity signal)")
else:
    print(f"    → Pressure affects both (mixed signal)")

# High vs low pressure: PnL breakdown
pz_high = pz > np.percentile(pz, 80)
pz_low = pz < np.percentile(pz, 20)

for label, mask in [("High Pz (top 20%)", pz_high), ("Low Pz (bottom 20%)", pz_low)]:
    sp = float(np.mean(batch_pnl_spread[100:100+n_eff][mask]))
    adv = float(np.mean(batch_pnl_adverse[100:100+n_eff][mask]))
    net = float(np.mean(batch_pnl_net[100:100+n_eff][mask]))
    print(f"\n    {label}:")
    print(f"      Spread: {sp:>+12,.1f}  Adverse: {adv:>+12,.1f}  Net: {net:>+12,.1f}")


# ═════════════════════════════════════════════════════════════════════
# Final Causal Verdict
# ═════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  Final Causal Verdict")
print(f"{'═'*65}")

scores = {
    "Lag causality (lead > lag)": lead_strength > lag_strength * 1.2,
    "Shock isolation (PnL worst before)": worst_period == "BEFORE",
    "Cross-regime (all stable)": all_stable,
    "Null model (real > null)": not null_wins,
}

passed = sum(scores.values())
for test, result in scores.items():
    print(f"  [{'PASS' if result else 'fail'}] {test}")

print(f"\n  {passed}/{len(scores)} causal tests passed")

if passed >= 3:
    print(f"  PressureMemory is CAUSALLY VALID — alpha has market causal weight.")
elif passed >= 2:
    print(f"  PressureMemory is PARTIALLY CAUSAL — needs robustness hardening.")
else:
    print(f"  PressureMemory is CORRELATIONAL — not yet proven causal.")

print(f"{'═'*65}")

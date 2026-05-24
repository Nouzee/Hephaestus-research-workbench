"""
Causal Alignment — per-signal optimal lead-lag discovery.

For each precursor signal, sweeps lag offsets to find the shift that
maximizes correlation with future impact. Only positive leads (signal
before impact) are eligible for prediction; negative leads (signal after
impact) indicate the signal is a reaction, not a precursor.

Output: a lag config JSON that the toxicity scorer uses to align signals
before fusion. This is an offline-only step — the config is frozen before
any online/deployment use.

Strict constraint: only use lag >= 0 for prediction (signal[t-lag] -> impact[t]).
Future data (negative lag) is recorded for diagnostic purposes only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


# ===========================================================================
# Config
# ===========================================================================

SOURCE_PARQUET = (
    r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
)

# Batch granularity (must match pipeline)
BATCH_SIZE = 2048

# Lag sweep range (batches)
MAX_LEAD = 20   # signal leads impact by up to 20 batches
MAX_LAG = 5     # signal lags impact by up to 5 batches (diagnostic only)

# Output
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CONFIG_PATH = CACHE_DIR / "causal_alignment.json"


# ===========================================================================
# Signal extractors (per-batch, raw data -> scalar)
# ===========================================================================

def extract_depth_evap(depth: np.ndarray) -> float:
    """Depth evaporation: -(depth_end - depth_start) / depth_start. Positive = vanishing."""
    if len(depth) < 10:
        return 0.0
    d0 = np.median(depth[:10])
    d1 = np.median(depth[-10:])
    return float(max(-(d1 - d0) / max(d0, 1e-12), 0.0))


def extract_obi_impulse(signed_imb: np.ndarray) -> float:
    """OBI impulse: 95th percentile / mean of |signed_imbalance|, log-scaled."""
    abs_imb = np.abs(signed_imb)
    mu = np.mean(abs_imb)
    p95 = np.percentile(abs_imb, 95)
    ratio = p95 / max(mu, 1e-12)
    return float(np.log1p(ratio))


def extract_spread_shock(spread: np.ndarray) -> float:
    """Spread shock: max / mean spread, log-scaled."""
    mu = np.mean(spread)
    mx = np.max(spread)
    return float(np.log1p(mx / max(mu, 1e-12)))


def extract_cancel_burst(duration_ms: np.ndarray) -> float:
    """Cancel burst proxy: median / 5th percentile duration. Positive = burst."""
    if len(duration_ms) < 10:
        return 0.0
    med = np.median(duration_ms)
    p5 = np.percentile(duration_ms, 5)
    return float(np.log1p(med / max(p5, 1e-12)))


def extract_gram_aux(gram_trace: np.ndarray, idx: int) -> float:
    """Gram auxiliary: trace velocity at batch index."""
    if idx < 1:
        return 0.0
    return float(max(gram_trace[idx] - gram_trace[idx - 1], 0.0))


# ===========================================================================
# Impact label
# ===========================================================================

def compute_impact(mid_px: np.ndarray, fwd_ticks: int = 50) -> np.ndarray:
    """Future |return| per tick."""
    ret = np.zeros(len(mid_px), dtype=np.float64)
    ret[:-fwd_ticks] = np.abs(
        (mid_px[fwd_ticks:] - mid_px[:-fwd_ticks])
        / (np.abs(mid_px[:-fwd_ticks]) + 1e-12)
    )
    return ret


# ===========================================================================
# Lag sweep
# ===========================================================================

def sweep_lag(
    signal: np.ndarray,
    impact: np.ndarray,
    max_lead: int = MAX_LEAD,
    max_lag: int = MAX_LAG,
) -> dict:
    """
    Sweep lag offsets for one signal.

    Positive offset k = signal[t-k] -> impact[t] (signal LEADS, predictive).
    Negative offset k = signal[t+|k|] -> impact[t] (signal LAGS, reactive).

    Returns dict with optimal lead, correlations, and diagnostic info.
    """
    n = min(len(signal), len(impact))
    s = signal[:n]
    imp = impact[:n]

    correlations = {}
    best_lead = 0
    best_corr = -999.0

    # Sweep from -max_lag (reactive) to +max_lead (predictive)
    for k in range(-max_lag, max_lead + 1):
        if k >= 0:
            # signal[t-k] -> impact[t]: signal leads
            corr = np.corrcoef(s[:-k], imp[k:])[0, 1] if k > 0 else np.corrcoef(s, imp)[0, 1]
        else:
            # signal[t+|k|] -> impact[t]: signal lags (impact leads)
            k_abs = -k
            corr = np.corrcoef(s[k_abs:], imp[:-k_abs])[0, 1]

        correlations[str(k)] = float(corr)

        # Only positive leads are eligible for prediction
        if k >= 0 and corr > best_corr:
            best_corr = corr
            best_lead = k

    return {
        "optimal_lead": best_lead,       # batches (>= 0)
        "optimal_corr": float(best_corr),
        "all_correlations": correlations,
        "is_predictive": best_corr > 0.02,  # minimum meaningful correlation
    }


# ===========================================================================
# Main
# ===========================================================================

def run_alignment() -> dict:
    """
    Full causal alignment pipeline:
      1. Load raw data
      2. Extract per-batch signals
      3. Compute impact labels
      4. Sweep lag per signal
      5. Save config
    """
    print("=" * 60)
    print("  Causal Alignment — Lag Sweep per Signal")
    print("=" * 60)

    # Load
    print("\n[1] Loading raw data ...")
    t0 = time.perf_counter()
    raw = pl.read_parquet(SOURCE_PARQUET)
    N_raw = raw.shape[0]

    # Trim to align with MatrixBuilder (drops ~50 leading NaN rows)
    offset = 50
    mid_px = raw["mid_px"].to_numpy().astype(np.float64)[offset:]
    depth = raw["total_depth"].to_numpy().astype(np.float64)[offset:]
    signed_imb = raw["signed_imbalance"].to_numpy().astype(np.float64)[offset:]
    spread = raw["spread"].to_numpy().astype(np.float64)[offset:]
    duration = raw["duration_ms"].to_numpy().astype(np.float64)[offset:]
    del raw

    N = len(mid_px)
    n_batches = N // BATCH_SIZE
    print(f"  Ticks: {N:,}  Batches: {n_batches}  time={time.perf_counter()-t0:.1f}s")

    # Extract per-batch signals
    print("\n[2] Extracting per-batch signals ...")
    t0 = time.perf_counter()

    signals = {
        "depth_evap": np.zeros(n_batches, dtype=np.float32),
        "obi_impulse": np.zeros(n_batches, dtype=np.float32),
        "spread_shock": np.zeros(n_batches, dtype=np.float32),
        "cancel_burst": np.zeros(n_batches, dtype=np.float32),
        "gram_aux": np.zeros(n_batches, dtype=np.float32),
    }

    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = s + BATCH_SIZE
        signals["depth_evap"][b] = extract_depth_evap(depth[s:e])
        signals["obi_impulse"][b] = extract_obi_impulse(signed_imb[s:e])
        signals["spread_shock"][b] = extract_spread_shock(spread[s:e])
        signals["cancel_burst"][b] = extract_cancel_burst(duration[s:e])

    # Compute Gram trace proxy (from online dict would be ideal; use depth volatility as placeholder)
    # In production, this comes from GramTracker. For alignment, use a simple proxy.
    # Will be replaced with actual Gram trace in pipeline.
    gram_trace = np.zeros(n_batches, dtype=np.float32)
    # Use spread volatility as Gram activity proxy for alignment purposes
    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = s + BATCH_SIZE
        gram_trace[b] = float(np.std(spread[s:e]) / max(np.mean(spread[s:e]), 1e-12))
    for b in range(1, n_batches):
        signals["gram_aux"][b] = float(max(gram_trace[b] - gram_trace[b-1], 0.0))

    print(f"  time={time.perf_counter()-t0:.1f}s")

    # Compute impact
    print("\n[3] Computing impact labels ...")
    t0 = time.perf_counter()
    impact_per_tick = compute_impact(mid_px, fwd_ticks=50)
    impact_per_batch = np.array([
        np.mean(impact_per_tick[b*BATCH_SIZE:(b+1)*BATCH_SIZE])
        for b in range(n_batches)
    ])
    print(f"  Impact mean={impact_per_batch.mean():.6e}  std={impact_per_batch.std():.6e}  "
          f"time={time.perf_counter()-t0:.1f}s")

    # Sweep per signal
    print("\n[4] Lag sweep per signal:")
    print(f"  {'Signal':<20s} {'Best Lead':>10s} {'Corr':>10s} {'Predictive?':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")

    results = {}
    for name, sig in signals.items():
        r = sweep_lag(sig, impact_per_batch)
        results[name] = r
        print(f"  {name:<20s} {r['optimal_lead']:>10d} {r['optimal_corr']:>+10.4f} "
              f"{'YES' if r['is_predictive'] else 'no':>12s}")

    # Build config: only include predictive signals with their optimal leads
    config = {
        "alignment": {},
        "weights": {
            "depth_evap": 1.0,
            "obi_impulse": 1.0,
            "spread_shock": 1.0,
            "cancel_burst": 0.5,
            "gram_aux": 0.3,
        },
        "notes": {
            "depth_evap": "leading indicator — liquidity vanishes before impact",
            "obi_impulse": "coincident — order flow spike at impact",
            "spread_shock": "reactive — spread widens after impact; diagnostic only",
            "cancel_burst": "duration collapse as activity surge proxy",
            "gram_aux": "Gram velocity as topological auxiliary",
        },
        "batch_size": BATCH_SIZE,
        "max_lead_batches": MAX_LEAD,
    }

    for name, r in results.items():
        config["alignment"][name] = {
            "optimal_lead": int(r["optimal_lead"]),
            "correlation": float(r["optimal_corr"]),
            "is_predictive": bool(r["is_predictive"]),
        }

    # Enforce causality: spread_shock must NOT be used for prediction if it lags
    for name in config["alignment"]:
        lead = config["alignment"][name]["optimal_lead"]
        is_pred = config["alignment"][name]["is_predictive"]
        if lead < 0 or not is_pred:
            config["alignment"][name]["role"] = "diagnostic"
        else:
            config["alignment"][name]["role"] = "predictive"

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[5] Config saved -> {CONFIG_PATH}")
    print(f"\n  Signal roles:")
    for name, cfg in config["alignment"].items():
        print(f"    {name:20s}  lead={cfg['optimal_lead']:>3d}  "
              f"corr={cfg['correlation']:+.4f}  role={cfg['role']}")

    return config


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    run_alignment()

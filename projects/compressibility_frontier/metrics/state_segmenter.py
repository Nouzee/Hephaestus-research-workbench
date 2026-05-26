"""
State Segmenter — split market data into regimes for compression analysis.

Segments:
  LOW_VOL, MID_VOL, HIGH_VOL  — by realized volatility terciles
  RECOVERY                     — after volatility spikes, returning to baseline
  FRAGILE                      — high spread + low depth simultaneously

Returns per-regime indices for batch-level feature matrices.
"""

from __future__ import annotations

import numpy as np


def segment_by_volatility(realized_vol: np.ndarray) -> dict[str, np.ndarray]:
    """
    Split batches into LOW / MID / HIGH volatility by terciles.

    Returns dict of {regime_name: boolean mask}.
    """
    lo = np.percentile(realized_vol, 33)
    hi = np.percentile(realized_vol, 67)

    return {
        "LOW_VOL": realized_vol < lo,
        "MID_VOL": (realized_vol >= lo) & (realized_vol < hi),
        "HIGH_VOL": realized_vol >= hi,
    }


def segment_by_fragility(
    spread_bps: np.ndarray,
    total_depth: np.ndarray,
    spread_pct: float = 80,
    depth_pct: float = 20,
) -> dict[str, np.ndarray]:
    """
    Identify fragile states: high spread AND low depth simultaneously.

    FRAGILE = spread > spread_pct percentile AND depth < depth_pct percentile.
    HEALTHY = spread < 50th AND depth > 50th.
    TRANSITION = everything else.
    """
    spread_hi = np.percentile(spread_bps, spread_pct)
    depth_lo = np.percentile(total_depth, depth_pct)
    spread_lo = np.percentile(spread_bps, 50)
    depth_hi = np.percentile(total_depth, 50)

    fragile = (spread_bps > spread_hi) & (total_depth < depth_lo)
    healthy = (spread_bps < spread_lo) & (total_depth > depth_hi)
    transition = ~fragile & ~healthy

    return {
        "FRAGILE": fragile,
        "HEALTHY": healthy,
        "TRANSITION": transition,
    }


def segment_by_recovery(
    realized_vol: np.ndarray,
    vol_window: int = 20,
    recovery_threshold: float = 0.7,
) -> dict[str, np.ndarray]:
    """
    Identify recovery periods: after a vol spike, now declining.

    A batch is in RECOVERY if:
      - Recent vol (last 20 batches) had a spike above 90th percentile
      - Current vol is below 70th percentile (declining from the spike)
    """
    n = len(realized_vol)
    vol_hi = np.percentile(realized_vol, 90)

    recovery = np.zeros(n, dtype=bool)
    spike_detected = np.zeros(n, dtype=bool)

    for t in range(vol_window, n):
        recent = realized_vol[t - vol_window:t]
        if np.max(recent) > vol_hi:
            spike_detected[t] = True

    for t in range(vol_window, n):
        if spike_detected[t] and realized_vol[t] < np.percentile(realized_vol, 70):
            recovery[t] = True

    normal = ~recovery

    return {"RECOVERY": recovery, "NON_RECOVERY": normal}


def segment_all(
    realized_vol: np.ndarray,
    spread_bps: np.ndarray,
    total_depth: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Run all segmenters and return combined regime masks.

    Returns dict with keys:
      LOW_VOL, MID_VOL, HIGH_VOL, FRAGILE, HEALTHY, RECOVERY
    each mapping to a boolean mask array.
    """
    regimes = {}
    regimes.update(segment_by_volatility(realized_vol))
    regimes.update(segment_by_fragility(spread_bps, total_depth))
    regimes.update(segment_by_recovery(realized_vol))
    return regimes


def regime_summary(regimes: dict[str, np.ndarray]) -> dict:
    """Print and return regime distribution statistics."""
    total = len(next(iter(regimes.values())))
    summary = {}
    for name, mask in regimes.items():
        n = int(np.sum(mask))
        pct = n / total * 100
        summary[name] = {"batches": n, "pct": float(pct)}
    return summary

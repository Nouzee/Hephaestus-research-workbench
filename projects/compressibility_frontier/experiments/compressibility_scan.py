"""
Compressibility Scan — run compression metrics on each market regime.

Answers: which market states are most/least compressible?
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from projects.compressibility_frontier.metrics.compressibility_metrics import (
    reconstruction_residual, effective_rank, atom_usage_entropy,
    temporal_redundancy, compressibility_summary,
)
from projects.compressibility_frontier.metrics.state_segmenter import segment_all


def run_scan(
    feature_matrix: np.ndarray,       # (N, M) standardized features
    dictionary: np.ndarray,           # (K, M) dictionary atoms
    alpha_matrix: np.ndarray,         # (N, K) sparse coefficients
    realized_vol: np.ndarray,         # (N,) realized volatility per batch
    spread_bps: np.ndarray,           # (N,) spread in bps
    total_depth: np.ndarray,          # (N,) total order book depth
) -> dict:
    """
    Run full compressibility scan across all regimes.

    Returns nested dict: {regime_name: {metric_name: value}}
    """
    # Segment
    regimes = segment_all(realized_vol, spread_bps, total_depth)

    results = {}
    for regime_name, mask in regimes.items():
        n_samples = int(np.sum(mask))
        if n_samples < 20:
            results[regime_name] = {
                "n_samples": n_samples,
                "error": "insufficient samples (need >= 20)",
            }
            continue

        X_seg = feature_matrix[mask]
        A_seg = alpha_matrix[mask]

        metrics = compressibility_summary(X_seg, dictionary, A_seg)
        metrics["n_samples"] = n_samples
        results[regime_name] = metrics

    return results


def print_scan_table(results: dict):
    """Print formatted scan results table."""
    print(f"\n{'═'*80}")
    print(f"  Compressibility Frontier — Regime Scan Results")
    print(f"{'═'*80}")

    header = (f"  {'Regime':<14s} {'N':>6s} {'Residual':>8s} {'EffRank':>8s} "
              f"{'AtomEnt':>8s} {'TempRed':>8s} {'Composite':>9s} {'Verdict':>14s}")
    print(header)
    print(f"  {'─'*14} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*14}")

    for regime_name, m in sorted(results.items()):
        n = m.get("n_samples", 0)
        if "error" in m:
            print(f"  {regime_name:<14s} {n:>6d}  {'(skipped)':>50s}")
            continue

        resid = m["reconstruction_residual"]
        erank = m["effective_rank"]
        entropy = m["atom_entropy"]
        redun = m["temporal_redundancy"]
        comp = m["composite_compressibility"]

        # Verdict
        if comp > 0.6:
            verdict = "HIGHLY COMPRESSIBLE"
        elif comp > 0.4:
            verdict = "moderate"
        elif comp > 0.25:
            verdict = "low structure"
        else:
            verdict = "NEAR NOISE"

        print(f"  {regime_name:<14s} {n:>6d} {resid:>8.4f} {erank:>8.2f} "
              f"{entropy:>8.3f} {redun:>8.3f} {comp:>9.4f} {verdict:>14s}")

    print(f"{'═'*80}")

    return results

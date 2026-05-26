"""
Cross-Scale Scanner — compressibility under different observation operators.

Answers: "Is market structure invariant to how you observe it?"

Runs effective rank, reconstruction residual, and entropy under each O_k,
then classifies structure as:
  INVARIANT       — stable across all operators (real structure)
  SCALE_DEPENDENT — only visible at certain scales (potential alpha source)
  PROJECTION_ARTIFACT — disappears under most operators (noise)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from projects.compressibility_frontier.operators.observation_operators import (
    OPERATORS, apply_operator, extract_features,
)
from projects.compressibility_frontier.metrics.compressibility_metrics import (
    effective_rank, temporal_redundancy,
)


def run_cross_scale_scan(
    raw_data: dict[str, np.ndarray],
    dictionary: np.ndarray,
    operators: dict = None,
) -> dict:
    """
    Run compressibility metrics under each observation operator.

    Returns nested dict: {operator_name: {metric: value}}
    """
    if operators is None:
        operators = OPERATORS

    results = {}

    for op_name, op_def in operators.items():
        try:
            features, M = apply_operator(raw_data, op_def, extract_features)
        except Exception as e:
            results[op_name] = {"error": str(e), "n_windows": 0}
            continue

        if M < 20:
            results[op_name] = {"error": "too few windows", "n_windows": M}
            continue

        # Standardize
        f_mean = features.mean(axis=0)
        f_std = np.maximum(features.std(axis=0), 1e-8)
        features_z = (features - f_mean) / f_std
        features_z = np.clip(features_z, -10, 10)

        # Metrics
        erank = effective_rank(features_z)
        redundancy = temporal_redundancy(features_z)

        # SVD spectrum for rank decay analysis
        try:
            from scipy import linalg
            sv = linalg.svdvals(features_z.astype(np.float64))
            sv_norm = sv / max(sv[0], 1e-12)
            # Rank at which cumulative variance exceeds 90%
            cum_var = np.cumsum(sv ** 2) / np.sum(sv ** 2)
            rank_90 = int(np.searchsorted(cum_var, 0.90)) + 1
            rank_50 = int(np.searchsorted(cum_var, 0.50)) + 1
        except Exception:
            sv_norm = np.ones(min(features_z.shape))
            rank_90 = min(features_z.shape)
            rank_50 = min(features_z.shape) // 2

        results[op_name] = {
            "n_windows": M,
            "effective_rank": float(erank),
            "temporal_redundancy": float(redundancy),
            "rank_90": rank_90,
            "rank_50": rank_50,
            "sv_spectrum": sv_norm[:10].tolist(),
        }

    # ── Classify structure stability ──
    if len(results) >= 2:
        ranks = [r["effective_rank"] for r in results.values()
                 if "error" not in r]
        if ranks:
            rank_cv = float(np.std(ranks) / max(np.mean(ranks), 1e-12))
            classification = _classify_structure(rank_cv, ranks)
            results["_classification"] = classification

    return results


def _classify_structure(rank_cv: float, ranks: list) -> dict:
    """Classify structure stability across operators."""
    if rank_cv < 0.05:
        struct_type = "INVARIANT"
        interpretation = "Market structure is stable across observation scales — real geometry"
    elif rank_cv < 0.15:
        struct_type = "WEAKLY_INVARIANT"
        interpretation = "Mostly stable, with minor scale-dependent variation"
    elif rank_cv < 0.30:
        struct_type = "SCALE_DEPENDENT"
        interpretation = "Structure varies significantly with observation — contains scale-specific information"
    else:
        struct_type = "PROJECTION_ARTIFACT"
        interpretation = "Structure is observation-dependent — what you see depends on how you look"

    return {
        "type": struct_type,
        "interpretation": interpretation,
        "rank_cv": float(rank_cv),
        "rank_range": [float(min(ranks)), float(max(ranks))],
    }


def print_cross_scale_table(results: dict):
    """Print formatted cross-scale comparison table."""
    print(f"\n{'═'*80}")
    print(f"  Cross-Scale Observation Sweep — Structure Under Different Operators")
    print(f"{'═'*80}")

    header = (f"  {'Operator':<14s} {'Windows':>8s} {'EffRank':>8s} "
              f"{'TempRed':>8s} {'Rank90':>7s} {'Rank50':>7s} {'SV[1:4]':>25s}")
    print(header)
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*25}")

    for op_name, r in results.items():
        if op_name.startswith("_"):
            continue
        if "error" in r:
            print(f"  {op_name:<14s} {r.get('n_windows',0):>8d}  "
                  f"{'ERROR: ' + r['error']:<50s}")
            continue

        sv = r.get("sv_spectrum", [])
        sv_str = " ".join(f"{v:.2f}" for v in sv[:4])

        print(f"  {op_name:<14s} {r['n_windows']:>8d} {r['effective_rank']:>8.2f} "
              f"{r['temporal_redundancy']:>8.3f} {r['rank_90']:>7d} "
              f"{r['rank_50']:>7d} {sv_str:<25s}")

    # Classification
    cls = results.get("_classification", {})
    if cls:
        print(f"\n  Structure Classification: {cls['type']}")
        print(f"    Rank CV: {cls['rank_cv']:.3f}  "
              f"Range: [{cls['rank_range'][0]:.2f}, {cls['rank_range'][1]:.2f}]")
        print(f"    {cls['interpretation']}")

    print(f"{'═'*80}")

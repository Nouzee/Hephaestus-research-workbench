"""
Report Builder — generate compressibility frontier summary.

Outputs:
  1. Regime compressibility ranking
  2. SVD spectrum comparison
  3. Structural interpretation
"""

from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def build_report(scan_results: dict, regime_segments: dict, output_dir: str = "") -> dict:
    """
    Build a structured compressibility report.

    Returns dict with: ranking, svd_spectra, interpretation, frontier_map.
    """
    # ── Ranking ──
    ranking = []
    for regime, m in scan_results.items():
        if "error" in m:
            continue
        ranking.append({
            "regime": regime,
            "composite": m["composite_compressibility"],
            "n_samples": m["n_samples"],
            "residual": m["reconstruction_residual"],
            "effective_rank": m["effective_rank"],
            "atom_entropy": m["atom_entropy"],
        })

    ranking.sort(key=lambda r: r["composite"], reverse=True)

    # ── Interpretation ──
    if len(ranking) >= 3:
        best = ranking[0]
        worst = ranking[-1]
        spread = best["composite"] - worst["composite"]

        if spread > 0.2:
            interpretation = (
                f"Market structure varies significantly across regimes "
                f"(spread={spread:.3f}). {best['regime']} is most compressible, "
                f"{worst['regime']} is least. "
                f"This supports the hypothesis that market microstructure "
                f"has regime-dependent information density."
            )
        else:
            interpretation = (
                f"Compressibility is relatively uniform across regimes "
                f"(spread={spread:.3f}). Either the dictionary captures "
                f"a universal market structure, or the regime definitions "
                f"don't isolate different information states."
            )
    else:
        interpretation = "Insufficient regimes for comparison."

    report = {
        "rankings": ranking,
        "interpretation": interpretation,
        "n_regimes": len(ranking),
        "compression_range": {
            "max": ranking[0]["composite"] if ranking else 0,
            "min": ranking[-1]["composite"] if ranking else 0,
        },
    }

    # ── Save ──
    if output_dir:
        path = Path(output_dir) / "compressibility_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved → {path}")

    return report


def print_report(report: dict):
    """Print human-readable report."""
    print(f"\n{'═'*65}")
    print(f"  Compressibility Frontier — Final Report")
    print(f"{'═'*65}")

    print(f"\n  Regime Ranking (most → least compressible):")
    print(f"  {'Rank':<5s} {'Regime':<14s} {'Composite':>9s} {'EffRank':>8s} {'AtomEnt':>8s}")
    print(f"  {'─'*5} {'─'*14} {'─'*9} {'─'*8} {'─'*8}")
    for i, r in enumerate(report["rankings"]):
        print(f"  {i+1:<5d} {r['regime']:<14s} {r['composite']:>9.4f} "
              f"{r['effective_rank']:>8.2f} {r['atom_entropy']:>8.3f}")

    print(f"\n  Compression Range: "
          f"{report['compression_range']['min']:.3f} — "
          f"{report['compression_range']['max']:.3f}")

    print(f"\n  Interpretation:")
    print(f"    {report['interpretation']}")

    print(f"{'═'*65}")

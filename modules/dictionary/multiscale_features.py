"""
Multi-Scale Feature Extraction + Scale-Level Causality

Applies causal wavelet decomposition to each precursor signal, then
runs per-scale lead-lag analysis against future impact.

Answers: which scale of which signal is the true precursor?
  - HF: sudden shock (precedes or coincides with impact?)
  - MF: sustained pressure (builds before impact?)
  - LF: regime drift (too slow to be useful?)

Output: scale-level lag config for toxicity scorer fusion.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from modules.dictionary.causal_wavelet import CausalDecomposer, CausalWaveletConfig


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class MultiscaleConfig:
    """Configuration for multi-scale feature extraction and causality."""

    # EMA spans per scale
    hf_span: int = 2
    mf_span: int = 10
    lf_span: int = 50

    # Lead-lag sweep range (batches)
    max_lead: int = 20

    # Output path
    config_path: str = ""

    def __post_init__(self):
        if not self.config_path:
            self.config_path = str(
                Path(__file__).resolve().parent / "cache" / "multiscale_causality.json"
            )


# ===========================================================================
# Core: Multi-Scale Feature Extractor + Causality Analyzer
# ===========================================================================

class MultiscaleAnalyzer:
    """
    Decompose signals into HF/MF/LF, then find optimal lead per (signal, scale).

    Usage
    -----
    >>> msa = MultiscaleAnalyzer()
    >>> msa.fit(signals_dict, impact_array)
    >>> config = msa.export_config()
    """

    def __init__(self, config: Optional[MultiscaleConfig] = None):
        self.config = config or MultiscaleConfig()
        cfg = self.config

        self.decomposer = CausalDecomposer(CausalWaveletConfig(
            hf_span=cfg.hf_span, mf_span=cfg.mf_span, lf_span=cfg.lf_span,
        ))

        self.scale_names = ["HF", "MF", "LF"]
        self.signal_names = ["depth_evap", "obi_impulse", "spread_shock", "cancel_burst"]

        # Results: {signal: {scale: {"lead": int, "corr": float, "role": str}}}
        self.results: dict = {}

    # ── Fit: decompose + sweep ────────────────────────────────────────

    def fit(self, signals: dict[str, np.ndarray], impact: np.ndarray) -> "MultiscaleAnalyzer":
        """
        Decompose each signal into HF/MF/LF, sweep lags, find best per-scale lead.

        Parameters
        ----------
        signals : dict of {name: (N,) array} for each precursor signal
        impact  : (N,) array of future |return| per batch
        """
        cfg = self.config

        print(f"\n{'─'*70}")
        print(f"  Multi-Scale Causality Analysis")
        print(f"{'─'*70}")
        print(f"  {'Signal':<16s} {'Scale':>5s} {'Best Lead':>10s} {'Corr':>10s} {'Role':>15s}")
        print(f"  {'─'*16} {'─'*5} {'─'*10} {'─'*10} {'─'*15}")

        for sig_name in self.signal_names:
            if sig_name not in signals:
                continue

            sig = signals[sig_name].astype(np.float64)
            n = min(len(sig), len(impact))
            sig = sig[:n]
            imp = impact[:n]

            # Decompose
            hf, mf, lf = self.decomposer.decompose(sig)
            scales = {"HF": hf, "MF": mf, "LF": lf}

            self.results[sig_name] = {}

            for scale_name, scale_sig in scales.items():
                # Sweep lags
                best_lead, best_corr = 0, -999.0
                for lag in range(cfg.max_lead + 1):
                    if lag == 0:
                        c = np.corrcoef(scale_sig, imp)[0, 1]
                    else:
                        c = np.corrcoef(scale_sig[:-lag], imp[lag:])[0, 1]
                    if not np.isnan(c) and c > best_corr:
                        best_corr, best_lead = c, lag

                # Role assignment
                if best_corr > 0.03 and best_lead > 0:
                    role = "precursor"       # leads impact, strong
                elif best_corr > 0.01:
                    role = "confirm"         # coincident or weak lead
                elif best_lead > 5:
                    role = "diagnostic"      # leads but too weak
                else:
                    role = "noise"           # no predictive value

                self.results[sig_name][scale_name] = {
                    "lead": int(best_lead),
                    "corr": float(best_corr),
                    "role": role,
                }

                print(f"  {sig_name:<16s} {scale_name:>5s} {best_lead:>10d} {best_corr:>+10.4f} {role:>15s}")

        return self

    # ── Get optimal scale per signal ──────────────────────────────────

    def best_scale(self, sig_name: str) -> Optional[str]:
        """Return the scale with highest correlation for a signal."""
        if sig_name not in self.results:
            return None
        best = None
        best_corr = -999.0
        for scale, info in self.results[sig_name].items():
            if info["corr"] > best_corr:
                best_corr = info["corr"]
                best = scale
        return best

    # ── Export config for toxicity scorer ─────────────────────────────

    def export_config(self) -> dict:
        """Build multiscale lag config for ToxicityScorer."""
        config = {
            "scales": {
                "HF": {"span": self.config.hf_span, "alpha": float(self.decomposer.alpha_hf)},
                "MF": {"span": self.config.mf_span, "alpha": float(self.decomposer.alpha_mf)},
                "LF": {"span": self.config.lf_span, "alpha": float(self.decomposer.alpha_lf)},
            },
            "signals": {},
            "weights": {
                # Weights adjusted by role: precursor > confirm > diagnostic > noise
                "HF": {"depth_evap": 1.5, "obi_impulse": 1.5, "spread_shock": 0.5, "cancel_burst": 0.3},
                "MF": {"depth_evap": 1.0, "obi_impulse": 1.0, "spread_shock": 0.5, "cancel_burst": 0.3},
                "LF": {"depth_evap": 0.3, "obi_impulse": 0.3, "spread_shock": 1.0, "cancel_burst": 0.0},
            },
        }

        for sig_name, scales in self.results.items():
            config["signals"][sig_name] = {}
            for scale_name, info in scales.items():
                config["signals"][sig_name][scale_name] = {
                    "lead": info["lead"],
                    "corr": info["corr"],
                    "role": info["role"],
                }

        # Save
        path = Path(self.config.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n  Multiscale config saved -> {path}")
        return config

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of which scale dominates per signal."""
        lines = []
        for sig_name in self.signal_names:
            if sig_name not in self.results:
                continue
            best_s = self.best_scale(sig_name)
            info = self.results[sig_name].get(best_s, {}) if best_s else {}
            lines.append(
                f"  {sig_name:<16s}: best={best_s or '?'} "
                f"lead={info.get('lead','?')} corr={info.get('corr',0):+.4f} "
                f"role={info.get('role','?')}"
            )
        return "\n".join(lines)

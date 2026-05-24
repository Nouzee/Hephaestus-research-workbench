"""
Signal Router — automated multi-scale decomposition + optimal scale selection.

Priority 1 of the Phase 2 architecture.

For each signal:
  1. Decompose into HF/MF/LF via causal EMA filter bank
  2. Run causal lead-lag sweep per scale against future impact
  3. Auto-select the optimal scale (highest corr with positive lead)
  4. Output a routing config consumed by ToxicityScorer and FSM layers

The router answers: "which scale of which signal is the true precursor?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from modules.dictionary.causal_wavelet import CausalDecomposer, CausalWaveletConfig


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class SignalRouterConfig:
    """Configuration for automated signal routing."""

    # EMA spans
    hf_span: int = 2
    mf_span: int = 10
    lf_span: int = 50

    # Lead sweep
    max_lead: int = 20
    min_corr_threshold: float = 0.02   # below this -> "noise"

    # Output
    config_path: str = ""

    def __post_init__(self):
        if not self.config_path:
            self.config_path = str(
                Path(__file__).resolve().parent / "cache" / "signal_router.json"
            )


# ===========================================================================
# Core: Signal Router
# ===========================================================================

class SignalRouter:
    """
    Automated multi-scale routing for precursor signals.

    Usage
    -----
    >>> router = SignalRouter()
    >>> router.fit(signals_dict, impact_array)
    >>> config = router.routing_table()
    >>> # config ready for ToxicityScorer ingestion
    """

    def __init__(self, config: Optional[SignalRouterConfig] = None):
        self.config = config or SignalRouterConfig()
        cfg = self.config

        self.decomposer = CausalDecomposer(CausalWaveletConfig(
            hf_span=cfg.hf_span, mf_span=cfg.mf_span, lf_span=cfg.lf_span,
        ))
        self.scale_names = ["HF", "MF", "LF"]

        # Routing table: {signal_name: {scale: {lead, corr, role}}}
        self.routes: dict = {}

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(self, signals: dict[str, np.ndarray], impact: np.ndarray) -> "SignalRouter":
        """
        Decompose each signal, sweep lags per scale, select optimal routing.

        Parameters
        ----------
        signals : {name: (N,) array}
        impact  : (N,) future |return| per batch
        """
        cfg = self.config

        print(f"\n{'─'*65}")
        print(f"  Signal Router — Multi-Scale Decomposition + Optimal Routing")
        print(f"{'─'*65}")

        for sig_name, sig in signals.items():
            sig = sig.astype(np.float64)
            n = min(len(sig), len(impact))
            sig_arr = sig[:n]
            imp_arr = impact[:n]

            hf, mf, lf = self.decomposer.decompose(sig_arr)
            scales = {"HF": hf, "MF": mf, "LF": lf}

            self.routes[sig_name] = {}
            best_scale = None
            best_overall_corr = -999.0

            for scale_name, scale_sig in scales.items():
                best_lead, best_corr = 0, -999.0
                for lag in range(cfg.max_lead + 1):
                    c = (np.corrcoef(scale_sig[:-lag], imp_arr[lag:])[0, 1]
                         if lag > 0 else np.corrcoef(scale_sig, imp_arr)[0, 1])
                    if not np.isnan(c) and c > best_corr:
                        best_corr, best_lead = c, lag

                # Role classification
                if best_corr > 0.05 and best_lead > 0:
                    role = "precursor"
                elif best_corr > cfg.min_corr_threshold:
                    role = "confirm"
                elif best_lead > 5:
                    role = "diagnostic"
                else:
                    role = "noise"

                self.routes[sig_name][scale_name] = {
                    "lead": int(best_lead),
                    "corr": float(best_corr),
                    "role": role,
                }

                if best_corr > best_overall_corr and role in ("precursor", "confirm"):
                    best_overall_corr = best_corr
                    best_scale = scale_name

            # Mark optimal scale
            if best_scale:
                self.routes[sig_name]["_optimal_scale"] = best_scale

        self._print_table()
        return self

    # ── Routing table ────────────────────────────────────────────────

    def routing_table(self) -> dict:
        """Export routing config for downstream consumers."""
        return {
            "scales": {
                "HF": {"span": self.config.hf_span, "alpha": float(self.decomposer.alpha_hf)},
                "MF": {"span": self.config.mf_span, "alpha": float(self.decomposer.alpha_mf)},
                "LF": {"span": self.config.lf_span, "alpha": float(self.decomposer.alpha_lf)},
            },
            "signals": self.routes,
        }

    def export(self) -> str:
        """Save routing table to JSON."""
        path = Path(self.config.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.routing_table(), f, indent=2)
        print(f"  Routing config saved -> {path}")
        return str(path)

    # ── Helpers ──────────────────────────────────────────────────────

    def optimal_scale(self, sig_name: str) -> Optional[str]:
        return self.routes.get(sig_name, {}).get("_optimal_scale")

    def get_lead(self, sig_name: str, scale: str) -> int:
        return self.routes.get(sig_name, {}).get(scale, {}).get("lead", 0)

    def get_role(self, sig_name: str, scale: str) -> str:
        return self.routes.get(sig_name, {}).get(scale, {}).get("role", "noise")

    # ── Display ──────────────────────────────────────────────────────

    def _print_table(self):
        print(f"  {'Signal':<16s} {'HF':>5s} {'lead':>5s} {'corr':>8s}  "
              f"{'MF':>5s} {'lead':>5s} {'corr':>8s}  "
              f"{'LF':>5s} {'lead':>5s} {'corr':>8s}  {'Best':>6s}")
        print(f"  {'─'*16} {'─'*5} {'─'*5} {'─'*8}  "
              f"{'─'*5} {'─'*5} {'─'*8}  "
              f"{'─'*5} {'─'*5} {'─'*8}  {'─'*6}")

        for sig_name, scales in self.routes.items():
            if sig_name.startswith("_"):
                continue
            best = self.optimal_scale(sig_name) or "?"
            parts = [f"  {sig_name:<16s}"]
            for sn in ["HF", "MF", "LF"]:
                info = scales.get(sn, {})
                role_mark = {"precursor": "+", "confirm": "~", "diagnostic": ".", "noise": " "}
                rm = role_mark.get(info.get("role", "noise"), " ")
                parts.append(f"{rm}{info.get('lead',0):>4d} {info.get('corr',0):>+7.4f}")
            parts.append(f"{best:>6s}")
            print("  ".join(parts))

        print(f"{'─'*65}")


# ===========================================================================
# Factory
# ===========================================================================

def create_router(hf_span=2, mf_span=10, lf_span=50) -> SignalRouter:
    return SignalRouter(SignalRouterConfig(hf_span=hf_span, mf_span=mf_span, lf_span=lf_span))

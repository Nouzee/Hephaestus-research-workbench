"""
Cross-Regime Mode Extractor — solve the latent dynamical basis of markets.

Takes the ~8D MarketState and decomposes it into:
  φ₁...φ₈  — stable basis vectors (what each mode IS)
  z₁(t)...z₈(t) — time series (what each mode DOES over time)

Then classifies each mode by:
  - Timescale: fast (microstructure) / medium (flow) / slow (regime)
  - Economic role: spread capture driver / adverse selection driver / vol driver
  - Regime sensitivity: does this mode collapse in FRAGILE?
  - Action mapping: which trading action does this mode control?
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from projects.compressibility_frontier.metrics.state_segmenter import segment_all


# ===========================================================================
# Mode Extractor
# ===========================================================================

class ModeExtractor:
    """
    Extract stable latent modes from MarketState feature matrix.

    Usage
    -----
    >>> me = ModeExtractor()
    >>> me.fit(feature_matrix, var_names)
    >>> me.print_modes()
    >>> z = me.project(feature_matrix)  # (N, 8) mode time series
    """

    def __init__(self, n_modes: int = 8):
        self.n_modes = n_modes
        self.phi: np.ndarray = None          # (D, K) basis vectors
        self.explained_var: np.ndarray = None # (K,) explained variance ratio
        self.singular_values: np.ndarray = None
        self.var_names: list[str] = []
        self.mode_labels: list[str] = []     # human-readable mode names
        self.mode_metadata: list[dict] = []  # per-mode classification

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,       # (N, D) standardized
        var_names: list[str],
    ) -> "ModeExtractor":
        """
        Compute SVD on the feature matrix to extract stable modes.

        φ_i = columns of V from SVD: X = U Σ V^T
        z_i(t) = X @ φ_i  (projection onto mode i)
        """
        N, D = features.shape
        K = min(self.n_modes, D)
        self.var_names = var_names

        # SVD
        U, S, Vt = linalg.svd(features.astype(np.float64), full_matrices=False)
        self.phi = Vt[:K, :].T              # (D, K) — each column is a mode
        self.singular_values = S[:K]
        self.explained_var = (S[:K] ** 2) / np.sum(S ** 2)

        return self

    # ── Project ──────────────────────────────────────────────────────

    def project(self, features: np.ndarray) -> np.ndarray:
        """Project data onto mode basis → z_i(t) time series. Returns (N, K)."""
        return features @ self.phi

    # ── Classify each mode ───────────────────────────────────────────

    def classify_modes(
        self,
        z_series: np.ndarray,           # (N, K) mode time series
        pnl_spread: np.ndarray,          # (N,) spread capture PnL
        pnl_adverse: np.ndarray,         # (N,) adverse selection PnL
        realized_vol: np.ndarray,        # (N,) realized volatility
        regimes: dict[str, np.ndarray],  # regime masks
    ) -> "ModeExtractor":
        """
        Classify each mode by timescale, economic role, and regime sensitivity.

        Parameters
        ----------
        z_series     : (N, K) mode activation time series
        pnl_spread   : (N,) per-batch spread capture
        pnl_adverse  : (N,) per-batch adverse selection
        realized_vol : (N,) per-batch volatility
        regimes      : {regime_name: boolean mask}
        """
        K = z_series.shape[1]
        self.mode_labels = []
        self.mode_metadata = []

        for k in range(K):
            z = z_series[:, k]
            meta = {}

            # ── 1. Timescale: autocorrelation decay ──
            acf_1 = np.corrcoef(z[:-1], z[1:])[0, 1] if len(z) > 2 else 0
            # How many lags until ACF drops below 0.5?
            half_life = 1
            for lag in range(1, min(50, len(z) // 2)):
                acf = np.corrcoef(z[:-lag], z[lag:])[0, 1]
                if abs(acf) < 0.5:
                    half_life = lag
                    break

            if half_life <= 2:
                timescale = "fast (microstructure)"
            elif half_life <= 10:
                timescale = "medium (flow)"
            else:
                timescale = "slow (regime)"

            meta["timescale"] = timescale
            meta["acf_1"] = float(acf_1)
            meta["half_life"] = half_life

            # ── 2. Economic role ──
            corr_spread = np.corrcoef(z, pnl_spread)[0, 1]
            corr_adverse = np.corrcoef(z, pnl_adverse)[0, 1]
            corr_vol = np.corrcoef(z, realized_vol)[0, 1]

            meta["corr_spread_capture"] = float(corr_spread)
            meta["corr_adverse_selection"] = float(corr_adverse)
            meta["corr_volatility"] = float(corr_vol)

            # Determine primary role
            abs_corrs = {
                "spread capture driver": abs(corr_spread),
                "adverse selection driver": abs(corr_adverse),
                "volatility driver": abs(corr_vol),
            }
            primary_role = max(abs_corrs, key=abs_corrs.get)

            # Only assign role if correlation is meaningful
            if max(abs_corrs.values()) > 0.05:
                meta["economic_role"] = primary_role
            else:
                meta["economic_role"] = "structural (no direct PnL link)"

            # ── 3. Regime sensitivity ──
            if "FRAGILE" in regimes and "HEALTHY" in regimes:
                z_fragile = z[regimes["FRAGILE"]]
                z_healthy = z[regimes["HEALTHY"]]

                if len(z_fragile) > 10 and len(z_healthy) > 10:
                    mean_f = np.mean(np.abs(z_fragile))
                    mean_h = np.mean(np.abs(z_healthy))
                    ratio = mean_f / max(mean_h, 1e-12)

                    if ratio < 0.5:
                        regime_sensitivity = "COLLAPSES in fragile"
                    elif ratio > 2.0:
                        regime_sensitivity = "AMPLIFIES in fragile"
                    elif ratio > 1.3:
                        regime_sensitivity = "elevated in fragile"
                    elif ratio < 0.7:
                        regime_sensitivity = "suppressed in fragile"
                    else:
                        regime_sensitivity = "stable across regimes"

                    meta["fragile_healthy_ratio"] = float(ratio)
                else:
                    regime_sensitivity = "insufficient samples"
                    meta["fragile_healthy_ratio"] = 1.0
            else:
                regime_sensitivity = "regime data unavailable"
                meta["fragile_healthy_ratio"] = 1.0

            meta["regime_sensitivity"] = regime_sensitivity

            # ── 4. Mode composition (top 3 contributing variables) ──
            phi_k = self.phi[:, k]
            top_idx = np.argsort(np.abs(phi_k))[::-1][:3]
            composition = [
                {"variable": self.var_names[i], "weight": float(phi_k[i])}
                for i in top_idx
            ]
            meta["top_variables"] = composition

            # ── 5. Human-readable label ──
            label = self._generate_label(k, meta, composition)
            self.mode_labels.append(label)
            meta["label"] = label

            self.mode_metadata.append(meta)

        return self

    def _generate_label(self, k: int, meta: dict, composition: list) -> str:
        """Generate a human-readable mode name."""
        # Use top variable as primary descriptor
        top_var = composition[0]["variable"] if composition else "unknown"
        role = meta.get("economic_role", "")
        regime = meta.get("regime_sensitivity", "")

        # Shorten variable names
        short_names = {
            "trade_arrival_rate": "arrival",
            "signed_imbalance": "imb",
            "size_dispersion": "size_disp",
            "flow_persistence": "flow_persist",
            "cancel_burst_ratio": "cancel_burst",
            "buy_sell_volume_ratio": "buy_sell_ratio",
            "spread_bps": "spread",
            "total_depth": "depth",
            "depth_imbalance": "depth_imb",
            "queue_pressure": "queue_press",
            "spread_volatility": "spread_vol",
            "liquidity_tension": "liq_tension",
            "depth_replenish_corr": "depth_replen",
            "realized_volatility": "real_vol",
            "immediate_impact_corr": "impact_corr",
            "nonlinear_response": "nonlinear",
            "volatility_persistence": "vol_persist",
            "impact_memory": "imp_memory",
            "flow_memory": "flow_mem",
            "regime_stability": "reg_stability",
        }
        top_short = short_names.get(top_var, top_var[:12])

        if "COLLAPSES" in regime:
            return f"M{k}: {top_short} [COLLAPSIBLE]"
        elif "AMPLIFIES" in regime:
            return f"M{k}: {top_short} [FRAGILITY]"
        elif "spread capture" in role:
            return f"M{k}: {top_short} [SPREAD]"
        elif "adverse" in role:
            return f"M{k}: {top_short} [ADVERSE]"
        elif "volatility" in role:
            return f"M{k}: {top_short} [VOL]"
        else:
            return f"M{k}: {top_short} [STRUCTURAL]"

    # ── Mode → Action mapping ────────────────────────────────────────

    def action_map(self) -> dict:
        """Map each mode to a suggested trading action."""
        if not self.mode_metadata:
            return {}

        mapping = {}
        for k, meta in enumerate(self.mode_metadata):
            role = meta.get("economic_role", "")
            regime = meta.get("regime_sensitivity", "")
            scale = meta.get("timescale", "")

            if "adverse" in role:
                action = "inventory_skew + spread_widening"
            elif "spread capture" in role:
                action = "quote_aggressiveness"
            elif "volatility" in role or "COLLAPSES" in regime:
                action = "size_reduction + risk_throttle"
            elif "fast" in scale:
                action = "execution_timing"
            elif "slow" in scale:
                action = "regime_position_sizing"
            else:
                action = "monitor_only"

            mapping[self.mode_labels[k]] = {
                "mode_index": k,
                "role": role,
                "scale": scale,
                "action": action,
            }

        return mapping

    # ── Report ───────────────────────────────────────────────────────

    def print_modes(self):
        """Print full mode decomposition report."""
        K = len(self.mode_metadata)
        if K == 0:
            print("No modes extracted. Run fit() + classify_modes() first.")
            return

        print(f"\n{'═'*75}")
        print(f"  Latent Dynamical Modes — Market State Decomposition")
        print(f"{'═'*75}")

        print(f"\n  Explained variance by mode:")
        cum = 0
        for k in range(K):
            cum += self.explained_var[k]
            bar = "#" * int(self.explained_var[k] * 50) + "." * (50 - int(self.explained_var[k] * 50))
            print(f"    M{k}: {self.explained_var[k]:>6.1%}  {bar}  (cumulative: {cum:.1%})")

        print(f"\n  Mode Details:")
        for k in range(K):
            meta = self.mode_metadata[k]
            label = self.mode_labels[k]

            print(f"\n  ┌─ {label}")
            print(f"  │  Variance: {self.explained_var[k]:.1%}  "
                  f"Timescale: {meta['timescale']}  (half-life={meta['half_life']} batches)")
            print(f"  │  Economic role: {meta['economic_role']}")
            print(f"  │    corr(spread)={meta['corr_spread_capture']:+.3f}  "
                  f"corr(adverse)={meta['corr_adverse_selection']:+.3f}  "
                  f"corr(vol)={meta['corr_volatility']:+.3f}")
            print(f"  │  Regime: {meta['regime_sensitivity']}"
                  f"{' (ratio=' + str(round(meta.get('fragile_healthy_ratio', 1), 2)) + ')' if 'fragile_healthy_ratio' in meta else ''}")
            print(f"  │  Composition:")
            for v in meta.get("top_variables", []):
                sign = "+" if v["weight"] > 0 else "-"
                print(f"  │    {sign} {v['variable']:<25s} ({abs(v['weight']):.3f})")

        # Action map
        print(f"\n  Mode → Action Mapping:")
        am = self.action_map()
        for mode_name, info in am.items():
            print(f"    {mode_name:<40s} → {info['action']}")

        print(f"{'═'*75}")

    # ── Collapse report ──────────────────────────────────────────────

    def collapse_report(self) -> dict:
        """Identify which modes collapse in fragile states."""
        collapsing = []
        amplifying = []
        for k, meta in enumerate(self.mode_metadata):
            rs = meta.get("regime_sensitivity", "")
            if "COLLAPSES" in rs:
                collapsing.append(self.mode_labels[k])
            elif "AMPLIFIES" in rs:
                amplifying.append(self.mode_labels[k])

        return {
            "collapsing_modes": collapsing,
            "amplifying_modes": amplifying,
            "collapse_interpretation": (
                f"{len(collapsing)} modes collapse in fragile, "
                f"{len(amplifying)} amplify. "
                f"The market simplifies by suppressing {collapsing} "
                f"while {amplifying} dominate."
            ) if collapsing else "No clear collapse pattern detected.",
        }

"""
A-Share Regime Segmentation — Phase 1

Extracts microstructure features from L2 orderbook + trade data,
clusters into market regimes using HDBSCAN/spectral methods.

Features per window (100 tick):
  LIQUIDITY: spread, orderbook slope, bid/ask imbalance, queue refill,
             cancellation intensity, depth collapse rate
  FLOW: trade arrival rate, signed order flow, flow persistence, meta-order proxy
  IMPACT: realized volatility, nonlinear price response, local elasticity

Output: regime labels, persistence, occupancy, transition frequencies.
"""

from __future__ import annotations

import glob, time
from pathlib import Path
import numpy as np
import polars as pl


# ===========================================================================
# Feature Extractor — per-window microstructure features
# ===========================================================================

class L2FeatureExtractor:
    """
    Extract microstructure features from aligned L2 + trade data.
    Window size: configurable (default 100 ticks).
    """

    def __init__(self, window_size: int = 100):
        self.W = window_size

    def extract_window(
        self,
        ob_slice: dict,     # orderbook data for this window
        msg_slice: dict,    # message/trade data for this window
    ) -> dict:
        """Extract features from one window. Returns dict of scalar features."""
        eps = 1e-12

        # ── Orderbook features ──
        # Helper: sum across L2 levels for each timestamp
        def _level_sum(key_pattern, start=1, end=10):
            return sum(ob_slice[key_pattern.format(i)] for i in range(start, end+1))

        bid_qty_total = _level_sum("BidOrderQty{}")
        ask_qty_total = _level_sum("OfferOrderQty{}")

        # Spread (bps) — per-tick, then aggregate
        mid = (ob_slice["OfferPrice1"] + ob_slice["BidPrice1"]) / 2.0
        spread = (ob_slice["OfferPrice1"] - ob_slice["BidPrice1"]) / np.maximum(mid, eps) * 10000

        # Bid/ask imbalance (per-tick, then mean)
        depth_imbalance_arr = (bid_qty_total - ask_qty_total) / np.maximum(bid_qty_total + ask_qty_total, eps)

        # Queue refill speed: mean absolute change in total depth
        total_depth_arr = bid_qty_total + ask_qty_total
        depth_changes = np.abs(np.diff(total_depth_arr))
        queue_refill = float(np.mean(depth_changes)) if len(depth_changes) > 1 else 0.0

        # Depth collapse: 1 - min/max ratio
        depth_min = float(np.min(total_depth_arr))
        depth_max = float(np.max(total_depth_arr))
        depth_collapse = 1.0 - depth_min / max(depth_max, eps)

        # Spread volatility
        spread_vol = float(np.std(spread)) if len(spread) > 1 else 0.0

        # ── Trade/message features ──
        direction = msg_slice.get("Direction", np.zeros(1))
        size = msg_slice.get("Size", np.zeros(1))
        T = len(direction)
        time_span = float(msg_slice["Time (sec)"][-1] - msg_slice["Time (sec)"][0]) if T > 1 else 1.0
        arrival_rate = T / max(time_span, eps)

        # Signed order flow
        signed_flow = np.sum(direction * size)
        signed_imbalance = signed_flow / max(np.sum(np.abs(size)), eps) if np.sum(np.abs(size)) > 0 else 0.0

        # Flow persistence
        if T > 5:
            dir_arr = direction.astype(np.float64)
            flow_persist = np.corrcoef(dir_arr[:-1], dir_arr[1:])[0, 1]
            flow_persist = 0.0 if np.isnan(flow_persist) else float(flow_persist)
        else:
            flow_persist = 0.0

        # Meta-order proxy
        meta_order = signed_flow / max(np.sqrt(T), eps)

        # Cancellation intensity
        qty_changes_arr = np.abs(np.diff(total_depth_arr))
        trade_sizes_arr = size.astype(np.float64)
        cancel_intensity = float(np.mean(qty_changes_arr) - np.mean(trade_sizes_arr)) if len(qty_changes_arr) > 0 else 0.0

        # ── Price impact features ──
        mid_return = float((mid[-1] - mid[0]) / max(mid[0], eps)) if len(mid) > 1 else 0.0

        # Realized volatility
        if len(mid) > 5:
            mid_ret = np.diff(mid) / (np.abs(mid[:-1]) + eps)
            realized_vol = float(np.std(mid_ret) * np.sqrt(len(mid_ret)))
        else:
            realized_vol = 0.0

        # Nonlinear price response
        if T > 10 and len(mid) > 5:
            size_arr = size.astype(np.float64)
            med_size = np.median(size_arr)
            small_mask = size_arr < med_size
            large_mask = size_arr >= med_size
            n_ret = min(len(mid_ret), len(size_arr))
            impact_small = float(np.mean(np.abs(mid_ret[:n_ret][small_mask[:n_ret]]))) if np.any(small_mask[:n_ret]) else 0.0
            impact_large = float(np.mean(np.abs(mid_ret[:n_ret][large_mask[:n_ret]]))) if np.any(large_mask[:n_ret]) else 0.0
            nonlinear = impact_large / max(impact_small, eps)
        else:
            nonlinear = 1.0

        # Local elasticity
        elasticity = abs(mid_return) / max(abs(signed_imbalance), eps) if abs(signed_imbalance) > 1e-8 else 0.0

        # Impact asymmetry
        if len(mid) > 5:
            mid_ret_arr = np.diff(mid) / (np.abs(mid[:-1]) + eps)
            pos_impact = float(np.mean(mid_ret_arr[mid_ret_arr > 0])) if np.any(mid_ret_arr > 0) else 0.0
            neg_impact = float(np.mean(np.abs(mid_ret_arr[mid_ret_arr < 0]))) if np.any(mid_ret_arr < 0) else 0.0
            impact_asym = pos_impact / max(neg_impact, eps)
        else:
            impact_asym = 1.0

        # Orderbook slope
        ob_slope = ask_qty_total / np.maximum(bid_qty_total, eps)

        return {
            # Liquidity (8 features)
            "spread_bps": float(np.mean(spread)),
            "spread_volatility": float(spread_vol),
            "depth_imbalance": float(np.mean(depth_imbalance_arr)),
            "queue_refill_speed": float(queue_refill),
            "depth_collapse_ratio": float(depth_collapse),
            "cancel_intensity": float(cancel_intensity),
            "total_depth": float(np.mean(total_depth_arr)),
            "orderbook_slope": float(np.mean(ob_slope)),
            # Flow (4 features)
            "arrival_rate": float(arrival_rate),
            "signed_imbalance": float(signed_imbalance),
            "flow_persistence": float(flow_persist),
            "meta_order_proxy": float(meta_order),
            # Impact (4 features)
            "realized_volatility": float(realized_vol),
            "nonlinear_response": float(nonlinear),
            "local_elasticity": float(elasticity),
            "impact_asymmetry": float(impact_asym),
        }


# ===========================================================================
# Feature matrix builder — processes all data into (N_windows, D) matrix
# ===========================================================================

def build_feature_matrix(
    data_dir: str,
    window_size: int = 100,
    max_days: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all daily parquet files into a feature matrix.

    Returns:
      feature_matrix: (N_windows, D) standardized features
      day_boundaries: indices where days end
      raw_meta: dict with timestamps, prices for later analysis
    """
    extractor = L2FeatureExtractor(window_size=window_size)

    msg_files = sorted(glob.glob(str(Path(data_dir) / "message_*.parquet")))
    ob_files = sorted(glob.glob(str(Path(data_dir) / "orderbook_*.parquet")))

    if max_days:
        msg_files = msg_files[:max_days]
        ob_files = ob_files[:max_days]

    all_features = []
    day_boundaries = []
    timestamps = []
    mid_prices = []

    print(f"  Processing {len(msg_files)} days...")

    for day_idx, (mf, of) in enumerate(zip(msg_files, ob_files)):
        msg_df = pl.read_parquet(mf)
        ob_df = pl.read_parquet(of)

        N = msg_df.shape[0]
        n_windows = N // window_size

        if n_windows < 5:
            continue

        # Convert to numpy for fast slicing
        msg_dict = {col: msg_df[col].to_numpy() for col in msg_df.columns}
        ob_dict = {col: ob_df[col].to_numpy() for col in ob_df.columns}

        for w in range(n_windows):
            s, e = w * window_size, (w + 1) * window_size
            msg_slice = {k: v[s:e] for k, v in msg_dict.items()}
            ob_slice = {k: v[s:e] for k, v in ob_dict.items()}

            feats = extractor.extract_window(ob_slice, msg_slice)
            all_features.append(list(feats.values()))
            timestamps.append(msg_dict["Time (sec)"][s])

            # Track mid price
            mid = (ob_dict["OfferPrice1"][s:e] + ob_dict["BidPrice1"][s:e]) / 2.0
            mid_prices.append(float(np.mean(mid)))

        day_boundaries.append(len(all_features))
        if (day_idx + 1) % 10 == 0:
            print(f"    [{day_idx+1}/{len(msg_files)}] days, {len(all_features)} windows")

    feature_matrix = np.array(all_features, dtype=np.float32)
    day_boundaries = np.array(day_boundaries, dtype=np.int32)
    mid_prices = np.array(mid_prices, dtype=np.float64)
    timestamps = np.array(timestamps, dtype=np.float64)

    # Standardize
    f_mean = feature_matrix.mean(axis=0)
    f_std = np.maximum(feature_matrix.std(axis=0), 1e-8)
    feature_matrix_z = (feature_matrix - f_mean) / f_std
    feature_matrix_z = np.clip(feature_matrix_z, -10, 10)

    print(f"  Total: {feature_matrix_z.shape[0]} windows × {feature_matrix_z.shape[1]} features")
    return feature_matrix_z, day_boundaries, {"timestamps": timestamps, "mid_prices": mid_prices}


# ===========================================================================
# Feature names
# ===========================================================================

FEATURE_NAMES = [
    # Liquidity
    "spread_bps", "spread_volatility", "depth_imbalance",
    "queue_refill_speed", "depth_collapse_ratio", "cancel_intensity",
    "total_depth", "orderbook_slope",
    # Flow
    "arrival_rate", "signed_imbalance", "flow_persistence", "meta_order_proxy",
    # Impact
    "realized_volatility", "nonlinear_response", "local_elasticity", "impact_asymmetry",
]

"""
Observation Operators — multiple ways to project raw ticks into observed states.

O_k ∈ {
    tick          — raw tick-level (no aggregation)
    tick_50       — 50-tick windows
    tick_200      — 200-tick windows
    tick_2048     — current batch size (baseline)
    volume_100    — every ~100 contracts traded
    trade_50      — every 50 trades
    time_1s       — 1-second bars
}

Each operator defines: how to chunk raw data into observation windows.
The key question: is market structure INVARIANT to the operator?
"""

from __future__ import annotations

import numpy as np

# ===========================================================================
# Operator definitions
# ===========================================================================

OPERATORS = {
    "tick_50": {
        "type": "tick_count",
        "window": 50,
        "description": "50-tick windows (microstructure scale)",
    },
    "tick_200": {
        "type": "tick_count",
        "window": 200,
        "description": "200-tick windows (short-term scale)",
    },
    "tick_2048": {
        "type": "tick_count",
        "window": 2048,
        "description": "2048-tick windows (batch baseline)",
    },
    "trade_50": {
        "type": "trade_count",
        "window": 50,
        "description": "Every 50 trades (trade-time)",
    },
    "volume_100": {
        "type": "cumulative_volume",
        "window": 100,
        "description": "Every ~100 contracts (volume clock)",
    },
}


# ===========================================================================
# Operator applier
# ===========================================================================

def apply_operator(
    raw_data: dict[str, np.ndarray],   # {col: (N,) array}
    operator: dict,
    feature_func,
) -> np.ndarray:
    """
    Apply an observation operator to raw tick data.

    Returns (M, D) feature matrix where M = number of observation windows.

    Parameters
    ----------
    raw_data   : dict of column → (N,) array (mid_px, spread, depth, imb, duration)
    operator   : operator definition from OPERATORS
    feature_func : callable(raw_window) → (D,) feature vector

    Returns
    -------
    features : (M, D) array
    """
    N = len(next(iter(raw_data.values())))
    op_type = operator["type"]
    W = operator["window"]

    if op_type == "tick_count":
        # Fixed tick count windows
        M = N // W
        features = np.zeros((M, 9), dtype=np.float32)
        for i in range(M):
            s, e = i * W, (i + 1) * W
            window = {k: v[s:e] for k, v in raw_data.items()}
            features[i] = feature_func(window)
        return features, M

    elif op_type == "trade_count":
        # Every W trades
        trade_px = raw_data.get("trade_px")
        if trade_px is None:
            raise ValueError("trade_px required for trade_count operator")

        # Split by cumulative trade count
        n_trades = len(trade_px)
        M = n_trades // W
        features = np.zeros((M, 9), dtype=np.float32)
        for i in range(M):
            s, e = i * W, min((i + 1) * W, n_trades)
            window = {k: v[s:e] for k, v in raw_data.items() if len(v) == n_trades}
            features[i] = feature_func(window)
        return features, M

    elif op_type == "cumulative_volume":
        # Every W contracts of cumulative volume
        trade_sz = raw_data.get("trade_sz")
        if trade_sz is None:
            raise ValueError("trade_sz required for volume operator")

        cum_vol = np.cumsum(np.abs(trade_sz))
        total_vol = cum_vol[-1]
        M = int(total_vol / W)
        if M < 10:
            M = 10

        features = np.zeros((M, 9), dtype=np.float32)
        boundaries = np.linspace(0, total_vol, M + 1)
        for i in range(M):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (cum_vol >= lo) & (cum_vol < hi)
            if mask.sum() < 5:
                continue
            indices = np.where(mask)[0]
            window = {k: v[indices[0]:indices[-1] + 1]
                      for k, v in raw_data.items()
                      if len(v) == len(cum_vol)}
            features[i] = feature_func(window)
        return features, M

    else:
        raise ValueError(f"Unknown operator type: {op_type}")


# ===========================================================================
# Feature extractor (batch-level, same as market_decon LayerDecomposer)
# ===========================================================================

def extract_features(window: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract a 9-dim feature vector from a raw tick window.

    Features:
      0. mid_px return
      1. spread_bps
      2. depth_mean
      3. signed_imbalance
      4. realized_volatility
      5. trade_arrival_rate
      6. flow_persistence
      7. queue_pressure
      8. spread_volatility
    """
    eps = 1e-12
    mid_px = window.get("mid_px")
    spread = window.get("spread")
    depth = window.get("total_depth")
    signed_imb = window.get("signed_imbalance")
    duration = window.get("duration_ms")

    T = len(mid_px)
    if T < 5:
        return np.zeros(9, dtype=np.float32)

    # 0. mid-price return over window
    mid_ret = (mid_px[-1] - mid_px[0]) / max(abs(mid_px[0]), eps)

    # 1. spread in bps
    spread_bps = np.mean(spread) / max(np.mean(mid_px), eps) * 10000

    # 2. depth
    depth_mean = np.mean(depth)

    # 3. signed imbalance
    imb = float(np.mean(signed_imb))

    # 4. realized volatility
    if T > 2:
        rets = np.diff(mid_px) / (np.abs(mid_px[:-1]) + eps)
        realized_vol = float(np.std(rets) * np.sqrt(T))
    else:
        realized_vol = 0.0

    # 5. trade arrival rate
    if duration is not None and T > 0:
        total_time_s = np.sum(duration) / 1000.0
        arrival_rate = T / max(total_time_s, eps)
    else:
        arrival_rate = float(T)

    # 6. flow persistence (lag-1 autocorr of sign)
    if T > 5:
        signs = np.sign(signed_imb)
        if np.std(signs) > 0:
            flow_persist = np.corrcoef(signs[:-1], signs[1:])[0, 1]
            flow_persist = 0.0 if np.isnan(flow_persist) else float(flow_persist)
        else:
            flow_persist = 0.0
    else:
        flow_persist = 0.0

    # 7. queue pressure: arrival / depth
    queue_pressure = arrival_rate / max(depth_mean, eps)

    # 8. spread volatility
    spread_vol = float(np.std(spread) / max(np.mean(spread), eps))

    return np.array([
        mid_ret, spread_bps, depth_mean, imb,
        realized_vol, arrival_rate, flow_persist,
        queue_pressure, spread_vol,
    ], dtype=np.float32)

"""
Matrix Builder — 高频盘口数据 → N×M 特征矩阵 X

Task 1 of Dictionary Learning pipeline.
Reads events_features.parquet via Polars, computes multi-timescale
order book features, and outputs a clean N×M NumPy matrix ready for
MiniBatchDictionaryLearning.

Feature groups (M = 9):
  G1 — Multi-level OBI (4): imbalance, imbalance_ma_20, imbalance_ma_100, signed_imbalance
  G2 — Spread        (1): spread
  G3 — Depth Δ%      (2): depth_roc_10, depth_roc_50
  G4 — Micro-px Δ%   (2): micro_px_ret_5, micro_px_ret_20

All derived features use backward-looking windows only (no future leak).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MatrixBuilderConfig:
    """Configuration for the HFT matrix assembly step."""

    # Data source
    source_parquet: str = (
        r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"
    )

    # OB imbalance windows (ticks)
    obi_short_window: int = 20   # ~few seconds
    obi_long_window: int = 100   # ~tens of seconds

    # Depth rate-of-change windows
    depth_short_window: int = 10
    depth_long_window: int = 50

    # Micro-price return windows
    micro_short_window: int = 5
    micro_long_window: int = 20

    # Output
    dtype: str = "float32"       # float32 saves memory, sufficient for dict learning
    drop_nan: bool = True        # must be True — derived features have leading NaNs
    standardize: bool = True     # per-column Z-score → equal footing for optimizer
    seed: int = 42               # for any shuffling / downsampling later


# ═══════════════════════════════════════════════════════════════════════
# Column names (used after feature computation)
# ═══════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # G1: Multi-level OBI
    "imbalance",
    "imbalance_ma_20",
    "imbalance_ma_100",
    "signed_imbalance",
    # G2: Spread
    "spread",
    # G3: Depth change rate
    "depth_roc_10",
    "depth_roc_50",
    # G4: Micro-price momentum
    "micro_px_ret_5",
    "micro_px_ret_20",
]

FEATURE_GROUPS = {
    "OBI":   [0, 1, 2, 3],  # imbalance family
    "SPRD":  [4],            # spread
    "DEPTH": [5, 6],         # depth Δ% family
    "MICRO": [7, 8],         # micro-px Δ% family
}


# ═══════════════════════════════════════════════════════════════════════
# Matrix Builder
# ═══════════════════════════════════════════════════════════════════════

class MatrixBuilder:
    """
    Assemble high-frequency order book data into a clean N×M matrix
    suitable for sparse dictionary learning (MiniBatchDictionaryLearning).

    Usage
    -----
    >>> builder = MatrixBuilder()
    >>> X, meta = builder.assemble()
    >>> print(X.shape)   # (N, 9)
    """

    def __init__(self, config: Optional[MatrixBuilderConfig] = None):
        self.config = config or MatrixBuilderConfig()
        self._df: Optional[pl.DataFrame] = None
        self._X: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None

    # ── Step 1: load ────────────────────────────────────────────────

    def load_raw(self) -> pl.DataFrame:
        """
        Read the source parquet into a Polars DataFrame.
        Only projects columns needed for feature computation to keep
        memory footprint low.
        """
        t0 = time.perf_counter()
        path = self.config.source_parquet

        needed = [
            "ts_ms", "imbalance", "signed_imbalance",
            "spread", "total_depth", "micro_px",
        ]

        df = pl.read_parquet(path, columns=needed)
        elapsed = time.perf_counter() - t0
        print(f"[MatrixBuilder] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols "
              f"in {elapsed:.2f}s")
        return df

    # ── Step 2: derive features ─────────────────────────────────────

    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute derived features using pure Polars expressions.

        All windows are backward-looking:
          - shift(n) + rolling_mean → no future information leaks into the feature
          - pct_change uses shift(1) internally → safe after we manually shift

        Returns a DataFrame with only the 9 feature columns.
        """
        cfg = self.config

        expr_imbalance_ma_20 = (
            pl.col("imbalance")
            .shift(1)                     # exclude current tick from "past" window
            .rolling_mean(cfg.obi_short_window, min_periods=cfg.obi_short_window // 2)
            .alias("imbalance_ma_20")
        )

        expr_imbalance_ma_100 = (
            pl.col("imbalance")
            .shift(1)
            .rolling_mean(cfg.obi_long_window, min_periods=cfg.obi_long_window // 2)
            .alias("imbalance_ma_100")
        )

        # Depth rate-of-change: (depth_t - depth_{t-k}) / (|depth_{t-k}| + ε)
        expr_depth_roc_10 = (
            (pl.col("total_depth") - pl.col("total_depth").shift(cfg.depth_short_window))
            / (pl.col("total_depth").shift(cfg.depth_short_window).abs() + 1e-8)
        ).alias("depth_roc_10")

        expr_depth_roc_50 = (
            (pl.col("total_depth") - pl.col("total_depth").shift(cfg.depth_long_window))
            / (pl.col("total_depth").shift(cfg.depth_long_window).abs() + 1e-8)
        ).alias("depth_roc_50")

        # Micro-price returns (simple, backward-looking)
        expr_micro_ret_5 = (
            pl.col("micro_px").pct_change(cfg.micro_short_window)
        ).alias("micro_px_ret_5")

        expr_micro_ret_20 = (
            pl.col("micro_px").pct_change(cfg.micro_long_window)
        ).alias("micro_px_ret_20")

        # Build feature DataFrame
        feats = df.select([
            # G1 — OBI family
            pl.col("imbalance"),
            expr_imbalance_ma_20,
            expr_imbalance_ma_100,
            pl.col("signed_imbalance"),
            # G2 — Spread
            pl.col("spread"),
            # G3 — Depth Δ% family
            expr_depth_roc_10,
            expr_depth_roc_50,
            # G4 — Micro-px Δ% family
            expr_micro_ret_5,
            expr_micro_ret_20,
        ])

        return feats

    # ── Step 3: clean → numpy ───────────────────────────────────────

    def to_matrix(self, df: pl.DataFrame) -> np.ndarray:
        """
        Drop NaN rows, clip extreme outliers, convert to numpy,
        then apply per-column Z-score standardization.
        """
        total_before = df.shape[0]

        if self.config.drop_nan:
            df = df.drop_nulls()
            dropped = total_before - df.shape[0]
            pct = dropped / total_before * 100
            print(f"[MatrixBuilder] Dropped {dropped:,} rows ({pct:.2f}%) "
                  f"due to NaN (window boundaries)")

        X = df.to_numpy().astype(self.config.dtype)

        # Clip at 5σ to prevent outliers from dominating dictionary atoms
        for j in range(X.shape[1]):
            col = X[:, j]
            mu, sigma = np.nanmean(col), np.nanstd(col)
            if sigma > 0:
                lo, hi = mu - 5 * sigma, mu + 5 * sigma
                X[:, j] = np.clip(col, lo, hi)

        # Per-column Z-score standardization
        if self.config.standardize:
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)
            print(f"[MatrixBuilder] Standardized (Z-score per column)")

        print(f"[MatrixBuilder] Final matrix: {X.shape}  "
              f"dtype={X.dtype}  memory={X.nbytes / 1024**2:.1f} MB")
        return X

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Reverse standardization (back to original units)."""
        if self.scaler_ is None:
            raise RuntimeError("No scaler fitted — call assemble() first.")
        return self.scaler_.inverse_transform(X_scaled)

    # ── Step 4: orchestrate ─────────────────────────────────────────

    def assemble(self) -> Tuple[np.ndarray, pl.DataFrame]:
        """
        Run the full pipeline: load → derive → clean → output.

        Returns
        -------
        X    : (N, M) np.ndarray  — feature matrix for dictionary learning
        meta : pl.DataFrame       — the cleaned feature DataFrame (optional)
        """
        df = self.load_raw()
        feats = self.compute_features(df)
        self._df = feats   # keep for inspection
        self._X = self.to_matrix(feats)
        return self._X, self._df

    # ── Properties ──────────────────────────────────────────────────

    @property
    def X(self) -> Optional[np.ndarray]:
        return self._X

    @property
    def df(self) -> Optional[pl.DataFrame]:
        return self._df

    def feature_report(self) -> None:
        """Print a summary of each feature column."""
        if self._df is None:
            print("[MatrixBuilder] No data yet — call assemble() first.")
            return

        stats = self._df.describe()
        for name in FEATURE_NAMES:
            col_stats = stats[name]
            print(f"  {name:24s}  μ={col_stats['mean']:+.4e}  "
                  f"σ={col_stats['std']:.4e}  "
                  f"[{col_stats['min']:.4e}, {col_stats['max']:.4e}]")


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def create_matrix_builder(**kwargs) -> MatrixBuilder:
    """Factory: create a MatrixBuilder with optional config overrides."""
    config = MatrixBuilderConfig(**kwargs)
    return MatrixBuilder(config)


# ═══════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    builder = MatrixBuilder()
    X, feats = builder.assemble()
    print(f"\nFeature matrix: {X.shape}")
    print(f"Feature groups: {FEATURE_GROUPS}")
    builder.feature_report()

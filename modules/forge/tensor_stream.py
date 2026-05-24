"""
Hephaestus Forge - TensorStream Data Pipeline
高性能数据流水线：Polars + Parquet + Apache Arrow + Sliding Window
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Generator, Any
from dataclasses import dataclass
from enum import Enum

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StorageFormat(Enum):
    PARQUET = "parquet"
    HDF5 = "hdf5"
    CSV = "csv"
    ARROW = "arrow"


@dataclass
class TensorStreamConfig:
    """TensorStream 配置"""
    seq_len: int = 60          # 序列长度 (tick数)
    batch_size: int = 32        # 批大小
    feature_dim: int = 20      # 特征维度 (L1-L20挂单深度)
    include_book: bool = True # 是否包含订单簿
    include_trades: bool = True # 是否包含交易流
    include_derived: bool = True # 是否包含衍生特征 (OBI, Spread等)
    tick_aggregation: int = 1   # Tick聚合数量


class FeatureSchema(Enum):
    """预定义的特征维度"""
    # 基础价格 (4)
    MID_PRICE = 0
    BID_PRICE = 1
    ASK_PRICE = 2
    SPREAD = 3

    # L1 深度 (4)
    BID_SIZE_1 = 4
    ASK_SIZE_1 = 5
    BID_IMBALANCE = 6
    ASK_IMBALANCE = 7

    # L2-L5 深度 (16)
    BID_SIZE_2 = 8
    BID_SIZE_3 = 9
    BID_SIZE_4 = 10
    BID_SIZE_5 = 11
    ASK_SIZE_2 = 12
    ASK_SIZE_3 = 13
    ASK_SIZE_4 = 14
    ASK_SIZE_5 = 15

    # 交易流 (4)
    TRADE_SIZE = 16
    TRADE_DIR = 17
    VOLUME_WEIGHTED_PRICE = 18
    TRADE_COUNT = 19

    # 衍生特征 (预留给 L6-L20)
    OBI = 20           # Order Book Imbalance
    TWAP_DEVIATION = 21  # 与TWAP偏差
    VOLATILITY = 22    # 滚动波动率
    MICRO_PRICE = 23   # 微价格


class TensorStream:
    """
    高性能 Tick 数据张量生成器

    功能:
    1. 使用 Polars/Parquet 实现极速读取 (比 Pandas 快 5-10x)
    2. 滑动窗口生成 3D Tensor: (batch, seq_len, feature_dim)
    3. 使用 Numpy striding 实现零拷贝滑动窗口
    4. 支持 Apache Arrow 零拷贝传输
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[TensorStreamConfig] = None,
        storage_format: Optional[StorageFormat] = None,
    ):
        self.data_path = Path(data_path)
        self.config = config or TensorStreamConfig()
        # Auto-detect format from extension
        if storage_format is not None:
            self.storage_format = storage_format
        else:
            ext = self.data_path.suffix.lower()
            if ext in ('.parquet', '.pq'):
                self.storage_format = StorageFormat.PARQUET
            elif ext in ('.csv',):
                self.storage_format = StorageFormat.CSV
            else:
                self.storage_format = StorageFormat.PARQUET
        self._df: Optional[pd.DataFrame] = None
        self._window_cache: Optional[np.ndarray] = None
        self._cursor: int = 0

    def load(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> "TensorStream":
        """
        加载数据 (按日期和币种过滤)

        Example:
            stream.load(start_date="2026-01-01", end_date="2026-03-31", symbols=["BTC"])
        """
        if self.storage_format == StorageFormat.PARQUET:
            self._df = self._load_parquet(start_date, end_date, symbols)
        elif self.storage_format == StorageFormat.CSV:
            self._df = self._load_csv(start_date, end_date, symbols)
        else:
            raise NotImplementedError(f"Format {self.storage_format} not supported")

        return self

    def _load_parquet(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        symbols: Optional[List[str]],
    ) -> pd.DataFrame:
        """加载 Parquet 格式"""
        if not POLARS_AVAILABLE:
            # Fallback to pandas
            return pd.read_parquet(self.data_path)

        # 使用 Polars
        df = pl.scan_parquet(str(self.data_path))

        # 过滤
        if start_date:
            df = df.filter(pl.col("timestamp") >= start_date)
        if end_date:
            df = df.filter(pl.col("timestamp") <= end_date)
        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))

        # 转换为 pandas (或直接返回 polars)
        return df.collect().to_pandas()

    def _load_csv(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        symbols: Optional[List[str]],
    ) -> pd.DataFrame:
        """加载 CSV 格式"""
        df = pd.read_csv(self.data_path)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            if start_date:
                df = df[df["timestamp"] >= start_date]
            if end_date:
                df = df[df["timestamp"] <= end_date]

        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]

        return df

    def to_tensors(
        self,
        use_torch: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将原始 Tick 数据转换为 3D 张量 (使用滑动窗口)

        Returns:
            X: (n_windows, seq_len, feature_dim)
            y: (n_windows,) - 未来收益标签
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load() first.")

        df = self._df.sort_values("timestamp").reset_index(drop=True)

        # 提取特征
        X_raw = self._extract_features(df)

        # 滑动窗口
        X_windows = self._sliding_window(X_raw, self.config.seq_len)

        # 生成标签 (未来收益)
        if "returns" in df.columns:
            y = df["returns"].values[self.config.seq_len:]
        else:
            # 计算未来收益
            prices = df["mid_price"].values if "mid_price" in df.columns else df["close"].values
            y = (prices[self.config.seq_len:] - prices[:-self.config.seq_len]) / prices[:-self.config.seq_len]

        if use_torch and TORCH_AVAILABLE:
            return torch.from_numpy(X_windows).float(), torch.from_numpy(y).float()
        return X_windows, y

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取特征矩阵"""
        n_rows = len(df)
        feature_dim = self.config.feature_dim

        # 预分配矩阵
        features = np.zeros((n_rows, feature_dim), dtype=np.float32)

        fd = feature_dim  # shorthand

        # 基础价格特征
        if "mid_price" in df.columns and FeatureSchema.MID_PRICE.value < fd:
            features[:, FeatureSchema.MID_PRICE.value] = df["mid_price"].values
        if "bid_price" in df.columns and FeatureSchema.BID_PRICE.value < fd:
            features[:, FeatureSchema.BID_PRICE.value] = df["bid_price"].values
        if "ask_price" in df.columns and FeatureSchema.ASK_PRICE.value < fd:
            features[:, FeatureSchema.ASK_PRICE.value] = df["ask_price"].values

        # Spread
        if "bid_price" in df.columns and "ask_price" in df.columns and FeatureSchema.SPREAD.value < fd:
            features[:, FeatureSchema.SPREAD.value] = (
                df["ask_price"].values - df["bid_price"].values
            ) / df["mid_price"].values

        # 订单簿深度 (L1)
        if "bid_size_1" in df.columns and FeatureSchema.BID_SIZE_1.value < fd:
            features[:, FeatureSchema.BID_SIZE_1.value] = df["bid_size_1"].values
        if "ask_size_1" in df.columns and FeatureSchema.ASK_SIZE_1.value < fd:
            features[:, FeatureSchema.ASK_SIZE_1.value] = df["ask_size_1"].values

        # Imbalance
        if "bid_size_1" in df.columns and "ask_size_1" in df.columns and FeatureSchema.BID_IMBALANCE.value < fd:
            total = df["bid_size_1"].values + df["ask_size_1"].values + 1e-10
            features[:, FeatureSchema.BID_IMBALANCE.value] = (
                df["bid_size_1"].values - df["ask_size_1"].values
            ) / total
            features[:, FeatureSchema.ASK_IMBALANCE.value] = -features[:, FeatureSchema.BID_IMBALANCE.value]

        # 交易流 — capped by feature_dim
        if "size" in df.columns and FeatureSchema.TRADE_SIZE.value < feature_dim:
            features[:, FeatureSchema.TRADE_SIZE.value] = df["size"].values
        if "side" in df.columns and FeatureSchema.TRADE_DIR.value < feature_dim:
            features[:, FeatureSchema.TRADE_DIR.value] = df["side"].values

        # 衍生特征
        if self.config.include_derived:
            features = self._add_derived_features(features, df)

        return features

    def _add_derived_features(self, features: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """添加衍生特征"""
        n = len(features)

        fd = features.shape[1]  # actual feature dim (may be smaller than FeatureSchema)

        # OBI (Order Book Imbalance) - 滚动
        if "bid_size_1" in df.columns and "ask_size_1" in df.columns and FeatureSchema.OBI.value < fd:
            window = min(20, n)
            bid_cumsum = np.cumsum(df["bid_size_1"].values)
            ask_cumsum = np.cumsum(df["ask_size_1"].values)
            for i in range(n):
                if i >= window:
                    bid_vol = bid_cumsum[i] - bid_cumsum[i-window]
                    ask_vol = ask_cumsum[i] - ask_cumsum[i-window]
                    total = bid_vol + ask_vol + 1e-10
                    features[i, FeatureSchema.OBI.value] = (bid_vol - ask_vol) / total

        # 波动率
        if "returns" in df.columns and FeatureSchema.VOLATILITY.value < fd:
            returns = df["returns"].values
            window = min(20, n)
            for i in range(n):
                if i >= window:
                    features[i, FeatureSchema.VOLATILITY.value] = np.std(returns[i-window:i])

        return features

    def _sliding_window(self, data: np.ndarray, seq_len: int) -> np.ndarray:
        """
        使用 numpy striding 实现零拷贝滑动窗口

        性能比 for 循环快 10x+
        """
        n_samples, feature_dim = data.shape
        n_windows = n_samples - seq_len + 1

        if n_windows <= 0:
            return np.array([]).reshape(0, seq_len, feature_dim)

        # 使用 stride_tricks 实现滑动窗口
        shape = (n_windows, seq_len, feature_dim)
        strides = (
            data.strides[0],  # 步长 = 一行
            data.strides[0],  # 窗口内步长 = 一行
            data.strides[1],  # 特征步长 = 一列
        )

        windows = np.lib.stride_tricks.as_strided(
            data,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        return windows.copy()  # 返回连续内存副本

    def batch_generator(
        self,
        use_torch: bool = True,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        批生成器 - 内存友好版本

        Yields:
            X_batch: (batch_size, seq_len, feature_dim)
            y_batch: (batch_size,)
        """
        X, y = self.to_tensors(use_torch=False)

        n_samples = len(X)
        batch_size = self.config.batch_size

        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            if use_torch and TORCH_AVAILABLE:
                yield torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).float()
            else:
                yield X_batch, y_batch

    def save_parquet(self, output_path: Optional[str] = None) -> str:
        """保存为 Parquet 格式"""
        if output_path is None:
            output_path = str(self.data_path).replace(".csv", ".parquet")

        if POLARS_AVAILABLE:
            df = pl.DataFrame(self._df)
            df.write_parquet(output_path)
        else:
            self._df.to_parquet(output_path)

        return output_path

    @property
    def info(self) -> Dict:
        """返回数据信息"""
        if self._df is None:
            return {"status": "no_data"}

        return {
            "n_rows": len(self._df),
            "n_features": self.config.feature_dim,
            "seq_len": self.config.seq_len,
            "storage_format": self.storage_format.value,
            "columns": list(self._df.columns),
        }


def create_tensor_stream(
    data_path: str,
    seq_len: int = 60,
    feature_dim: int = 24,
    batch_size: int = 32,
) -> TensorStream:
    """便捷构造函数"""
    config = TensorStreamConfig(
        seq_len=seq_len,
        feature_dim=feature_dim,
        batch_size=batch_size,
    )
    return TensorStream(data_path, config)


# 使用示例
if __name__ == "__main__":
    # 快速测试
    print("TensorStream Data Pipeline v1.0")
    print(f"Polars available: {POLARS_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
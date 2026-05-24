"""
Alpha Factory - 因子工厂插件系统
所有因子必须继承 BaseAlpha 类，实现 compute() 方法
"""
from __future__ import annotations

import os
import re
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


@dataclass
class AlphaConfig:
    """因子配置"""
    name: str
    description: str = ""
    params: Dict[str, Any] = None  # 超参数
    lookback: int = 20  # 回看窗口
    normalize: bool = True  # 是否归一化
    decay: Optional[float] = None  # 指数衰减因子


class BaseAlpha(ABC):
    """
    因子基类 - 所有自定义因子必须继承此类

    Example:
        class MyAlpha(BaseAlpha):
            def compute(self, data) -> np.ndarray:
                # 返回形状为 (Time, Assets) 的信号矩阵
                return signals
    """

    def __init__(self, config: AlphaConfig):
        self.config = config
        self.name = config.name
        self.params = config.params or {}

    @abstractmethod
    def compute(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        计算因子值

        Args:
            data: 输入数据 (Time x Assets)

        Returns:
            信号矩阵 (Time x Assets)
        """
        pass

    def __call__(self, data) -> np.ndarray:
        return self.compute(data)

    def normalize_output(self, signals: np.ndarray) -> np.ndarray:
        """归一化输出"""
        if not self.config.normalize:
            return signals

        # 横截面归一化: (x - mean) / std
        mean = np.nanmean(signals, axis=1, keepdims=True)
        std = np.nanstd(signals, axis=1, keepdims=True)
        std[std == 0] = 1  # 避免除零

        return (signals - mean) / std

    def apply_decay(self, signals: np.ndarray) -> np.ndarray:
        """应用指数衰减"""
        if self.config.decay is None:
            return signals

        # 简化的指数衰减
        decay_factor = self.config.decay
        n = signals.shape[0]

        for i in range(1, n):
            signals[i] = signals[i] * decay_factor + signals[i-1] * (1 - decay_factor)

        return signals


class AlphaFactory:
    """
    因子工厂 - 自动扫描、加载和管理因子

    功能:
    1. 扫描 alphas/ 目录下的所有因子
    2. 自动实例化因子
    3. 计算并入库
    """

    def __init__(self, alphas_dir: Optional[str] = None):
        if alphas_dir is None:
            alphas_dir = Path(__file__).parent.parent / "alphas"
        self.alphas_dir = Path(alphas_dir)
        self.alphas: Dict[str, BaseAlpha] = {}
        self._registry: Dict[str, type] = {}

    def scan(self) -> List[str]:
        """扫描并加载目录下的所有因子"""
        if not self.alphas_dir.exists():
            return []

        loaded = []
        for file in self.alphas_dir.glob("*.py"):
            if file.stem.startswith("_"):
                continue

            # 动态导入
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 查找 BaseAlpha 子类
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BaseAlpha) and
                        obj is not BaseAlpha):
                        self._registry[name] = obj
                        loaded.append(name)
            except Exception as e:
                print(f"Failed to load {file}: {e}")

        return loaded

    def register(self, name: str, alpha_class: type) -> None:
        """手动注册因子"""
        if issubclass(alpha_class, BaseAlpha):
            self._registry[name] = alpha_class

    def create(self, name: str, params: Optional[Dict] = None) -> BaseAlpha:
        """创建因子实例"""
        if name not in self._registry:
            raise ValueError(f"Alpha {name} not found. Available: {list(self._registry.keys())}")

        config = AlphaConfig(
            name=name,
            params=params or {},
        )
        return self._registry[name](config)

    def compute(
        self,
        name: str,
        data: Union[pd.DataFrame, np.ndarray],
        params: Optional[Dict] = None,
    ) -> np.ndarray:
        """计算单个因子"""
        alpha = self.create(name, params)
        return alpha.compute(data)

    def compute_all(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        alpha_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """计算所有因子并返回 DataFrame"""
        if alpha_names is None:
            alpha_names = list(self._registry.keys())

        results = {}
        for name in alpha_names:
            try:
                results[name] = self.compute(name, data)
            except Exception as e:
                print(f"Error computing {name}: {e}")

        return pd.DataFrame(results)

    def list_alphas(self) -> List[str]:
        """列出所有可用因子"""
        return list(self._registry.keys())


# 内置因子示例
class MomentumAlpha(BaseAlpha):
    """动量因子"""

    def compute(self, data) -> np.ndarray:
        # 假设 data 有 'returns' 列
        if isinstance(data, pd.DataFrame):
            if "returns" not in data.columns:
                # 尝试计算
                prices = data.get("close", data.get("mid_price"))
                if prices is not None:
                    returns = prices.pct_change()
                else:
                    return np.zeros(len(data))
            else:
                returns = data["returns"]
            return self.normalize_output(returns.rolling(self.config.lookback).sum().values)
        else:
            # numpy 数组
            return self.normalize_output(
                np.nansum(data[-self.config.lookback:], axis=-1)
            )


class ReversalAlpha(BaseAlpha):
    """反转因子"""

    def compute(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            returns = data["returns"] if "returns" in data.columns else 0
            return self.normalize_output(-returns.rolling(self.config.lookback).sum().values)
        return np.zeros_like(data)


class SpreadAlpha(BaseAlpha):
    """价差因子"""

    def compute(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            bid = data.get("bid_price", data.get("mid_price") * 0.999)
            ask = data.get("ask_price", data.get("mid_price") * 1.001)
            mid = data.get("mid_price")
            if mid is not None:
                spread = (ask - bid) / mid
                return self.normalize_output(spread.rolling(self.config.lookback).mean().values)
        return np.zeros_like(data) if isinstance(data, np.ndarray) else np.zeros(len(data))


class OBIMAlpha(BaseAlpha):
    """订单簿失衡因子"""

    def compute(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            bid_size = data.get("bid_size_1", data.get("size"))
            ask_size = data.get("ask_size_1", data.get("size"))
            if bid_size is not None and ask_size is not None:
                total = bid_size + ask_size + 1e-10
                return self.normalize_output(
                    (bid_size - ask_size) / total
                )
        return np.zeros_like(data) if isinstance(data, np.ndarray) else np.zeros(len(data))


# 默认注册
DEFAULT_ALPHAS = {
    "momentum": MomentumAlpha,
    "reversal": ReversalAlpha,
    "spread": SpreadAlpha,
    "OBI": OBIMAlpha,
}


def create_factory(alphas_dir: Optional[str] = None) -> AlphaFactory:
    """创建因子工厂"""
    factory = AlphaFactory(alphas_dir)

    # 注册内置因子
    for name, alpha_class in DEFAULT_ALPHAS.items():
        factory.register(name, alpha_class)

    # 扫描自定义因子
    factory.scan()

    return factory
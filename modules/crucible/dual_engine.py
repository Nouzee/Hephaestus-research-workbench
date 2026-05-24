"""
Dual-Engine Backtester - 双引擎回测系统
1. Vectorized Engine: 纯 NumPy/Polars 矩阵运算，快速初筛
2. Event-Driven Engine: 模拟 L2 订单簿流，HFT 精度
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class EngineType(Enum):
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"


@dataclass
class BacktestConfig:
    """回测配置"""
    engine_type: EngineType = EngineType.VECTORIZED
    start_date: str = "2026-01-01"
    end_date: str = "2026-12-31"
    initial_capital: float = 100_000.0
    commission: float = 0.0004  # 手续费 (0.04%)
    slippage: float = 0.0001    # 滑点 (0.01%)
    maker_rebate: float = 0.0002   # Maker 回扣
    latency_ms: int = 0           # 延迟模拟 (ms)
    max_position: float = 1.0       # 最大仓位
    verbose: bool = False


@dataclass
class Signal:
    """交易信号"""
    timestamp: Any
    symbol: str
    direction: int       # 1=long, -1=short, 0=flat
    size: float
    price: float
    reason: str = ""


@dataclass
class Position:
    """持仓"""
    symbol: str
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    turnover: float
    n_trades: int
    win_rate: float
    avg_holding_period: float
    equity_curve: np.ndarray
    trades: List[Dict]


def _compute_common_metrics(
    equity: np.ndarray,
    trades: List[Dict],
    initial_capital: float,
) -> dict:
    """共享指标计算 — 消除向量化引擎和事件驱动引擎之间的重复代码"""
    if len(equity) < 2:
        return dict(total_return=0, sharpe_ratio=0, calmar_ratio=0,
                    max_drawdown=0, turnover=0, n_trades=0, win_rate=0)

    total_return = (equity[-1] - equity[0]) / equity[0]
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390) if np.std(returns) > 0 else 0

    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown)
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    total_volume = sum(abs(t.get("size", 0)) * t.get("price", 0) for t in trades)
    turnover = total_volume / initial_capital if initial_capital > 0 else 0

    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    win_rate = wins / len(trades) if trades else 0

    return dict(
        total_return=total_return,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        turnover=turnover,
        n_trades=len(trades),
        win_rate=win_rate,
    )


class BaseStrategy(ABC):
    """策略基类"""

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        positions: Dict[str, Position],
    ) -> List[Signal]:
        """生成交易信号"""
        pass

    @abstractmethod
    def on_bar(self, bar: pd.Series, positions: Dict[str, Position]) -> Optional[Signal]:
        """逐 bar 处理 (事件驱动)"""
        pass


class VectorizedEngine:
    """
    向量化引擎 - 纯矩阵运算，极速筛选

    适用于:
    - 日级/分钟级因子回测
    - 大批量因子初筛
    - 不考虑订单簿细节
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.capital = config.initial_capital

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> BacktestResult:
        """运行回测"""
        start_time = time.time()

        data = data.sort_values("timestamp").reset_index(drop=True)

        # 向量化计算信号
        signals = self._vectorized_signals(data, strategy)

        # 模拟执行
        equity = [self.capital]
        for i, row in data.iterrows():
            if i in signals:
                sig = signals[i]
                self._execute(sig, equity[-1])

            # 更新权益
            equity.append(self._calculate_equity(row, equity[-1]))

        self.equity_curve = np.array(equity[:-1])
        metrics = _compute_common_metrics(self.equity_curve, self.trades, self.config.initial_capital)
        result = BacktestResult(avg_holding_period=0, equity_curve=self.equity_curve, trades=self.trades, **metrics)

        if self.config.verbose:
            print(f"Vectorized engine: {time.time() - start_time:.2f}s")

        return result

    def _vectorized_signals(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> Dict[int, Signal]:
        """向量化生成信号"""
        # 使用策略的向量化方法
        if hasattr(strategy, "compute_signals"):
            return strategy.compute_signals(data)
        return {}

    def _execute(self, signal: Signal, capital: float) -> None:
        """执行交易"""
        cost = signal.size * signal.price * (1 + self.config.slippage)
        self.trades.append({
            "timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "size": signal.size,
            "price": signal.price,
            "cost": cost,
        })

    def _calculate_equity(self, bar: pd.Series, capital: float) -> float:
        return capital  # 简化 — vectorized engines use direct PnL accounting


class EventDrivenEngine:
    """
    事件驱动引擎 - 模拟 L2 订单簿流

    适用于:
    - HFT 策略 (Tick 级)
    - 需要订单簿深度
    - 考虑延迟和滑点
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Tuple[Any, float]] = []
        self.current_time = None

    def run(
        self,
        data: pd.DataFrame,
        orderbook: Optional[pd.DataFrame],
        strategy: BaseStrategy,
    ) -> BacktestResult:
        """运行回测"""
        start_time = time.time()

        data = data.sort_values("timestamp").reset_index(drop=True)

        for idx, row in data.iterrows():
            self.current_time = row["timestamp"]

            # 获取当前订单簿状态
            ob_state = self._get_orderbook_state(orderbook, idx) if orderbook is not None else None

            # 生成信号
            for symbol, pos in self.positions.items():
                signal = strategy.on_bar(row, self.positions)
                if signal:
                    self._submit_order(signal, ob_state)

            # 处理待成交订单
            self._process_pending_orders(ob_state)

            # 更新持仓
            self._update_positions(row)

            # 记录权益
            equity = self._calculate_equity(row)
            self.equity_curve.append((self.current_time, equity))

        equity_arr = np.array([e[1] for e in self.equity_curve])
        metrics = _compute_common_metrics(equity_arr, self.trades, self.config.initial_capital)
        result = BacktestResult(avg_holding_period=0, equity_curve=equity_arr, trades=self.trades, **metrics)

        if self.config.verbose:
            print(f"Event-driven engine: {time.time() - start_time:.2f}s")

        return result

    def _get_orderbook_state(
        self,
        orderbook: pd.DataFrame,
        idx: int,
    ) -> Dict:
        """获取订单簿状态"""
        if idx >= len(orderbook):
            return {}

        row = orderbook.iloc[idx]
        return {
            "bid_prices": [row.get(f"bid_price_{i}") for i in range(1, 6)],
            "bid_sizes": [row.get(f"bid_size_{i}") for i in range(1, 6)],
            "ask_prices": [row.get(f"ask_price_{i}") for i in range(1, 6)],
            "ask_sizes": [row.get(f"ask_size_{i}") for i in range(1, 6)],
            "spread": row.get("spread", 0),
            "mid_price": row.get("mid_price", 0),
        }

    def _submit_order(self, signal: Signal, ob_state: Dict) -> None:
        """提交订单 — 使用时间戳模拟延迟 (no actual sleep)"""
        effective_time = self.current_time
        if self.config.latency_ms > 0:
            import pandas as pd
            effective_time = pd.Timestamp(self.current_time) + pd.Timedelta(ms=self.config.latency_ms)

        self.pending_orders.append({
            "signal": signal,
            "ob_state": ob_state,
            "submit_time": self.current_time,
            "effective_time": effective_time,
        })

    def _process_pending_orders(self, ob_state: Dict) -> None:
        """处理待成交订单"""
        filled = []

        for order in self.pending_orders:
            sig = order["signal"]
            ob = order["ob_state"]

            # 模拟成交逻辑
            filled_price = self._calculate_fill_price(sig, ob)

            self.trades.append({
                "timestamp": self.current_time,
                "symbol": sig.symbol,
                "direction": sig.direction,
                "size": sig.size,
                "price": filled_price,
            })

            filled.append(order)

        # 移除已成交
        for f in filled:
            self.pending_orders.remove(f)

    def _calculate_fill_price(self, signal: Signal, ob_state: Dict) -> float:
        """计算成交价格"""
        if not ob_state:
            return signal.price

        # 简化: 使用对手盘第一档 + 滑点
        if signal.direction == 1:  # 买入
            price = ob_state.get("ask_prices", [signal.price])[0]
            price *= (1 + self.config.slippage)
        else:  # 卖出
            price = ob_state.get("bid_prices", [signal.price])[0]
            price *= (1 - self.config.slippage)

        return price

    def _update_positions(self, bar: pd.Series) -> None:
        """更新持仓"""
        for symbol, pos in self.positions.items():
            if pos.size != 0:
                # 更新未实现盈亏
                if "mid_price" in bar:
                    pos.unrealized_pnl = pos.size * (bar["mid_price"] - pos.entry_price)

    def _calculate_equity(self, bar: pd.Series) -> float:
        """计算权益"""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        realized = sum(p.realized_pnl for p in self.positions.values())
        return self.config.initial_capital + unrealized + realized



class DualEngine:
    """
    双引擎管理器

    Example:
        engine = DualEngine(config)
        result = engine.run(data, strategy, engine_type="vectorized")
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        orderbook: Optional[pd.DataFrame] = None,
        engine_type: str = "vectorized",
    ) -> BacktestResult:
        """运行回测"""
        if engine_type == "vectorized":
            engine = VectorizedEngine(self.config)
            return engine.run(data, strategy)
        else:
            engine = EventDrivenEngine(self.config)
            return engine.run(data, orderbook, strategy)

    def run_both(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        orderbook: Optional[pd.DataFrame] = None,
    ) -> Tuple[BacktestResult, BacktestResult]:
        """同时运行两个引擎，对比结果"""
        vec_result = self.run(data, strategy, engine_type="vectorized")
        evt_result = self.run(data, strategy, orderbook, engine_type="event_driven")
        return vec_result, evt_result


def create_engine(
    engine_type: str = "vectorized",
    initial_capital: float = 100_000.0,
    commission: float = 0.0004,
    **kwargs,
) -> DualEngine:
    """创建引擎"""
    config = BacktestConfig(
        engine_type=EngineType.VECTORIZED if engine_type == "vectorized" else EngineType.EVENT_DRIVEN,
        initial_capital=initial_capital,
        commission=commission,
        **kwargs,
    )
    return DualEngine(config)
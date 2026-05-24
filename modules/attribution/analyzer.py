"""
Attribution Module - 归因分析工坊
Markout 计算、Fill Probability、Turnover 合规检查、Markdown 报告生成
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING_LOW = "warning_low"
    WARNING_HIGH = "warning_high"
    VIOLATION = "violation"


@dataclass
class MarkoutResult:
    """Markout 计算结果"""
    window: int
    mean: float
    std: float
    median: float
    positive_rate: float
    adverse_selection_bps: float


@dataclass
class FillResult:
    """成交概率结果"""
    overall_rate: float
    buy_rate: float
    sell_rate: float
    n_filled: int
    n_total: int


@dataclass
class TurnoverResult:
    """换手率结果"""
    turnover_ratio: float
    avg_position: float
    total_volume: float
    status: str
    freq_trades_per_min: float


@dataclass
class DiagnosticResult:
    """综合诊断结果"""
    is_poisoned: bool
    risk_factors: List[str]
    warnings: List[str]
    recommendations: List[str]


class MarkoutCalculator:
    """
    Markout 计算器 - 逆向选择监测

    功能:
    1. 计算 t+1, t+5, t+30, t+60 窗口的价格偏移
    2. 分析逆向选择成本 (Adverse Selection)
    3. 识别"毒性"市场条件
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [1, 5, 30, 60]

    def calculate(
        self,
        trades_df: pd.DataFrame,
        side_col: str = "side",
        price_col: str = "price",
        timestamp_col: str = "timestamp",
        size_col: str = "size",
    ) -> Dict[str, MarkoutResult]:
        """计算 Markout"""
        if trades_df.empty or len(trades_df) < max(self.windows):
            return {}

        trades = trades_df.sort_values(timestamp_col).reset_index(drop=True)

        results = {}
        prices = trades[price_col].values
        sides = trades[side_col].values
        sizes = trades[size_col].values if size_col in trades.columns else np.ones(len(trades))

        for window in self.windows:
            if len(prices) <= window:
                continue

            # 价格变化 (bps)
            prices_now = prices[:-window]
            prices_future = prices[window:]
            price_changes = (prices_future - prices_now) / prices_now * 10000

            # 逆向选择: 订单方向 * 价格变化
            # 买入后价格上涨 = 负向选择 (坏)
            # 卖出后价格下跌 = 正向选择 (好)
            adsel = sides[:-window] * price_changes

            results[f"markout_{window}"] = MarkoutResult(
                window=window,
                mean=np.mean(price_changes),
                std=np.std(price_changes),
                median=np.median(price_changes),
                positive_rate=np.mean(price_changes > 0),
                adverse_selection_bps=np.mean(adsel),
            )

        return results

    def weighted_adverse_selection(
        self,
        trades_df: pd.DataFrame,
    ) -> float:
        """
        加权逆向选择成本

        公式: AS = sum(sign * size * price_change) / sum(|size|)
        """
        if trades_df.empty:
            return 0.0

        trades = trades_df.sort_values("timestamp").reset_index(drop=True)

        sizes = trades["size"].values
        sides = trades["side"].values
        prices = trades["price"].values

        weighted_as = 0.0
        total_volume = 0.0

        for i in range(len(trades) - 1):
            price_change = (prices[i+1] - prices[i]) / prices[i]
            adsel = sides[i] * price_change * sizes[i]
            weighted_as += adsel
            total_volume += abs(sizes[i])

        if total_volume > 0:
            return weighted_as / total_volume * 10000  # bps
        return 0.0


class FillProbabilityAnalyzer:
    """
    成交概率分析器

    功能:
    1. 计算整体成交率
    2. 按买卖方向分组
    3. 检测流动性枯竭
    """

    def calculate(
        self,
        trades_df: pd.DataFrame,
        is_filled_col: str = "is_filled",
        side_col: str = "side",
    ) -> FillResult:
        """计算成交概率"""
        if is_filled_col not in trades_df.columns:
            return FillResult(0.85, 0.85, 0.85, 0, 0)

        n_total = len(trades_df)
        n_filled = trades_df[is_filled_col].sum()
        fill_rate = n_filled / n_total if n_total > 0 else 0

        # 按方向分组
        if side_col in trades_df.columns:
            buy_trades = trades_df[trades_df[side_col] == 1]
            sell_trades = trades_df[trades_df[side_col] == -1]

            buy_rate = buy_trades[is_filled_col].sum() / len(buy_trades) if len(buy_trades) > 0 else 0
            sell_rate = sell_trades[is_filled_col].sum() / len(sell_trades) if len(sell_trades) > 0 else 0
        else:
            buy_rate = sell_rate = fill_rate

        return FillResult(
            overall_rate=fill_rate,
            buy_rate=buy_rate,
            sell_rate=sell_rate,
            n_filled=n_filled,
            n_total=n_total,
        )


class TurnoverChecker:
    """
    换手率合规检查器

    硬约束: 500x - 15000x
    """

    MIN_TURNOVER = 500
    MAX_TURNOVER = 15000

    def calculate(
        self,
        trades_df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None,
        size_col: str = "size",
        price_col: str = "price",
    ) -> TurnoverResult:
        """计算换手率"""
        if trades_df.empty:
            return TurnoverResult(0, 0, 0, "no_data", 0)

        sizes = trades_df[size_col].values if size_col in trades_df.columns else np.ones(len(trades_df))
        prices = trades_df[price_col].values if price_col in trades_df.columns else np.ones(len(trades_df))

        total_volume = np.sum(np.abs(sizes * prices))

        # 平均持仓
        avg_position = 1.0
        if positions_df is not None and "position" in positions_df.columns:
            avg_position = np.mean(np.abs(positions_df["position"].values))

        turnover_ratio = total_volume / avg_position if avg_position > 0 else 0

        # 状态判断
        if turnover_ratio < self.MIN_TURNOVER:
            status = "TOO_LOW"
        elif turnover_ratio > self.MAX_TURNOVER:
            status = "TOO_HIGH"
        else:
            status = "COMPLIANT"

        # 交易频率
        if "timestamp" in trades_df.columns:
            trades = trades_df.sort_values("timestamp")
            durations = (trades["timestamp"].diff().dt.total_seconds().fillna(1)).values
            durations = durations[durations > 0]
            avg_duration = np.mean(durations) if len(durations) > 0 else 60
            freq_trades_per_min = 60 / avg_duration if avg_duration > 0 else 0
        else:
            freq_trades_per_min = 0

        return TurnoverResult(
            turnover_ratio=turnover_ratio,
            avg_position=avg_position,
            total_volume=total_volume,
            status=status,
            freq_trades_per_min=freq_trades_per_min,
        )


class AttributionAnalyzer:
    """
    Hephaestus 归因分析器

    整合 Markout、Fill Probability、Turnover Compliance
    """

    def __init__(self):
        self.markout_calc = MarkoutCalculator()
        self.fill_analyzer = FillProbabilityAnalyzer()
        self.turnover_checker = TurnoverChecker()

    def analyze(
        self,
        trades_df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict, DiagnosticResult]:
        """
        运行完整归因分析

        Returns:
            (metrics_dict, diagnostic_result)
        """
        # Markout
        markout_results = self.markout_calc.calculate(trades_df)
        weighted_as = self.markout_calc.weighted_adverse_selection(trades_df)

        # Fill Probability
        fill_result = self.fill_analyzer.calculate(trades_df)

        # Turnover
        turnover_result = self.turnover_checker.calculate(trades_df, positions_df)

        # 整合结果
        metrics = {
            "n_trades": len(trades_df),
            "weighted_adverse_selection_bps": weighted_as,
            "fill_rate": fill_result.overall_rate,
            "buy_fill_rate": fill_result.buy_rate,
            "sell_fill_rate": fill_result.sell_rate,
            "turnover_ratio": turnover_result.turnover_ratio,
            "turnover_status": turnover_result.status,
            "trades_per_min": turnover_result.freq_trades_per_min,
        }

        # 诊断
        diagnostic = self._diagnose(metrics)

        return metrics, diagnostic

    def _diagnose(self, metrics: Dict) -> DiagnosticResult:
        """综合诊断"""
        is_poisoned = False
        risk_factors = []
        warnings = []
        recommendations = []

        # 逆向选择检查
        was = metrics.get("weighted_adverse_selection_bps", 0)
        if was < -5:
            risk_factors.append(f"严重逆向选择: {was:.2f} bps")
            is_poisoned = True
        elif was < 0:
            warnings.append(f"轻度逆向选择: {was:.2f} bps")

        # 换手率检查
        turnover = metrics.get("turnover_ratio", 0)
        status = metrics.get("turnover_status", "")
        if status in ["TOO_HIGH", "TOO_LOW"]:
            risk_factors.append(f"换手率异常: {status}")
            if status == "TOO_HIGH":
                is_poisoned = True

        # 成交率检查
        fill_rate = metrics.get("fill_rate", 0)
        if fill_rate < 0.5:
            warnings.append(f"成交率过低: {fill_rate:.1%}")

        # 建议
        if is_poisoned:
            recommendations.extend([
                "策略被特定市场条件毒化",
                "建议检查信号生成逻辑",
                "考虑添加波动率过滤",
            ])

        return DiagnosticResult(
            is_poisoned=is_poisoned,
            risk_factors=risk_factors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def generate_report(
        self,
        metrics: Dict,
        diagnostic: DiagnosticResult,
        output_path: Optional[str] = None,
    ) -> str:
        """生成 Markdown 报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        is_poisoned = diagnostic.is_poisoned

        report = f"""# Hephaestus Attribution Report
Generated: {timestamp}

## Summary
- **Total Trades**: {metrics.get('n_trades', 0):,}
- **Poisoned Status**: {'⚠️ YES' if is_poisoned else '✅ NO'}

---

## Markout Analysis (逆向选择)
- **Weighted Adverse Selection**: {metrics.get('weighted_adverse_selection_bps', 0):.4f} bps
- **Interpretation**: {'负向选择 > 5bps = 高风险' if metrics.get('weighted_adverse_selection_bps', 0) < -5 else '正常范围'}

---

## Fill Probability (成交概率)
- **Overall**: {metrics.get('fill_rate', 0):.2%}
- **Buy Side**: {metrics.get('buy_fill_rate', 0):.2%}
- **Sell Side**: {metrics.get('sell_fill_rate', 0):.2%}

---

## Turnover Compliance (换手率)
- **Turnover Ratio**: {metrics.get('turnover_ratio', 0):,.0f}x
- **Status**: {metrics.get('turnover_status', 'UNKNOWN')}
- **Frequency**: {metrics.get('trades_per_min', 0):.2f} trades/min
- **Threshold**: 500x - 15000x

---

## Diagnostics
### Risk Factors
{self._format_list(diagnostic.risk_factors)}

### Warnings
{self._format_list(diagnostic.warnings)}

### Recommendations
{self._format_list(diagnostic.recommendations)}

---

## Conclusion
{'## ⚠️ WARNING: Strategy is poisoned by specific market conditions' if is_poisoned else '## ✅ Strategy appears healthy'}

---
*Generated by Hephaestus Attribution Analyzer v1.0*
"""

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def _format_list(self, items: List[str]) -> str:
        if not items:
            return "- _None_"
        return "\n".join(f"- {item}" for item in items)


def create_analyzer() -> AttributionAnalyzer:
    """创建归因分析器"""
    return AttributionAnalyzer()
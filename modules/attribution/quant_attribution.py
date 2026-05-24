"""
Quant Thinking Attribution - 量化思维归因模块
强制将策略分解为: Hypothesis → Decision → Failure Mode

这是 quant 面试的核心能力:
- 不是"做了什么系统"
- 而是"市场不确定性如何被建模"
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd


class RegimeType(Enum):
    """市场状态"""
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    FLAT = "flat"
    HIGH_TOXICITY = "high_toxicity"  # 高毒性 (订单流)
    LOW_LIQUIDITY = "low_liquidity"


class PnLSource(Enum):
    """收益来源"""
    ALPHA = "alpha"  # 预测能力
    EXECUTION = "execution"  # 执行
    STRUCTURE = "structure"  # 结构收益 (如 maker rebate)
    RISK_PREMIUM = "risk_premium"  # 风险溢价
    IDLE = "idle"  # 躺赚


@dataclass
class Hypothesis:
    """
    标准 alpha hypothesis 格式:
    I believe X causes Y under Z regime.
    """
    driver: str           # X - 原因
    effect: str          # Y - 价格/收益变化
    regime: str          # Z - 成立条件
    confidence: float     # 置信度 0-1

    def __str__(self) -> str:
        return f"I believe {self.driver} causes {self.effect} under {self.regime} regimes."

    @classmethod
    def from_alpha_source(cls, source: str) -> "Hypothesis":
        """从 alpha 源码自动推断 hypothesis"""
        sources = {
            "micro_price": Hypothesis(
                driver="order_book_imbalance",
                effect="mid_price_reversion",
                regime="low_volatility",
                confidence=0.6,
            ),
            "inventory_skew": Hypothesis(
                driver="inventory_imbalance",
                effect="price_convergence",
                regime="mean_reversion",
                confidence=0.5,
            ),
            "OBI": Hypothesis(
                driver="order_book_imbalance_extreme",
                effect="liquidation_forced_execution",
                regime="high_volatility",
                confidence=0.7,
            ),
            "cooldown": Hypothesis(
                driver="adverse_selection_risk",
                effect="execution_quality_preservation",
                regime="high_toxicity",
                confidence=0.6,
            ),
        }
        return sources.get(source.lower(), Hypothesis(
            driver="unknown",
            effect="unknown",
            regime="any",
            confidence=0.0,
        ))


@dataclass
class DecisionRule:
    """
    标准决策规则:
    When do I trade / not trade?
    """
    entry_condition: str      # 入场条件
    exit_condition: str       # 出场条件
    size_rule: str           # 仓位规则
    filter_regimes: List[str] # 过滤的市场状态

    def should_trade(self, regime: str, signal: float) -> bool:
        """判断是否应该交易"""
        if regime in self.filter_regimes:
            return False
        if abs(signal) < 0.3:  # 假设信号阈值
            return False
        return True

    def __str__(self) -> str:
        return f"Entry: {self.entry_condition}, Exit: {self.exit_condition}"


@dataclass
class FailureMode:
    """
    标准失败模式:
    What breaks first?
    """
    primary_failure: str       # 首要失败点
    secondary_failures: List[str]  # 次要失败点
    recovery_approach: str   # 恢复方法

    def rank_failures(self) -> List[Tuple[str, float]]:
        """
        返回按概率排序的失败点 (name, probability)
        """
        failures = [(self.primary_failure, 0.5)]
        for f in self.secondary_failures:
            failures.append((f, 0.2 / len(self.secondary_failures)))
        return failures


@dataclass
class QuantAttributionResult:
    """
    归因结果 - 按 trader 框架组织
    """
    # Hypothesis
    hypothesis: Optional[Hypothesis] = None

    # Decision
    decision_rule: Optional[DecisionRule] = None

    # Failure Analysis
    failure_modes: List[FailureMode] = field(default_factory=list)

    # PnL Decomposition
    pnl_sources: Dict[str, float] = field(default_factory=dict)

    # Edge Analysis
    edge_exists: bool = False
    edge_source: str = ""
    edge_confidence: float = 0.0

    # Regime Analysis
    regime_performance: Dict[str, Dict] = field(default_factory=dict)


class QuantThinkingAttributor:
    """
    Quant 思维归因器

    把工程师思维翻译成 trader 思维:
    ❌ "我做了 micro-price + inventory skew + cooldown"
    ✅ "I believe order_book_imbalance drives mid_price_reversion under low_volatility regimes"
    """

    def __init__(self):
        self.results: List[QuantAttributionResult] = []

    def analyze(
        self,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        regime_indicators: Optional[pd.DataFrame] = None,
    ) -> QuantAttributionResult:
        """
        执行完整的 quant 思维归因
        """
        result = QuantAttributionResult()

        # 1. 提取 hypothesis (从策略参数)
        result.hypothesis = self._extract_hypothesis(trades_df, positions_df)

        # 2. 提取 decision rule
        result.decision_rule = self._extract_decision_rule(positions_df)

        # 3. 失败模式分析
        result.failure_modes = self._analyze_failure_modes(trades_df, positions_df)

        # 4. PnL 来源分解
        result.pnl_sources = self._decompose_pnl(trades_df, positions_df)

        # 5. Edge 分析
        result.edge_exists, result.edge_source, result.edge_confidence = \
            self._analyze_edge(trades_df, positions_df)

        # 6. Regime 表现
        result.regime_performance = self._analyze_regime_performance(
            trades_df, positions_df, regime_indicators
        )

        return result

    def _extract_hypothesis(
        self,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> Hypothesis:
        """
        从数据中推断当前的 hypothesis

        这是最关键的一步: 把工程转成 belief
        """
        # 分析 micro-price 预测能力
        if "micro_price_signal" in trades_df.columns:
            signal_col = "micro_price_signal"
        elif "imbalance" in trades_df.columns:
            signal_col = "imbalance"
        else:
            # 默认推断
            return Hypothesis(
                driver="micro_price",
                effect="mid_price_movement",
                regime="any",
                confidence=0.5,
            )

        # 计算预测能力 (简化)
        if signal_col in trades_df.columns and "returns" in trades_df.columns:
            corr = trades_df[signal_col].corr(trades_df["returns"])
            confidence = min(abs(corr) * 2, 1.0)  # 缩放
        else:
            confidence = 0.5

        # 分析驱动因素
        driver = "order_book_imbalance"  # 假设
        effect = "mid_price_reversion"   # 假设
        regime = "low_volatility"        # 假设

        return Hypothesis(
            driver=driver,
            effect=effect,
            regime=regime,
            confidence=confidence,
        )

    def _extract_decision_rule(self, positions_df: pd.DataFrame) -> DecisionRule:
        """提取决策规则"""
        # 从参数中推断
        return DecisionRule(
            entry_condition="abs(micro_price_signal) > threshold",
            exit_condition="cooldown_ms elapsed OR OBI_breach",
            size_rule="fixed_position * alpha_scaling",
            filter_regimes=["high_toxicity", "low_liquidity"],
        )

    def _analyze_failure_modes(
        self,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> List[FailureMode]:
        """
        分析失败模式 - 这是 quant 思维的核心
        """
        failures = []

        # Markout 分析 -> 逆向选择失败
        if "markout_100ms" in trades_df.columns:
            markout = trades_df["markout_100ms"].mean()
            if markout < -2.0:
                failures.append(FailureMode(
                    primary_failure="adverse_selection_regime_shift",
                    secondary_failures=[
                        "micro_price_signal_decay",
                        "execution_breakdown",
                    ],
                    recovery_approach="check if PnL drop correlates with volatility spike",
                ))
            else:
                failures.append(FailureMode(
                    primary_failure="execution_breakdown",
                    secondary_failures=["latency_increase", "queue_position_worsening"],
                    recovery_approach="measure execution latency trend",
                ))

        # 换手率 -> cooldown 太短
        if "turnover" in trades_df.columns:
            turnover = trades_df["turnover"].iloc[-1] if len(trades_df) > 0 else 0
            if turnover > 1500:
                failures.append(FailureMode(
                    primary_failure="over_trading",
                    secondary_failures=["signal_noise_amplification"],
                    recovery_approach="increase cooldown",
                ))

        # 默认失败模式
        if not failures:
            failures.append(FailureMode(
                primary_failure="unknown",
                secondary_failures=["regime_shift", "alpha_decay"],
                recovery_approach="backtest on OOS period",
            ))

        return failures

    def _decompose_pnl(
        self,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        分解 PnL 来源 - 不是"赚了多少钱"
        而是"收益从哪里来"
        """
        pnl = {}

        if "revenue_spread" in trades_df.columns:
            pnl["spread_capture"] = trades_df["revenue_spread"].sum()

        if "revenue_rebate" in trades_df.columns:
            pnl["maker_rebate"] = trades_df["revenue_rebate"].sum()

        if "loss_adverse_selection" in trades_df.columns:
            pnl["adverse_selection"] = trades_df["loss_adverse_selection"].sum()

        if "pnl" in trades_df.columns:
            pnl["total"] = trades_df["pnl"].sum()

        return pnl

    def _analyze_edge(
        self,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> Tuple[bool, str, float]:
        """
        分析是否真的存在 edge
        """
        # 检查 Sharpe
        if "returns" in trades_df.columns:
            returns = trades_df["returns"].dropna()
            if len(returns) > 30:
                mean_ret = returns.mean()
                std_ret = returns.std()

                if std_ret > 0:
                    sharpe = mean_ret / std_ret * np.sqrt(252 * 390)

                    if sharpe > 1.0:
                        return True, "alpha_predictive", 0.8
                    elif sharpe > 0:
                        return True, "weak_alpha", 0.4
                    else:
                        return False, "no_edge", 0.0

        return False, "not_analysis", 0.0

    def _analyze_regime_performance(
        self,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        regime_indicators: Optional[pd.DataFrame],
    ) -> Dict[str, Dict]:
        """
        按 regimes 分析表现
        """
        if regime_indicators is None or "regime" not in regime_indicators.columns:
            return {}

        merged = trades_df.merge(
            regime_indicators[["timestamp", "regime"]],
            on="timestamp",
            how="left"
        )

        performance = {}
        for regime in merged["regime"].unique():
            regime_trades = merged[merged["regime"] == regime]
            if len(regime_trades) > 0:
                performance[regime] = {
                    "n_trades": len(regime_trades),
                    "avg_pnl": regime_trades.get("pnl", 0).mean(),
                    "win_rate": (regime_trades.get("pnl", 0) > 0).mean(),
                }

        return performance

    def generate_report(
        self,
        result: QuantAttributionResult,
        output_path: Optional[str] = None,
    ) -> str:
        """生成 trader 风格的报告"""
        lines = [
            "# Quant Thinking Attribution Report",
            "",
            "## 1. Hypothesis (核心信念)",
            result.hypothesis and str(result.hypothesis) or "Not defined",
            "",
            "## 2. Decision Rule (何时交易)",
            result.decision_rule and str(result.decision_rule) or "Not defined",
            "",
            "## 3. Failure Analysis (哪里会先死)",
        ]

        for fm in result.failure_modes:
            lines.append(f"### Primary: {fm.primary_failure}")
            lines.append(f"Recovery: {fm.recovery_approach}")
            lines.append("")

        lines.extend([
            "## 4. PnL Decomposition",
            f"- Total: {result.pnl_sources.get('total', 0):.4f}",
            f"- Spread: {result.pnl_sources.get('spread_capture', 0):.4f}",
            f"- Rebate: {result.pnl_sources.get('maker_rebate', 0):.4f}",
            f"- Adverse Selection: {result.pnl_sources.get('adverse_selection', 0):.4f}",
            "",
            "## 5. Edge Analysis",
            f"- Edge Exists: {result.edge_exists}",
            f"- Edge Source: {result.edge_source}",
            f"- Confidence: {result.edge_confidence:.1%}",
            "",
            "## 6. Regime Performance",
        ])

        for regime, perf in result.regime_performance.items():
            lines.append(f"- {regime}: {perf}")

        lines.extend([
            "",
            "---",
            "### Trader 思维框架",
            "1. 有 hypothesis 吗? (不是工程是信念)",
            "2. 有 decision rule 吗? (何时交易何时不交易)",
            "3. 知道哪里会先死吗?",
        ])

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report

    def run_training_exercise(
        self,
        project_name: str,
        description: str,
    ) -> str:
        """
        训练模板 - 强制回答三个问题

        以后做每个项目都强制写这三行:
        """
        template = f"""
        ## 项目: {project_name}
        Description: {description}

        ### 训练模板 (必须完成)

        ### 1. Hypothesis
        I believe [X] causes [Y] under [Z] regimes.

        ### 2. Decision Rule
        - Entry: [when]
        - Exit: [when]
        - Size: [how much]

        ### 3. Failure Mode
        If performance collapses, the first suspect is:
        [ ] Alpha decay
        [ ] Execution breakdown
        [ ] Regime shift

        ---
        """
        return template


# ===== 训练例子 =====
# 用 BTC 项目改写

BTC_HFT_HYPOTHESIS = """
项目: BTC HFT Market Making

### 1. Hypothesis (从 README 改写)
I believe short-term order book imbalance causes predictable mid-price reversion
under stable volatility regimes.

### 2. Decision Rule
- Entry: abs(micro_price_signal) > 0.3 AND volatility is stable
- Exit: cooldown_ms elapsed OR |OBI| > 0.82
- Size: fixed_position * alpha_scaling * (1 - inventory_skew)

### 3. Failure Mode
If Sharpe drops from -48 to worse:
- First suspect: Regime shift in order flow toxicity
- The micro-price signal lost predictive power
- Execution became purely adversarial during high volatility

This is an EDGE problem, not parameter tuning problem.
"""
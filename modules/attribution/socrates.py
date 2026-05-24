"""
Socrates Agent — AI 助理集成
1. Alpha 翻译官: 根据研报数学描述生成 BaseAlpha 模板
2. 数学审查: 检测因子共线性 / 极端值漏洞
3. 归因解读: 自动生成归因自然语言摘要
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MathReviewResult:
    has_issues: bool
    warnings: List[str]
    suggestions: List[str]
    multicollinearity_alerts: List[str]
    outlier_alerts: List[str]


@dataclass
class AttributionNarrative:
    summary: str
    key_driver: str
    risk_exposure: str
    recommendation: str


class SocratesAgent:
    """
    Socrates AI — 量化研究助理

    三个核心能力:
    1. Alpha 翻译: 论文公式 -> Python 代码模板
    2. 数学审查: 检测共线性 / 极端值
    3. 归因解读: 数字 -> 自然语言
    """

    def __init__(self):
        self.templates: Dict[str, str] = {
            "momentum": self._momentum_template(),
            "reversal": self._reversal_template(),
            "imbalance": self._imbalance_template(),
        }

    # === Alpha Translation ===

    def translate_to_alpha(
        self,
        description: str,
        formula_hint: Optional[str] = None,
    ) -> str:
        """
        将研报数学描述转换为 BaseAlpha 模板

        Example:
            agent.translate_to_alpha(
                "5-period returns, skip most recent period",
                "R_t = (P_t - P_{t-5}) / P_{t-5}"
            )
        """
        keywords = description.lower()

        if "momentum" in keywords or "return" in keywords:
            template = self._momentum_template()
        elif "reversal" in keywords or "mean revert" in keywords:
            template = self._reversal_template()
        elif "imbalance" in keywords or "order book" in keywords:
            template = self._imbalance_template()
        elif "volatility" in keywords or "vol" in keywords:
            template = self._volatility_template()
        else:
            template = self._generic_template()

        # 替换参数
        if formula_hint:
            template = template.replace("{{formula}}", formula_hint)

        return template

    def _momentum_template(self) -> str:
        return '''
from modules.forge.base_alpha import BaseAlpha, AlphaConfig
import numpy as np

class MomentumAlpha(BaseAlpha):
    """Momentum: P_t / P_{t-N} - 1"""

    def compute(self, data):
        if hasattr(data, "columns"):  # DataFrame
            prices = data["mid_price"]
            returns = prices.pct_change(periods=self.config.lookback)
            return self.normalize_output(returns.values)
        else:  # numpy
            return self.normalize_output(data[-self.config.lookback:])
'''

    def _reversal_template(self) -> str:
        return '''
class ReversalAlpha(BaseAlpha):
    """Short-term reversal: -1 * short-term returns"""

    def compute(self, data):
        if hasattr(data, "columns"):
            returns = data["mid_price"].pct_change(periods=3)
            return self.normalize_output(-returns.values)
        return np.zeros_like(data)
'''

    def _imbalance_template(self) -> str:
        return '''
class OBIAlpha(BaseAlpha):
    """Order Book Imbalance: (bid - ask) / (bid + ask)"""

    def compute(self, data):
        if hasattr(data, "columns"):
            bid = data["bid_size_1"]
            ask = data["ask_size_1"]
            obi = (bid - ask) / (bid + ask + 1e-10)
            return self.normalize_output(obi.values)
        return np.zeros_like(data)
'''

    def _volatility_template(self) -> str:
        return '''
class VolatilityAlpha(BaseAlpha):
    """Rolling volatility as signal"""

    def compute(self, data):
        if hasattr(data, "columns"):
            rets = data["mid_price"].pct_change()
            vol = rets.rolling(self.config.lookback).std()
            return self.normalize_output(vol.values)
        return np.zeros_like(data)
'''

    def _generic_template(self) -> str:
        return '''
class CustomAlpha(BaseAlpha):
    """TODO: describe your alpha hypothesis"""

    def compute(self, data):
        # TODO: implement {{formula}}
        return self.normalize_output(np.zeros_like(data))
'''

    # === Math Review ===

    def review_factor_code(self, factor_values: np.ndarray, factor_names: Optional[List[str]] = None) -> MathReviewResult:
        """
        审查因子代码

        检测:
        1. 共线性 (相关系数 > 0.9)
        2. 极端值 (超过 5 倍标准差)
        3. 常值因子
        """
        warnings = []
        suggestions = []
        mc_alerts = []
        outlier_alerts = []

        if factor_values.ndim == 1:
            factor_values = factor_values.reshape(-1, 1)
        T, K = factor_values.shape

        if K < 2:
            return MathReviewResult(False, [], [], [], [])

        if factor_names is None:
            factor_names = [f"f{i}" for i in range(K)]

        # 共线性检测
        corr = np.corrcoef(factor_values.T)
        for i in range(K):
            for j in range(i+1, K):
                if abs(corr[i, j]) > 0.9:
                    mc_alerts.append(
                        f"{factor_names[i]} vs {factor_names[j]}: corr={corr[i,j]:.3f}"
                    )

        # 极端值检测
        for k in range(K):
            vals = factor_values[:, k]
            mean, std = np.mean(vals), np.std(vals)
            if std < 1e-10:
                warnings.append(f"{factor_names[k]}: near-constant factor")
                continue

            n_outliers = np.sum(np.abs(vals - mean) > 5 * std)
            if n_outliers > T * 0.05:
                outlier_alerts.append(
                    f"{factor_names[k]}: {n_outliers} extreme values ({n_outliers/T:.1%})"
                )

        has_issues = len(mc_alerts) > 0 or len(outlier_alerts) > 0

        if mc_alerts:
            suggestions.append("Consider PCA or ridge regression for collinear factors")
        if outlier_alerts:
            suggestions.append("Consider winsorization or robust scaling")

        return MathReviewResult(
            has_issues=has_issues,
            warnings=warnings,
            suggestions=suggestions,
            multicollinearity_alerts=mc_alerts,
            outlier_alerts=outlier_alerts,
        )

    # === Attribution Narrative ===

    def interpret_attribution(
        self,
        metrics: Dict[str, float],
        diagnostics: Dict,
    ) -> AttributionNarrative:
        """
        将归因数字转换为自然语言
        """
        lines = []
        key_driver = "none"
        risk = "unknown"

        # Markout 解读
        was = metrics.get("weighted_adverse_selection_bps", 0)
        if was < -5:
            lines.append(
                f"策略面临严重的逆向选择成本 ({was:.1f} bps)。"
                "这意味着你的挂单在信息劣势方被成交 —— "
                "对方知道方向比你准，抢在了你前面。"
            )
            key_driver = "adverse selection"
        elif was < 0:
            lines.append(
                f"轻度逆向选择 ({was:.1f} bps)，在可接受范围内。"
            )
            key_driver = "mild adverse selection"
        else:
            lines.append("未检测到明显的逆向选择。")

        # Turnover 解读
        turnover = metrics.get("turnover_ratio", 0)
        if turnover > 15000:
            lines.append(
                f"换手率 {turnover:,.0f}x 超过上限，"
                "过度交易可能放大了噪音而非信号。"
            )
            risk = "overtrading"
        elif turnover < 500:
            lines.append(
                f"换手率 {turnover:,.0f}x 过低，"
                "策略过于保守，无法覆盖固定成本。"
            )
            risk = "undertrading"
        else:
            lines.append(f"换手率 {turnover:,.0f}x 在合规范围内。")
            risk = "normal"

        # 综合建议
        if key_driver.startswith("adverse"):
            rec = "建议: 降低做市价差 / 增加冷却时间 / 添加 OBI 熔断过滤。"
        elif risk == "overtrading":
            rec = "建议: 增加信号阈值，只在高置信度时挂单。"
        elif risk == "undertrading":
            rec = "建议: 降低信号阈值或增加报价频率。"
        else:
            rec = "策略表现正常，但需持续监控 regime shift。"

        return AttributionNarrative(
            summary="\n".join(lines),
            key_driver=key_driver,
            risk_exposure=risk,
            recommendation=rec,
        )

    def review_attribution(self, metrics: Dict, diagnostics: Dict) -> str:
        """生成完整的归因解读报告"""
        narrative = self.interpret_attribution(metrics, diagnostics)

        return f"""
=== Socrates Attribution Review ===

{narrative.summary}

### Key Driver
{narrative.key_driver}

### Risk Exposure
{narrative.risk_exposure}

### Recommendation
{narrative.recommendation}

--- Socrates v0.1 ---
"""


def create_socrates() -> SocratesAgent:
    """创建 Socrates 助手"""
    return SocratesAgent()

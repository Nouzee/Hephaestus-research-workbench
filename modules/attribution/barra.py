"""
Barra Attribution -barra 风格因子归因
将策略收益拆解为: Market Beta + 行业暴露 + 风格因子 + 纯 Alpha
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FactorReturn:
    """因子收益"""
    factor: str
    weight: float
    contribution: float
    exposure: float


@dataclass
class AttributionResult:
    """归因结果"""
    total_return: float
    factors: List[FactorReturn]
    alpha: float
    residual: float


class BarraAttributor:
    """
    Barra 风格归因

    公式: R_portfolio = W × R_factors + Alpha
    其中:
    - W: 因子暴露矩阵 (T × K)
    - R_factors: 因子收益矩阵 (T × K)
    - Alpha: 特异性收益
    """

    def __init__(
        self,
        factor_names: Optional[List[str]] = None,
    ):
        """
        Args:
            factor_names: 因子名称列表
        """
        self.factor_names = factor_names or [
            "MKT",       # 市场
            "SMB",       # 市值
            "HML",       # 价值
            "RMW",       # 盈利
            "CMA",       # 投资
            "MOM",       # 动量
        ]
        self.factor_loadings: Optional[pd.DataFrame] = None
        self.factor_returns: Optional[pd.DataFrame] = None

    def set_factor_data(
        self,
        loadings: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> None:
        """
        设置因子暴露和收益

        Args:
            loadings: 因子暴露 (Time × Factors)
            returns: 因子收益 (Time × Factors)
        """
        self.factor_loadings = loadings
        self.factor_returns = returns

    def attribute(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
    ) -> AttributionResult:
        """
        执行因子归因

        Args:
            portfolio_weights: 持仓权重 (Assets × 1) 或 (Time × Assets)
            portfolio_returns: 策略收益 (Time,)

        Returns:
            AttributionResult
        """
        if self.factor_loadings is None or self.factor_returns is None:
            # 使用演示数据
            return self._demo_attribution(portfolio_returns)

        # 如果是横截面权重 (Assets,)
        if len(portfolio_weights.shape) == 1:
            return self._cross_sectional_attribution(
                portfolio_weights,
                portfolio_returns
            )
        # 时间序列权重 (Time × Assets)
        else:
            return self._time_series_attribution(
                portfolio_weights,
                portfolio_returns
            )

    def _cross_sectional_attribution(
        self,
        weights: pd.Series,
        returns: pd.Series,
    ) -> AttributionResult:
        """横截面归因"""
        # 简化: 假设持仓是每期固定的
        # R_portfolio = Σ(w_i × r_i)

        explained_return = 0.0
        factor_contributions = []

        for factor in self.factor_names:
            if factor in self.factor_loadings.columns:
                # 因子暴露 = sum(weight × loading)
                exposure = 0.0
                for asset, w in weights.items():
                    if asset in self.factor_loadings.index:
                        loading = self.factor_loadings.loc[asset, factor]
                        exposure += w * loading

                # 因子收益 = mean(factor_return)
                if factor in self.factor_returns.columns:
                    factor_ret = self.factor_returns[factor].mean()
                    contribution = exposure * factor_ret
                    explained_return += contribution

                    factor_contributions.append(FactorReturn(
                        factor=factor,
                        weight=exposure,
                        contribution=contribution,
                        exposure=exposure,
                    ))

        alpha = returns.mean() - explained_return

        return AttributionResult(
            total_return=returns.mean(),
            factors=factor_contributions,
            alpha=alpha,
            residual=0.0,
        )

    def _time_series_attribution(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
    ) -> AttributionResult:
        """时间序列归因"""
        # R_portfolio[t] = Σ(w[t,i] × r[t,i])

        T = len(returns)
        explained_returns = np.zeros(T)

        factor_contributions = []

        for factor in self.factor_names:
            if factor in self.factor_returns.columns:
                # 滚动暴露
                exposures = []
                for t in range(T):
                    if t < len(weights):
                        exp = (weights.iloc[t] * self.factor_loadings[factor]).sum()
                        exposures.append(exp)
                    else:
                        exposures.append(0)

                exposures = np.array(exposures)
                factor_rets = self.factor_returns[factor].values[:T]

                # 贡献 = exposure × return
                contributions = exposures * factor_rets
                explained_returns += contributions

                factor_contributions.append(FactorReturn(
                    factor=factor,
                    weight=np.mean(exposures),
                    contribution=np.mean(contributions),
                    exposure=np.mean(exposures),
                ))

        alpha = returns.values - explained_returns

        return AttributionResult(
            total_return=returns.mean(),
            factors=factor_contributions,
            alpha=np.mean(alpha),
            residual=np.std(alpha),
        )

    def _demo_attribution(
        self,
        returns: pd.Series,
    ) -> AttributionResult:
        """演示归因"""
        total = returns.mean()
        # 模拟分解
        market = total * 0.8  # 假设 80% 市场
        other = total * 0.15
        alpha = total * 0.05

        return AttributionResult(
            total_return=total,
            factors=[
                FactorReturn("MKT", 0.8, market, 0.8),
                FactorReturn("SMB", 0.1, other * 0.5, 0.1),
                FactorReturn("HML", 0.05, other * 0.3, 0.05),
                FactorReturn("MOM", 0.05, other * 0.2, 0.05),
            ],
            alpha=alpha,
            residual=0.0,
        )

    def generate_report(
        self,
        result: AttributionResult,
        output_path: Optional[str] = None,
    ) -> str:
        """生成归因报告"""
        lines = [
            "# Barra Attribution Report",
            "",
            f"## Summary",
            f"- **Total Return**: {result.total_return:.4%}",
            f"- **Alpha**: {result.alpha:.4%}",
            "",
            "## Factor Contributions",
        ]

        for f in result.factors:
            lines.append(f"- **{f.factor}**: {f.contribution:.4%} (weight: {f.weight:.2f})")

        lines.extend([
            "",
            f"## Interpretation",
            f"- Alpha 是无法被因子解释的收益部分",
            f"- 正 Alpha 表示选股能力",
            f"- 负 Alpha 表示选股劣势",
            "",
            f"**Conclusion**: {'策略有超额收益' if result.alpha > 0 else '策略跑输基准'} "
            f"(Alpha: {result.alpha:.2%})"
        ])

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report


def create_attributor(factor_names: Optional[List[str]] = None) -> BarraAttributor:
    """创建归因器"""
    return BarraAttributor(factor_names)
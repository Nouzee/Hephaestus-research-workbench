"""
Anti-Cheat Detector — 自动化防作弊检验
1. Future data leak detection (look-ahead bias)
2. IC/IR decay curve
3. Turnover consistency check
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FutureLeakResult:
    has_leak: bool
    original_score: float
    shifted_score: float
    score_drop_ratio: float
    warning: str


@dataclass
class ICDecayResult:
    horizons: List[int]
    ic_values: List[float]
    half_life: int         # steps until IC drops to 50%
    decay_rate: float
    is_decaying: bool
    warning: str


@dataclass
class OverfitResult:
    n_params: int
    n_observations: int
    ratio: float
    in_sample_score: float
    out_of_sample_score: float
    gap: float
    is_overfit: bool
    warning: str


@dataclass
class AntiCheatReport:
    future_leak: Optional[FutureLeakResult]
    ic_decay: Optional[ICDecayResult]
    overfit: Optional[OverfitResult]
    passed: bool
    violations: List[str]


class AntiCheatDetector:
    """
    防作弊检测器

    三个核心检测:
    1. 未来函数: 信号滞后一期，对比差异
    2. IC 衰减: T+1, T+2, T+5 预测能力曲线
    3. 过拟合: 参数/观测比 + 样本内外差距
    """

    def __init__(self, alpha_threshold: float = 0.05):
        self.alpha_threshold = alpha_threshold
        self.results: List[AntiCheatReport] = []

    def detect_future_leak(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        threshold: float = 0.3,
    ) -> FutureLeakResult:
        """
        检测未来函数

        方法: 将信号滞后一期，看预测能力是否显著下降
        如果下降 >30%，说明原始信号可能存在 look-ahead bias
        """
        T = min(len(signals), len(returns))

        if T < 10:
            return FutureLeakResult(False, 0, 0, 0, "insufficient data")

        # 原始信号 vs 未来收益
        original_corr = np.corrcoef(signals[:T-1], returns[1:T])[0, 1]

        # 滞后一期信号 vs 未来收益
        shifted_corr = np.corrcoef(signals[:T-2], returns[2:T])[0, 1]

        # 差异
        score_drop = (original_corr - shifted_corr) / (abs(original_corr) + 1e-10)
        has_leak = score_drop > threshold

        return FutureLeakResult(
            has_leak=has_leak,
            original_score=original_corr,
            shifted_score=shifted_corr,
            score_drop_ratio=score_drop,
            warning="possible look-ahead bias" if has_leak else "ok",
        )

    def compute_ic_decay(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        horizons: Optional[List[int]] = None,
    ) -> ICDecayResult:
        """
        IC 衰减测试

        测量因子在 T+1, T+2, T+5 的预测能力衰减
        """
        if horizons is None:
            horizons = [1, 2, 5, 10, 20]

        T = min(len(signals), len(returns))
        ic_values = []

        for h in horizons:
            if T <= h + 1:
                ic_values.append(0.0)
                continue
            corr = np.corrcoef(signals[:T-h], returns[h:T])[0, 1]
            ic_values.append(corr)

        ic_values = np.array(ic_values)
        ic_values = np.nan_to_num(ic_values, 0)

        # 半衰期
        half_life = len(horizons)
        if abs(ic_values[0]) > 1e-10:
            for i, ic in enumerate(ic_values):
                if abs(ic) < abs(ic_values[0]) / 2:
                    half_life = horizons[i] if i < len(horizons) else horizons[-1]
                    break

        # 衰减率 (线性拟合)
        decay_rate = 0.0
        if len(horizons) > 1:
            decay_rate = (ic_values[0] - ic_values[-1]) / (horizons[-1] - horizons[0] + 1e-10)

        is_decaying = abs(ic_values[-1]) < abs(ic_values[0]) * 0.5

        return ICDecayResult(
            horizons=horizons,
            ic_values=ic_values.tolist(),
            half_life=half_life,
            decay_rate=decay_rate,
            is_decaying=is_decaying,
            warning="rapid decay, check signal quality" if is_decaying else "stable",
        )

    def detect_overfit(
        self,
        n_params: int,
        n_observations: int,
        in_sample_score: float,
        out_of_sample_score: float,
        gap_threshold: float = 0.3,
    ) -> OverfitResult:
        """
        过拟合检测

        规则:
        1. 参数/观测 > 0.01 -> 过参数化风险
        2. IS - OOS > 30% -> 过拟合
        """
        ratio = n_params / n_observations if n_observations > 0 else 1.0
        gap = abs(in_sample_score - out_of_sample_score) / (abs(in_sample_score) + 1e-10)
        is_overfit = ratio > 0.01 or gap > gap_threshold

        return OverfitResult(
            n_params=n_params,
            n_observations=n_observations,
            ratio=ratio,
            in_sample_score=in_sample_score,
            out_of_sample_score=out_of_sample_score,
            gap=gap,
            is_overfit=is_overfit,
            warning=(
                "overfit detected: IS/OOS gap too large" if gap > gap_threshold
                else "over-parameterized" if ratio > 0.01
                else "ok"
            ),
        )

    def run_all(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        n_params: Optional[int] = None,
        n_observations: Optional[int] = None,
        in_sample_score: Optional[float] = None,
        out_of_sample_score: Optional[float] = None,
    ) -> AntiCheatReport:
        """运行全部检测"""
        violations = []

        # 1. 未来函数
        leak = self.detect_future_leak(signals, returns)
        if leak.has_leak:
            violations.append(f"FUTURE_LEAK: {leak.warning}")

        # 2. IC 衰减
        ic_decay = self.compute_ic_decay(signals, returns)
        if ic_decay.is_decaying:
            violations.append(f"IC_DECAY: {ic_decay.warning}")

        # 3. 过拟合
        overfit = None
        if n_params is not None:
            overfit = self.detect_overfit(
                n_params, n_observations or len(returns),
                in_sample_score or 0, out_of_sample_score or 0,
            )
            if overfit.is_overfit:
                violations.append(f"OVERFIT: {overfit.warning}")

        passed = len(violations) == 0

        report = AntiCheatReport(
            future_leak=leak,
            ic_decay=ic_decay,
            overfit=overfit,
            passed=passed,
            violations=violations,
        )
        self.results.append(report)
        return report

    def generate_report(self, result: AntiCheatReport) -> str:
        """生成报告"""
        status = "[OK] PASSED" if result.passed else "[!] FAILED"

        lines = [
            "# Anti-Cheat Report",
            f"## Status: {status}",
            "",
        ]

        if result.future_leak:
            fl = result.future_leak
            lines.extend([
                "## 1. Future Data Leak Detection",
                f"- Original signal correlation: {fl.original_score:.4f}",
                f"- Shifted signal correlation: {fl.shifted_score:.4f}",
                f"- Score drop: {fl.score_drop_ratio:.1%}",
                f"- Verdict: {'PASS' if not fl.has_leak else 'FAIL — possible look-ahead bias'}",
                "",
            ])

        if result.ic_decay:
            ic = result.ic_decay
            lines.extend([
                "## 2. IC Decay Curve",
                "| Horizon | IC |",
                "|---------|-----|",
            ])
            for h, v in zip(ic.horizons, ic.ic_values):
                lines.append(f"| T+{h} | {v:.4f} |")
            lines.extend([
                "",
                f"- Half-life: {ic.half_life} steps",
                f"- Verdict: {'PASS' if not ic.is_decaying else 'FAIL — rapid decay'}",
                "",
            ])

        if result.overfit:
            of = result.overfit
            lines.extend([
                "## 3. Overfit Detection",
                f"- Params / Obs: {of.ratio:.4f} ({of.n_params}/{of.n_observations})",
                f"- IS - OOS gap: {of.gap:.1%}",
                f"- Verdict: {'PASS' if not of.is_overfit else 'FAIL — overfitting'}",
                "",
            ])

        if result.violations:
            lines.extend([
                "## Violations",
                *(f"- {v}" for v in result.violations),
            ])
        else:
            lines.append("## All checks passed")

        return "\n".join(lines)


def run_anti_cheat(
    signals: np.ndarray,
    returns: np.ndarray,
    **kwargs,
) -> AntiCheatReport:
    """便捷函数"""
    detector = AntiCheatDetector()
    return detector.run_all(signals, returns, **kwargs)

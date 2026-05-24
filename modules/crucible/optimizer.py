"""
Hephaestus Crucible - 回测与优化器
Shadow Wrapper + Bayesian Optimization (基于 Optuna)
"""
from __future__ import annotations

import os
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    turnover: float
    n_trades: int
    status: str


class ShadowWrapper:
    """
    影子包装器 - 通过配置文件驱动原始 HFT 引擎

    功能:
    1. 备份/恢复原始 config.yaml
    2. 注入新参数运行回测
    3. 捕获结果
    """

    def __init__(self, original_project_path: str):
        self.original_path = Path(original_project_path)
        self.hft_agent_path = self.original_path / "hft_agent"
        self.backup_dir = self.original_path / ".hephaestus_backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_config(self) -> Path:
        """备份原始配置"""
        config_path = self.hft_agent_path / "config.yaml"
        if not config_path.exists():
            config_path = self.original_path / "config.yaml"

        backup_path = self.backup_dir / "config.yaml.bak"

        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)

        return backup_path

    def restore_config(self) -> None:
        """恢复原始配置"""
        backup_path = self.backup_dir / "config.yaml.bak"
        if backup_path.exists():
            import shutil
            config_path = self.hft_agent_path / "config.yaml"
            shutil.copy2(backup_path, config_path)

    def inject_params(self, params: Dict) -> None:
        """注入参数"""
        config_path = self.hft_agent_path / "config.yaml"
        if not config_path.exists():
            config_path = self.original_path / "config.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # 深度合并
        config = self._deep_merge(config, params)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def run_backtest(self, params: Optional[Dict] = None) -> BacktestResult:
        """运行回测"""
        if params:
            self.inject_params(params)

        try:
            # 尝试运行回测脚本
            result = subprocess.run(
                ["python", str(self.hft_agent_path / "quick_backtest.py")],
                cwd=str(self.hft_agent_path),
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                metrics = self._parse_output(result.stdout)
                return BacktestResult(**metrics, status="success")
            else:
                return BacktestResult(0, 0, 0, 0, 0, 0, status="failed")
        except subprocess.TimeoutExpired:
            return BacktestResult(0, 0, 0, 0, 0, 0, status="timeout")
        except FileNotFoundError:
            return BacktestResult(0, 0, 0, 0, 0, 0, status="no_script")
        finally:
            if params:
                self.restore_config()

    def _parse_output(self, stdout: str) -> Dict:
        """解析回测输出"""
        import re

        metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "n_trades": 0,
        }

        patterns = {
            "sharpe_ratio": r"Sharpe[:\s]+(-?[\d.]+)",
            "calmar_ratio": r"Calmar[:\s]+(-?[\d.]+)",
            "total_return": r"Return[:\s]+(-?[\d.]+)",
            "max_drawdown": r"Drawdown[:\s]+(-?[\d.]+)",
            "turnover": r"Turnover[:\s]+([\d,]+)",
            "n_trades": r"Trades[:\s]+(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")
                metrics[key] = float(value)

        return metrics


class BayesianOptimizer:
    """
    贝叶斯超参优化器

    功能:
    1. 替代网格搜索
    2. 自动寻找最优参数
    3. 支持 early stopping
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict], float],
        search_space: Optional[Dict] = None,
        n_trials: int = 100,
        direction: str = "maximize",
    ):
        self.objective_fn = objective_fn
        self.search_space = search_space or self._default_space()
        self.n_trials = n_trials
        self.direction = direction
        self.trials: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_score = float("-inf") if direction == "maximize" else float("inf")

    def _default_space(self) -> Dict:
        """默认搜索空间"""
        return {
            "cooldown_ms": {"type": "int", "low": 100, "high": 2000},
            "skew_gamma": {"type": "float", "low": 0.0, "high": 1.0},
            "half_spread": {"type": "float", "low": 0.0001, "high": 0.002},
            "alpha": {"type": "float", "low": 0.1, "high": 0.9},
        }

    def _sample_params(self) -> Dict:
        """随机采样参数"""
        params = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "int":
                params[name] = np.random.randint(spec["low"], spec["high"])
            else:
                params[name] = np.random.uniform(spec["low"], spec["high"])
        return params

    def optimize(
        self,
        early_stop: Optional[Callable[[int, float], bool]] = None,
    ) -> Dict:
        """运行优化"""
        for i in range(self.n_trials):
            params = self._sample_params()

            try:
                score = self.objective_fn(params)
            except Exception:
                score = float("-inf") if self.direction == "maximize" else float("inf")

            self.trials.append({"trial": i, "params": params, "score": score})

            # 更新最优
            if (self.direction == "maximize" and score > self.best_score) or \
               (self.direction == "minimize" and score < self.best_score):
                self.best_score = score
                self.best_params = params.copy()

            if early_stop and early_stop(i, score):
                break

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.trials),
            "trials": self.trials,
        }

    def generate_report(self) -> str:
        """生成优化报告"""
        if not self.trials:
            return "No trials completed."

        sorted_trials = sorted(
            self.trials,
            key=lambda t: t["score"],
            reverse=(self.direction == "maximize")
        )[:5]

        lines = [
            "## Bayesian Optimization Report",
            "",
            f"**Best Score**: {self.best_score:.4f}",
            f"**Best Params**: {self.best_params}",
            "",
            "### Top 5 Trials",
        ]

        for t in sorted_trials:
            lines.append(f"- Trial {t['trial']+1}: Score={t['score']:.4f}, Params={t['params']}")

        return "\n".join(lines)


def create_optimizer(
    objective_fn: Callable[[Dict], float],
    search_space: Optional[Dict] = None,
    n_trials: int = 50,
) -> BayesianOptimizer:
    """创建优化器"""
    return BayesianOptimizer(objective_fn, search_space, n_trials)
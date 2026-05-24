"""
Experiment Logger - 实验记录器
追踪每个实验的参数和性能
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict


class ExperimentLogger:
    """
    实验记录器 - 记录每次实验的参数和结果

    功能:
    1. 记录实验参数
    2. 记录性能指标
    3. 生成实验摘要
    4. 追踪历史趋势
    """

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            workspace_root = Path(__file__).parent.parent.parent
        self.workspace_root = Path(workspace_root)
        self.ledger_path = self.workspace_root / "master_ledger.csv"
        self._init_ledger()

    def _init_ledger(self) -> None:
        """初始化 ledger 文件"""
        if not self.ledger_path.exists():
            with open(self.ledger_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "experiment_id",
                    "experiment_name",
                    "command",
                    "environment",
                    "params_json",
                    "metrics_json",
                    "status",
                    "notes",
                ])

    def log_experiment(
        self,
        experiment_name: str,
        command: str,
        params: Dict,
        metrics: Optional[Dict] = None,
        status: str = "running",
        notes: str = "",
    ) -> str:
        """记录实验"""
        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_name[:20]}"

        with open(self.ledger_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                experiment_id,
                experiment_name,
                command,
                self._detect_env(),
                json.dumps(params),
                json.dumps(metrics or {}),
                status,
                notes,
            ])

        return experiment_id

    def update_experiment(self, experiment_id: str, metrics: Dict, status: str = "completed", notes: str = "") -> None:
        """更新实验记录"""
        # 读取所有记录
        rows = []
        with open(self.ledger_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["experiment_id"] == experiment_id:
                    row["metrics_json"] = json.dumps(metrics)
                    row["status"] = status
                    if notes:
                        row["notes"] = notes
                rows.append(row)

        # 写回
        with open(self.ledger_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_experiment_history(self, limit: int = 50) -> list:
        """获取实验历史"""
        rows = []
        with open(self.ledger_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        return rows[-limit:]

    def get_best_experiment(self, metric: str = "sharpe_ratio") -> Optional[Dict]:
        """获取最佳实验"""
        rows = self.get_experiment_history(limit=1000)

        best = None
        best_value = float("-inf")

        for row in rows:
            if row["status"] == "completed" and row["metrics_json"]:
                metrics = json.loads(row["metrics_json"])
                if metric in metrics and metrics[metric] > best_value:
                    best = row
                    best_value = metrics[metric]

        return best

    def generate_summary(self) -> str:
        """生成实验摘要"""
        rows = self.get_experiment_history(limit=100)

        total = len(rows)
        completed = sum(1 for r in rows if r["status"] == "completed")
        failed = sum(1 for r in rows if r["status"] == "failed")

        # 计算趋势
        recent_returns = []
        for row in rows[-10:]:
            if row["metrics_json"]:
                m = json.loads(row["metrics_json"])
                if "total_return" in m:
                    recent_returns.append(m["total_return"])

        trend = "unknown"
        if len(recent_returns) >= 2:
            if recent_returns[-1] > recent_returns[0]:
                trend = "improving"
            else:
                trend = "declining"

        summary = f"""
## Hephaestus 实验摘要

| 指标 | 值 |
|------|-----|
| 总实验数 | {total} |
| 已完成 | {completed} |
| 失败 | {failed} |
| 趋势 | {trend} |

最近 10 个实验: {len(recent_returns)} 个有收益数据
"""
        return summary

    def _detect_env(self) -> str:
        """检测环境"""
        import sys
        if sys.platform == "win32":
            return "windows"
        else:
            return "linux"

    def search_experiments(
        self,
        pattern: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """搜索实验"""
        rows = self.get_experiment_history(limit=limit)

        results = []
        for row in rows:
            if status and row["status"] != status:
                continue
            if pattern and pattern.lower() not in row["experiment_name"].lower():
                continue
            results.append(row)

        return results


# 便捷函数
def create_logger(workspace_root: Optional[str] = None) -> ExperimentLogger:
    """创建实验记录器"""
    return ExperimentLogger(workspace_root)
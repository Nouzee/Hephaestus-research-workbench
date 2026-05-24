"""
Task Orchestrator - 任务编排层
解析用户指令并分配工作流
"""

import os
import sys
import json
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


# 指令集定义
COMMAND_SET = {
    "analyze_attribution": {
        "description": "运行归因分析",
        "module": "modules.attribution",
        "entry": "run_analysis",
    },
    "integrate_mamba": {
        "description": "集成 Mamba 模型",
        "module": "modules.forge.mamba",
        "entry": "integrate",
    },
    "run_cross_val": {
        "description": "交叉验证",
        "module": "modules.crucible",
        "entry": "cross_validate",
    },
    "optimize_params": {
        "description": "Bayesian 超参优化",
        "module": "modules.crucible",
        "entry": "bayesian_optimize",
    },
    "generate_report": {
        "description": "生成 Markdown 报告",
        "module": "modules.attribution",
        "entry": "generate_report",
    },
    "sync_data": {
        "description": "同步数据",
        "module": "core.data_loader",
        "entry": "sync",
    },
}


def detect_environment() -> str:
    """自动检测 Windows 还是 WSL 环境"""
    if sys.platform == "win32":
        return "windows"
    elif os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower():
        return "wsl"
    else:
        return "linux"


class TaskOrchestrator:
    """
    Hephaestus 的大脑 - 解析指令并分配工作流

    功能:
    1. 解析用户指令
    2. 创建带时间戳的实验文件夹
    3. 自动检测环境
    4. 路由任务到对应模块
    """

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            workspace_root = Path(__file__).parent.parent
        self.workspace_root = Path(workspace_root)
        self.environment = detect_environment()
        self.current_experiment = None
        self.experiment_log = []

    def parse_command(self, command: str) -> Dict[str, Any]:
        """解析用户指令"""
        # 支持多种格式: analyze_attribution, --analyze_attribution, /analyze_attribution
        cmd_clean = command.strip().lstrip("-/").split()[0]

        if cmd_clean in COMMAND_SET:
            return {
                "command": cmd_clean,
                "description": COMMAND_SET[cmd_clean]["description"],
                "module": COMMAND_SET[cmd_clean]["module"],
                "entry": COMMAND_SET[cmd_clean]["entry"],
            }
        else:
            # 模糊匹配
            for known_cmd in COMMAND_SET:
                if known_cmd.startswith(cmd_clean):
                    return {
                        "command": known_cmd,
                        "description": COMMAND_SET[known_cmd]["description"],
                        "module": COMMAND_SET[known_cmd]["module"],
                        "entry": COMMAND_SET[known_cmd]["entry"],
                    }
            return {"error": f"Unknown command: {command}. Available: {list(COMMAND_SET.keys())}"}

    def create_experiment(self, name: str, config: Optional[Dict] = None) -> Path:
        """
        为新实验创建带时间戳的文件夹

        Example:
            orchestrator.create_experiment("markout_analysis_001")
            -> workspace_root/experiments/2026-04-30_143000_markout_analysis_001/
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        exp_name = f"{timestamp}_{name}"
        exp_path = self.workspace_root / "experiments" / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)

        # 保存 config
        if config:
            config_file = exp_path / "config.yaml"
            import yaml
            with open(config_file, "w") as f:
                yaml.dump(config, f)

        # 保存参数到 master_ledger
        self.log_experiment(exp_name, config or {})

        self.current_experiment = exp_path
        return exp_path

    def log_experiment(self, experiment_name: str, params: Dict) -> None:
        """记录实验到 master_ledger.csv"""
        ledger_path = self.workspace_root / "master_ledger.csv"

        import csv
        file_exists = ledger_path.exists()

        with open(ledger_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "experiment_name", "environment", "params_json"])
            writer.writerow([
                datetime.now().isoformat(),
                experiment_name,
                self.environment,
                json.dumps(params),
            ])

    def route_task(self, command: str, **kwargs) -> Any:
        """
        路由任务到对应模块

        Example:
            orchestrator.route_task("analyze_attribution", source="path/to/data")
        """
        parsed = self.parse_command(command)
        if "error" in parsed:
            return {"error": parsed["error"]}

        # 创建实验目录
        exp_path = self.create_experiment(command)

        # 动态导入模块并执行
        try:
            module_name = parsed["module"]
            entry_name = parsed["entry"]

            # 安全导入 (不允许危险的 __import__)
            if ".." in module_name or "/" in module_name:
                return {"error": "Invalid module path"}

            module = __import__(module_name, fromlist=[entry_name])
            func = getattr(module, entry_name, None)

            if func is None:
                return {"error": f"Function {entry_name} not found in {module_name}"}

            # 执行
            result = func(experiment_path=exp_path, environment=self.environment, **kwargs)
            return result

        except Exception as e:
            return {"error": str(e), "experiment_path": str(exp_path)}

    def list_commands(self) -> list:
        """列出所有可用指令"""
        return [
            {"command": cmd, **info}
            for cmd, info in COMMAND_SET.items()
        ]

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "environment": self.environment,
            "current_experiment": str(self.current_experiment) if self.current_experiment else None,
            "experiments_count": len(list((self.workspace_root / "experiments").glob("*"))),
            "available_commands": list(COMMAND_SET.keys()),
        }


# CLI 入口点
def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Hephaestus - 量化锻造工台")
    parser.add_argument("command", nargs="?", help="执行的命令")
    parser.add_argument("--source", help="数据源路径")
    parser.add_argument("--target", help="优化目标")
    parser.add_argument("--list", action="store_true", help="列出所有可用命令")
    parser.add_argument("--status", action="store_true", help="显示状态")

    args = parser.parse_args()

    orchestrator = TaskOrchestrator()

    if args.list:
        print("可用命令:")
        for cmd_info in orchestrator.list_commands():
            print(f"  {cmd_info['command']}: {cmd_info['description']}")
        return

    if args.status:
        status = orchestrator.get_status()
        print(f"环境: {status['environment']}")
        print(f"实验数量: {status['experiments_count']}")
        print(f"可用命令: {status['available_commands']}")
        return

    if args.command:
        result = orchestrator.route_task(
            args.command,
            source=args.source,
            target=args.target,
        )
        if "error" in result:
            print(f"错误: {result['error']}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"完成: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
"""
Data Loader — 数据加载器
从原始 HFT 项目加载数据，不修改原始文件
"""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class DataLoader:
    """从原始 HFT 项目加载数据"""

    def __init__(self, source_root: str):
        self.source_root = Path(source_root)

    def load_backtest_results(self, filename: str = "backtest_results.csv") -> pd.DataFrame:
        for path in [self.source_root / filename,
                     self.source_root / "hft_agent" / filename]:
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError(f"backtest results not found in {self.source_root}")

    def load_config(self) -> Dict:
        import yaml
        for path in [self.source_root / "config.yaml",
                     self.source_root / "hft_agent" / "config.yaml"]:
            if path.exists():
                with open(path) as f:
                    return yaml.safe_load(f) or {}
        return {}

    def list_available_files(self) -> List[Dict]:
        results = []
        for path in self.source_root.rglob("*"):
            if path.suffix in {".csv", ".parquet", ".json", ".yaml"}:
                results.append({
                    "filename": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                })
        return results

    def sync(self, target: Optional[str] = None, **kwargs) -> Dict:
        return {
            "status": "synced",
            "source": str(self.source_root),
            "files": self.list_available_files(),
        }

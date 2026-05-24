"""Hephaestus Crucible — 回测与优化器"""
from .dual_engine import (
    DualEngine, VectorizedEngine, EventDrivenEngine,
    BacktestConfig, BacktestResult as EngineBacktestResult,
    BaseStrategy,
)
from .optimizer import (
    ShadowWrapper, BayesianOptimizer,
    BacktestResult, create_optimizer,
)

__all__ = [
    "ShadowWrapper", "BayesianOptimizer",
    "BacktestResult", "DualEngine",
    "VectorizedEngine", "EventDrivenEngine",
    "BacktestConfig", "BaseStrategy",
    "EngineBacktestResult", "create_optimizer",
]
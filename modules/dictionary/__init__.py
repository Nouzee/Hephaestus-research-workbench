"""
Dictionary Learning + Toxicity Detection — 自适应字典 + 毒性识别

Pipeline:
  matrix_builder  — Task 1: 高频盘口数据 → N×M 特征矩阵 X
  dict_trainer    — Task 2: MiniBatchDictionaryLearning → 字典 D + 稀疏系数 α
  online_dict     — Phase 2: 动态在线字典学习 (Mairal et al. 2010)
  gram_tracker    — Phase 3: Gram 拓扑追踪 (原子协方差 + 漂移检测)
  hmm_regime      — Phase 4: HMM 隐 Markov 状态识别
  toxicity_scorer — Phase 5: 条件偏离毒性评分
"""

__all__ = [
    "matrix_builder", "dict_trainer",
    "online_dict", "gram_tracker", "hmm_regime", "toxicity_scorer",
    "pnl_backtest", "precursor_scorer",
    "causal_wavelet", "multiscale_features",
    "signal_router", "pressure_memory",
]

from modules.dictionary import (
    matrix_builder, dict_trainer,
    online_dict, gram_tracker, hmm_regime, toxicity_scorer,
    pnl_backtest, precursor_scorer,
    causal_wavelet, multiscale_features,
    signal_router, pressure_memory,
)

# Hephaestus — BTC 量化研究工台

一个从脏数据、假 alpha 和 OOM 中活下来的训练与归因管线。

当前主线：在 OKX BTC-USDT-SWAP L2 订单簿上做回归预测。能跑通。效果还在挣扎。

## 架构

```
Hephaestus/
├── run_pipeline.py          # 主入口：数据→训练→评估 （日常最常用）
├── core/
│   ├── training_frame.py    # WindowDataset / 指标 / 训练循环 / 特征消融
│   ├── data_generator.py    # events_features.parquet → 每日归一化 → label
│   ├── concat_tool.py       # manifest → train/test parquet
│   ├── orchestrator.py      # 任务路由
│   └── logger.py            # master_ledger.csv
├── models/
│   └── model_zoo.py         # HybridTransformerLSTM + 轻量变体
├── modules/
│   ├── forge/               # TensorStream / Alpha Factory / Mamba / NeuralSDE / STGNN
│   ├── attribution/         # Markout / 逆向选择 / Anti-Cheat / Socrates agent
│   └── crucible/            # Bayesian 优化 / dual-engine backtester
├── .claude/
│   └── skills/              # Claude Code Skills (train-model, data-inspect, result-analyze)
├── output/                  # 训练产物（按实验名 + label tag 组织）
├── data/                    # 缓存 parquet / demo 数据
└── CLAUDE.md                # 给 Claude Code 的项目说明书
```

## 快速开始

```bash
# 主流程：从已有特征文件直接训练
python run_pipeline.py --skip-data

# 改了 label 参数 → 重建数据
python run_pipeline.py --rebuild

# 快速冒烟
python run_pipeline.py --rebuild --epochs 2
```

或用 Skills（在 Claude Code 里）：

```
/train-model --epochs 5
/data-inspect
/result-analyze
```

## 为什么这样设计

**为什么不做端到端？**

原始 backtester 的撮合逻辑经过了 4 个月迭代调整（延迟建模、队列位置、OBI 熔断）。重写会引入 simulator mismatch。Hephaestus 作为 shadow research layer 叠在上面 — 保持执行语义不变，只在研究层折腾。

**为什么用预处理的 events_features.parquet 而不是从 D 盘 tar.gz 实时加载？**

从 31 天 tar.gz 实时加载 → 45 分钟。预处理 parquet → 3 秒。研究迭代速度差 1000 倍。

**为什么 normalization 在 label 计算之后？**

label 公式是 (m_plus - m_minus) / m_minus。mid_px 归一化后出现负值，m_minus ≤ 0 会把所有 label 过滤掉。花了两小时才定位到这个 bug。

**为什么 zero-masking 而不是删列做 ablation？**

删列会改变模型输入维度 → model capacity 不是唯一变量 → ablation 无意义。zero-masking 保持完全相同架构，差异只能来自特征可用性。

## Benchmarks

*指 pipeline 实测，非理论值*

| 阶段 | 数据量 | 耗时 | 备注 |
|---|---|---|---|
| events_features.parquet 加载 | 3.9M 行/月 | ~3s | pandas read_parquet |
| 每日文件生成 + concat | 3.9M 行 | ~90s | 含 label 计算 + 滚动归一化 |
| 训练 1 epoch (HybridTransformerLSTM, bs=256×4) | 2.7M train rows | ~120s | 5070 Ti, CUDA 12.8 |
| 全 pipeline (2 epochs, 1 seed) | 3.9M 行 | ~6 min | 含数据阶段 |

## Known Failure Modes

这些是实际踩过的坑：

- **Label 全 NaN**：label 计算在归一化之后 → mid_px 有负值 → m_minus ≤ 0 → 全部过滤。必须在归一化**之前**算。
- **test_windows: 0**：同上，验证集没有任何有效样本。
- **loss = 3.5e9**：TARGET_SCALE 设太大（1e4）且模型初始化未做 Xavier → 输出爆炸。降到 100 + Xavier init + 梯度裁剪修复。
- **model shape mismatch (64000×12 vs 11×32)**：模型 input_dim 没算上 TimeDiff 拼接列。特征数 + 1。
- **RTX 5070 Ti 无法用官方 PyTorch**：Blackwell 架构需要 CUDA 12.6+。解决方案：PyTorch 2.12 nightly + cu128。
- **Bash 里调 Windows Python 各种 exit code 49**：Windows Store Python alias 问题。用 anaconda 的 python.exe 全路径。

## Failed Experiments

### HybridTransformerLSTM + LABEL_H=1.0s
- Pearson IC ≈ 0.005, Spearman IC ≈ -0.08
- 1 秒前向 horizon 对 BTC 几乎没有可预测信号
- 噪声完全淹没了任何模式

### TARGET_SCALE=1e4
- 训练 loss 直接 3.5e9 起步
- BTC $80K 价格下 1e4 缩放把 label 放大了 100 倍
- 降到 100 才稳定

### 高频项目的 3-class 分类
- train_model.py 的方向预测 (卖/持有/买) accuracy ≈ 35%
- 略好于随机 (33%)，但远不够 executable
- 原因：top-of-book 数据没有足够的信息量区分方向

## 数据坑

OKX 加密 tick 数据实际遇到的问题：

- 同一毫秒内事件顺序可能颠倒 → 按 ts_ms sort 不能保证因果正确
- 极端行情下 orderbook snapshot 的 bid/ask depth 瞬时脱节
- trade side 字段偶尔缺失 → 需要 heuristically 推断
- tar.gz 内的 JSONL 可能有断行 → json.loads 会炸
- 2026-01-01 的 events_features.parquet 实际没有数据（交易所维护？）→ 日期 split 要容错

## Open Questions

- 11 个 top-of-book 特征是否包含足够信息？是否需要深度档位特征？
- 当前 500-tick lookback 是否对 BTC 过拟合特定波动率 regime？
- HybridTransformerLSTM 在小样本（1个月）上是否天生过参数化？
- 离线 IC 转在线 PnL 的 gap 有多大？（还没接入真实撮合引擎验证）

## Research Directions

不是模型列表，是有待验证的问题：

- **Queue-aware feature learning**：从 top-of-book 推断排队位置
- **Regime-adaptive quoting**：波动率飙升时自动收紧/放松 spread
- **Latency-sensitive replay**：在回放中注入真实网络延迟，验证策略对延迟的鲁棒性
- **Cross-horizon markout**：同时预测 100ms / 1s / 10s 的多尺度收益，而不是单一 horizon
- **Inventory-aware sizing**：不是二元 buy/sell，而是连续仓位大小

## Philosophy

大多数 alpha idea 死在归因阶段，不是模型阶段。

Hephaestus 的设计目标不是"做出最准的预测"，而是——**在假 alpha 污染核心撮合引擎之前，快速、可追溯、可复现地证伪它**。

90% 的实验应该失败。剩下 10% 才值得进入回测。

## 依赖

```
pytorch>=2.5 (nightly for RTX 50 series)
numpy, pandas, scikit-learn
polars (可选，用于 L2DataLoader)
```

## 鸣谢

- 中山大学岭南学院 "面向经济学家的 AI 编程" 课程教材 (ai.lingnan.top) — Skills / Agent Teams 架构参考
- 高频项目 train_model.py — HybridTransformerLSTM 原始实现
- ExperimentC — 训练框架原型（WindowDataset / 指标 / 特征消融）

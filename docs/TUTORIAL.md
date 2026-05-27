# Hephaestus 全方位项目概述

**状态**：训练管线能跑通，模型效果还在调。5070 Ti GPU 可用。

---

## 一、项目是什么

在 OKX BTC-USDT-SWAP 400档 L2 订单簿 + 逐笔成交数据上，用深度学习做 mid-price 回归预测。不是 demo，不是课程作业。目标是能跑的量化研究工台——能加载真实数据、能训练、能评估、能对比实验。

## 二、数据流全景

```
D:\btc\raw data\
  ├── order/  (31个 tar.gz, 每个 ~2GB, L2 快照 JSONL)
  └── trades/ (31个 zip, 逐笔成交 CSV)
        │
        ▼  [L2DataLoader]  合并快照+成交 → events DataFrame (~17列)
        │
        ▼  [高频项目 pipeline]  提取11个特征 → events_features.parquet (53MB, 3.9M行)
        │
        ▼  [core/data_generator.py]  按日切分 → 滚动5天归一化 → 计算 forward return label
        │
        ▼  [core/concat_tool.py]  拼接 → train.parquet + test.parquet
        │
        ▼  [core/training_frame.py]  WindowDataset (K=500 tick窗口, H=10s前向)
        │
        ▼  [models/model_zoo.py]  HybridTransformerLSTM → scalar prediction
        │
        ▼  指标: Pearson IC / Spearman IC / MAE / RMSE / Directional Accuracy
```

**关键决策**：不用 D 盘原始 tar.gz 实时加载（45分钟），用预处理 parquet（3秒）。

## 三、核心组件

### 3.1 训练框架 (`core/training_frame.py`)

从 ExperimentC 项目移植并适配。核心类：

- **WindowDataset**：滑动窗口数据集。输入 (T, n_features)，输出 (1, T, n_features) + (T, 1) 的 time delta。自动过滤 NaN label。
- **train_one_epoch / eval_one_epoch**：标准训练循环。支持梯度累积（grad_accum_steps）。
- **train_and_test_pipeline**：完整 train/val/test 编排。自动保存 best model。
- **回归指标**：safe_pearsonr, safe_spearmanr, safe_r2_score, safe_directional_accuracy。都做了 NaN/Inf 防护。
- **特征消融**：zero-masking 模式。不改模型结构，只把被消融的特征列置零。

### 3.2 数据生成 (`core/data_generator.py`)

- 从 `events_features.parquet`（53MB，高频项目预处理）加载
- 时间戳列 "Time" + 时间差 "TimeDiff" (log1p) + 11 个特征列
- 滚动 5 天归一化（参考 ExperimentC）
- Label：forward return = (m_plus - m_minus) / m_minus，其中 K tick 回溯，H 秒前向

### 3.3 模型 (`models/model_zoo.py`)

| 模型 | 来源 | 参数 | 输入 |
|---|---|---|---|
| `HybridTransformerLSTM` | 高频项目 train_model.py | ~500K | (B, 12, T) |
| `HybridTransformerLSTM_Small` | 同上精简 | ~100K | (B, 12, T) |

架构：Input Proj → PositionalEncoding → 3×TransformerEncoder → 2×LSTM → Regressor

**注意**：输入 dim = 11 特征 + 1 TimeDiff = 12。改特征数时记得同步改模型默认值。

### 3.4 Claude Code Skills (`.claude/skills/`)

| Skill | 用途 |
|---|---|
| `train-model` | 调参 → 训练 → 监控 → 报告 |
| `data-inspect` | 检查 label 分布 / 特征统计 / 窗口有效性 |
| `result-analyze` | 收集 metrics → 汇总 → 诊断 → 改进建议 |

### 3.5 其他模块

- **modules/forge/**：TensorStream（Polars 数据流水线）、Alpha Factory（因子插件）、Mamba SSM、Neural SDE、ST-GNN。目前独立存在，未接入训练管线。
- **modules/attribution/**：Markout 计算、逆向选择分析、Anti-Cheat 检测、Socrates AI 解读。
- **modules/crucible/**：Bayesian 超参优化、dual-engine backtester。

## 四、关键配置速查

所有配置在 `run_pipeline.py` 第 55-101 行：

```python
# 数据
EVENTS_PARQUET = r"C:\...\events_features.parquet"
LABEL_K = 500        # 回溯 tick 数
LABEL_H = 10.0       # 前向 horizon（秒）
TARGET_SCALE = 100   # label 缩放倍数

# 模型
MODEL_NAME = "HybridTransformerLSTM"

# 训练
BATCH_SIZE = 256
GRAD_ACCUM_STEPS = 4   # 等效 batch = 1024
EPOCHS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 实验
FEATURE_SETS = ["PV+NA+DT"]
SEEDS = [0]
```

## 五、输出目录结构

```
output/hephaestus_btc_regression/<LABEL_TAG>/
├── tmp_daily/          # 每日临时文件（可删）
├── concat/             # train/test parquet
├── runs/               # 每次训练的产物
│   └── PV_NA_DT/
│       └── seed_0/
│           ├── best_model.pt
│           ├── training_log.txt
│           ├── test_metrics.json
│           └── test_predictions.parquet
├── summary/
│   ├── all_runs.csv
│   └── summary.csv
└── logs/
    └── master_log.txt
```

## 六、踩过的坑及解决方案

| 症状 | 原因 | 修复 |
|---|---|---|
| loss = 3.5e9 | TARGET_SCALE=1e4 太大 + 无 Xavier init | 降到 100 + Xavier + 梯度裁剪 |
| test_windows: 0 | label 在归一化后计算，mid_px 有负值 | label 必须在归一化**之前**算 |
| model shape mismatch | input_dim 没算 TimeDiff 列 | input_dim = n_features + 1 |
| pearson_ic 始终 NaN | data 里只有 1 天 test，label 全 NaN | 修复 label 计算顺序后解决 |
| RTX 5070 Ti 用不了 | 官方 PyTorch 不支持 Blackwell | 装 PyTorch 2.12 nightly + cu128 |
| GPU 训练慢于预期 | pin_memory=False | 改为 True |

## 七、当前实验结果

*截至 2026-05-13，1个月 BTC 数据，HybridTransformerLSTM，70/30 时序切分*

| 指标 | 值 | 判断 |
|---|---|---|
| Train loss | ~3.5e9 → 缓慢下降 | 在学，但非常慢 |
| Val Pearson IC | ~0.005 | ≈ 无预测能力 |
| Val Spearman IC | ~-0.08 | ≈ 随机 |
| Directional Acc | N/A | test 阶段崩溃 |

**诊断**：11 个 top-of-book 特征 + 1 个月数据，信息量不足以支撑 Transformer+LSTM 级别的模型。下一步方向：

1. 加特征（深度档位、微观结构统计量）
2. 换更简单模型（防止过参数化）
3. 减小 K（500 tick ≈ 50-100 秒，可能太长的无关历史）
4. 用全月 31 天数据（目前只有约 2 天有效）

## 八、怎么跑

```bash
# 激活环境
conda activate base

# 快速冒烟（2 epoch, 约 6 分钟）
python run_pipeline.py --rebuild --epochs 2

# 正式训练（30 epoch, 建议 overnight）
python run_pipeline.py --rebuild --epochs 30

# 只训练不复算数据
python run_pipeline.py --skip-data --epochs 10
```

## 九、项目文件索引

| 要做什么 | 看哪个文件 |
|---|---|
| 改超参数 | `run_pipeline.py` 第 55-101 行 |
| 换模型 | `models/model_zoo.py` |
| 加新特征 | `core/data_generator.py` 的 FEATURE_COLS |
| 改训练逻辑 | `core/training_frame.py` |
| 改 label 定义 | `core/data_generator.py` 的 compute_labels |
| 改数据源 | `core/data_generator.py` 的 load_events_parquet |
| 了解项目给 AI 看 | `CLAUDE.md` |
| 人类看 | `README.md` |
| 训练、检查数据、分析结果 | `.claude/skills/` 下的 SKILL.md |

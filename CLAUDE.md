# Hephaestus — BTC L2 量化训练工台

面向经济学家的 AI 编程课程配套项目。在 BTC L2 订单簿数据上训练深度学习模型做回归预测。

## 项目结构

```
Hephaestus/
├── run_pipeline.py          # 主入口：一键运行完整训练管线
├── core/
│   ├── training_frame.py    # 训练框架（WindowDataset / 指标 / 循环 / 特征消融）
│   ├── data_generator.py    # 数据适配器（events_features.parquet → 每日文件 → label）
│   ├── concat_tool.py       # 拼接工具（manifest → train/test parquet）
│   ├── orchestrator.py      # 任务编排器
│   └── logger.py            # 实验日志
├── models/
│   └── model_zoo.py         # 模型库（HybridTransformerLSTM + 变体）
├── modules/
│   ├── forge/               # 模型定义（Mamba / NeuralSDE / STGNN / Alpha Factory）
│   ├── attribution/         # 归因分析 / 防作弊 / Socrates
│   └── crucible/            # 优化器
├── configs/                  # 配置文件
├── data/                     # 缓存数据
├── output/                   # 训练输出（自动生成）
└── .claude/
    ├── settings.json
    └── skills/               # Claude Code Skills
```

## 核心训练管线

```
events_features.parquet  →  core/data_generator.py  →  每日 LOB + label
        │                                                    │
        │                                         core/concat_tool.py
        │                                                    │
        │                                         train/test parquet
        │                                                    │
        └─────────────────────────→  core/training_frame.py  ←  models/model_zoo.py
                                              │
                                         metrics / model.pt
```

## 关键约定

- **数据源**：`C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet`
- **特征列**：11个（mid_px, spread, imbalance, total_depth, bid_px, ask_px, bid_sz, ask_sz, signed_imbalance, abs_imbalance, spread_ticks）
- **模型**：`HybridTransformerLSTM`（回归版，输入11特征 + 1时间差 = 12维）
- **标签**：K=500 tick 回溯, H=10s 前向收益, 无log
- **缩放**：`TARGET_SCALE=100`
- **设备**：auto（5070 Ti CUDA 12.8）
- **Feature ablation**：零掩码（不改模型结构），group 为 PV / N / A
- **SKIP_FINISHED**：已完成 run 自动跳过，支持断点续跑
- **全局配置**：修改 `run_pipeline.py:55-101` 的配置区，不要散改

## 运行命令

```bash
# 快速冒烟
python run_pipeline.py --skip-data

# 重建数据 + 训练
python run_pipeline.py --rebuild

# 调整epoch
python run_pipeline.py --skip-data --epochs 30
```

## 调试提示

- 标签必须在**归一化之前**计算（mid_px 归一化后有负值，label 全被过滤）
- 模型 `input_dim` 必须是特征数+1（拼接了 TimeDiff 列）
- `features_groups` 里的列索引对应 `FEATURE_COLS` 列表的实际位置
- 输出文件都在 `output/` 下按实验名 + label标签组织
- 如果要换模型，改 `MODEL_NAME` 和 `MODEL_MODULE_NAME` 两个配置即可

## 当前状态

- [x] 训练管线能跑通
- [x] GPU 可用（5070 Ti / CUDA 12.8 / PyTorch 2.12 nightly）
- [ ] 模型效果还不行（Pearson IC ≈ 0），需要调参/换模型/加特征

## 相关参考

- 高频项目：`C:\Users\ZaneLaw\Desktop\Zane\高频项目\`
- 教材：`http://ai.lingnan.top`（中山大学岭南学院 面向经济学家的 AI 编程）
- 教材第17章：量化投资系统实战

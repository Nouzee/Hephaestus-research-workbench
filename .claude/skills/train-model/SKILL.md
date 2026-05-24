---
name: train-model
description: 训练BTC L2回归模型。修改超参数后触发训练，自动跳过已完成runs。支持 --rebuild 重建数据、--epochs 调整轮数、--feature-sets 选择特征集。
---

# 模型训练 Skill

严格按照以下步骤执行。

## 第一步：确认配置

读取 `run_pipeline.py` 顶部的全局配置区（第50-101行），确认当前参数：

- MODEL_NAME（模型选择）
- EPOCHS（训练轮数）
- BATCH_SIZE / GRAD_ACCUM_STEPS（等效batch）
- LEARNING_RATE / WEIGHT_DECAY
- FEATURE_SETS / SEEDS
- LABEL_K / LABEL_H（标签定义）

如果用户给出了具体的参数调整要求，直接修改对应变量。

## 第二步：执行训练

```bash
cd c:\Users\ZaneLaw\Desktop\Zane\Hephaestus
python run_pipeline.py [--rebuild] [--skip-data] [--epochs N]
```

**规则**：
- 第一次运行或改了 LABEL 参数 → 加 `--rebuild`
- 只改模型/训练参数 → 加 `--skip-data`
- 不确定 → 加 `--rebuild` 保底

## 第三步：监控输出

关注以下关键行：

- `Train: X events, Test: Y events` — 数据量正常
- `val_loss 持续下降` — 训练正常
- `val_pearson 上升` — 模型在学信号
- `New best model saved` — checkpoint 保存

**异常信号**：
- `loss=3e9` 量级 → scale 爆炸，检查 TARGET_SCALE
- `val_pearson < 0` 或接近 0 → 模型没学到东西
- `test_windows: 0` → label 全 NaN，检查归一化顺序
- `Pipeline FAILED` → 看 traceback 定位具体错误

## 第四步：报告结果

训练完成后，报告：
1. 最优 epoch 的 val_loss / val_pearson / val_spearman
2. 测试集 Pearson IC / Spearman IC / Directional Accuracy
3. 与上次训练结果的对比（如果存在 `output/` 下的历史记录）

## 训练技巧（来自教材第17章 + 老学长经验）

- **梯度累积**：`GRAD_ACCUM_STEPS=4` 等效 batch=1024，比单纯加大 BATCH_SIZE 更稳定
- **学习率**：先从 1e-4 开始，loss 不降则试 5e-5 或 1e-3
- **特征消融**：先跑 `ALL` 确认管线通，再加 `PV` / `PV+N` / `PV+NA+DT` 对比
- **Seeds**：正式实验用 [0,1,2] 看均值±标准差，调参阶段用 [0] 即可
- **早停**：val_loss 连续 5 epoch 不降就停，改 `training_frame.py` 可加 patience

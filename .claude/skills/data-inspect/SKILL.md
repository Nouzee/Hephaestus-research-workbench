---
name: data-inspect
description: 检查BTC L2数据质量。查看特征分布、label统计、缺失值、异常值。在训练前或训练效果异常时使用。
---

# 数据检查 Skill

## 第一步：加载数据

```python
import pandas as pd
import numpy as np

# 加载训练用的 concat 数据
train_lob = pd.read_parquet("output/hephaestus_btc_regression/<LABEL_TAG>/concat/train_lob_concat.parquet")
train_label = pd.read_parquet("output/hephaestus_btc_regression/<LABEL_TAG>/concat/train_label_concat.parquet")

print(f"LOB shape: {train_lob.shape}")
print(f"Label shape: {train_label.shape}")
print(f"Columns: {train_lob.columns.tolist()}")
```

## 第二步：检查Label分布

```python
y = train_label.iloc[:, 0].to_numpy(dtype=np.float64)
finite = y[np.isfinite(y)]

print(f"Total: {len(y):,}")
print(f"Valid (finite): {len(finite):,} ({len(finite)/len(y)*100:.1f}%)")
print(f"NaN: {np.isnan(y).sum():,}")
print(f"Inf: {np.isinf(y).sum():,}")

if len(finite) > 0:
    print(f"\nLabel stats (RAW, unscaled):")
    print(f"  Mean:    {np.mean(finite):.8f}")
    print(f"  Std:     {np.std(finite):.8f}")
    print(f"  Min:     {np.min(finite):.8f}")
    print(f"  Max:     {np.max(finite):.8f}")
    print(f"  Q01:     {np.quantile(finite, 0.01):.8f}")
    print(f"  Q05:     {np.quantile(finite, 0.05):.8f}")
    print(f"  Q50:     {np.quantile(finite, 0.50):.8f}")
    print(f"  Q95:     {np.quantile(finite, 0.95):.8f}")
    print(f"  Q99:     {np.quantile(finite, 0.99):.8f}")
    print(f"  |y|<1e-6: {(np.abs(finite)<1e-6).sum():,} ({(np.abs(finite)<1e-6).mean()*100:.2f}%)")
```

**健康标准**：
- Valid ratio > 50%（否则检查 label 计算逻辑）
- NaN ratio < 50%
- Mean ≈ 0（否则 label 有偏）
- Q01 到 Q99 之间跨度合理（BTC 10s 收益通常在 ±0.1% 内）

## 第三步：检查特征分布

```python
feat_cols = [c for c in train_lob.columns if c not in ("Time", "TimeDiff")]
X = train_lob[feat_cols].to_numpy(dtype=np.float64)

for i, col in enumerate(feat_cols):
    col_data = X[:, i]
    finite_col = col_data[np.isfinite(col_data)]
    print(f"\n{col}:")
    print(f"  NaN: {np.isnan(col_data).sum():,}")
    print(f"  Mean={np.mean(finite_col):.4f}  Std={np.std(finite_col):.4f}")
    print(f"  Min={np.min(finite_col):.4f}  Max={np.max(finite_col):.4f}")
```

**健康标准**：
- Std ≈ 1.0（归一化后）
- 无全零列（如果不是 ablation 模式）

## 第四步：检查样本窗口

```python
# 验证 WindowDataset 有效性
T = 500  # LABEL_K
X_all = X
dt_all = train_lob["TimeDiff"].to_numpy(dtype=np.float32)
y_all = y

valid_mask = np.isfinite(y_all)
valid_end = np.where(valid_mask)[0]
valid_end = valid_end[valid_end >= (T - 1)]

print(f"\nWindowDataset stats:")
print(f"  Total rows: {len(X_all):,}")
print(f"  Valid windows: {len(valid_end):,}")
print(f"  Train windows (80%): {int(len(valid_end)*0.8):,}")
print(f"  Val windows (20%): {int(len(valid_end)*0.2):,}")
```

**健康标准**：Valid windows > 10000（太少则无法训练）

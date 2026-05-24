---
name: result-analyze
description: 分析训练结果。读取 output/ 下的 metrics 和 predictions，对比不同实验，给出改进建议。在训练完成后使用。
---

# 结果分析 Skill

## 第一步：收集结果

```python
import pandas as pd
import numpy as np
import json, os, glob

# 找所有完成的 runs
runs_dir = "output/hephaestus_btc_regression/<LABEL_TAG>/runs"
metrics_files = glob.glob(f"{runs_dir}/**/test_metrics.json", recursive=True)

results = []
for f in metrics_files:
    with open(f) as fh:
        d = json.load(fh)
    results.append(d)

df = pd.DataFrame(results)
print(f"Found {len(df)} completed runs")
```

## 第二步：核心指标总览

```python
key_cols = ["feature_set", "seed", "pearson_ic", "spearman_ic",
            "mae", "rmse", "directional_accuracy", "best_val_pearson"]
if all(c in df.columns for c in key_cols):
    print(df[key_cols].to_string(index=False))
```

## 第三步：按特征集汇总

```python
for fs in sorted(df["feature_set"].unique()):
    g = df[df["feature_set"] == fs]
    if g.empty: continue
    print(f"\n=== {fs} (n={len(g)}) ===")
    for metric in ["pearson_ic", "spearman_ic", "directional_accuracy"]:
        if metric in g.columns:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"  {metric}: {vals.mean():.6f} ± {vals.std(ddof=1):.6f}")
```

## 第四步：诊断 + 建议

根据指标组合判断问题：

| 现象 | 诊断 | 建议 |
|---|---|---|
| Pearson IC ≈ 0, Spearman IC ≈ 0 | 模型没学到任何东西 | 1. 检查label是否正确 2. 增大epoch/lr 3. 换更简单的模型 |
| Pearson IC > 0 但 Spearman IC < 0 | 模型过拟合线性关系，忽略了排序 | 加正则化、加dropout、减少模型容量 |
| Directional Acc ≈ 0.5 | 随机猜测水平 | 同第一行 |
| train_pearson >> test_pearson | 过拟合 | 加weight_decay、加dropout、early stopping |
| val_loss下降但test指标差 | 验证集泄露 / 分布偏移 | 检查时间顺序是否严格、train/test是否有重叠 |
| 所有指标都接近0但非NaN | 特征无效 | 做特征消融对比：ALL vs PV vs PV+N 看哪个有用 |

## 第五步：对比历史最佳

如果有之前的实验结果（`output/` 下其他 LABEL_TAG 目录），加载对比：

```python
# 找历史 summary
old_csvs = glob.glob("output/hephaestus_btc_regression/*/summary/summary.csv")
for f in sorted(old_csvs):
    print(f"\n--- {f} ---")
    old = pd.read_csv(f)
    print(old.to_string(index=False))
```

输出当前 vs 历史最佳的差距，给出下一步调参方向。

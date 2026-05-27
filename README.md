# Hephaestus — LLM-Assisted Quant Research Infrastructure

面向市场微观结构研究的模块化实验工作台。  
覆盖特征提取、状态识别、回测执行、归因分析——以 **pipeline-as-experiment** 方式组织。

---

## 项目概述

Hephaestus 是一个持续演化的量化研究基础设施，覆盖 A 股 L2 订单簿和 BTC 永续合约两个市场。

研究工作流以 LLM 辅助方式组织：实验设计、代码生成、结果分析环节由 LLM 辅助加速，研究者主导假设形成与决策判断。项目的核心价值在于将高频研究中的各环节模块化，形成可复现、可迭代的实验管线。

---

## 架构

```
Raw L2 Data (orderbook + trades)
    │
    ▼
Feature Engine (16 microstructure variables)
    │
    ▼
State / Regime Layer (clustering + risk scoring)
    │
    ▼
Signal / Filter Layer (state selection)
    │
    ▼
Execution / Backtest Engine (stochastic fill simulation)
    │
    ▼
Attribution & Reporting (PnL decomposition, state economics)
```

---

## 仓库结构

```
Hephaestus/
├── modules/                 # 核心库
│   ├── probability/         #   随机过程建模工具 (11 modules)
│   ├── execution/           #   成交模拟 + PnL 归因
│   ├── risk/                #   风控状态机
│   └── research/            #   市场解构实验模块
│
├── experiments/             # 实验管线 (37 scripts)
│   ├── ashare/              #   A 股研究
│   └── btc/                 #   BTC 研究
│
├── strategies/
│   └── hephaestus-ssp/      # A 股稀疏状态执行策略
│
├── examples/                # 可运行示例
│   └── demo_pipeline.py     #   最小可运行 demo
│
├── docs/                    # 理论文档
│   ├── LEXICON.md
│   ├── DEFINITIONS.md
│   ├── INTERFACE.md
│   └── TRANSLATION.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 研究覆盖

### A 股 L2 (000333, 美的集团, 81 天 10 档深度)

| 阶段 | 内容 | 状态 |
|------|------|------|
| Regime 发现 | 8 种市场状态自动识别 | 已验证 |
| 风险排序 | Quantile-based risk scoring | 已验证 |
| 状态选择 | CORE state filtering, rolling walk-forward stable | 已验证 |
| 状态经济学 | Per-state EV, fill prob, adverse ratio | 已验证 |
| 策略验证 | Binary filter vs continuous, anti-tests | 已验证 |

### BTC 永续合约 (3.9M ticks)

| 阶段 | 内容 | 状态 |
|------|------|------|
| 字典学习 | Sparse coding + 多尺度分解 | 已完成 |
| 因果验证 | Lag causality, null models, confounder tests | 已完成 |
| 市场解构 | 3-layer decomposition, mode extraction | 已完成 |
| 动力学分析 | State transition estimation | 实验中 |

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 demo（无需真实数据）
python examples/demo_pipeline.py
```

Demo 使用随机生成的模拟数据展示完整的 feature → regime → filter → backtest 流程。

---

## 关键技术发现

**已验证**
- Toxicity Inversion：紧价差 = adverse selection 主导的亏损区；宽价差 = mean reversion 保护
- Stress attractor 持续性 0.939，跨 12 周 CV=0.18
- 风险相变点在 30%ile，15 个滚动窗口全部一致
- Binary state filter 严格优于连续权重
- CORE 状态跨窗重叠率 76%

**实验中**
- 市场整体无内在表示尺度，但在给定 regime 下存在条件结构
- 状态转移矩阵通过 3/3 反证检验，但单变量控制不可行

---

## 研究 Workflow

1. **Hypothesis** — 研究者主导假设，LLM 辅助细化
2. **Pipeline Generation** — 生成自包含的实验脚本
3. **Execution** — 运行，产出结构化结果
4. **Analysis** — LLM 辅助解读，研究者做决策
5. **Iteration** — 深化 / 转向 / 放弃

每条 pipeline 从数据加载到结论输出可独立复现。

---

## 文档

| 文档 | 用途 |
|------|------|
| `docs/LEXICON.md` | 72 术语标准词典 |
| `docs/DEFINITIONS.md` | 核心概念严格定义 |
| `docs/INTERFACE.md` | 模块接口规范 |
| `docs/TRANSLATION.md` | 术语 → 概率对象映射 |

---

## 技术栈

Python · Polars · NumPy · scikit-learn · SciPy · hmmlearn  
LLM-assisted workflow · Pipeline-as-experiment

---

## 规模

```
138 Python 文件    34,000+ 行
37 条实验管线     2 个市场覆盖
11 模块工具层     4 份理论文档
```

---

## 关于本项目

Hephaestus 是一个持续演化的量化研究工作台。核心价值在于建立了一套可复现、可迭代的研究基础设施——覆盖从原始数据到策略验证的完整链路。LLM 辅助加速了实验设计和迭代速度，但不替代研究者的判断和决策。

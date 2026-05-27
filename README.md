# Hephaestus — Agent-Oriented Quant Research Workbench

面向市场微观结构研究的模块化研究工作台。  
覆盖高频数据处理、状态识别、实验编排、策略回测与结果归因的完整链路。

---

## 项目概述

Hephaestus 是一个长期演化的量化研究基础设施，目前覆盖 BTC 永续合约和 A 股 L2 订单簿两个市场。

研究工作流以人机协同方式组织：LLM 辅助实验设计、代码生成与结果分析，研究者主导假设形成与决策判断。项目的核心价值在于将高频研究中的特征提取、状态识别、回测执行、归因分析等环节模块化，形成可复现、可迭代的实验工作流。

---

## 架构

```
                        Raw L2 Data (orderbook + trades)
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │    Feature Engine         │
                    │  (16 microstructure vars) │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │       State / Regime Layer        │
              │  (unsupervised clustering +       │
              │   quantile-based risk scoring)    │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │        Signal / Filter Layer      │
              │  (state selection, CORE filter)   │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │    Execution / Backtest Engine    │
              │  (stochastic fill, MC evaluation) │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │     Attribution & Reporting       │
              │  (PnL decomposition, state econ)  │
              └──────────────────────────────────┘
```

---

## 仓库结构

```
Hephaestus/
├── modules/
│   ├── probability/      # 随机过程建模工具
│   │   ├── stochastic_state      — 统一状态表示
│   │   ├── transition_kernel     — 状态转移估计
│   │   ├── hazard_model          — 风险评分函数
│   │   ├── policy                — 策略分布抽象
│   │   ├── mc_backtest           — 蒙特卡罗评估
│   │   └── execution_engine      — 组件集成执行
│   │
│   ├── execution/        # 成交模拟 + PnL 归因
│   ├── risk/             # 风控状态机 + 仓位管理
│   ├── research/         # 市场解构实验模块
│   └── dictionary/       # 字典学习 + 信号路由
│
├── experiments/                  # 独立实验管线 (37 scripts)
│   ├── ashare/                   # A 股研究
│   └── btc/                      # BTC 研究
│
├── strategies/
│   └── hephaestus-ssp/           # A 股稀疏状态执行策略
│
├── docs/                         # 理论文档
│   ├── LEXICON.md                # 术语词典
│   ├── DEFINITIONS.md            # 核心概念严格定义
│   ├── INTERFACE.md              # 模块接口规范
│   └── TRANSLATION.md            # 术语映射表
│
├── projects/                     # 研究子项目
├── strategies/                   # 策略封装
├── requirements.txt
└── README.md
```

---

## 研究覆盖

### A 股 L2 (000333, 美的集团, 81 天 10 档深度)

| 阶段 | 内容 | 状态 |
|------|------|------|
| Regime 发现 | 8 种市场状态自动识别 (KMeans on 16 L2 features) | 已验证 |
| 风险排序 | Quantile-based risk scoring; tox inversion 验证 | 已验证 |
| 状态选择 | 168 → 15 CORE states, rolling walk-forward stable | 已验证 |
| 状态经济学 | Per-state EV, fill prob, adverse ratio, queue position | 已验证 |
| 策略验证 | Binary filter vs continuous weighting; anti-tests | 已验证 |

### BTC 永续合约 (3.9M ticks)

| 阶段 | 内容 | 状态 |
|------|------|------|
| 字典学习 | Sparse coding + 多尺度分解 | 已完成 |
| 因果验证 | Lag causality, null models, confounder tests | 已完成 |
| 市场解构 | 3-layer decomposition, mode extraction | 已完成 |
| 动力学分析 | State transition estimation, stability tests | 实验中 |

---

## 快速开始

```bash
# A-Share: 全量 regime 发现
python experiments/ashare/regime_discovery.py

# Rolling walk-forward: CORE 稳定性验证
python experiments/ashare/rolling_validation.py

# 状态经济学表
python experiments/ashare/state_economics.py

# BTC: 市场解构管线
python experiments/btc/market_decomposition.py

# MSDP 最小尺度发现
python experiments/btc/scale_discovery.py
```

---

## 关键技术发现

### 已验证

- R5 Stress Attractor：持续性 0.939，跨 12 周 CV=0.18
- Toxicity Inversion：紧价差 = 结构性亏损区（adverse selection 主导）；宽价差 = 结构性盈利区（mean reversion 保护）
- 相变点在 tox 分位 30%ile，15 个滚动窗口全部一致
- Binary state filter 严格优于连续权重方案
- CORE 状态跨窗重叠率 76%，方向稳定性 100%

### 实验中

- 市场整体无内在表示尺度（MSDP CASE_C），但在给定 regime 下存在条件结构
- A-matrix 通过 3/3 反证检验，但单变量控制不可行
- Predictability ceiling 约 8% R²

---

## 研究 Workflow

Hephaestus 的研究流程以 **pipeline-as-experiment** 方式组织：

1. **Hypothesis** — 研究者主导假设形成，LLM 辅助细化
2. **Pipeline Generation** — 生成独立的实验脚本
3. **Execution** — 运行实验，产出结构化结果
4. **Analysis** — LLM 辅助解读输出，研究者做决策判断
5. **Iteration** — 基于分析结果决定深化、转向或放弃

每条 pipeline 是自包含的：从数据加载到最终结论输出，可独立复现。

---

## 文档体系

| 文档 | 用途 |
|------|------|
| `docs/LEXICON.md` | 72 术语标准词典，统一项目语言 |
| `docs/DEFINITIONS.md` | 10 核心概念严格分析学定义 (7 字段模板) |
| `docs/INTERFACE.md` | 模块间概率对象接口规范 |
| `docs/TRANSLATION.md` | 原始术语 → 概率对象映射 |

---

## 技术栈

Python · Polars · NumPy · scikit-learn · SciPy · hmmlearn  
LLM-assisted workflow · Pipeline-as-experiment · Multi-market coverage

---

## 项目规模

```
138 Python 文件    34,000+ 行
37 条实验管线     11 模块随机过程工具
2 个市场覆盖      81d A-Share + 3.9M tick BTC
4 份理论文档      72 术语标准词典
```

---

## 关于本项目

Hephaestus 是一个持续演化的量化研究工作台。项目的核心价值不在于单一的盈利策略，而在于建立了一套可复现、可迭代、可扩展的研究基础设施——覆盖从原始数据到策略验证的完整链路。LLM 作为 workflow 辅助工具嵌入研究流程的各个环节，加速了实验设计和迭代速度。

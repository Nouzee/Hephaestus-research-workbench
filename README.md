# Hephaestus — Agent-Oriented Quant Research Workbench

面向市场微观结构研究的模块化研究工作台。  
用于组织高频数据处理、状态识别、实验编排、策略回测与结果归因流程。

---

## 项目概述

Hephaestus 是一个长期演化的量化研究系统，覆盖 BTC 永续合约和 A 股 L2 订单簿两个市场。

研究工作流以人机协同方式组织：利用 LLM 辅助实验设计、代码生成、结果分析与研究迭代，构建面向微观结构研究的半自动化 workflow。核心价值在于将研究过程中的数据处理、特征提取、状态识别、回测执行、归因分析等环节模块化，支持快速实验迭代和结果复现。

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
              │  (KMeans regime + quantile tox)   │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │        Signal / Filter Layer      │
              │  (tox inversion, CORE selector)   │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │    Execution / Backtest Engine    │
              │  (stochastic fill, MC paths)      │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │     Attribution & Reporting       │
              │  (state economics, PnL decomp)    │
              └──────────────────────────────────┘
```

---

## 仓库结构

```
Hephaestus/
├── modules/
│   ├── probability/      # 随机过程建模层 (11 modules)
│   │   ├── stochastic_state      — S_t 统一随机状态
│   │   ├── transition_kernel     — P(Z_{t+1}|Z_t) 马尔可夫核
│   │   ├── hazard_model          — h(X_t) 连续危险率
│   │   ├── policy                — π(a|S_t) 随机策略
│   │   ├── mc_backtest           — E[R|π] via 蒙特卡罗路径
│   │   ├── stochastic_geometry   — Fisher metric / entropy flow
│   │   ├── execution_engine      — 组件集成随机控制
│   │   ├── scale_flow            — KL/Wasserstein/决策稳定性
│   │   ├── msdp                  — 重整化群不动点检测
│   │   └── scale_consistency     — 反证测试套件
│   │
│   ├── dictionary/       # 字典学习 + 多尺度分解 + 信号路由
│   ├── execution/        # 成交模型 + 硬化执行 + PnL 归因
│   ├── risk/             # 分层 FSM + 仓位缩放 + 库存偏斜
│   ├── state/            # MarketState + 一致性扫描
│   ├── research/         # 市场解构 + 因果图 + 冲击核 + 生成器
│   ├── forge/            # Mamba / NeuralSDE / STGNN / Alpha 工厂
│   ├── attribution/      # 归因分析 / Barra / 防作弊
│   └── crucible/         # 回测引擎 / 优化器
│
├── projects/
│   ├── ashare/                       # A 股 regime 分割引擎
│   └── compressibility_frontier/     # 可压缩性边界研究
│
├── strategies/
│   └── hephaestus-ssp/               # A 股稀疏状态执行策略
│
├── run_*.py (37 pipelines)           # 独立实验管线
│
├── LEXICON.md                        # 72 术语标准词典
├── DEFINITIONS.md                    # 10 核心概念严格定义
├── INTERFACE.md                      # 9 概率对象 + 模块接口
└── TRANSLATION.md                    # 80+ 术语 → 概率对象映射
```

---

## 研究覆盖

### BTC 永续合约 (3.9M ticks)

| 阶段 | 内容 |
|------|------|
| 字典学习 | Sparse coding, K=3 atoms, 因果小波 |
| 毒性检测 | Gram topology, HMM regime, 多信号前兆 |
| 因果验证 | Lag causality, shock isolation, null models, confounder |
| 市场解构 | 3-layer decomposition (OF/LQ/PI), 8D mode extraction |
| 动力学 | A-matrix identification, SVCT falsification, controllability |
| 信息几何 | 6-mode minimal basis, predictability ceiling R²=8% |
| 尺度分析 | Observation operator sweep, MSDP (CASE_C: no intrinsic scale) |

### A 股 L2 (000333, 美的集团, 81 天 10 档深度)

| 阶段 | 内容 |
|------|------|
| Regime 发现 | 8 market states via KMeans on 16 L2 features |
| Toxicity 分析 | Tox inversion (tight spread = loss, wide = profit) |
| CORE 选择 | 168 → 15 states stable across rolling windows |
| 状态经济学 | EV/fill, A/E ratio, fill prob, queue position per state |
| Pareto 重构 | 3D Pareto front on (EV, A/E, Fill%) |
| 策略验证 | Rolling walk-forward (20/5, 15/5), anti-tests |

---

## 快速开始

```bash
# A-Share: 全量 regime 发现 (59 days, ~30s)
python run_ashare_full.py

# Rolling walk-forward: CORE 稳定性验证
python run_rolling_v2.py

# 状态经济学表: EV / adverse / fill_prob per state
python run_state_economics.py

# Pareto CORE 重构
python run_pareto_core.py

# BTC: 市场解构管线
python run_market_decon.py

# MSDP 最小尺度发现
python run_msdp.py
```

---

## 关键技术发现

### 微观结构

- R5 Stress Attractor：持续性 0.939，跨 12 周 CV=0.18
- Toxicity Inversion：紧价差 = adverse selection 主导的结构性亏损区；宽价差 = mean reversion 保护的结构性盈利区
- 相变点在 tox 分位 30%ile，15 个滚动窗口全部一致
- FRAGILE 状态下不是模式坍缩，而是全模式方差放大 (2-3x)

### 动力系统

- A-matrix 通过 3/3 反证检验（Null model, Time reversal, Confounder nulling）
- 系统不可单变量控制（Gain=0），表现为自归一化随机场
- Predictability ceiling: R² ≈ 8%
- MSDP 结论：市场整体无内在表示尺度 (CASE_C)，但在给定 regime 下存在条件结构

### 策略层面

- Binary state filter 严格优于连续权重方案 (PnL +111%)
- CORE 状态跨窗重叠率 76%，方向稳定性 100%
- Pareto CORE：三维支配对该数据过于严格 (19→3)，建议两步过滤

---

## 研究 Workflow

Hephaestus 的研究流程以 **pipeline-as-experiment** 方式组织：

1. **Hypothesis** — 在 LLM 辅助下形成可测试的研究问题
2. **Pipeline Generation** — 生成独立的 `run_*.py` 实验脚本
3. **Execution** — 运行实验，产出结构化结果
4. **Analysis** — 在 LLM 辅助下解读输出，形成下一步假设
5. **Iteration** — 基于分析结果决定：深化 / 转向 / 放弃

每条 pipeline 是自包含的：从数据加载到最终结论输出，可独立复现。

---

## 文档体系

| 文档 | 用途 |
|------|------|
| `LEXICON.md` | 72 术语标准词典，统一项目语言 |
| `DEFINITIONS.md` | 10 核心概念严格分析学定义 (7 字段模板) |
| `INTERFACE.md` | 模块间概率对象接口规范 |
| `TRANSLATION.md` | 原始术语 → 概率对象映射 |

---

## 技术栈

Python · Polars · NumPy · scikit-learn · SciPy · hmmlearn  
Agent-assisted workflow (Claude) · Pipeline-as-experiment · 多轮迭代研究

---

## 项目规模

```
138 Python 文件    34,000+ 行
37 条实验管线     11 模块随机过程引擎
7 个子包          4 份理论文档
2 个市场覆盖      81d A-Share + 3.9M tick BTC
```

---

## 关于本项目

Hephaestus 是一个持续演化的量化研究工作台。项目的核心价值不在于单一的盈利策略，而在于建立了一套可复现、可迭代、可扩展的研究基础设施——覆盖从原始数据到策略验证的完整链路。LLM 作为 workflow 辅助工具嵌入研究流程的各个环节，加速了实验设计、代码生成和结果分析的循环速度。

# Hephaestus — AI Agent-Driven Market Microstructure Engine

> 一个由 Claude 作为研究 agent 驱动的金融市场微观结构概率引擎。  
> 132 个模块、34 条自主实验管线、11 模块随机过程推理层。  
> 从原始 L2 订单簿到可执行策略——全链路 agent 自主完成。

---

## 核心定位

**Hephaestus 不是一个量化回测框架。它是一个 AI agent 驱动的市场微观结构研究系统。**

传统 quant 流程：人工假设 → 人工编码 → 人工回测 → 人工分析。  
Hephaestus 流程：**agent 自主提出假设 → 自主生成实验管线 → 自主分析结果 → 自主迭代收敛。**

34 条 `run_*.py` 管线全部由 agent 在对话中自主生成、调试、执行、分析。每条管线代表一个完整的研究阶段，从数据加载到最终结论输出，无需人工介入。

---

## 研究路线图

```
BTC L2 字典学习
  → Gram + HMM 毒性检测
  → 多信号前兆 + FSM 风控
  → 因果验证（4 实验：lag / shock / null / regime）
  → 市场解构（3 层生成分解）
  → 8D 模式提取（SVD 动力学基）
  → 模式动力学（A-matrix 辨识，SVCT 伪证检验）
  → 最小信息基底（6-mode, predictability ceiling = 8% R²）
  → 信息几何 / 漂移起源 / 重整化不动点检测

A-Share L2 Regime 图谱
  → 8 种市场状态自动发现（KMeans on 16 L2 features）
  → R5 Stress Attractor（持续性 0.939, 跨 12 周 CV=0.18）
  → Toxicity Inversion 发现（紧价差 = 结构性亏损, 宽价差 = 保护）
  → 168 → 15 CORE states（稀疏状态参与）
  → Pareto CORE 重构
  → Production v4: out-of-sample 盈利
```

---

## Agent 能力展示

### 自主实验设计

Agent 在对话中独立完成以下实验的完整设计-执行-分析闭环：

| 实验 | Agent 自主完成的任务 |
|------|-------------------|
| **SVCT 伪证检验** | 设计 Null model / Time reversal / Confounder nulling 三类反证，判定 A-matrix 是真实动力学结构 |
| **MSDP 最小尺度发现** | 构建 5 级尺度层级，设计 SNR/variance/monotonicity 守门条件，执行 80 路径 MC 扫描，输出 CASE A/B/C 分类 |
| **Rolling Walk-Forward** | 自主发现固定阈值失效，切换为相对分位数校准，跑 7+8 窗双方案验证，输出 CORE 稳定性报告 |
| **Pareto CORE 重构** | 将经验 state set 重构为 Pareto-optimal 执行流形，发现纯 Pareto 对该数据过于严格，输出 CASE_B 并给出两步过滤替代方案 |
| **State Economics Table** | 自主补全 EV/fill_prob/adverse/queue_position 四维状态经济学表，识别 R2 为高 EV 危险陷阱 |

### 自主代码架构

Agent 在对话中自主完成了项目从"单文件脚本"到"模块化概率引擎"的架构演进：

```
modules/probability/          ← Agent 在对话中自主设计并实现的 11 模块随机过程引擎
modules/research/             ← Agent 自主设计市场解构实验室
modules/execution/            ← Agent 自主设计执行模拟层
projects/ashare/              ← Agent 自主设计 A 股 regime 分割引擎
strategies/hephaestus-ssp/    ← Agent 自主封装的生产策略包
```

### 自主收敛与纠错

Agent 在对话中多次展示自主纠错能力：
- Tox 阈值：发现固定绝对值失效 → 自主切换为滚动分位数校准
- Tox inversion：发现方向稳定但边界敏感 → 自主补充连续 rank curve 验证
- 连续权重：自主发现 binary filter 严格优于 sigmoid weighting → 输出"continuous approximation is strictly worse"结论并停止该方向
- R2 危险态：通过 state economics 表自主识别 R2 为高 EV 高 adverse 陷阱

---

## Probability Engine v1

Agent 在对话中自主设计并实现的随机过程推理层：

| 模块 | Agent 定义的概率对象 |
|------|-------------------|
| `stochastic_state` | $S_t = (X_t, Z_t, H_t, M_t)$ — 统一随机状态 |
| `transition_kernel` | $P(Z_{t+1} \mid Z_t)$ — 马尔可夫核 |
| `hazard_model` | $h(X_t) \in [0,1]$ — 连续危险率函数 |
| `policy` | $a_t \sim \pi(a \mid S_t)$ — 随机策略分布 |
| `mc_backtest` | $E[R \mid \pi]$ via Monte Carlo 路径 |
| `stochastic_geometry` | Fisher metric + entropy production + drift field |
| `execution_engine` | 全组件集成随机控制 |
| `scale_flow` | 分布级 KL/Wasserstein/决策稳定性不动点检测 |
| `msdp` | 重整化群实验，CASE A/B/C 强制三分类输出 |

每个模块都由 agent 在对话中完成：数学定义 → Python 实现 → 单元测试 → 集成验证。

---

## 关键技术发现

### 微观结构

- 市场在 FRAGILE 状态下不是"坍缩"而是"全模式放大"(2-3x 方差)
- Stress attractor 持续性 0.939，跨 12 周 CV=0.18
- Toxicity Inversion：价差宽度与实际毒性反向——紧价差才是真正的毒
- 紧价差 = 结构性亏损区（adverse selection 主导），宽价差 = 结构性盈利区（mean reversion 保护）
- 相变点在 tox 分位 30%ile——**所有 15 个滚动窗口全部一致**

### 动力系统

- A-matrix 是真实动力学结构（时间打乱后 ρ 暴跌 8x，3/3 反证测试通过）
- 系统不可单变量控制（Gain=0）——自归一化随机场
- 可预测性上界：R²=8%
- 6-mode 最小信息基（98.3% coverage）
- **MSDP 结论：市场整体无内在表示尺度（CASE_C），但在给定 regime 下存在条件结构**

### 策略

- Binary state filter 严格优于连续权重（+111% PnL）
- Pareto CORE 发现：三维 Pareto 对该数据过于严格（19→3 状态）
- CORE 重叠率 76% 跨窗稳定
- Out-of-sample 盈利（SSP Production v4）

---

## 项目规模

```
132 Python 文件    32,000+ 行代码
34 条独立实验管线  11 模块随机过程引擎
7 个子包           4 份理论文档
2 个完整市场研究   81 天 A-Share + 3.9M tick BTC
```

---

## 文档体系

Agent 在对话中自主建立的四层理论文档：

| 文档 | 内容 |
|------|------|
| `LEXICON.md` | 72 术语标准词典，12 类别，命名规范 + 禁止用法 |
| `DEFINITIONS.md` | 10 核心概念严格分析学定义（7 字段模板，Parts A-G） |
| `INTERFACE.md` | 9 核心概率对象 + 10 模块接口规范 |
| `TRANSLATION.md` | 80+ 原始术语 → 概率对象映射表 |

---

## 运行示例

```bash
# A-Share 全量 regime 发现
python run_ashare_full.py

# 滚动验证 CORE 稳定性
python run_rolling_v2.py

# 状态经济学表
python run_state_economics.py

# MSDP 最小尺度发现
python run_msdp.py

# Pareto CORE 重构
python run_pareto_core.py
```

---

## 技术栈

Python · Polars · NumPy · scikit-learn · SciPy · hmmlearn  
Claude Agent SDK · 自主实验设计 · 多轮对话研究

---

## 关于本项目

Hephaestus 展示的核心能力不是"写出了多少策略"，而是：

> **一个人 + 一个 AI agent，在对话中自主完成从原始数据探索到理论收敛的完整研究闭环。**

Agent 不是代码补全工具。Agent 是研究伙伴——提出假设、设计实验、分析结果、纠错迭代、建立理论。本项目中的所有模块、实验、文档，都是在人机对话中由 agent 自主完成的。

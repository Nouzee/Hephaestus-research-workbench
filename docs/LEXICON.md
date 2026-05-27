# Hephaestus 标准词典 v1

> 市场是一个部分可观测的随机动力系统，通过状态、转移、传播核、危险率和条件收益来描述；稀疏状态参与是其执行层，最小传播单元是其理论层，最小表示基底是其压缩层。

---

## 1. 核心对象

### S_t — latent market state
潜在市场状态。不是单一特征，而是隐藏的市场结构状态。一切 regime、stress、backbone、attractor 的底层对象。

### X_t — observation / observable state
观测到的微观结构特征。例如：spread, depth, imbalance, arrival, cancel, refill, vol, tox, flow_persistence。由 orderbook / trades 提取。

### Z_t — regime label
离散化市场状态标签。取值：{R0, R1, R2, R3, R4, R5, R6, R7, CORE}。用于状态分桶、策略门控、转移统计。

### A_t — action / execution decision
执行动作。例如：quote size, spread width, skew, cancel rate, participation rate, inventory cap。策略层输出。

### R_t — reward / payoff
收益随机变量。拆解为：spread capture + inventory PnL + adverse selection + slippage + fees。

### T — transition kernel
P(S_{t+1} | S_t)。状态转移核，描述 regime / state 如何演化。

### K(τ) — propagation kernel
局部事件到未来状态的传播核。描述 spread expansion, cancel burst, depth collapse 等事件如何影响未来分布。

### H_t — hazard / stress arrival probability
危险率。P(R5 within k windows | S_t)。stress early warning。

### U_t — minimal propagating unit
最小传播单元。能够稳定改变未来条件分布的局部事件。候选：spread shock, cancel burst, imbalance burst, depth evaporation, arrival shock。

---

## 2. 结构对象

### backbone / 1D backbone
市场状态空间中的主漂移方向。当前解释：flow_persistence + nonlinear_response 形成的主轴。

### mode / latent mode
由分解得到的潜在动力学模式。例如 M0~M7。用于状态压缩、动力学分析。

### CORE state
稳定正期望的可交易状态。跨样本、跨段稳定为正的状态集合。实际参与的唯一优先状态空间。

### attractor
状态系统中高持续性、易停留的盆地。例如 R5 stress attractor。解释状态锁定与风险聚集。

### metastability
局部稳定、偶发跳转的准稳定状态。解释 regime 的暂态停留与切换。

### directional field
具有方向性的动力学场。强调系统不是对称噪声，而是有时间箭头的演化过程。

### self-normalizing field
系统对单变量干预不敏感，会自吸收扰动。当前 controllability 实验的结论。

---

## 3. 概率对象

| 符号 | 含义 | 用途 |
|------|------|------|
| P(S_t \| X_{≤t}) | 给定历史观测的状态后验 | 状态识别 |
| P(Z_t \| X_t) | 给定当前观测的 regime 后验 | 在线 regime 识别 |
| P(S_{t+1} \| S_t) | 一阶状态转移分布 | 状态动力学核心 |
| P(R_t \| S_t, A_t) | 条件收益分布 | 策略评估与优化 |
| P(stress \| S_t) | 当前状态下进入 stress 的概率 | 风险预警 |
| P(trade \| S_t) | 当前状态下参与交易的概率 | 参与门控 |
| E[R_t \| S_t, A_t] | 条件期望收益 | 正期望状态筛选 |
| Var[R_t \| S_t, A_t] | 条件收益方差 | 风险评估 |

---

## 4. SSP 稀疏参与术语

| 术语 | 定义 |
|------|------|
| **SSP** | Sparse State Participation。只在少数高期望状态中参与交易。Hephaestus-SSP 执行核心。 |
| **sparse state** | 低频但高价值的状态。tox 4-6 + CORE state + 特定时段。 |
| **non-trade state** | 不交易状态。tox 0-3。默认退出区。 |
| **positive expectancy state** | E[R_t \| S_t, A_t] > 0 的状态。 |
| **state filter** | 对状态进行交易准入筛选的规则。策略门控层。 |
| **participation frontier** | 参与率与期望收益之间的最优边界。 |

---

## 5. 微观结构风险词典

| 术语 | 定义 |
|------|------|
| **toxicity** | 结构复合评分。当前更接近 spread-width / mean-reversion opportunity axis，不是字面"毒性"。 |
| **low-tox state** | tox 0-3。结构性亏损区。策略：withdraw / minimal quote。 |
| **high-tox state** | tox 4-6。稳定正期望区。策略：active participation。 |
| **adverse selection** | 成交后价格朝不利方向运动。做市核心风险项。 |
| **spread capture** | 挂单成交后赚取的价差收益。做市基础收益项。 |
| **inventory risk** | 持仓导致的价格波动风险。 |
| **liquidity stress** | 流动性压力态。R5 stress attractor。 |
| **queue collapse** | 盘口队列失稳 / 深度坍缩。压力前兆。 |
| **cancel burst** | 撤单强烈爆发。压力传播信号。 |
| **refill wave** | 盘口回补浪潮。市场自我修复信号。 |
| **arrival burst** | 成交到达率爆发。交易洪峰 / 冲击释放。 |

---

## 6. 动力学词典

| 术语 | 定义 |
|------|------|
| **drift** | 系统的平均演化方向。解释 backbone 与主趋势。 |
| **diffusion** | 随机扰动项。解释不可约随机性。 |
| **SDE** | dX_t = f(X_t)dt + Σ(X_t)dW_t。描述状态演化。 |
| **Lyapunov stability** | 扰动是否发散。判定系统局部稳定性。 |
| **local instability** | 局部发散 / 不稳定增长。解释 stress 放大。 |
| **regime-dependent dynamics** | 不同状态下动力学不同。核心事实之一。 |
| **time asymmetry** | 前向与反向时间结构不同。证明因果方向存在。 |
| **self-exciting** | 状态会自我强化。解释正反馈结构。 |
| **controllability** | 系统是否可由单变量稳定控制。当前结论：不可。 |

---

## 7. 信息几何与压缩词典

| 术语 | 定义 |
|------|------|
| **effective rank** | 有效维度。衡量结构压缩程度。 |
| **minimal basis** | 保留主要信息的最小表示基底。 |
| **compression** | 从高维观测中抽取更小表示。Hephaestus 基础能力。 |
| **observability** | 从观测中恢复状态的能力。 |
| **predictability ceiling** | 可预测性的上限。BTC 当前约 8% R²。 |
| **flat geometry** | 无明显曲率的空间结构。非复杂流形。 |
| **curvature** | 空间中局部几何弯曲。A 股 / BTC 某些层面弱或无。 |
| **information geometry** | 用概率分布的几何来描述市场。未来概率论化主体语言。 |
| **measure** | 对状态空间中事件的概率度量。 |
| **conditional distribution** | 在给定状态下的收益/转移分布。核心分析对象。 |

---

## 8. 语法层术语

| 术语 | 定义 |
|------|------|
| **market grammar** | 市场状态与转移的"语法"。描述哪些状态能合法组成高收益序列。 |
| **probabilistic grammar** | 用概率而非硬规则来描述市场语法。Hephaestus 理论化目标。 |
| **event sequence** | 局部事件按时间形成的序列。如 spread expansion → cancel burst → depth collapse → stress。 |
| **transition phrase** | 一段局部状态转移模式。语法单元。 |
| **propagation motif** | 可重复的传播模式。最小传播结构。 |
| **canonical state sequence** | 标准化的状态转移路径。如 R5 → R3 → recovery。 |
| **illegal sequence** | 与高收益结构不一致的状态序列。过滤低价值路径。 |

---

## 9. 决策规则术语

| 术语 | 定义 |
|------|------|
| **active quote** | 积极报价。高期望状态中使用。 |
| **withdraw** | 退出 / 不参与。低期望状态中使用。 |
| **minimal quote** | 极小参与。过渡状态或边缘状态。 |
| **size multiplier** | 仓位放大系数。状态条件化执行。 |
| **spread multiplier** | 报价宽度调节系数。控制 adverse selection。 |
| **cancel policy** | 撤单规则。应对压力和毒性状态。 |
| **participation policy** | 状态条件下的参与规则。Hephaestus-SSP 核心执行逻辑。 |

---

## 10. 命名规范

1. **所有对象优先写成概率对象**。不写"某状态好/坏"，写"P(positive expectancy | S_t)"。
2. **所有规则优先写成条件分布**。不写 if-else，写"P(R_t | S_t, A_t)"。
3. **所有风险优先写成 hazard / toxicity / adverse selection**。不写"危险"，写"hazard 上升""toxic fill 变高"。
4. **所有结构优先写成 state / transition / propagation**。不写"模式"，写"状态、转移、传播核"。
5. **所有压缩优先写成 basis / observability / minimal representation**。不写"降维"，写"最小表示基底"。

---

## 11. 禁止用法

1. **不要把 alpha 当成主体概念**。alpha 只是某些条件分布下的正期望现象。
2. **不要把 regime 当成真理**。regime 是离散标签，不是本体。
3. **不要把 tox 当成字面毒性**。在当前系统里它更像结构复合评分。
4. **不要把 GNN / RL 当成目标本身**。它们只是后续对状态语言的计算工具。
5. **不要把 manifold 当成默认假设**。当前实验更支持 flat field + 1D backbone + sparse profitable states。

---

## 12. 项目命名空间

| 名称 | 含义 |
|------|------|
| **Hephaestus** | 整个研究框架 |
| **Hephaestus-SSP** | A 股稀疏状态执行策略 |
| **BTC-HFT-MarketMaker** | 姊妹项目：BTC 在线高频做市引擎 |

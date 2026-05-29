# 技术深度解析 — 面试备用

---

## 一、字典学习（BTC 阶段）

### 1.1 做什么

把 L2 orderbook 的 10 档买卖挂单量（20 维向量）压缩成少数几个"市场模式"。

### 1.2 数学

```
X: T × 20 矩阵（每行是一个时刻的 orderbook 状态，20 列 = 10 档 bid qty + 10 档 ask qty）

NMF 分解：
X ≈ W × H

W: T × k  （每个时刻在 k 个模式上的权重）
H: k × 20 （k 个模式各自的价格-数量分布）
```

例如 k=6 时，可能分解出：
- **Mode 1**：bid 侧深度集中，ask 薄 → "买盘支撑"
- **Mode 2**：ask 侧深度集中，bid 薄 → "卖盘压力"
- **Mode 3**：bid/ask 均衡，深度厚 → "稳定流动性"
- **Mode 4**：bid/ask 都薄，深度塌陷 → "流动性撤离"
- **Mode 5**：spread 宽，深度薄 → "高波动 / 新闻冲击"
- **Mode 6**：bid/ask 深度不对称 + spread 变化 → "方向性压力"

### 1.3 模式动力学

有了每个时刻的模式权重 W[t]，可以构建转移矩阵：

```
P(mode_j | mode_i) = count(mode_i → mode_j) / count(mode_i)
```

得到 6×6 或 8×8 的转移概率矩阵 A。然后分析：
- **自转移概率**：对角线元素，模式持续多久
- **吸收态**：哪些模式一旦进入就出不来
- **主要转移路径**：市场如何在模式间流转

### 1.4 为什么失败了

三个致命问题：

**问题 1：模式是描述性的，不是预测性的**

做了 Granger 因果检验：用过去的模式权重预测未来的 mid-price 变动。p 值不显著。模式告诉你"当前市场长什么样"，但不告诉你"接下来会怎样"。

**问题 2：R² 天花板 = 8%**

即使最优的基向量组合，也只能解释 mid-price 变动的 8%。剩下 92% 是噪声或更高阶的结构。

**问题 3：不可控**

做了 Controllability Test（MCT）：假设你能精确控制某个模式的权重（比如增加 bid 深度），能否把市场推向更有利的状态？答案是不能——单变量干预被系统吸收，其他维度自动补偿。

### 1.5 关键教训

```
信号识别（找到模式）
    ≠
可交易 edge（模式能预测未来方向）
    ≠
经济可行（扣除成本后盈利）
```

BTC 失败在第二步。这个教训直接塑造了之后 A 股的研究方向：**不再问"市场是什么样的"，而是问"哪些状态下被动做市是盈利的"**。

---

## 二、状态分割（A 股阶段核心方法）

### 2.1 从 orderbook 到特征

从每 tick 的 L2 orderbook 快照提取 16 个微观结构特征：

| 类别 | 特征 | 含义 |
|---|---|---|
| 价格 | Spread, Mid-price change | 交易成本、方向 |
| 深度 | Bid/Ask total depth (5 levels) | 流动性供给 |
| 平衡 | OBI = (BidDepth-AskDepth)/(BidDepth+AskDepth) | 买卖压力方向 |
| 波动 | Spread volatility, mid volatility | 不确定性 |
| 流量 | Trade count, trade size, direction imbalance | 成交强度 |
| 形态 | Depth concentration (level 1 vs levels 2-5) | 深度分布形状 |

每 100 tick 为一个窗口，计算窗口内每个特征的统计量（均值、标准差、偏度等）。

### 2.2 Regime 聚类

```
X: N_windows × 16 特征矩阵
↓ 标准化（前 20 天训练集的 mean/std）
↓ KMeans(n_clusters=8)
8 个市场 regime
```

8 个 regime 事后被标注为：

| Regime | 特征 | 经济学含义 |
|---|---|---|
| **R0** | 深度崩溃，高波动 | 流动性撤离 / 恐慌 |
| **R1** | Ask 重，bid 薄 | 卖盘压力主导 |
| **R2** | 深度极厚，spread 适中 | 深度流动性（陷阱！） |
| **R3** | 成交激增 | 大量交易到达 |
| **R4** | Bid 重，ask 薄 | 买盘支撑 |
| **R5** | 压力吸引子 | 持续性卖压 |
| **R6** | 主动买盘 + 宽 spread | 买方主导的波动 |
| **R7** | 主动卖盘 | 卖方主导 |

### 2.3 Toxicity 量化

**核心公式**：
```
Tox = Spread / TotalDepth
```

经济学直觉：
- **低 Tox**（窄 spread，厚深度）→ 竞争激烈，你的挂单排在队列后面，adverse selection 会吃掉你的 spread → **亏损**
- **高 Tox**（宽 spread，薄深度）→ 做市商少，spread 宽，mean reversion 保护你的仓位 → **盈利**

**关键：这一点是反直觉的。** 传统观念认为"低 tox = 安全"，但被动做市的视角完全相反——低 tox 意味着太多人在同一个地方抢同一个 spread。

### 2.4 三维状态编码

每个窗口被分配一个状态标签：

```
State = R{r}_q{tq}_T{td}

r: regime (0-7)          ← 市场微观结构类型
tq: toxicity quantile    ← spread/depth 的百分位（0=窄, 1=中, 2=宽）
td: time of day          ← 早盘/中盘/尾盘
```

总共 8 × 3 × 3 = 72 个 q2（宽 spread）状态 + 72 个 q1 + 72 个 q0 = 216 个可能状态。

### 2.5 相变点：30th Percentile

**关键发现：PnL/fill 在 tox ≈ 30th percentile 处发生突变。**

- tox < 30th：E[PnL/fill] ≈ 0 或负
- tox > 30th：E[PnL/fill] > 0，且单调递增

这个相变点**在所有 15 个滚动验证窗口中一致出现**。它不是过拟合——它是市场微观结构的一个稳健特征。

**为什么是相变而不是渐变？**
因为被动做市的盈亏取决于 queue position。在低 tox 状态，你的挂单排在几十万股的深度后面，永远成交不了；即使成交了，也是在 adverse selection 最严重的时候（有人用市价单扫掉了前面的深度，说明方向性强）。在高 tox 状态，前面只有少量深度，你的挂单快速成交，且 spread 够宽可以覆盖偶尔的不利选择。

### 2.6 从 216 到 21：ECORE 过滤

不是所有正 EV 状态都可以执行。需要同时满足：

```
ECORE 过滤条件：
1. ExecEV(s) = P(fill|s) × (SpreadCapture(s) - Adverse(s)) > 0
2. A/E ratio < 1.0（adverse 不能超过 spread）
3. P(fill) > 20%（必须有足够成交概率）
4. Queue wait < 100 ticks（不能排队太久）
5. NOT execution trap（像 R2 这种 EV 高但 fill 0.7% 的状态）
```

最终保留 **21 个 ECORE 状态**，排除 R2（执行陷阱）。

---

## 三、L2 队列模拟（EVL）

### 3.1 为什么 Bernoulli 不行

原来的 fill model：每条 quote 以固定概率 P=0.30 成交。这是完全不现实的——fill 概率高度依赖市场状态。R2（深度流动性）的 fill 是 0.7%，R6（主动买盘+宽 spread）的 fill 是 81%。

### 3.2 队列模拟引擎

对每 tick 的每条 quote：

```
1. 选择方向（bid 或 ask，随机 50/50）
2. 记录当前队列位置：
   bid: queue_position = BidOrderQty1(t) + OUR_SIZE
   ask: queue_position = OfferOrderQty1(t) + OUR_SIZE

3. 向前遍历后续 tick（最多 100 tick）：
   对每个后续 tick k：
     - 如果有人以我们的价格成交：
       queue_position -= TradeSize(k)
     - 如果深度缩减（撤单）：
       queue_position -= (OldDepth - NewDepth)
     - 如果 queue_position <= 0 且有成交：
       → FILL！记录成交时刻和 markout

4. 如果没有在 100 tick 内成交：
   → 未成交（记录为 unfilled quote）
```

### 3.3 输出

对每个状态 s，计算：
- **Real fill probability**：成交数 / 总挂单数
- **Markout**：fill 后 100 tick 的 mid 变动（bps），按方向符号化
- **Queue wait**：从挂单到成交的 tick 数
- **Bernoulli comparison**：如果只用 P=0.30，会成交多少次？

### 3.4 关键发现

| 指标 | Bernoulli 模型 | EVL 真实值 | 差距 |
|---|---|---|---|
| 平均 fill rate | 30% | **82.4%** | 2.75x |
| R2 fill rate | 30%（假设） | **0.7%** | 43x 高估 |
| R2 markout | 未建模 | **-49 bps** | 灾难级 |

**R2 为什么是陷阱**：
- 深度极厚（看起来安全）
- Spread 较宽（看起来利润高）
- 但你的挂单排在几十万股后面，永远到不了队首
- 即使成交了，也是因为有巨大的方向性订单扫掉了所有深度——此时 mid 已经移动了 49 bps 对你不利

---

## 四、库存控制（IECORE）

### 4.1 问题：库存随机游走

对称被动做市的自然结果：
```
bid fill (+1 share) 和 ask fill (-1 share) 以相同概率到达
→ 库存 = 随机游走
→ autocorr ≈ 1.0（单位根）
→ half-life ≈ 693 窗口（≈ 69,300 tick）——不存在自然均值回归
```

没有控制时，库存可以从 +45,000 漂移到 -40,000，完全不可预测。MTM 波动（inventory × Δmid）主导 PnL 方差。

### 4.2 Skewing 引擎

三个机制，按库存偏离程度逐级激活：

**Level 1：Size Skew**（始终在线）
```
inv_frac = inventory / max_position_limit   # ∈ [-1, 1]

if inv_frac > 0（long）:
    bid_mult = 1.0 - skew × inv_frac   # 减少买盘（0 → 1）
    ask_mult = 1.0 + skew × inv_frac   # 增加卖盘（1 → 2）

if inv_frac < 0（short）:
    bid_mult = 1.0 + skew × |inv_frac| # 增加买盘
    ask_mult = 1.0 - skew × |inv_frac| # 减少卖盘
```

**Level 2：Price Skew**（始终在线）
```
Long: bid 价格 shade 更差（远离 mid），ask 价格 shade 更好（靠近 mid）
      → 减少买入成交，增加卖出成交
Short: 反过来
```

**Level 3：Suppression**（极端库存时激活）
```
|inv_frac| > 70%: 同侧 fill 概率 × 0.3
|inv_frac| > 90%: 同侧 fill 概率 = 0（完全停止）
```

### 4.3 State-Conditioned Skew

不同状态的 adverse ratio 不同。高 A/E 状态（如 R6，A/E=0.45）的 spread capture 本来就有近一半被 adverse 吃掉——在这里加强 skew 可能把剩下的 edge 也扭曲掉。

```
skew_strength = base_skew × (1 - A/E_state)

R1（A/E=0.08）：skew ≈ 0.5 × 0.92 = 0.46（强 skew）
R6（A/E=0.45）：skew ≈ 0.5 × 0.55 = 0.28（弱 skew）
```

### 4.4 结果

| 指标 | 无控制 | Moderate (skew=0.5) | 改善 |
|---|---|---|---|
| 库存标准差 | 13,939 | 1,114 | **-92%** |
| Max DD | 1.72B | 176M | **-90%** |
| 总 PnL | +14.2B | +15.2B | **不变** |
| Half-life | 693w | 40w | **17x 更快回归** |

**核心结论：库存控制不创造 alpha。它降低噪音，让已有的执行 edge 不被 MTM 波动淹没。**

---

## 五、三重 Edge 框架

这是整个项目最核心的思维框架。面试时讲清楚这个，基本就能证明你懂 quant research 的本质。

### Layer 1: Statistical Edge（统计层面）

**定义**：存在统计显著的市场微观结构模式，能区分盈利和亏损状态。

**本项目证据**：
- Toxicity Inversion：高 tox = 正 EV，低 tox = 负/零 EV
- 相变点在 30th percentile，15 个滚动窗口一致
- 反证测试：shuffle tox → edge 消失；shuffle state → edge 消失

**关键点**：统计 edge 是必要的，但不是充分的。很多学术论文停在这一层。

### Layer 2: Execution Edge（执行层面）

**定义**：在考虑真实执行约束（队列位置、fill 概率、adverse selection、库存管理、MTM 会计）后，信号仍然是正 EV。

**本项目证据**：
- L2 队列模拟：真实 fill 82.4%，不是 Bernoulli 30%
- 库存控制：随机游走 → 均值回归，PnL 不变
- Delta 中性分解：Edge 是纯 spread capture，不是方向暴露
- 会计审计：修正 3 个 bug 后，edge 仍然为正

**关键点**：执行层面是对统计 edge 的第一次"现实检验"。很多 quant 策略在这一层被淘汰——它们依赖 unrealistic 的执行假设（比如 "以 mid 成交" 或 "固定 fill 率"）。

### Layer 3: Economic Viability（经济可行性）

**定义**：在扣除所有交易成本（手续费 + 印花税 + 滑点 + 冲击成本）后，edge 仍然为正。

**本项目证据**：
- 零售费率（佣金 2.5bps + 印花税 5bps → 每笔 3.2-8.2 bps）：**Edge 被摧毁**
- 机构费率（佣金 0.5bps）：601899 接近盈亏平衡
- **做市商计划（免印花税 + 0.1bp 佣金）：601899 ETR = 3.14，强可行**
- 低价股（~10 CNY）在零售费率下也可行：ETR = 1.07

**关键点**：大多数 quant 项目和几乎所有学术论文都停在前两层。到达第三层需要：完整的交易成本模型、对费率结构的精确理解、承认 "edge 存在但不可交易" 的诚实。

### 三层框架的核心洞察

```
Layer 1 问：能区分好坏吗？     → 分类问题
Layer 2 问：能执行出来吗？     → 工程问题
Layer 3 问：扣除成本后还赚吗？ → 经济问题

三层都通过 → 可部署的策略
只通过前两层 → 有价值的 research，但需要更好的市场条件
只通过第一层 → 学术论文，不可交易
```

### 面试时怎么讲这个框架

> "我在这个项目里学到的最重要的东西是：quant edge 不是二元的'有/没有'，而是三层递进的。第一层是统计显著性——你的模式是真实的还是过拟合的。第二层是执行可行性——在真实队列模拟、库存约束、MTM 会计下，信号还能不能赚钱。第三层是经济可行性——扣除印花税、佣金、滑点、冲击成本后还有没有剩。大部分 quant 项目停在第一层，好的项目到第二层，极少数到第三层。Hephaestus 的价值不在于'找到了一个赚钱策略'，而在于建立了一套能从第一层追问到第三层的完整审计体系。"

---

## 六、你可能被追问的点

**Q: 16 个特征具体是什么？为什么选这些？**

从微观结构文献（Cont, Stoikov, Avellaneda 等）中选择了覆盖五个维度的特征：价格（spread, mid return）、深度（bid/ask total depth）、平衡（OBI = order book imbalance）、波动（spread volatility）、流量（trade count, size, direction）。不是所有特征都有用，但覆盖面保证了 regime 聚类的信息充分性。

**Q: 为什么用 KMeans 而不是其他聚类方法？**

KMeans 的优势是可解释性——每个 regime 的聚类中心有明确的经济学含义。HMM 也试过，但没有显著改进分类质量，且增加了过拟合风险。对于 59 天数据的探索性研究，简单方法 + 严格的 walk-forward 验证比复杂方法 + 一次性的全样本拟合更可靠。

**Q: Tox = Spread/Depth 为什么 work？**

这不是随便选的。从做市商的角度看，spread 是你每笔交易能赚的最大金额，depth 是跟你抢这笔钱的竞争者数量。Tox 本质上就是"每单位竞争的潜在利润"。当 Tox 高，竞争者少，利润大；当 Tox 低，竞争激烈，adverse selection 吃掉一切。

**Q: 如果做市商计划可以做，为什么还没人在做？**

可能有人在做，只是没公开。但更关键的是：做市商计划通常需要交易所批准，有严格的报价义务（比如必须连续报价、最大 spread 限制等），不是谁都能拿到的。这个项目的分析假设了"获得做市商费率但没有额外的报价义务约束"——这是一个需要进一步验证的假设。

**Q: 这个系统能直接部署吗？**

不能。它是一个研究系统，不是生产系统。缺少：实时数据接入、订单管理、风险监控、异常处理、与券商的 API 对接。但它提供了部署前必须完成的验证清单：会计审计、执行审计、库存控制、OOS 验证、摩擦分析——这些都是生产系统上线前必须通过的关卡。

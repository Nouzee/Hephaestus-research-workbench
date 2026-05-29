# Hephaestus 面试简报 — tenX 绰瑞资产 AI Agent 实习岗

---

## 一、项目定位（一句话）

**Hephaestus 是一个 LLM 辅助的量化研究基础设施，用于市场微观结构分析、实验编排和事件驱动回测。它不是单一策略，而是一套完整的研究方法论。**

---

## 二、研究全貌：从幻觉到现实约束

### 起点：为什么开始这个项目

量化研究的三个核心问题：
1. 市场微观结构是否包含可预测的模式？
2. 这些模式能否转化为可执行的交易信号？
3. **在真实交易成本下，信号是否仍然盈利？**

大多数 quant 项目停在问题 2。这个项目的核心贡献是**一直追问到问题 3**。

### 研究时间线

```
BTC 字典学习失败
    │  Sharpe -40，adverse selection 主导，系统不可控
    │  核心教训：信号识别 ≠ 可交易 edge
    │
    ▼
A 股微观结构发现（000333 美的集团，59 天 L2 数据）
    │  发现 Toxicity Inversion：宽 spread = 盈利，窄 spread = 亏损
    │  168 个状态缩减为 15 个 CORE 状态
    │
    ▼
执行层验证（EVL）
    │  L2 队列模拟替换 Bernoulli fill
    │  真实 fill 82.4% vs Bernoulli 30%（2.75x 差距）
    │  发现 R2 执行陷阱：EV 最高但 fill 0.7%
    │
    ▼
会计审计（BIA）
    │  发现 3 个致命会计 bug：无库存、无 MTM、无 bid/ask 拆分
    │  Sharpe 7.82 → 0.61（修正后）
    │  100% 胜率 / 0 回撤 → 65% 胜率 / 4.3B 回撤
    │
    ▼
库存控制（IECORE）
    │  对称报价 → 库存随机游走（autocorr ≈ 1.0）
    │  Skewing 引擎：风险降 90%，PnL 不变
    │
    ▼
Delta 中性审计（DNA）
    │  PnL = 执行 PnL + 库存 PnL
    │  执行 PnL = 100%，库存 PnL = 0%
    │  Edge 是纯执行的，不是方向暴露
    │
    ▼
样本外验证（FOOS）
    │  前 40 天训练，后 19 天冻结 OOS
    │  Edge 存活，无衰退（PnL +12%）
    │
    ▼
摩擦与压力测试（EFL-SRV）
    │  A 股真实费率：佣金 2.5bps + 印花税 5bps = 每笔 3-8bps
    │  零售费率下 Edge 被完全摧毁（16/16 场景失败）
    │
    ▼
跨资产转移（X601899）
    │  601899（紫金矿业，~30 CNY）：Edge 厚度是 000333 的 2.63 倍
    │  股价越低 → 手续费越低 → spread/fee 比越好
    │
    ▼
可行性边界分析（VF）
    │  做市商计划（免印花税 + 0.1bp 佣金）：601899 ETR = 3.14，强可行
    │  零售费率下 10 CNY 以下的股票 + 宽 spread 也可行
```

---

## 三、关键技术方法

### 3.1 市场状态分割（Regime Segmentation）

**目标**：将连续的市场微观结构离散化为可分析的状态

**方法**：
- 从 L2 orderbook 提取 16 个微观结构特征（spread, depth, OBI, volatility, trade intensity 等）
- 前 20 天数据训练 KMeans（8 个 regime）
- 每个 100-tick 窗口分配一个 regime 标签

**Toxicity 量化**：
```
Tox = Spread / Depth
```
- 按 per-window percentile 分三档（0-30%, 30-70%, 70-100%）
- 关键发现：**Tox 越高（宽 spread），被动做市越盈利**
  - 窄 spread = 激烈排队竞争 = adverse selection 吃掉 spread
  - 宽 spread = mean reversion 保护 = 结构性盈利

**状态编码**：`R{r}_q{tq}_T{td}` = Regime × Tox_quantile × Time_of_day
- 总共 8 × 3 × 3 = 72 个 q2 状态
- ECORE 过滤后保留 21 个可执行状态

### 3.2 L2 队列执行模拟（EVL）

**问题**：Bernoulli P(fill)=0.30 与真实 fill 差距 2.75x

**方法**：
- 跟踪 10 档 orderbook 深度
- 在每 tick 以 OUR_SIZE=100 股挂单
- 模拟队列位置变化（成交消耗 + 撤单缩减）
- 当队列位置归零且同价位有成交 → fill
- 记录 fill 后 100 tick 的 markout

**关键发现**：
- 真实平均 fill rate = 82.4%（远高于 Bernoulli 30%）
- R2（深度流动性）是执行陷阱：EV 最高但 fill 仅 0.7%，markout -49bps
- 队列等待时间 25-33 ticks，稳定可预测

### 3.3 库存控制引擎（IECORE）

**问题**：对称报价产生库存随机游走（half-life = 693 窗口）

**方法**：
- **Size skew**：bid_mult/ask_mult ∈ [0, 2]，调整各方 fill 概率
- **Price skew**：在报价价格上加 shade（bps），吸引/排斥成交
- **Suppression**：|inv| > 70% limit → 削弱同侧报价；> 90% → 完全停止
- **State-conditioned**：高 A/E 状态弱 skew（保护执行优势），低 A/E 状态强 skew

**结果**：库存 std -92%，Max DD -90%，总 PnL 不变。Skewing 降低噪音但不触及信号。

### 3.4 Delta-Neutral 分解（DNA）

**核心公式**：
```
Total PnL_t  = Equity_t - Equity_{t-1}
Inventory PnL_t = inv_{t-1} × (mid_t - mid_{t-1})
Execution PnL_t = Total PnL_t - Inventory PnL_t
```

**验证**：
- Beta 回归：PnL = α + β × Δmid + ε
- β ≈ 0, R² ≈ 2%：mid 几乎不解释 PnL
- 执行 PnL 占总 PnL 100%，库存贡献 ~0%

### 3.5 可行性边界分析（VF）

**Edge Thickness Ratio (ETR)**：
```
ETR = GrossEdge / Friction = (SpreadCapture - Adverse) / (Fees + Slippage + Impact)
```

**核心洞察**：
- A 股费用与成交金额成正比（bps of notional）
- Spread capture 与 tick size 相关（raw units）
- → **股价越低，ETR 越高**
- 000333 (~80 CNY)：ETR = 0.13
- 601899 (~30 CNY)：ETR = 0.35
- 10 CNY 股票（同 raw spread）：ETR = 1.07 → **零售费率下可行**

---

## 四、BTC 部分的教训

### 为什么 BTC 失败了

**方法**：NMF 字典学习 → 8 个市场模式 → 模式动力学矩阵 A

**发现**：
1. 模式是可解释的（bid-heavy, ask-heavy, balanced, volatile）
2. 但模式是**描述性的**，不是**预测性的**（Granger 因果检验失败）
3. R² 天花板 = 8%（即使最优基也只能解释 8% 方差）
4. 系统**不可控**（单变量干预无法改变市场状态）
5. 连续被动做市 Sharpe ≈ -40

**核心教训**：
> 找到统计模式很容易。找到能 survive adverse selection + 执行成本的模式很难。执行经济学才是瓶颈，信号识别不是。

---

## 五、项目独特价值

### 大多数 quant 项目到这里就停了

```
"我们发现了 15 个正 EV 状态，Sharpe 7.82！"
```

### 这个项目选择继续追问

```
1. 确认会计正确吗？     → BIA：3 个 bug，Sharpe 实际 0.61
2. 确认是执行 edge 吗？  → DNA：是的，0% 方向暴露
3. 确认不是过拟合吗？    → FOOS：冻结 OOS 存活
4. 确认能覆盖交易成本吗？ → EFL-SRV：零售费率不行
5. 那什么条件下可行？    → VF：做市商计划 / 低价股 / 宽 spread
```

### 三层 Edge 框架

| 层级 | 定义 | 本项目结论 |
|---|---|---|
| **Statistical Edge** | 统计显著的模式 | ✅ Toxicity Inversion 真实存在 |
| **Execution Edge** | 可执行的信号 | ✅ Delta-neutral, 库存受控 |
| **Economic Viability** | 扣除成本后盈利 | ⚠️ 需要 MM 费率或特定市场条件 |

---

## 六、关键技术栈

- **Polars**：高性能数据处理（59 天 × 460K ticks/天）
- **NumPy/SciPy**：数值计算、聚类、统计检验
- **scikit-learn**：KMeans 状态聚类、NMF 字典学习
- **自定义 L2 队列模拟器**：10 档 orderbook 跟踪
- **LLM（Claude）**：实验编排、代码生成、审计框架设计
- **Git/GitHub**：三仓库管理（Hephaestus, Alfred, BTC-HFT）

---

## 七、可能的面试问题准备

**Q: 为什么叫 Hephaestus？**
赫菲斯托斯是希腊神话中的锻造之神。这个项目的核心隐喻是：在熔炉中反复锻造、检验、推翻，直到剩下的东西是真正坚固的。研究过程本身比最终策略更重要。

**Q: 你在这个项目中的角色是什么？**
我是研究和实验的 orchestator（编排者）。LLM 协助代码生成、审计框架设计和文档撰写。我负责研究方向的决策、假设提出、结果解读和下一步规划。

**Q: 最大的失败是什么？**
两个。第一，BTC 字典学习——花了几周找到漂亮的模式，但 Granger 因果检验全部失败。第二，EBACKTEST 的 "Sharpe 7.82, 0 回撤"——后来发现是三个会计 bug 造成的假象。

**Q: 最大的收获是什么？**
学会了 "default to falsification" 的思维方式。每次得到一个好结果，第一反应不是兴奋，而是设计反证实验来推翻它。BIA（会计审计）、DNA（delta 中性审计）、EFL-SRV（摩擦压力测试）都是这种思维的产物。

**Q: 如果给你更多资源，你会怎么推进？**
1. 获取做市商费率（不需要印花税）的实盘环境
2. 在 10-20 CNY 价格区间的 A 股上验证（低价 + 宽 spread）
3. 迁移到加密货币市场（maker fee 0-0.2bps，无印花税）
4. 将整个 pipeline 产品化，使非技术用户也能运行审计流程

---

## 八、一句话总结

> 我们建造了一个研究系统，它发现了真实的微观结构 edge，通过了五层审计（会计、Sharpe、Delta、OOS、摩擦），最终精确地指出了这个 edge 在什么条件下可以存活——不是在什么条件下看起来很厉害。

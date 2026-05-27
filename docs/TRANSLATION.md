# Hephaestus Translation Table v1

> 原始项目语言 → 概率对象。每个工程概念必须对应一个数学对象。

---

## 状态与观测

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| market state | $S_t$ | 随机过程 | DEF §B.1 |
| latent state | $S_t$ | 随机过程 | DEF §B.1 |
| observation | $X_t = O(S_t)$ | 随机变量 | DEF §A.2 |
| L2 features | $X_t \in \mathbb{R}^{16}$ | 随机向量 | — |
| regime | $Z_t \in \{0,\dots,7\}$ | 离散随机变量 | DEF §B.2 |
| regime label | $Z_t$ | 同上 | — |
| CORE state | $u \in \mathcal{C}$ | 状态元组 | DEF §B.5 |
| non-trade state | $u \notin \mathcal{C}$ | 补集 | — |
| observable state space | $\mathcal{X}$ | 可测空间 | DEF §A.2 |
| latent state space | $\mathcal{S}$ | 可测空间 | DEF §A.1 |

## 动力学与结构

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| backbone | $\mathbf{v}_1$ | 单位向量 | DEF §B.3 |
| 1D backbone | $\mathbf{v}_1$ | 同上 | — |
| A-matrix | $A$ | 线性算子 | DEF §B.3 |
| mode | $\phi_i$ | 基向量 | — |
| effective rank | $d_{\text{eff}}$ | 标量 | LEX §51 |
| drift | $f(S_t)$ | 向量场 | LEX §42 |
| diffusion | $\Sigma(S_t)$ | 矩阵场 | LEX §43 |
| spectral radius | $\rho(A)$ | 标量 | — |
| time asymmetry | $\|A - A_{\text{rev}}\|_F$ | 标量 | LEX §48 |
| self-exciting | $A_{ii} > 0$ | 标量条件 | LEX §49 |

## 收益与风险

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| PnL | $R_t$ | 随机变量 | DEF §A.4 |
| spread capture | $R_t^{\text{spread}}$ | 随机变量分量 | DEF §A.4 |
| adverse selection | $R_t^{\text{adverse}}$ | 随机变量分量 | DEF §A.4 |
| inventory PnL | $R_t^{\text{inventory}}$ | 随机变量分量 | — |
| positive expectancy state | $E[R_t \mid u] > 0$ | 条件期望不等式 | DEF §B.5 |
| tox score | $\tau(X_t, Z_t, d_t)$ | 可测评分 | DEF §B.6 |
| low-tox | $\tau \leq 3$ | 集合条件 | LEX §32 |
| high-tox | $\tau \geq 4$ | 集合条件 | LEX §33 |
| tox inversion | $\text{sign}(E[R_t \mid \tau \geq 4]) \neq \text{sign}(E[R_t \mid \tau \leq 3])$ | 符号翻转 | — |

## 传播与事件

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| stress event | $e_t^{\text{stress}}$ | 事件指示 | DEF §B.4 |
| spread expansion | $e_t^{\text{spread}}$ | 事件指示 | — |
| cancel burst | $e_t^{\text{cancel}}$ | 事件指示 | — |
| depth collapse | $e_t^{\text{depth}}$ | 事件指示 | — |
| arrival shock | $e_t^{\text{arrival}}$ | 事件指示 | — |
| propagation | $K(\tau \mid e \to v)$ | 条件核 | DEF §B.8 |
| MPU | $u^*$ | 事件类型 | DEF §B.10 |
| stress cascade | $\{u^*_1 \to u^*_2 \to \dots\}$ | 事件序列 | — |
| precursor chain | $\{e_{t-k_i}\}$ ordered by lead | 有序事件集 | — |

## 吸引子与盆地

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| stress attractor | $\mathcal{A}_{\text{stress}}$ | 可测子集 | DEF §B.4 |
| attractor basin | $\mathcal{B} \subset \mathcal{S}$ | 可测子集 | — |
| profitable basin | $\mathcal{B}^+$ | 可测子集 | DEF §C.3 |
| R5 | $Z_t = 5$ | 离散状态 | — |
| metastability | $0.70 < P(Z_{t+1}=z \mid Z_t=z) < 0.90$ | 持续性区间 | LEX §14 |
| state locking | $P(Z_{t+1}=z \mid Z_t=z) > 0.90$ | 持续性条件 | — |

## 决策与参与

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| participation | $\pi: \mathcal{U} \to \mathcal{A}$ | 可测映射 | DEF §B.7 |
| sparse participation | $\pi_{\mathcal{C}}$ | 条件决策 | DEF §B.7 |
| active quote | $\pi(u) = (s>0, w, 0)$ | 动作元组 | LEX §68 |
| withdraw | $\pi(u) = (0, \cdot, 0)$ | 动作元组 | LEX §69 |
| size multiplier | $s \in [0, s_{\max}]$ | 标量 | LEX §71 |
| spread multiplier | $w \in [1, w_{\max}]$ | 标量 | LEX §72 |
| participation frontier | $\{(\alpha, E[R \mid \pi_\alpha])\}$ | Pareto 前沿 | — |

## 验证与稳定性

| 原始词 | 概率对象 | 类型 | 定义位置 |
|--------|---------|------|---------|
| walk-forward | 时间序列交叉验证 | 方法论 | — |
| stability | $\text{CV}(E[R_t \mid u]) < 0.5$ 跨段 | 条件 | — |
| state overlap | $|\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}}| / |\mathcal{C}_{\text{train}}|$ | Jaccard | — |
| null model | $\rho(A_{\text{real}}) > \rho(A_{\text{shuffled}})$ | 不等式 | — |
| controllability | $\partial \rho / \partial \epsilon = 0$ | 导数条件 | LEX §50 |

## 展示层 → 理论层映射

以下词仅在展示/叙事层使用，理论层必须替换为对应的概率对象：

| 展示层叙事词 | 理论层替换 |
|------------|----------|
| "市场语法" | $P(\text{合法序列})$ over $\mathcal{G}$ |
| "市场语言" | 状态转移短语的生成规则 |
| "原子" | 字典基向量或模式 |
| "毒性流" | $P(\text{adverse fill} \mid S_t)$ 高 的状态 |
| "结构骨架" | $\mathbf{v}_1 \in \mathcal{S}$ |
| "做市直觉" | $\text{sign}(E[R_t \mid \tau])$ 的分段结构 |
| "风险盆地" | $\mathcal{A}_{\text{stress}}$ 的邻域 |
| "状态图谱" | $(Z_t)$ 的转移矩阵可视化 |

---

## 废弃 / 降级词

以下词不应再出现在理论文档中：

| 废弃词 | 原因 | 替代 |
|--------|------|------|
| "alpha" | 误导性：暗示可预测性 | $E[R_t \mid S_t] > 0$ |
| "因子" | 线性因子模型假设 | $X_t$ 或 $O(S_t)$ |
| "signal" | 歧义：观测 vs 交易信号 | $X_t$（观测）或 $E[R_t \mid S_t]$（期望） |
| "indicator" | 同上 | 同上 |
| "predictor" | 项目不追求预测 | $X_t$ 作为协变量 |
| "feature importance" | 降维思维 | $d_{\text{eff}}$ 或 ablation |

---

## 版本

| 版本 | 日期 | 变更 |
|------|------|------|
| v1 | 2026-05-26 | 初始翻译映射。7 类 × 80+ 条目。 |

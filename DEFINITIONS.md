# Hephaestus Definition Spec v1

> 将市场变成一个可定义、可测量、可推导的分析学对象。

---

## Part A. Primitive Objects

所有原始对象直接采用标准数学定义，不重新定义。

### A.1 State Space

$$
\mathcal{S} \subset \mathbb{R}^d
$$

$d$ 为有效维度。$\mathcal{S}$ 为可测空间 $(\mathcal{S}, \mathcal{B}(\mathcal{S}))$。

### A.2 Observation Space

$$
\mathcal{X} \subset \mathbb{R}^m
$$

$m = 16$（当前 L2 微观结构特征维度）。观测映射 $O: \mathcal{S} \to \mathcal{X}$ 为可测函数。

### A.3 Action Space

$$
\mathcal{A} = [0, s_{\max}] \times [1, w_{\max}] \times \{0,1\}
$$

$(q, w, c)$ 分别表示 size multiplier、spread multiplier、cancel flag。

### A.4 Reward

$$
R_t: \Omega \to \mathbb{R}
$$

$$
R_t = \underbrace{\frac{s_t \cdot w_t}{2}}_{\text{spread capture}} - \underbrace{\mathbf{1}_{\text{adverse}} \cdot |\Delta m_{t+\tau}| \cdot m_t}_{\text{adverse selection}} - \underbrace{\kappa \cdot |I_t|}_{\text{inventory cost}}
$$

其中 $s_t$ 为 spread，$m_t$ 为 mid-price，$I_t$ 为净持仓，$\tau$ 为前瞻窗口。

### A.5 Event

$$
e_t \in \mathcal{E} = \{\text{spread expansion}, \text{cancel burst}, \text{depth collapse}, \text{arrival shock}, \dots\}
$$

事件为状态空间中的可测子集上的指示函数在时间段上的激活。

### A.6 Transition

$$
T: \mathcal{S} \times \mathcal{B}(\mathcal{S}) \to [0,1]
$$

$T(s, B) = P(S_{t+1} \in B \mid S_t = s)$ 为 Markov 转移核。

---

## Part B. Hephaestus-Specific Objects

每个自定义概念按 7 字段模板严格定义。

---

### B.1 State $S_t$

**① 对象类型**：随机过程，取值于可测空间 $(\mathcal{S}, \mathcal{B}(\mathcal{S}))$

**② 所在空间**：状态空间 $\mathcal{S} \subset \mathbb{R}^d$

**③ 严格定义**：

$\{S_t\}_{t \in \mathbb{N}}$ 为定义在 $(\Omega, \mathcal{F}, P)$ 上、取值于 $(\mathcal{S}, \mathcal{B}(\mathcal{S}))$ 的离散时间随机过程。

$S_t$ 满足部分可观测性：存在可测映射 $O: \mathcal{S} \to \mathcal{X}$ 使得 $X_t = O(S_t)$，但 $O$ 不可逆。

**④ 可测性**：$S_t: \Omega \to \mathcal{S}$ 为 $\mathcal{F}/\mathcal{B}(\mathcal{S})$-可测。

**⑤ 估计方法**：通过 SVD/PCA 分解观测矩阵 $X \in \mathbb{R}^{N \times m}$ 获得潜在模式 $Z$，$S_t$ 取为 $Z_t$ 的前 $d$ 个主分量。$d$ 由 effective rank 确定（90% 方差）。

**⑥ 不变量**：$S_t$ 在观测算子的非退化线性变换下协变。在正交旋转下协变（PCA 不唯一）。有效维度 $d$ 跨观测尺度稳定。

**⑦ 失效条件**：
- $d \to m$（满秩）：空间退化为平凡
- 观测算子 $O$ 发生结构性变化（换交易所、换撮合规则）
- 样本量 $N < 10d$

---

### B.2 Regime $Z_t$

**① 对象类型**：离散随机变量，取值于有限集 $\mathcal{Z}$

**② 所在空间**：$\mathcal{Z} = \{0, 1, \dots, K-1\}$，配备离散 $\sigma$-代数

**③ 严格定义**：

$Z_t = \phi(X_t)$，其中 $\phi: \mathcal{X} \to \mathcal{Z}$ 为聚类映射（当前：KMeans, $K=8$）。

$\phi$ 满足：$\phi$ 在训练集 $\mathcal{D}_{\text{train}}$ 上拟合，在测试集 $\mathcal{D}_{\text{test}}$ 上仅做预测。$\phi$ 的拟合仅使用 $X_{\leq t}$（无未来泄露）。

$Z_t$ 的转移满足：

$$
P(Z_{t+1}=j \mid Z_t=i) = T_{ij}
$$

其中 $T_{ij}$ 从样本频率估计。

**④ 可测性**：$Z_t: \Omega \to \mathcal{Z}$ 可测（$\mathcal{Z}$ 有限，任何映射可测）。

**⑤ 估计方法**：KMeans on $\{X_t\}_{t \in \text{train}}$ with $K=8$。Spectral clustering 替代方案可用。

**⑥ 不变量**：Regime 标签在聚类算法的随机种子下仅相差置换。Regime 之间的转移结构（强边、吸引子）在 $K$ 的局部扰动下稳定。

**⑦ 失效条件**：
- $K$ 选取不当（肘部法则失效）
- 数据包含 regime shift 但训练集未覆盖
- 聚类退化为单一主导状态（HMM collapse）

---

### B.3 Backbone $\mathbf{v}_1$

**① 对象类型**：向量场的主方向（单位向量）

**② 所在空间**：$\mathcal{S}$ 的切空间 $T\mathcal{S} \cong \mathbb{R}^d$

**③ 严格定义**：

令 $A$ 为局部线性化转移算子：

$$
A = \arg\min_{W \in \mathbb{R}^{d \times d}} \|Z_{t+1} - W Z_t\|^2 + \lambda\|W\|_2^2
$$

Backbone $\mathbf{v}_1$ 定义为 $A$ 的最大实特征值对应的特征向量：

$$
A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1, \quad \Re(\lambda_1) = \max_i \Re(\lambda_i)
$$

$\mathbf{v}_1$ 归一化：$\|\mathbf{v}_1\|_2 = 1$。

**④ 可测性**：$\mathbf{v}_1$ 由有限样本 OLS 估计，是经验协方差矩阵的可测函数，因此可测。

**⑤ 估计方法**：对 $Z_{1:N-1}$ 和 $Z_{2:N}$ 做 Ridge 回归（$\lambda = 10^{-4}$），取主导特征向量。训练集：前 60% 数据。

**⑥ 不变量**：$\mathbf{v}_1$ 在 $A$ 的相似变换下协变。若 $\lambda_1 \gg \lambda_2$，则 backbone 方向对噪声不敏感。

**⑦ 失效条件**：
- $\Re(\lambda_1) \approx \Re(\lambda_2)$（简并）：无主导方向
- 样本量不足以稳定 $A$（$N < 10d$）
- $A$ 的特征值全部接近 0（无漂移系统）

---

### B.4 Stress Attractor $\mathcal{A}_{\text{stress}}$

**① 对象类型**：状态空间中的可测子集（状态集合）

**② 所在空间**：$(\mathcal{S}, \mathcal{B}(\mathcal{S}))$ 的子集

**③ 严格定义**：

$\mathcal{A}_{\text{stress}} \subset \mathcal{Z}$ 为满足以下条件的 regime 集合：

1. 持续性：$P(Z_{t+1}=z \mid Z_t=z) > 0.90$
2. 吸收性：$\forall z' \neq z, P(Z_{t+1}=z' \mid Z_t=z) < 0.10$
3. 高波动：$\text{Var}(X_t \mid Z_t=z) > \text{Var}(X_t)$（条件方差超过无条件方差）
4. 宽价差：$E[\text{spread}_t \mid Z_t=z] > Q_{0.80}(\text{spread})$

在当前数据集上：$\mathcal{A}_{\text{stress}} = \{R5\}$（spread -7.3σ, realized_vol +4.5σ, 持续性 0.939）。

**④ 可测性**：$\mathcal{A}_{\text{stress}}$ 由有限样本的聚类标签和统计阈值定义，是有限可测划分。

**⑤ 估计方法**：KMeans 聚类 → 逐 regime 统计持续性 + 条件方差 + 条件价差 → 阈值判定。

**⑥ 不变量**：$\mathcal{A}_{\text{stress}}$ 在 $K$ 增加时表现为吸收盆地的细分而非消失。在观测算子的连续变化下稳定。

**⑦ 失效条件**：
- 数据集未包含 stress 事件（所有 regime 持续性 < 0.85）
- 聚类未能分离 stress 状态
- Stress 的定义特征（高波动 + 宽价差）在数据中解耦（出现高波动但窄价差，或反之）

---

### B.5 CORE State $\mathcal{C}$

**① 对象类型**：可测状态集合（离散有限集）

**② 所在空间**：$\mathcal{Z} \times \mathcal{T} \times \mathcal{V}$（regime × tox × time-of-day）

**③ 严格定义**：

令 $\mathcal{U} = \mathcal{Z} \times \mathcal{T} \times \mathcal{D}$ 为全状态空间（$8 \times 7 \times 3 = 168$ 个原子状态），其中 $\mathcal{T} = \{0,\dots,6\}$ 为 tox 分值，$\mathcal{D} = \{\text{OPEN}, \text{MID}, \text{CLOSE}\}$。

CORE 集合 $\mathcal{C} \subset \mathcal{U}$ 定义为：

$$
\mathcal{C} = \left\{ u \in \mathcal{U} \;\middle|\;
\begin{aligned}
& E[R_t \mid U_t = u] > 0 \\
& \text{且该正期望在 walk-forward 的至少 } 2/3 \text{ 段中保持} \\
& \text{且 } \text{tox}(u) \geq 4
\end{aligned}
\right\}
$$

其中 $|\mathcal{C}| = 15$（当前估计值）。

**④ 可测性**：$\mathcal{U}$ 有限，$\mathcal{C}$ 由经验均值的符号确定，可测。

**⑤ 估计方法**：全样本模拟 fill → 逐状态计算 E[PnL/fill] → walk-forward 3 段验证 → 筛选 tox ≥ 4 且跨段稳定为正的状态。

**⑥ 不变量**：$\mathcal{C}$ 在相似市场结构下保持核心成员（跨段重叠 > 70%）。在训练/验证/测试分割下，正期望状态集合的 Jaccard 相似度 > 0.60。

**⑦ 失效条件**：
- 市场微观结构发生根本变化（tick size 改变、交易机制改变）
- $\mathcal{C} = \emptyset$（不存在正期望状态）
- 样本量不足以在每个 $u$ 上积累足够 fill 记录（$n_{\text{fills}} < 200$）

---

### B.6 Tox Score $\tau(X_t)$

**① 对象类型**：可测评分函数（非概率对象）

**② 所在空间**：$\tau: \mathcal{X} \times \mathcal{Z} \times \mathcal{D} \to \{0, \dots, 6\}$

**③ 严格定义**：

$$
\tau(X_t, Z_t, d_t) = \sum_{i=1}^{5} w_i \cdot \mathbf{1}_{\text{condition}_i}
$$

其中：

| $i$ | Condition | $w_i$ |
|-----|-----------|------|
| 1 | $\text{spread}_t > Q_{0.67}(\text{spread})$ | 2 |
| 2 | $\text{spread}_t > Q_{0.33}(\text{spread})$ | 1 |
| 3 | $\text{depth}_t < Q_{0.33}(\text{depth})$ | 2 |
| 4 | $\text{depth}_t < Q_{0.67}(\text{depth})$ | 1 |
| 5 | $d_t = \text{OPEN}$ | 1 |

外加 regime 调整：$Z_t=7$ 时 +1，$Z_t=3$ 时 -1。

$\tau$ 取截断值 $\max(\tau, 0)$。

**经验意义**：$\tau$ 不是字面"毒性"——它是 spread width / mean-reversion opportunity 的结构复合评分。$\tau \geq 4$ 对应正期望区域，$\tau \leq 3$ 对应结构性亏损区域。

**④ 可测性**：$\tau$ 是有限个阈值函数的有限加权和，可测。

**⑤ 估计方法**：分位数阈值从训练集估计。权重由 ablation 实验验证（各成分独立贡献）。

**⑥ 不变量**：$\tau \geq 4 \implies E[R_t] > 0$ 的结构关系在 walk-forward 段中保持。阈值 $Q_p$ 的微小变化不改变 tox 的分类性质。

**⑦ 失效条件**：
- 分位数阈值在 regime shift 下失效（需要自适应阈值）
- $\tau$ 对 spread 过度依赖（ablation 显示 spread 是最强成分，但非唯一）
- 在价差分布完全不同的市场（如 crypto vs A-share）需要重新校准

---

### B.7 Participation $\pi_t$

**① 对象类型**：可测决策映射（从状态到动作）

**② 所在空间**：$\pi: \mathcal{U} \to \mathcal{A}$，其中 $\mathcal{A} \subset [0,1] \times [1, \infty) \times \{0,1\}$

**③ 严格定义**：

Sparse participation policy $\pi_{\mathcal{C}}$ 定义为：

$$
\pi_{\mathcal{C}}(u) =
\begin{cases}
(1.0 \sim 3.0, \; 1.0, \; 0) & \text{if } u \in \mathcal{C} \\
(0, \; \cdot, \; 0) & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{C}$ 为 CORE 状态集合。

更一般地，$\pi$ 为可测函数 $\pi: \mathcal{U} \to \mathcal{A}$，其最优性由：

$$
\pi^* = \arg\max_{\pi} E\left[\sum_t R_t \cdot \pi(U_t) \;\middle|\; \pi\right]
$$

定义（受限于 $\pi$ 只在可估计的 $\mathcal{C}$ 上非零）。

**④ 可测性**：$\pi$ 定义在有限空间 $\mathcal{U}$ 上，任何函数可测。

**⑤ 估计方法**：通过状态级条件期望 $E[R_t \mid U_t = u]$ 的符号确定是否参与；通过 size 扫描确定最优参与量。

**⑥ 不变量**：$\pi$ 的支撑集 $\{u: \pi(u) \neq 0\}$ 在 walk-forward 段中重叠 > 70%。

**⑦ 失效条件**：
- $\mathcal{C} = \emptyset$
- $\mathcal{C}$ 中的状态在 OOS 测试中期望翻负

---

### B.8 Propagation Kernel $K(\tau \mid e \to v)$

**① 对象类型**：条件概率核（转移核的推广）

**② 所在空间**：$K: \mathbb{N} \times \mathcal{E} \times \mathcal{B}(\mathcal{V}) \to [0,1]$

**③ 严格定义**：

令 $e \in \mathcal{E}$ 为局部事件（spread expansion, cancel burst, depth collapse 等），$v \subseteq \mathcal{V}$ 为未来状态的可测子集。

传播核 $K$ 定义为：

$$
K(\tau \mid e \to v) = P\big(S_{t+\tau} \in v \;\big|\; e \text{ occurs at } t, S_t\big)
$$

其中条件包括事件发生时刻的完整状态 $S_t$。

对于最小传播单元 $u \in \mathcal{U}_{\text{MPU}}$，进一步要求：

$$
\exists \tau_0 > 0: \forall \tau \geq \tau_0, \;
D_{\text{KL}}\big(P(S_{t+\tau} \mid e, S_t) \;\|\; P(S_{t+\tau} \mid S_t)\big) > \epsilon
$$

即事件 $e$ 对未来分布的因果影响（以 KL 散度度量）在 $\tau_0$ 后仍显著。

**④ 可测性**：$K(\tau \mid \cdot)$ 对每个 $\tau$ 为转移核，可测。

**⑤ 估计方法**：Event study 对齐 → 条件分布估计 → KL 散度时间曲线。

**⑥ 不变量**：传播的定性结构（事件的因果影响方向）在时间平移下不变。

**⑦ 失效条件**：
- 事件样本量不足（$n_{\text{events}} < 50$）
- 条件分布估计不稳定（高维稀疏）
- KL 散度退化（事件无因果影响力）

---

### B.9 Hazard Rate $h(t \mid S_t)$

**① 对象类型**：条件强度函数（随机强度）

**② 所在空间**：$h: \mathbb{N} \times \mathcal{S} \to \mathbb{R}^+$

**③ 严格定义**：

令 $\mathcal{A}_{\text{stress}}$ 为 stress attractor 集合。Hazard rate 定义为：

$$
h(t \mid S_t = s) = \lim_{\Delta \to 0} \frac{1}{\Delta} P\big(Z_{t+\Delta} \in \mathcal{A}_{\text{stress}} \;\big|\; Z_t \notin \mathcal{A}_{\text{stress}}, S_t = s\big)
$$

在离散时间下，以 $k$-步前向概率近似：

$$
h_k(s) = P\big(Z_{t+k} \in \mathcal{A}_{\text{stress}} \;\big|\; Z_t \notin \mathcal{A}_{\text{stress}}, X_t\big)
$$

**④ 可测性**：$h(\cdot \mid s)$ 为条件概率，对每个 $s$ 可测。

**⑤ 估计方法**：在训练集上统计每个 regime $z \notin \mathcal{A}_{\text{stress}}$ 在 $k$ 步内进入 stress 的频率。按 tox 分桶做 hazard curve。

**⑥ 不变量**：hazard 的排序（哪些状态更危险）在 walk-forward 段中保持。

**⑦ 失效条件**：
- Stress 事件太少（$n_{\text{stress entries}} < 20$）
- Hazard 估计的置信区间过宽（无法区分状态）

---

### B.10 Minimal Propagating Unit $u^*$

**① 对象类型**：事件（可测子集上的指示函数 + 传播力判定）

**② 所在空间**：$\mathcal{E} \times \mathbb{N}$（事件类型 × 传播时间）

**③ 严格定义**：

$u \in \mathcal{E}$ 是一个 minimal propagating unit (MPU)，如果：

1. 传播性：$\exists \tau_0 > 0$，$D_{\text{KL}}(P(S_{t+\tau_0} \mid u_t) \| P(S_{t+\tau_0})) > \epsilon$
2. 最小性：$\forall u' \subsetneq u$，$u'$ 不满足传播性（$u$ 不可再分解为更小的传播单元）
3. 可复制性：$u$ 在不同时段的不同样本中出现，且传播特性一致（时间平移不变）

候选集合：$\{\text{spread shock}_{+2\sigma}, \text{cancel burst}_{+2\sigma}, \text{depth evaporation}_{-2\sigma}, \text{arrival shock}_{+3\sigma}\}$

**④ 可测性**：$u$ 是基于阈值的事件指示函数，可测。

**⑤ 估计方法**：Event study → KL 散度时间衰减曲线 → 显著性检验 → 可分解性检验（ablation）。

**⑥ 不变量**：MPU 的定性结构（它不是哪个具体 tick，而是某一类事件模式）在时间平移和时间重参数化下不变。

**⑦ 失效条件**：
- 样本中缺乏可重复的事件类型
- KL 散度在所有 $\tau$ 下都不显著
- 无法区分复合事件和原子事件

---

## Part C. Composite Objects

这些对象由 Part A + Part B 组合定义。

### C.1 Probabilistic Grammar $\mathcal{G}$

$\mathcal{G} = (\mathcal{V}, \mathcal{R}, P)$ 其中：
- $\mathcal{V}$ = 状态转移词组（transition phrases）
- $\mathcal{R}$ = 合法组合规则（由转移矩阵 $T$ 的正支持集定义）
- $P$ = 词组出现的概率权重（由路径频率估计）

合法序列：$\{v_1 \to v_2 \to \dots \to v_k \mid \forall i, T(v_i, v_{i+1}) > 0\}$

### C.2 Hazard Surface $\mathcal{H}$

$\mathcal{H}: \mathcal{S} \times \mathbb{N} \to [0,1]$

$$
\mathcal{H}(s, k) = P(Z_{t+k} \in \mathcal{A}_{\text{stress}} \mid S_t = s)
$$

### C.3 Profitable Basin $\mathcal{B}^+$

$\mathcal{B}^+ \subset \mathcal{S}$ 为满足 $E[R_t \mid S_t \in \mathcal{B}^+] > 0$ 的连通区域。$\mathcal{C}$ 为 $\mathcal{B}^+$ 的离散有限近似。

---

## Part D. Axioms / Assumptions

### D.1 Partial Observability

$\exists$ 可测映射 $O: \mathcal{S} \to \mathcal{X}$ 但 $O^{-1}$ 不存在。市场状态不能从观测中唯一恢复。

### D.2 State Non-Stationarity

$P(S_{t+1} \mid S_t)$ 在 regime 之间不同（regime-dependent dynamics）。在单个 regime 内可近似平稳，但跨 regime 非平稳。

### D.3 Local Propagation

存在 $\delta > 0$，使得事件 $e_t$ 对未来分布的影响随 $\tau$ 衰减：

$$
\lim_{\tau \to \infty} D_{\text{KL}}\big(P(S_{t+\tau} \mid e_t) \;\|\; P(S_{t+\tau})\big) = 0
$$

### D.4 Existence of Positive Expectancy States

$\mathcal{C} \neq \emptyset$。在当前市场上成立（15 个 CORE state），但不作为普适公理——仅在经过验证的数据集上断言。

### D.5 Observation Operator Invariance

在观测算子的非退化连续变化下，有效维度 $d$、attractor 的存在性、CORE states 的符号结构保持不变。已验证：跨 5 种观测算子，effective rank CV = 2.6%。

---

## Part E. Failure Modes

所有定义在以下条件下可能失效：

| 条件 | 影响的对象 |
|------|-----------|
| 数据集不含 stress 事件 | $\mathcal{A}_{\text{stress}}$, $\mathcal{H}$ |
| 聚类无法分离 regime | $Z_t$, $\mathcal{A}_{\text{stress}}$, $\mathcal{C}$ |
| 简并特征值（$\lambda_1 \approx \lambda_2$） | backbone $\mathbf{v}_1$ |
| Fill 模拟与真实执行偏差过大 | $E[R_t \mid \cdot]$ 的所有估计 |
| 观测算子结构性变化 | 所有基于 $X_t$ 的定义 |
| 样本量不足以填满状态空间 | $\mathcal{C}$ 的可靠性 |
| $\mathcal{C} = \emptyset$ | $\pi_{\mathcal{C}}$ 退化为全零映射 |
| 传播事件样本量 < 50 | $K(\tau)$, MPU |

---

## Part F. 已冻结术语分类

| 类别 | 术语 | 说明 |
|------|------|------|
| **A. 标准数学** | random variable, conditional expectation, transition kernel, measurable function, probability measure, stopping time, hazard rate | 直接采用 |
| **B. Hephaestus 自定义** | backbone, stress attractor, CORE state, tox score, MPU, probabilistic grammar, sparse participation | 本文件已定义 |
| **C. 仅展示层** | "市场语法", "原子", "毒性流", "做市直觉" | 叙事性语言，不进入定义层 |

---

## Part G. 未定义待补充清单

以下术语在后续版本中需要严格定义：

- flat field（当前限于经验描述"无曲率空间"）
- directionality（当前限于"时间反转不对称 > 0.5"）
- self-normalizing field（当前限于"单变量干预 Gain = 0"）
- metastable basin（当前限于"持续性在 0.7-0.85 之间"）
- motif（当前限于经验聚类，未给出分析学定义）

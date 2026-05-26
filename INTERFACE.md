# Hephaestus Probability Interface v1

> 规定项目里每个模块必须输出什么概率对象。任何新模块必须实现至少一个接口。

---

## 1. 主语法（9 个核心对象）

所有模块的输出必须归属到以下 9 个概率对象之一：

| # | 对象 | 符号 | 类型 | 定义文件引用 |
|---|------|------|------|------------|
| 1 | 潜在状态 | $S_t$ | 随机过程 | DEF §B.1 |
| 2 | 观测 | $X_t = O(S_t)$ | 随机变量 | DEF §A.2 |
| 3 | 离散 regime | $Z_t$ | 离散随机变量 | DEF §B.2 |
| 4 | 动作 | $A_t$ | 可测决策映射 | DEF §B.7 |
| 5 | 收益 | $R_t$ | 随机变量 | DEF §A.4 |
| 6 | 状态转移核 | $P(S_{t+1} \mid S_t)$ | 转移核 | DEF §A.6 |
| 7 | 条件收益 | $P(R_t \mid S_t, A_t)$ | 条件分布 | DEF §A.4 |
| 8 | 危险率 | $P(\text{stress} \mid S_t)$ | 条件强度 | DEF §B.9 |
| 9 | 传播核 | $K(\tau \mid e \to v)$ | 条件核 | DEF §B.8 |

---

## 2. 模块接口规范

### 2.1 Dictionary Learning (`modules/dictionary/`)

**必须输出**：

$$
P(Z_t \mid X_t) \quad \text{— 稀疏编码系数作为 regime 后验}
$$

**输出格式**：`alpha: ndarray (N, K)` — 每行是 $X_t$ 在字典 $D$ 上的稀疏系数分布

**验证**：$0 \leq \|\alpha_t\|_0 / K \leq 1$（稀疏度 ∈ [0,1]）

---

### 2.2 Regime Segmentation (`projects/ashare/regime_segmentation.py`)

**必须输出**：

$$
P(Z_t \mid X_t) \quad \text{— 离散 regime 分配}
$$

$\{Z_t\}_{t=1}^N \quad \text{— 标签序列}$

$T_{ij} = P(Z_{t+1}=j \mid Z_t=i) \quad \text{— 转移矩阵}$

**输出格式**：
- `labels: ndarray (N,)` — 离散标签
- `transmat: ndarray (K, K)` — 转移频率

**验证**：$\sum_j T_{ij} = 1$，每行归一化

---

### 2.3 Toxicity Scoring (`toxicity_scorer.py` / `run_toxicity_filter.py`)

**必须输出**：

$\tau(X_t, Z_t, d_t) \in \{0,\dots,6\}$ — tox 评分

$E[R_t \mid \tau \geq 4] > 0 \quad \text{和} \quad E[R_t \mid \tau \leq 3] < 0$ — 条件期望符号

**输出格式**：`tox_scores: ndarray (N,)`

**验证**：正负期望分离（ablation 确认 3 成分独立贡献）

---

### 2.4 Mode Extraction (`mode_extractor.py`)

**必须输出**：

$S_t \in \mathbb{R}^d$ — 潜在状态（前 $d$ 个主分量）

$\{\phi_i\}_{i=1}^d$ — 模式基向量

**输出格式**：
- `z_series: ndarray (N, d)`
- `phi: ndarray (D, d)`

**验证**：$d$ 由 effective rank 确定（90% 累积方差）

---

### 2.5 Mode Dynamics (`mode_dynamics.py`)

**必须输出**：

$$
A = \arg\min_W \|Z_{t+1} - W Z_t\|^2
$$

$$
\mathbf{v}_1 = \arg\max_{\|v\|=1} \Re(\lambda(A))
$$

$$
\rho(A) = \max_i |\lambda_i(A)|
$$

**输出格式**：
- `A: ndarray (d, d)`
- `backbone: ndarray (d,)`
- `spectral_radius: float`

**验证**：$\rho(A) > 0$（系统有漂移），$\mathbf{v}_1$ 在 walk-forward 段中稳定

---

### 2.6 PnL Attribution (`run_toxicity_attribution.py`)

**必须输出**：

$$
E[R_t \mid \tau = k] \quad \forall k \in \{0,\dots,6\}
$$

$$
E[R_t \mid Z_t = z, \tau = k, d_t] \quad \text{— 全条件期望}
$$

**输出格式**：条件期望表（regime × tox × TOD → E[PnL/fill]）

**验证**：每个 bucket ≥ 200 fills（统计可靠性阈值）

---

### 2.7 CORE State Identification (`run_production_v4.py`)

**必须输出**：

$$
\mathcal{C} = \{u \in \mathcal{U} \mid E[R_t \mid u] > 0 \text{ 且跨段稳定}\}
$$

$$
\pi_{\mathcal{C}}(u) =
\begin{cases}
s(u) & u \in \mathcal{C} \\
0 & \text{otherwise}
\end{cases}
$$

**输出格式**：
- `CORE: set of (regime, tox, TOD) tuples`
- `size_map: dict mapping state → multiplier`

**验证**：$|\mathcal{C}| > 0$，跨 ≥2/3 walk-forward 段保持正期望

---

### 2.8 Impact Kernel (`impact_kernel.py`)

**必须输出**：

$$
K(\tau) = E[\Delta p_{t+\tau} \mid e_t, S_t]
$$

$\alpha$ — 衰减指数

$\tau_{1/2}$ — 半衰期

**输出格式**：
- `kernel: ndarray (max_lag+1,)`
- `decay_type: str`
- `half_life: float`

**验证**：$\lim_{\tau \to \max\_\text{lag}} |K(\tau)| < |K(0)| / 2$（衰减）

---

### 2.9 Stress Hazard (`state_machine.py` / HMM scaler)

**必须输出**：

$$
h_k(S_t) = P(Z_{t+k} \in \mathcal{A}_{\text{stress}} \mid S_t)
$$

**输出格式**：`hazard_curve: dict mapping regime → ndarray (k_max,)`

**验证**：$h_k$ 对 $k$ 单调非减（长期 hazard 不低于短期）

---

### 2.10 Falsification Tests (`run_svct.py` / `run_dynamics_falsification.py`)

**必须输出**：

| 测试 | 概率语句 |
|------|---------|
| Null model | $\rho(A_{\text{real}}) > \rho(A_{\text{null}})$ |
| Time reversal | $\|A_{\text{forward}} - A_{\text{reverse}}\|_F > 0.15$ |
| Confounder nulling | $\rho(A \mid g \text{ removed}) \approx \rho(A)$ |
| Controllability | $\partial \rho(A) / \partial \epsilon_{M1} = 0$ |

**输出格式**：pass/fail verdicts + 效应量

**验证**：3/4 通过 → 系统可被视为真实动力学结构

---

## 3. 接口遵循规则

1. **任何新模块必须实现至少一个接口**。如果不输出上述 9 个对象之一，需要说明为何豁免。
2. **输出必须包含可测性证明或引用**。如果不能证明可测，标记为"经验定义"。
3. **输出必须可被后续模块消费**。格式为 numpy/csv/json，不依赖 Python 对象引用。
4. **Breaking changes 必须递增版本号**。v1 → v2 需要说明旧定义失效在哪里。

---

## 4. 版本

| 版本 | 日期 | 变更 |
|------|------|------|
| v1 | 2026-05-26 | 初始接口规范。9 核心对象 + 10 模块接口。 |

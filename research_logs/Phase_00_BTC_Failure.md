# Phase 00 — BTC Passive Market Making Failure

## 背景

初始假设：BTC 永续合约的 L2 订单簿数据包含可通过被动做市来利用的持续微观结构模式。

数据：BTC/USDT 永续合约，L2 十档订单簿快照 + 成交消息。

## 初始假设

1. 被动做市（双边报价、捕获 spread）在考虑逆向选择后能够产生正期望收益
2. 字典学习 / NMF 可以将订单簿分解为可解释的"市场模式"
3. 模式动力学可以预测状态转换并为报价策略提供信息
4. 连续报价是一个可行的基线

## 方法

- **Market Decomposition** (`market_decomposition.py`): 3-layer LOB decomposition using NMF/sparse coding
- **Causal Validation** (`causal_validation.py`): 4 experiments (lag, shock, null, regime) to test if modes had predictive power
- **Structural Validity** (`structural_validity.py`): SVCT falsification tests on mode structure
- **Mode Dynamics** (`mode_dynamics.py`): 8×8 interaction matrix A between modes
- **Information Geometry** (`information_geometry.py`): MDS embedding, curvature analysis, flow fields
- **Minimal Basis** (`minimal_basis.py`): OMB — 6-mode minimal basis with 8% R² ceiling
- **Scale Discovery** (`scale_discovery.py`): MSDP — Minimal Scale Discovery Phase, fixed point detection
- **Drift Origin** (`drift_origin.py`): 1D backbone = nonlinear_response + flow_persistence
- **Controllability Test** (`controllability_test.py`): MCT — single-variable intervention
- **Compressibility** (`compressibility.py`): Observation operator sweep

## 核心发现

1. **Mode decomposition works technically**: NMF produces interpretable modes (bid-heavy, ask-heavy, balanced, volatile, etc.)
2. **Causality is absent**: Modes are descriptive, not predictive. Causal validation failed — mode transitions don't Granger-cause mid-price moves
3. **R² ceiling at 8%**: Even the optimal minimal basis explains only 8% of variance
4. **System is uncontrollable**: Single-variable intervention (MCT) cannot shift market state
5. **Structure is compressible but invariant**: Observation operator sweep shows structure is robust but not exploitable
6. **Fixed point is stable**: MSDP detected a stable fixed point — market returns to it regardless of perturbation

## 审计与反证

- SVCT falsification: modes survive structural validity checks but lack predictive power
- MSDP verdict: CASE_C — scale flow converges to fixed point, no exploitable drift
- MCT verdict: UNCONTROLLABLE — no single variable can perturb the system

## 被推翻的内容

1. **连续被动做市是死路**：模拟中 Sharpe 约为 -40。逆向选择占主导
2. **字典学习不产生 alpha**：模式在统计上有效但在经济上不可利用
3. **微价格预测不够**：即使使用对称技巧，信号质量仍然太低，无法支撑执行

## 当前理解

BTC 的核心教训：

> 信号识别不是瓶颈。执行经济学才是。

找到统计模式很容易。找到能经受逆向选择 + spread 捕获经济学考验的模式很难。这一洞察直接促成了向 A 股执行感知稀疏参与的转向。

## 未解决的问题

- Would the same mode decomposition approach work on more structured markets (equities vs crypto)?
- Is the problem specific to perpetual futures (no fundamental anchor) or to crypto microstructure generally?
- Could a different execution model (aggressive vs passive) salvage the signal?

## 关键文件

- `experiments/btc/market_decomposition.py`
- `experiments/btc/causal_validation.py`
- `experiments/btc/structural_validity.py`
- `experiments/btc/mode_dynamics.py`
- `experiments/btc/information_geometry.py`
- `experiments/btc/minimal_basis.py`
- `experiments/btc/scale_discovery.py`
- `experiments/btc/drift_origin.py`
- `experiments/btc/controllability_test.py`
- `experiments/btc/compressibility.py`

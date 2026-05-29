# Phase 01 — Sparse State Participation (SSP)

## 背景

BTC 失败后，转向 A 股 L2 数据（000333 美的集团，59 个交易日）。核心问题：如果连续被动做市是死路，能否识别出特定的盈利状态？

数据：000333 L2 订单簿（十档）+ 成交消息，59 天，约 460K tick/天。

## 初始假设

1. A 股微观结构可能比加密货币更有结构性（基本面锚定、交易时段、涨跌停限制）
2. 某些市场状态可能对被动报价具有正期望收益，而其他状态则是有害的
3. 毒性（spread/depth 比率）可能是关键的区分因素

## 方法

- **Regime Discovery** (`regime_discovery.py`): KMeans clustering on 16 L2 microstructure features (8 regimes)
- **L2FeatureExtractor** (`regime_segmentation.py`): 16 features including spread, depth, OBI, volatility, trade intensity
- **State classification**: R{r}_q{tq}_T{td} = Regime × Toxicity_quantile × Time_of_day
- **Toxicity**: spread / depth ratio, quantile-based (per-window calibration using train-only data)
- **Rolling Validation** (`rolling_validation.py`): 20/5 walk-forward, frozen rules
- **Toxicity Inversion** (`toxicity_inversion.py`): 5-hypothesis validation that wide-spread states are MORE profitable
- **Toxicity Attribution** (`toxicity_attribution.py`): PnL decomposition + toxic fill map (4.8M fills)
- **State Economics** (`state_economics.py`): Per-state EV, adverse ratio, fill probability (534K fills)
- **CORE Verification** (`core_verification.py`): Phase 1 CORE with 20/5 and 15/5 schemes, 7/7 tests passed
- **Policy Surface** (`policy_surface.py`): Binary vs continuous weighting comparison
- **Pareto CORE** (`pareto_core.py`): 3D Pareto front on (EV, A/E, Fill%)
- **Sparse Execution** (`sparse_execution.py`): Production v4 with CORE_15 filter + size scan

### 版本演进
- v1-v3：从绝对阈值逐步改进为逐窗口分位校准（滚动稳定性的关键突破）
- v4：最终的 15 个 CORE 状态的二值过滤器

## 核心发现

1. **Phase transition at 30th percentile**: Expected PnL/fill flips from negative/zero to positive at tox rank ≥ 30th percentile, confirmed across 15 rolling windows
2. **Toxicity Inversion discovered**: Tight spread (low tox) → structural LOSS from adverse selection. Wide spread (high tox) → structural PROFIT from mean reversion protection. This is counter-intuitive — the conventional "toxicity" interpretation is inverted for passive MM.
3. **168 possible states reduced to 15**: CORE = states with tox ≥ 4 (wide spread), cross-segment stable positive EV
4. **Binary filtering >> continuous weighting**: +111% PnL advantage. The phase transition is sharp, not smooth — you either participate or you don't.
5. **Continuous ranking is harmful**: Weighting positions by state quality degrades performance compared to hard binary filter
6. **8 market regimes identified**: R0(depth collapse), R1(ask heavy), R2(deep liquidity/trap), R3(trade surge), R4(bid heavy), R5(STRESS attractor), R6(active buy), R7(active sell)

## 审计与反证

- Rolling walk-forward with frozen rules (20/5 and 15/5)
- Anti-tests: tox shuffle, state shuffle — both destroy edge, confirming state structure is real
- Toxicity inversion validated across 15 rolling windows

## 被推翻的内容

1. "低毒性 = 安全" → 被反转：低 tox = 逆向选择，高 tox = 均值回归盈利
2. "更多状态 = 更多机会" → 尖锐相变，二值过滤器最优
3. "连续仓位管理" → 使表现恶化 111%

## 当前理解

并非所有市场状态都具有同等的可执行性。关键的区分因素是 spread 宽度（毒性分位），而非仅 regime。稀疏参与——仅在宽 spread、高 tox 状态下报价——是执行 edge 的基础。

## 未解决的问题

- Are the 15 CORE states stable out-of-sample?
- Does R2 (deep liquidity) kill execution in ALL markets or just 000333?
- Can we time entry/exit within CORE states?

## 关键文件

- `experiments/ashare/regime_discovery.py`
- `experiments/ashare/regime_segmentation.py`
- `experiments/ashare/rolling_validation.py`
- `experiments/ashare/toxicity_inversion.py`
- `experiments/ashare/toxicity_attribution.py`
- `experiments/ashare/state_economics.py`
- `experiments/ashare/core_verification.py`
- `experiments/ashare/policy_surface.py`
- `experiments/ashare/pareto_core.py`
- `experiments/ashare/sparse_execution.py`

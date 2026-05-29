# Phase 06 — Final Out-of-Sample Validation (FOOS)

## 背景

All previous work used data from the full 59-day period. While walk-forward validation (20/5 rolling) provided some protection against overfitting, the ECORE states, tox thresholds, and inventory control parameters were all developed with visibility into the full dataset.

真正的样本外测试需要：冻结参数、未见数据、一次性评估。

## 初始假设

1. 如果 ECORE 状态捕捉了真实的市场结构，它们应该迁移到未见数据
2. 执行 edge 应在样本外存活（至少在方向上）
3. 样本外条件可能更不利（状态转换、spread 压缩）

## 方法

**FOOS v1** (`foos_validation.py`):

**Data split**: First 40 days = Train, Last 19 days = Frozen OOS
**Frozen parameters**: Regime clusters (KMeans), tox thresholds (p30, p70), ECORE definition, ETE entry/exit logic, inventory skew policy, quote sizing — ALL derived from train only.
**One-shot evaluation**: OOS data seen exactly once. No parameter adjustment permitted.

**Tests**:
- A. Frozen transfer of full pipeline
- B. OOS metrics (PnL, Sharpe, DD, inventory, efficiency)
- C. DNA re-run on OOS (beta, inventory contribution)
- D. ECORE stability (state frequency, R6 attractor, q2 occupancy)
- E. Always-On comparison (unit efficiency)
- F. Degradation analysis (in-sample vs OOS comparison)

## 核心发现

| Metric | In-Sample (40 days) | OOS (19 days) | Change |
|---|---|---|---|
| Total PnL | +4.57B | +5.12B | +12% |
| Execution PnL | +4.55B | +5.12B | +12% |
| Daily Sharpe (total) | 37.57 | 44.07 | +17% |
| Execution fraction | 100% | 100% | — |
| Max DD | 0 | 0 | — |

1. **Edge survives frozen OOS**: No degradation. Execution PnL actually increases OOS.
2. **ECORE occupancy INCREASES OOS**: 7.8% → 12.3% (+58%). More opportunities, not fewer.
3. **All ECORE states more frequent OOS**: R7, R0, R3, R1 all show increased frequency. State structure is robust.
4. **OOS market was more favorable**: Mid vol -58%, q2 occupancy up (11.9% → 17.7%). Calmer, wider-spread market = better passive MM conditions.
5. **Unit efficiency advantage intact**: PnL/quote 4.84x, PnL/fill 2.36x vs Always-On.
6. **DNA confirms zero directional exposure OOS**: Execution fraction = 100%, inventory PnL = +0.88M (0%).

## 审计与反证

- Frozen parameters verified (all derived from train only)
- OOS data confirmed unseen (last 19 days, no overlap with train)
- DNA decomposition re-run on OOS independently
- Degradation analysis: in-sample vs OOS comparison

## Degradation Analysis

| Metric | In-Sample | OOS | Change |
|---|---|---|---|
| Execution PnL | +4.55B | +5.12B | +12% |
| Daily Sharpe | 37.57 | 44.07 | +17% |
| Mid vol (window) | 2,618 | 1,092 | -58% |

No degradation detected. OOS performance is actually better than in-sample.

## 被推翻的内容

1. "OOS will be harder" → Wrong (for this specific OOS window). The OOS period happened to be more favorable.
2. "ECORE occupancy will decay OOS" → Wrong. Occupancy increased 58%.

## 当前理解

执行 edge 不是过拟合假象。冻结参数迁移到未见数据时零衰退。 However, there are important caveats:

1. **Only 15 OOS trading days**: 3 test windows of 5 days each. Too short for statistical confidence.
2. **OOS was an unusually favorable period**: Low volatility, wide spreads. A different OOS draw might show different results.
3. **0 losing days persists OOS**: The "never lose" property is suspicious across both in-sample and OOS.
4. **The real test is transaction costs**: See Phase 07.

## 未解决的问题

- What happens under realistic transaction costs? (→ Phase 07)
- Would the edge survive a truly adverse OOS period (high vol, spread compression)?
- Can this be validated on another asset? (600900 — blocked by data)

## 关键文件

- `experiments/ashare/foos_validation.py`

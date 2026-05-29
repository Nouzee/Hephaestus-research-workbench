# Phase 07 — Execution Friction & Stress Validation (EFL-SRV)

## 背景

Phase 01-06 确立了真实的微观结构执行 edge 存在、经受会计审计、是 delta 中性的、且能迁移到未见数据。但所有分析都假设了无摩擦执行：零费用、零滑点、零市场冲击。

真实交易有成本。问题是：edge 能在成本下存活吗？

## 初始假设

1. The execution edge per fill is small (~77 CNY in spread capture per 100-share fill at 2 bps spread)
2. A-share transaction costs are material (3.2 bps buys, 8.2 bps sells including 5 bps stamp duty)
3. The edge might survive modest fees but fail under realistic costs

## 方法

**EFL-SRV v1** (`friction_stress.py`):

- **A. Explicit Fee Layer**: Commission (2.5bps), exchange fee (0.5bps), transfer fee (0.2bps), stamp duty (5bps, sells only). Per-fill deduction.
- **B. Slippage Layer**: Stale quote slippage at 1/2/3/5 bps
- **C. Queue Fade**: Fill probability degradation at 100%/70%/50%/30%/10%
- **D. Impact Cost**: Linear + sqrt market impact model at 1x/3x/5x
- **E. Stress Regimes**: High volatility days, low q2 occupancy days
- **F. Combined Severe Stress**: Fee×2, Slip×3bps, Fill×0.3, Impact×3, Adverse×2, Spread×0.5
- **G. Delta-Neutral Revalidation**: DNA decomposition under all stress scenarios
- **H. Capacity Test**: 0.5x/1x/2x/3x/5x position size

16 friction scenarios tested. Frozen pipeline (no parameter changes permitted).

## 核心发现

### Fee Economics: The Critical Calculation

```
Per-fill notional (100 shares × 80 CNY):    8,000 CNY
Per-fill fee (buy, 3.2 bps):                2.56 CNY
Per-fill fee (sell, 8.2 bps):               6.56 CNY
Average fee per fill:                       ~4.56 CNY

Per-fill spread capture (½ spread, 2 bps):  ~0.80 CNY
Per-fill spread capture (½ spread, 7 bps):  ~2.80 CNY

Fee / Spread capture ratio:                 1.6x (best case) to 5.7x (typical)
```

### Scenario Results

| Scenario | PnL | Sharpe | Status |
|---|---|---|---|
| Baseline (zero friction) | +5.78B | +35.7 | SURVIVES |
| **Realistic A-Share Fees** | **-32.2B** | **-32.8** | **DIES** |
| Fees + 5bp Slippage | -65.5B | -33.1 | DIES |
| Fill Decay 30% | -11.7B | -32.4 | DIES |
| Adverse ×3 | -70.0B | -34.8 | DIES |
| Spread ×0.5 | -43.8B | -33.4 | DIES |
| Heavy Impact (5x) | -160.3B | -33.9 | DIES |
| **Severe Combined** | **-53.8B** | **-33.9** | **DIES** |

**16/16 friction scenarios: edge destroyed. 0.5x capacity already negative.**

### Friction Breakdown (Realistic Fees)

Gross frictionless PnL: baseline level
Fees consume: ~45% of friction costs
Impact: ~43%
Slippage: ~12%

The dominant cost is fees (commission + stamp duty), followed by market impact at larger sizes.

### Why the Edge Dies

The spread capture per fill (0.8-2.8 CNY) is smaller than the transaction cost per fill (2.6-6.6 CNY) by a factor of 1.6-5.7x. This is a structural economic constraint, not an execution optimization problem.

Even in the best state (R1_q2_T0, spread = 505 raw units ≈ 6.7 bps):
- Round-trip fees: 11.4 bps
- Round-trip spread capture: 6.7 bps
- **Fees are 1.7x the spread**

No amount of state selection, timing optimization, or inventory control can overcome this. The spread is simply too narrow relative to A-share transaction costs.

## 审计与反证

- All 16 friction layers tested independently
- DNA decomposition re-run under stress (execution fraction still ~100% — fees affect both total and execution PnL equally)
- Capacity curve extrapolated to find break-even point (below 0.5x — i.e., no viable size exists)

## 被推翻的内容

1. **"The execution edge is economically viable"** → Wrong. The edge is STATISTICALLY real but ECONOMICALLY non-viable under A-share fees.
2. **"We can optimize our way out of fees"** → Wrong. The fee/spread ratio is structural, not tactical.

## 当前理解

### Hephaestus 证明了什么

1. **微观结构 edge 是真实的**：毒性反转、ECORE 状态选择、q1→q2 扩张——全部在多层审计中验证
2. **Delta 中性执行 edge 存在**：系统确实从 spread 捕获减去逆向选择中赚钱，零方向暴露
3. **库存控制有效**：偏斜降低风险 90%+ 而不触及 edge
4. **Edge 在冻结样本外存活**：不是过拟合假象

### Hephaestus 未能克服的

1. **A 股交易成本**：每笔 3.2-8.2 bps，费率结构消耗了可用 spread 捕获的 1.6-5.7 倍
2. **可行性所需 spread**：约 12 bps 往返（vs 当前约 2-7 bps）。这是市场结构约束，而非策略问题

## 可能的前进路径

1. **Institutional fee structure**: Negotiated commissions (0.5 bps), market-making stamp duty exemptions. Could reduce fees to ~1 bps, making edge viable.
2. **Different asset class**: BTC/crypto markets have 0-2.5 bps fees, wider spreads. The same execution logic may be viable there.
3. **Only widest-spread states**: R1_q2_T0 (6.7 bps) is closest to breakeven. Ultra-sparse participation in only the top 1-2 states.
4. **Larger position sizes with lower impact**: Requires institutional execution infrastructure.

## 关键文件

- `experiments/ashare/friction_stress.py`

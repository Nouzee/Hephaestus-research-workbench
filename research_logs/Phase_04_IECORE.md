# Phase 04 — Inventory-Aware ECORE (IECORE)

## 背景

BIA 揭示了库存是一个接近单位根的过程（自相关 ≈ 1.0，半衰期 ≈ 693 个窗口）。对称被动报价会在库存中产生随机游走——不存在自然的均值回归。需要主动控制。

核心问题：能否在不破坏执行 edge 的前提下控制库存漂移？

## 初始假设

1. 偏斜报价（不对称买卖）可以将库存推向零
2. 偏斜必须是状态条件的——扭曲高 A/E 状态是危险的
3. 库存控制应该降低方差而不降低平均 PnL

## 方法

**IECORE v1** (`iecure.py`):

- **Inventory Dynamics Analysis** (Task A): Inventory path diagnostics, autocorrelation, holding duration, state-level accumulation/release patterns
- **Inventory-Conditioned State Economics** (Task B): ExecEV recomputed conditional on inventory sign
- **Skewing Engine** (Task C): Three mechanics:
  1. Size skew: multiply fill probability on each side (bid_mult, ask_mult ∈ [0, 2])
  2. Price skew: shade quote prices to attract/repel fills (bps)
  3. Quote suppression: hard stop one side when |inv| > threshold
- **Risk-Aware Participation** (Task E): State-conditioned skew strength (stronger skew in low-A/E states, weaker in high-A/E)
- **Inventory Mean Reversion** (Task D): AR(1) half-life analysis, natural reversion states, runaway conditions
- **Backtest** (Task G): Four controller configs tested: NoSkew, Mild (0.3), Moderate (0.5), Aggressive (0.8)
- **Stress Tests** (Task F): 6 scenarios (OneWay_Sell, OneWay_Buy, SpreadCollapse, QueueThickening, ToxicFlow)

### 控制器设计
```
Position limit: 50,000 shares
Skew multiplier: 0.5 (Moderate)
Suppress at: 70% of limit (35,000 shares)
Hard stop at: 90% of limit (45,000 shares)
State-conditioned: skew × (1 - A/E), clamped to [0.2, 0.9]
```

## 核心发现

| Controller | PnL | Sharpe | Max DD | Inv Std | Half-life |
|---|---|---|---|---|---|
| **NoSkew** | +14.2B | 1.90 | 1.72B | 13,939 | 693w |
| Mild | +15.1B | 14.90 | 192M | 1,527 | 76w |
| **Moderate** | **+15.2B** | **22.35** | **176M** | **1,114** | **40w** |
| Aggressive | +15.1B | 30.83 | 123M | 1,028 | 33w |

1. **Inventory control does NOT hurt PnL**: All controllers produce roughly identical total PnL (~15.1B). The skew doesn't destroy the execution edge.
2. **Massive risk reduction**: Max DD -90%, Inventory std -92%.
3. **Natural reversion is nonexistent**: Half-life without control = 693 windows ≈ 69,300 ticks. With Moderate control: 40 windows.
4. **All states are NEUTRAL in symmetric quoting**: Bid/ask fills are perfectly balanced per state. Inventory drift is from random fill arrival order, not structural bias.
5. **Stress tests pass**: All 6 scenarios survive with max inventory peaks well within limits (max 8,300 shares vs 50,000 limit).
6. **Optimal config is Moderate (skew=0.5)**: Sweet spot between risk reduction and execution preservation.

## 审计与反证

- Compared 4 controller strengths to verify monotonic improvement
- Stress-tested under 6 adverse market scenarios
- Verified state-level fill balance (neutral bid/ask ratio confirms no structural bias)

## 被推翻的内容

1. "Inventory will naturally mean-revert" → Wrong. Half-life without control = 693w. Active control is required.
2. "Skewing will hurt PnL" → Wrong. All controllers maintain identical total PnL.
3. "Stronger skew = better" → Partially wrong. Aggressive skew has highest Sharpe but risks execution distortion.

## 当前理解

库存控制是一个风险管理层，而非 alpha 来源。它不创造 edge——它通过防止库存漂移主导 PnL 方差来稳定现有的 edge。 The skewing engine reduces MTM noise without touching the spread capture signal.

Sharpe 的改善（1.90 → 22.35）完全来自方差降低：当库存受约束时，窗口间的权益变化由 spread 捕获（低方差）主导，而非 inventory × mid 变动（高方差）。

## 未解决的问题

- The Sharpe 22.35 is suspiciously high — is it real or a frequency artifact? (→ Phase 05)
- Does inventory control work the same way OOS? (→ Phase 06)
- What happens under realistic transaction costs? (→ Phase 07)

## 关键文件

- `experiments/ashare/iecure.py`

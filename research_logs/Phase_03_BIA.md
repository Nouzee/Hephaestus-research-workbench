# Phase 03 — Backtest Integrity Audit (BIA)

## 背景

EBACKTEST v1 报告：+100M PnL，Sharpe 7.82，最大回撤 = 0，胜率 = 100%。这些数字很可疑。一个有数百万笔成交的被动做市策略不应该有零回撤。会计上出了问题。

## 初始假设

1. 策略可能确实有很强的风险调整后收益
2. 但零回撤 + 100% 胜率 + 数百万笔成交在物理上不可能
3. 默认假设："在所有现实层面被验证之前，回测是错误的"

## 方法

**BIA v1** (`backtest_audit.py`): Full accounting audit with proper tick-level MTM.

Tasks performed:
- **A. Mark-to-Market**: equity = cash + inventory × mid_price (not just realized spread capture)
- **B. Inventory**: track bid/ask fills separately, compute net position, holding duration, drift
- **C. Fill Accounting**: duplicate detection, bid/ask imbalance, overlapping fills
- **D. Queue Realism**: queue wait dynamics using EVL-calibrated parameters
- **E. Cancel Realism**: cancel lag, exchange reaction delay, stale quote exposure
- **F. Spread Crossing**: bid/ask overlap detection
- **G. PnL Path**: full equity curve, rolling drawdown, intraday volatility
- **H. Baseline Reconstruction**: Always-On MM with identical execution layer (replacing the old "30% random fill" baseline)

## Key Findings

### EBACKTEST 中发现的三个致命会计错误

1. **无库存跟踪**：原始代码按每笔成交计入 spread 捕获，而不跟踪成交是在买方还是卖方。 `pnl += n_fills_binom * (sp_capture - markout_raw)` — this treats every fill as a completed round-trip, which is wrong.

2. **No MTM of open positions**: Original code never computed `equity = cash + inventory × mid`. Only tracked realized spread capture. This is why Max DD = 0 — there was nothing to draw down.

3. **无买卖拆分**：所有成交被汇总而不跟踪方向。 Impossible to compute inventory, impossible to MTM, impossible to detect adverse selection on open positions.

### 修正后的指标

| Metric | Original EBACKTEST | BIA Corrected |
|---|---|---|
| Total PnL | +100M | +13.1B (ECORE) / +23.9B (Always-On) |
| Sharpe | 7.82 | 0.61 (ECORE) / 0.51 (Always-On) |
| Max DD | 0 | 4.3B (ECORE) / 6.9B (Always-On) |
| Win rate | 100% | 65% (ECORE) / 64.5% (Always-On) |
| Negative windows | 0 | 31,791 (ECORE) / 32,119 (Always-On) |

### 单位经济学存活

Despite the accounting bugs, the core economic signal survived correction:
- PnL/quote: ECORE = +10,364 vs Always-On = +3,901 (2.66x)
- PnL/fill: ECORE = +6,290 vs Always-On = +4,741 (1.33x)
- Inventory std: ECORE = 22,609 vs Always-On = 32,451 (30% lower)

## Audit Failures

Three flags remained after BIA:
1. ECORE ending inventory = 33,600 shares — material, MTM-dependent
2. Inventory autocorr ≈ 1.0 for both strategies — near-unit-root random walk
3. All state PnL estimates positive — non-ECORE states are low-efficiency, not negative

## 被推翻的内容

1. **Sharpe 7.82 was an accounting artifact**: The correct Sharpe is ~0.61. The original PnL calculation was missing 96% of the variance (inventory MTM).
2. **"0 drawdown" was impossible**: Real drawdown is 4-7B. The original code never tracked unrealized losses.
3. **"100% win rate" was a measurement error**: Corrected hit rate is ~65%.

## 当前理解

Backtest accounting is not an implementation detail. Three bugs — no inventory, no MTM, no bid/ask split — produced results that were off by orders of magnitude. The corrected metrics still show positive edge, but the magnitude is far smaller and more realistic.

关键教训：**每一个"好"的回测结果都应该被视为错误，直到被证明不是。**

## 未解决的问题

- Inventory is a unit root (autocorr ≈ 1.0) — how to control this? (→ Phase 04)
- Is the remaining Sharpe of 0.61 real or still inflated? (→ Phase 05, 06)
- Does the edge survive transaction costs? (→ Phase 07)

## 关键文件

- `experiments/ashare/backtest_audit.py`

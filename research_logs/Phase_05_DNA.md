# Phase 05 — Delta-Neutral Audit (DNA) + Sharpe Audit (SA)

## 背景

需要进行两项相互关联的审计：

**SA（Sharpe 审计）**：IECORE 报告的 Sharpe 为 22-30。有约 9 万个窗口和 sqrt(N) 年化，这可能是频率/聚合假象。

**DNA（Delta 中性审计）**：SA 发现 MTM 占 PnL 方差的 96%。如果策略只是"持有库存 + 搭乘中间价趋势"，那么 edge 就是方向性的，而非基于执行的。

## 初始假设

1. SA: The IECORE Sharpe is likely inflated by sqrt(90K) annualization
2. DNA: There might be a hidden directional beta that explains the "0 losing days" phenomenon
3. Default assumption: "the Sharpe is wrong until proven otherwise"

## 方法

**SA v1** (`sharpe_audit.py`):
- Multi-frequency Sharpe: window-level, daily, monthly
- Proper annualization: sqrt(252) for daily, sqrt(12) for monthly
- Bootstrap confidence intervals (10K samples)
- AR(1) adjustment for serial correlation
- Overlap audit (20/5 rolling scheme)
- Realized vs MTM variance decomposition

**DNA v1** (`delta_neutral.py`):
- PnL decomposition: Total PnL = Execution PnL + Inventory PnL
  - Inventory PnL_t = inv_{t-1} × (mid_t - mid_{t-1})
  - Execution PnL_t = Total PnL_t - Inventory PnL_t
- Beta regression: PnL_t = α + β × Δmid_t + ε
- Delta-neutral Sharpe recomputation
- Long/short symmetry test
- Trend sensitivity by subperiod (Early/Mid/Late)

## 核心发现

### SA：IECORE 的 Sharpe 是保守的，而非被夸大的

| Method | Sharpe | Notes |
|---|---|---|
| IECORE (sqrt-N window) | 20.43 | Most conservative |
| Daily, annualized | 33.66 | Standard method |
| Daily, AR(1)-adjusted | 19.71 | Corrected for autocorrelation |
| Non-overlap 5-day | 21.72 | 7 independent observations |

The negative autocorrelation of window-level PnL (AR(1) = -0.42) means daily aggregation REDUCES variance more than expected, making the daily Sharpe HIGHER than the window-level Sharpe. The IECORE method was not inflating — it was the lower bound.

**However**: Only 35 trading days, 0 losing days. Daily AR(1) = 0.489 → effective N = 12. The sample is too small for reliable Sharpe estimation.

### DNA：Edge 是纯执行的，零方向暴露

| Component | Symmetric (no control) | Moderate (inventory control) |
|---|---|---|
| Total PnL | +13.62B | +15.05B |
| Execution PnL | +15.09B (+111%) | +15.10B (+100%) |
| Inventory PnL | **-1.47B (-11%)** | **-0.05B (-0%)** |

1. **Inventory PnL is NEGATIVE without control**: The directional drift COSTS money (-11%), not makes it. The system is fighting a headwind.
2. **Moderate controller eliminates directional exposure**: Mean inventory = +8 (essentially zero). R² with mid = 2.2%.
3. **Delta-neutral Sharpe retention = 100%**: Execution Sharpe = 34.13, identical to total Sharpe.
4. **Long/short perfectly symmetric**: t-stat = -1.26 (not significant). The system makes money equally regardless of inventory sign.
5. **Stable across subperiods**: Execution PnL consistent in Early/Mid/Late, uncorrelated with cumulative mid change.

## 审计与反证

- SA: Bootstrap CI, rolling Sharpe stability, serial correlation adjustment
- DNA: Beta regression at multiple frequencies, variance decomposition
- SA verdict: CASE_B — Sharpe inflated by limited sample but edge clearly positive
- DNA verdict: CASE_A — Edge survives delta-neutral decomposition

## 被推翻的内容

1. **"IECORE Sharpe is inflated by sqrt(N)"** → Partially wrong. The frequency artifact goes the OTHER way — window-level Sharpe UNDERESTIMATES daily Sharpe due to negative autocorrelation.
2. **"Edge might be directional beta"** → Wrong. Inventory contribution is NEGATIVE. The edge is pure execution.
3. **"0 losing days = suspicious"** → Still suspicious (small sample), but at least not due to directional drift.

## 当前理解

Hephaestus 的 edge 是真正由执行驱动的。系统从 spread 捕获减去逆向选择中赚钱，而非搭乘市场趋势。 Inventory control eliminates the directional noise without touching the signal.

The consistently positive daily returns (0 losing days) remain a concern but are not explained by directional exposure. They may be due to: (a) genuinely strong edge, (b) small sample (35 days), or (c) some other unidentified smoothing mechanism.

## 未解决的问题

- What happens out-of-sample with frozen parameters? (→ Phase 06)
- What happens under realistic transaction costs? (→ Phase 07)

## 关键文件

- `experiments/ashare/sharpe_audit.py`
- `experiments/ashare/delta_neutral.py`

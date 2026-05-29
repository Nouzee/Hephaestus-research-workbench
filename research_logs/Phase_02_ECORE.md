# Phase 02 — Execution-Aware CORE Reconstruction (ECORE)

## 背景

Phase 01 识别了 15 个 CORE 状态（宽 spread、正期望收益）。但执行模型是幼稚的：Bernoulli P(fill) = 0.30，没有队列动态，没有逆向选择校准。真实的成交不是这样工作的。我们需要执行感知。

## 初始假设

1. L2 队列模拟将揭示哪些 CORE 状态是真正可执行的
2. 某些状态具有高 spread 捕获但无法成交的队列（执行陷阱）
3. 执行现实主义将减少可行状态的数量

## 方法

- **EVL** (`execution_validation.py`): L2 queue simulation replacing Bernoulli fills. 10-level orderbook tracking. Per-tick queue position evolution. Fill triggered when position reaches 0 and trade occurs at our price.
  - Parameters: OUR_SIZE = 100 shares, FUTURE_TICKS = 20, 10-day subset
  - Tracks: real fill probability, markout at 100 ticks post-fill, queue wait time
  - Compares: Bernoulli P(fill) vs Real P(fill) per state
- **ECORE** (`ecore.py`): Execution-aware CORE reconstruction. ExecEV(s) = P(fill|s) × (SpreadCapture(s) - Adverse(s)). Filters: ExecEV > 0, A/E < 1.0, P(fill) > 20%, queue_wait < 100 ticks.
- **ECD** (`ecore_dynamics.py`): State transition analysis. Transition matrix, self-persistence, R2 trap detection, positive flow paths, exit risk per state.
- **ETE** (`ecore_timing.py`): Entry/exit timing engine. Entry precursors, exit rates, survival model, executable chains.
- **EBACKTEST** (`ebacktest.py`): First integrated backtest. Entry on q1→q2 widening, hold while ECORE, exit on q2→q1 or non-ECORE. Used EVL fill rates and markouts (no Bernoulli).

## 核心发现

1. **Real fills = 82.4% vs Bernoulli 30%**: 2.75x gap. The old model was severely underestimating fill probability.
2. **R2 is an execution trap**: 0.7% fill rate, -49bps markout, A/E ratio = 4.42. R2 (deep liquidity regime) has huge spread capture but the queue never clears. Permanently excluded from ECORE.
3. **ECORE = 21 states** (vs 15 OLD CORE, vs 3 Pareto CORE):
   - Most R0/R3/R4/R5/R6/R7 states retained (fill > 80%, low markout)
   - R1 retained but low quotes — execution edge confirmed
   - Pareto CORE was too strict (only 3 states)
   - Old CORE retained R2 (trap)
4. **ECORE occupancy = 13.9%**, mean run = 2.9 windows
5. **Self-loops dominate**: R6→R6→R6 (753x most frequent chain). R6 is the execution attractor.
6. **Entry precursors**: ALL q1 (mid-spread) states. Transition q1→q2 is the widening onset signal.
7. **Exit rates 25-46%** per state — ECORE states are transient, not persistent.
8. **Survival model**: R6_q2_T0 = 4.6w longest, R3_q2_T1 = 2.0w shortest.
9. **First backtest result**: +100M PnL, Sharpe 7.82, 0 drawdown, 100% win rate.
   - **This was later found to be an accounting artifact (see Phase 03 BIA).**

## 审计与反证

- Queue simulation validated against real orderbook depth data
- R2 trap detection confirmed by both queue model AND state economics
- ECORE vs Pareto vs OLD comparison — ECORE is the only one that both includes high-fill states AND excludes traps
- EBACKTEST baseline comparison showed -43% vs always-quote (but baseline was flawed — see BIA)

## 被推翻的内容

1. "更多状态 = 更好的 CORE" → 错误。Pareto 过于严格（3 个状态），旧 CORE 过于宽松（包含 R2 陷阱）。ECORE 的 21 个状态是正确的平衡
2. "Bernoulli 成交模型够用" → 错误。真实成交率高出 2.75 倍，且依赖于状态
3. "EBACKTEST 显示 100% 胜率" → 这是一个会计错误。见 Phase 03

## 当前理解

执行经济学（成交概率、队列等待、markout）决定了哪些状态是真正可交易的。统计盈利性是必要的但不充分的。R2 是典型例子：EV 最高的状态，但 0.7% 的成交率使其毫无价值。执行层不是事后补充——它本身就是策略。

## 未解决的问题

- Is EBACKTEST PnL real or an accounting artifact? (→ Phase 03)
- Can we time entry/exit within ECORE states to improve hold duration?
- Does the q1→q2 widening signal generalize?

## 关键文件

- `experiments/ashare/execution_validation.py`
- `experiments/ashare/ecore.py`
- `experiments/ashare/ecore_dynamics.py`
- `experiments/ashare/ecore_timing.py`
- `experiments/ashare/ebacktest.py`

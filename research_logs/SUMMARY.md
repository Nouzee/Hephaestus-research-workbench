# Hephaestus — 研究轨迹总结

## 项目定位

Hephaestus 是一个 LLM 辅助的量化研究基础设施，用于市场微观结构分析。它被设计为一个实验编排系统，而非生产交易系统。研究遵循了一条系统性的路径：从假设生成，经过多层审计，最终到达经济可行性验证。

**主要标的**：000333 美的集团（A 股），L2 订单簿 + 成交数据，59 个交易日
**次要标的**：BTC/USDT 永续合约（已放弃 — 见 Phase 00）
**尝试跨资产验证**：600900 长江电力（数据阻塞 — 缺少订单簿文件）

## 研究弧线

```
BTC Failure (Phase 00)
    |
    v
Sparse State Participation (Phase 01)
    |-- Toxicity Inversion discovered
    |-- 15 CORE states identified
    v
Execution-Aware CORE (Phase 02)
    |-- EVL queue simulation
    |-- R2 execution trap found
    |-- ECORE: 21 states
    v
Backtest Integrity Audit (Phase 03)
    |-- THREE accounting bugs found
    |-- Original Sharpe 7.82 → corrected 0.61
    |-- MTM, inventory, bid/ask split fixed
    v
Inventory-Aware ECORE (Phase 04)
    |-- Inventory unit root documented
    |-- Skewing engine built
    |-- Risk reduced 90%, PnL unchanged
    v
Delta-Neutral + Sharpe Audit (Phase 05)
    |-- Edge is pure execution (0% directional)
    |-- Sharpe conservative, not inflated
    |-- 35 days, 0 losing days — sample too small
    v
Final OOS Validation (Phase 06)
    |-- Frozen 40/19 split
    |-- Edge survives OOS (+12% PnL)
    |-- ECORE occupancy increases OOS
    v
Friction & Stress Validation (Phase 07)
    |-- A-share fees tested
    |-- Edge DESTROYED by transaction costs
    |-- Fee/spread ratio: 1.6-5.7x
```

## 已证实的结论

### 已确立

1. **Toxicity Inversion**: Tight spread = adverse selection loss; Wide spread = mean reversion profit. Validated across 15 rolling windows. Counter-intuitive but robust.

2. **Sparse Participation > Continuous Quoting**: Binary state filter outperforms continuous weighting by 111%. The phase transition at the 30th toxicity percentile is sharp, not smooth.

3. **Execution Economics Determine Viability**: Statistical edge (R2 has highest EV) ≠ executable edge (R2 has 0.7% fill rate). The execution layer is not secondary — it IS the strategy.

4. **Delta-Neutral Execution Edge**: The system's PnL is pure spread capture minus adverse selection. Zero directional exposure (beta ≈ 0, R² with mid ≈ 2%). Confirmed by DNA decomposition.

5. **Inventory Control Works**: Skewing reduces inventory variance by 92% and drawdown by 90% without reducing total PnL. Inventory drift is a risk management problem, not an alpha problem.

6. **Edge Survives Frozen OOS**: Parameters frozen on first 40 days transfer to last 19 days with zero degradation. Not an overfitting artifact.

### 未确立 / 残留问题

1. **Only 35-59 trading days**: The entire research is based on at most 59 days of data. Statistical confidence is limited.

2. **Single asset**: Only tested on 000333. Cross-validation on 600900 was blocked by data format (no orderbook files). BTC pipeline was a different methodology (mode decomposition) and failed independently.

3. **Consistently zero losing days**: Across all audit phases, the "never lose" property persists at daily frequency. This is suspicious and may indicate an unidentified smoothing mechanism in the simulation.

4. **Simulation, not live trading**: All fills are simulated using EVL-calibrated parameters. No real orders were placed. Queue dynamics, fill probabilities, and markouts are estimated from historical data and may not reflect live conditions.

### 已被推翻 / 已放弃

1. **BTC continuous passive MM**: Sharpe ~-40. Adverse selection dominates. Abandoned.

2. **Dictionary learning for alpha**: Modes are descriptive, not predictive. R² ceiling = 8%. Abandoned.

3. **Bernoulli fill model**: Real fills are 2.75x higher and state-dependent. Replaced by EVL queue simulation.

4. **Continuous position weighting**: Degrades performance by 111%. Replaced by binary state filter.

5. **Sharpe 7.82 with 0 drawdown**: Accounting artifact. Three bugs: no inventory, no MTM, no bid/ask split.

6. **Economic viability in A-shares**: Destroyed by transaction costs. Fee/spread ratio of 1.6-5.7x makes the strategy non-viable at any tested size.

## Edge 的三个层次

Hephaestus 的研究轨迹揭示了"edge"的三个不同层次：

### 第一层：统计 Edge（Phase 01-02）
毒性反转、状态经济学、ECORE 过滤。这些是统计显著的微观结构模式。它们是真实的但不能直接交易。

### 第二层：执行 Edge（Phase 03-06）
当统计模式通过现实执行（队列模拟、库存跟踪、MTM 会计）过滤后，一个正期望收益的信号存活下来。这个 edge 是 delta 中性的，在样本外存活，且对库存风险具有稳健性。

### 第三层：经济可行性（Phase 07）
当加入交易成本后，edge 被摧毁。spread 捕获（0.8-2.8 CNY/笔）比费用（2.6-6.6 CNY/笔）小 1.6-5.7 倍。

**Hephaestus 的结论**：微观结构 edge 确实存在（第一层）、可以被执行（第二层）、但在 A 股零售费率结构下不具备经济可行性（第三层）。

## 关键数据

| 指标 | 数值 | 确立阶段 |
|---|---|---|
| 毒性相变点 | 30th percentile | Phase 01 |
| CORE → ECORE 状态数 | 15 → 21 | Phase 02 |
| 平均真实成交率 | 82.4% | Phase 02 |
| R2 成交率（陷阱） | 0.7% | Phase 02 |
| 发现的会计错误数 | 3 critical | Phase 03 |
| 修正后 Sharpe（无费用） | 0.61 → 1.90 → 22.35 (with inv control) | Phase 03-04 |
| 库存风险降低幅度 | -92% std, -90% DD | Phase 04 |
| 方向暴露 | ~0% (beta ≈ 0) | Phase 05 |
| 样本外衰退 | 0% (edge increases) | Phase 06 |
| 费用/spread 比率 | 1.6-5.7x | Phase 07 |
| Edge 存活的场景数 | 1/17 (baseline only) | Phase 07 |

## 仓库结构

```
Hephaestus/
├── README.md                    # Project overview (engineering focus)
├── requirements.txt
├── LICENSE (MIT)
├── .gitignore
├── hephaestus.py                # Entry point
├── experiments/
│   ├── ashare/                  # A-share pipeline (14 scripts)
│   └── btc/                     # BTC pipeline (10 scripts, abandoned)
├── projects/
│   └── ashare/
│       └── regime_segmentation.py  # L2FeatureExtractor
├── modules/
│   ├── probability/             # 11 stochastic process modules
│   ├── execution/               # fill_model, simulator, attribution
│   └── risk/                    # state_machine, FSM, inventory_skew
├── docs/
│   ├── LEXICON.md               # 72-term standard dictionary
│   ├── DEFINITIONS.md           # 10 core concepts
│   ├── INTERFACE.md             # 9 probability objects, 10 modules
│   ├── TRANSLATION.md           # 80+ term mappings
│   ├── llm_workflow.md          # Agent context
│   └── TUTORIAL.md
├── research_logs/               # This directory
│   ├── Phase_00_BTC_Failure.md
│   ├── Phase_01_SSP.md
│   ├── Phase_02_ECORE.md
│   ├── Phase_03_BIA.md
│   ├── Phase_04_IECORE.md
│   ├── Phase_05_DNA.md
│   ├── Phase_06_FOOS.md
│   ├── Phase_07_EFL_SRV.md
│   └── SUMMARY.md (this file)
├── examples/
│   └── demo_pipeline.py         # Runnable with synthetic data
└── reports/                     # Output reports (gitignored)
```

## 最终评估

Hephaestus 作为一个研究系统是成功的。它完成了以下事项：

1. 发现了真实的微观结构模式（毒性反转）
2. 构建了严格的执行模拟层（EVL）
3. 识别并修正了自身的会计错误（BIA）
4. 验证了 edge 是执行驱动的，而非方向性的（DNA）
5. 通过冻结样本外测试验证了稳健性（FOOS）
6. 正确地识别出 A 股费率使策略不具备经济可行性（EFL-SRV）

它没有产出一个可部署的交易策略。但它产出了一些可能更有价值的东西：一条从假设到经济现实检验的完整研究轨迹，每一层审计都被保留，每一次失败都被记录。

**这个系统不是一个策略。它是一种研究方法论。**

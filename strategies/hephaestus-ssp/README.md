# Hephaestus-SSP: Sparse State Policy

A-share L2 microstructure execution strategy.  
**ONLY trades 15 CORE states. ELSE flat.**

---

## Core Discovery

The market has **168 observable states** (8 regime × 7 toxicity × 3 time-of-day).  
Only **15 states** have positive expectancy. Trading everything else is structural loss.

### The Inversion

Conventional wisdom says: *wide spread = toxic, avoid.*  
**Reality**: tight spread = adverse selection dominates (NEGATIVE expectancy).  
Wide spread = protection + mean reversion (POSITIVE expectancy).

| Tox Score | Spread | Expectancy | Action |
|-----------|--------|-----------|--------|
| 0–3 | Tight | **NEGATIVE** | WITHDRAW |
| 4–6 | Wide | **POSITIVE** | ACTIVE (1x–3x size) |

---

## Architecture

```
Raw L2 Data (10-level orderbook + trades)
    ↓
Feature Extractor (16 microstructure features, 100-tick windows)
    ↓
KMeans Regime Classifier (8 regimes, fitted on TRAIN only)
    ↓
Toxicity Scoring (spread + depth + regime + time-of-day)
    ↓
CORE_15 State Filter (cross-segment stable, tox≥4, positive expectancy)
    ↓
Weighted Position Sizing (1.0x–3.0x, no saturation observed)
    ↓
Execution (stochastic fill simulation, 30% fill rate, adverse selection model)
```

---

## 15 CORE States

All have: tox ≥ 4, positive PnL in ≥ 2 of 3 walk-forward segments.

| State | Regime | Tox | Time | PnL (3-segment total) |
|-------|--------|-----|------|----------------------|
| R7_t4_MD | Active Sell | 4 | MID | +5.7M |
| R6_t5_OP | Active Buy | 5 | OPEN | +5.1M |
| R6_t4_MD | Active Buy | 4 | MID | +5.0M |
| R6_t4_OP | Active Buy | 4 | OPEN | +4.0M |
| R1_t4_OP | Ask Heavy | 4 | OPEN | +3.7M |
| R7_t5_OP | Active Sell | 5 | OPEN | +3.6M |
| R0_t4_OP | Depth Collapse | 4 | OPEN | +2.8M |
| R4_t4_OP | Bid Heavy | 4 | OPEN | +2.6M |
| R1_t5_OP | Ask Heavy | 5 | OPEN | +2.4M |
| R7_t5_MD | Active Sell | 5 | MID | +2.4M |
| R7_t4_CL | Active Sell | 4 | CLOSE | +2.3M |
| R7_t6_OP | Active Sell | 6 | OPEN | +2.2M |
| R1_t4_MD | Ask Heavy | 4 | MID | +2.0M |
| R0_t5_OP | Depth Collapse | 5 | OPEN | +1.7M |
| R4_t4_MD | Bid Heavy | 4 | MID | +1.4M |

---

## Performance (10-day TEST, out-of-sample)

| Strategy | PnL | Fills | PnL/Fill |
|----------|-----|-------|----------|
| Baseline (always quote) | -904.6M | 727K | -1,244 |
| **SSP 1.0x** | **+6.85M** | 48K | +145 |
| **SSP 2.0x** | **+13.9M** | 49K | +285 |
| **SSP 3.0x** | **+20.8M** | 48K | +430 |

93% fewer fills. No PnL saturation up to 3x size.

---

## 8 Market Regimes

| Regime | Name | Share | Persistence | Character |
|--------|------|-------|-------------|-----------|
| R5 | STRESS ATTRACTOR | 1.7% | 0.939 | Wide spread, high vol |
| R2 | Deep Liquidity | 5.1% | 0.859 | Thick book (≠ safe!) |
| R0 | Depth Collapse | 10.8% | 0.440 | Liquidity thinning |
| R1 | Ask Heavy | 22.5% | 0.606 | Selling pressure |
| R4 | Bid Heavy | 25.4% | 0.621 | Buying pressure |
| R6 | Active Buy Flow | 16.3% | 0.380 | Aggressive buying |
| R7 | Active Sell Flow | 17.2% | 0.387 | Aggressive selling |
| R3 | Trade Surge | 1.0% | 0.403 | Extreme arrival rate |

---

## Key Research Findings

1. **Market is ~8D but effectively 1D**: SVD on 16 microstructure features shows 8 effective dimensions, but 1D backbone dominates (flow_persistence + nonlinear_response).

2. **Stress is an ATTRACTOR, not noise**: R5 has persistence 0.939 — once entered, hard to leave. But wide spreads in R5 make it PROFITABLE for market makers.

3. **Toxicity is INVERTED**: The "toxic" label should be on TIGHT spreads, not wide ones. Adverse selection dominates when spreads are narrow.

4. **OPEN ≠ toxic**: OPEN session losses are an artifact of quoting at tight spreads. With tox filtering, OPEN becomes profitable.

5. **Deep liquidity is a TRAP**: R2 (deepest book) has the highest adverse selection. Thick books attract informed flow.

6. **Ablation confirms structure**: Removing spread, depth, or cancel from the tox score all independently degrade performance. The score is a composite structural signal, not a single-variable proxy.

---

## Files

```
strategies/hephaestus-ssp/
├── README.md                    # This file
├── regime_segmentation.py       # L2 feature extractor (16 features, 3 layers)
├── run_phase1_discovery.py      # Regime discovery (KMeans, spectral)
├── run_full_analysis.py         # Full 3-month pipeline
├── run_regime_backtest.py       # Regime-aware MM backtest
├── run_toxicity_attribution.py  # PnL decomposition + toxic fill map
├── run_toxicity_filter.py       # Toxicity filtering backtest
├── run_profit_turnaround.py     # Continuous toxicity PnL model
├── run_inverted_tox.py          # Inverted tox policy (H1-H5 validation)
├── run_production_bridge.py     # Stability + ablation + compression
└── run_sparse_execution.py      # Production v4: CORE_15 filter + size scan
```

---

## Quick Start

```bash
# 1. Full regime atlas (3 months, 59 days)
python run_full_analysis.py

# 2. Toxicity attribution
python run_toxicity_attribution.py

# 3. Inverted tox policy backtest
python run_inverted_tox.py

# 4. Sparse execution (production)
python run_sparse_execution.py
```

## Data Requirements

A-share L2 parquet files with:
- `orderbook_*.parquet`: 10-level bid/ask prices and quantities
- `message_*.parquet`: trade prints with direction, size, price

Expected in: `~/Desktop/000333/RawTrainData/`

---

## License

Research code. Not production trading advice.

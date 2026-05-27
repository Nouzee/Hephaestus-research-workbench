# Hephaestus — Quant Research Workbench

Agent-oriented microstructure research system. BTC perpetual + A-share L2.

## Key Paths
- Modules: `modules/` (probability, execution, risk, research, dictionary)
- Experiments: `experiments/` (ashare/, btc/)
- Docs: `docs/` (LEXICON, DEFINITIONS, INTERFACE, TRANSLATION)
- Strategies: `strategies/hephaestus-ssp/`

## Data
- A-share: `~/Desktop/000333/RawTrainData/` (message_*.parquet + orderbook_*.parquet)
- BTC: `~/Desktop/Zane/高频项目/刘子睿_HFT_Backtest/data/events_features.parquet`

## Core Conventions
- Frozen rules for CORE: 30%ile tox phase transition, binary filter, quantile-based thresholds
- No absolute spread/depth thresholds — per-window quantile calibration
- Regime K=8 via KMeans on 16 L2 features
- Tox inversion: tight spread = adverse selection loss, wide spread = protective

## Key Modules
- `modules/probability/` — 11-module stochastic process layer
- `modules/execution/` — fill simulation, hardened execution, PnL attribution
- `modules/risk/` — FSM, HMM scaler, inventory skew
- `projects/ashare/regime_segmentation.py` — L2 feature extractor

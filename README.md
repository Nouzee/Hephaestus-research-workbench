# Hephaestus — 量化研究基础设施

面向市场微观结构分析的实验编排系统。覆盖从 L2 订单簿特征提取、状态识别、执行模拟、多层级审计、到经济可行性边界分析的完整研究链路。

---

## 定位

Hephaestus 不是一个交易策略。它是一个研究方法论。

核心流程：假设生成 → 实验验证 → 多层审计 → 反证推翻 → 经济可行性边界分析。LLM 辅助代码生成与审计框架设计，研究者主导假设形成与决策判断。

---

## 研究弧线

```
BTC 字典学习（失败）
    │  Sharpe -40，逆向选择主导
    │  教训：信号识别 ≠ 可交易 edge
    ▼
A 股微观结构发现（000333）
    │  发现毒性反转：宽 spread = 盈利，窄 spread = 亏损
    │  相变点在 30% 分位，15 个滚动窗口一致
    ▼
执行感知 CORE（ECORE）
    │  L2 队列模拟替代 Bernoulli 成交
    │  发现 R2 执行陷阱（EV 最高但成交率 0.7%）
    ▼
回测完整性审计（BIA）
    │  发现 3 个致命会计错误
    │  Sharpe 7.82 → 0.61，0 回撤 → 4.3B 回撤
    ▼
库存感知 ECORE（IECORE）
    │  对称报价 → 库存随机游走
    │  Skewing 引擎：风险降 90%，PnL 不变
    ▼
Delta 中性 + Sharpe 审计（DNA + SA）
    │  执行 PnL 占总 PnL 100%，库存贡献 ~0%
    │  Edge 是纯执行的，不是方向暴露
    ▼
冻结样本外验证（FOOS）
    │  前 40 天训练，后 19 天 OOS
    │  Edge 存活，零衰退（PnL +12%）
    ▼
摩擦与压力测试（EFL-SRV）
    │  A 股零售费率下 Edge 被完全摧毁
    │  16/16 场景失败
    ▼
跨资产转移（X601899）→ 可行性边界分析（VF）
    │  601899 Edge 厚度是 000333 的 2.63 倍
    │  做市商计划下 ETR = 3.14，强可行
```

---

## 三层 Edge 框架

| 层级 | 问题 | 本项目结论 |
|---|---|---|
| 统计 Edge | 模式是否真实？ | 毒性反转真实存在，不是过拟合 |
| 执行 Edge | 能否执行出来？ | Delta 中性，库存受控，OOS 存活 |
| 经济可行性 | 扣除成本后还赚吗？ | 零售费率不可行，做市商计划可行 |

大多数 quant 项目停在第一层。Hephaestus 追问到了第三层。

---

## 仓库结构

```
Hephaestus/
├── experiments/ashare/         # A 股实验管线（14 个独立脚本）
│   ├── regime_discovery.py     #   市场状态聚类发现
│   ├── execution_validation.py #   L2 队列执行模拟（EVL）
│   ├── ecore.py                #   执行感知 CORE 重建
│   ├── ecore_dynamics.py       #   状态转移动力学（ECD）
│   ├── ecore_timing.py         #   进出场时机引擎（ETE）
│   ├── ebacktest.py            #   首次集成回测（后被 BIA 推翻）
│   ├── backtest_audit.py       #   回测完整性审计（BIA）
│   ├── iecure.py               #   库存感知 ECORE（IECORE）
│   ├── sharpe_audit.py         #   Sharpe 频率审计（SA）
│   ├── delta_neutral.py        #   Delta 中性分解（DNA）
│   ├── foos_validation.py      #   冻结样本外验证（FOOS）
│   ├── friction_stress.py      #   摩擦与压力测试（EFL-SRV）
│   ├── cross_asset_601899.py   #   跨资产转移（X601899）
│   └── viability_frontier.py   #   可行性边界分析（VF）
│
├── experiments/btc/            # BTC 实验管线（已放弃）
│   ├── market_decomposition.py
│   ├── causal_validation.py
│   ├── mode_dynamics.py
│   ├── scale_discovery.py
│   └── ...
│
├── research_logs/              # 研究轨迹记录
│   ├── Phase_00_BTC_Failure.md
│   ├── Phase_01_SSP.md
│   ├── Phase_02_ECORE.md
│   ├── Phase_03_BIA.md
│   ├── Phase_04_IECORE.md
│   ├── Phase_05_DNA.md
│   ├── Phase_06_FOOS.md
│   ├── Phase_07_EFL_SRV.md
│   ├── SUMMARY.md
│   ├── INTERVIEW_BRIEF.md
│   └── TECHNICAL_DEEP_DIVE.md
│
├── modules/                    # 核心库
│   ├── probability/            #   随机过程建模
│   ├── execution/              #   成交模拟 + PnL 归因
│   └── risk/                   #   风控状态机
│
├── projects/ashare/            # L2 特征提取器
│   └── regime_segmentation.py
│
├── docs/                       # 理论文档
│   ├── LEXICON.md              #   72 术语标准词典
│   ├── DEFINITIONS.md          #   核心概念严格定义
│   ├── INTERFACE.md            #   模块接口规范
│   └── TRANSLATION.md          #   术语 → 概率对象映射
│
├── examples/
│   └── demo_pipeline.py        #   最小可运行 demo
│
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

---

## 关键技术发现

**已验证**
- 毒性反转：宽 spread = 均值回归保护 → 结构性盈利；窄 spread = 逆向选择 → 结构性亏损
- 相变点在 30% 毒性分位，15 个滚动窗口一致，不是过拟合
- L2 队列真实成交率 82.4%（vs Bernoulli 模型的 30%）
- R2（深度流动性）是执行陷阱：成交率 0.7%，markout -49 bps，A/E = 4.42
- 二值状态过滤器严格优于连续权重（+111% PnL）
- 库存控制不创造 alpha，但将风险降低 90%+ 而不触及执行 edge
- Delta 中性分解：Edge 是纯粹的 spread 捕获，零方向暴露
- 冻结参数在样本外零衰退

**经济可行性**
- A 股零售费率下不可行（费用/spread 比 = 1.6-5.7x）
- 做市商计划（免印花税 + 0.1bp 佣金）下可行：601899 ETR = 3.14
- 低价股（~10 CNY）在零售费率下 ETR > 1.0
- 可行性改善主要来自股价差异（低股价 → 低费用 → 更高 ETR）

**BTC（已放弃）**
- 字典学习分解出可解释的市场模式，但 Granger 因果检验全部失败
- R² 天花板 = 8%，系统不可控
- 连续被动做市 Sharpe ≈ -40

---

## 快速开始

```bash
pip install -r requirements.txt
python examples/demo_pipeline.py  # 使用模拟数据运行完整流程
```

每个实验脚本可独立运行（需要对应的 L2 数据文件）。

---

## 文档

| 文档 | 用途 |
|---|---|
| `research_logs/SUMMARY.md` | 完整研究轨迹总结 |
| `research_logs/Phase_*.md` | 各阶段详细记录 |
| `docs/LEXICON.md` | 72 术语标准词典 |
| `docs/DEFINITIONS.md` | 核心概念严格定义 |

---

## 技术栈

Python · Polars · NumPy · scikit-learn · SciPy
LLM-assisted research workflow · Pipeline-as-experiment

---

## 许可

MIT License

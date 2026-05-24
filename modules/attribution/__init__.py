"""Attribution Module — 归因分析工坊"""
from .analyzer import (
    AttributionAnalyzer, MarkoutCalculator,
    FillProbabilityAnalyzer, TurnoverChecker,
    ComplianceStatus, MarkoutResult, FillResult,
    TurnoverResult, DiagnosticResult,
)
from .barra import BarraAttributor, FactorReturn, create_attributor
from .quant_attribution import (
    QuantThinkingAttributor, Hypothesis, DecisionRule,
    FailureMode, QuantAttributionResult,
    RegimeType, PnLSource, BTC_HFT_HYPOTHESIS,
)
from .anti_cheat import AntiCheatDetector, FutureLeakResult, ICDecayResult, OverfitResult, AntiCheatReport
from .socrates import SocratesAgent, MathReviewResult, AttributionNarrative, create_socrates

__all__ = [
    "AttributionAnalyzer", "MarkoutCalculator",
    "FillProbabilityAnalyzer", "TurnoverChecker",
    "ComplianceStatus", "MarkoutResult", "FillResult",
    "TurnoverResult", "DiagnosticResult",
    "BarraAttributor", "FactorReturn", "create_attributor",
    "QuantThinkingAttributor", "Hypothesis", "DecisionRule",
    "FailureMode", "QuantAttributionResult",
    "RegimeType", "PnLSource", "BTC_HFT_HYPOTHESIS",
    "AntiCheatDetector", "FutureLeakResult", "ICDecayResult",
    "OverfitResult", "AntiCheatReport",
    "SocratesAgent", "MathReviewResult", "AttributionNarrative", "create_socrates",
]
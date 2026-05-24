"""
Risk Controller — orchestrates ToxicityScorer + StateMachine -> MM actions.

Connects the multi-signal precursor scorer to the 4-state FSM and produces
per-tick market-making actions for PnL simulation.

Thin orchestrator — logic lives in ToxicityScorer and StateMachine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from modules.risk.state_machine import StateMachine, StateMachineConfig


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class RiskControllerConfig:
    """Configuration for the risk controller."""

    # Passed through to StateMachine
    recovery_cooldown: int = 10
    watch_max_consecutive: int = 5


# ===========================================================================
# Core: Risk Controller
# ===========================================================================

class RiskController:
    """
    Orchestrates precursor scorer -> FSM -> actions.

    Usage
    -----
    >>> rc = RiskController()
    >>> for batch in stream:
    ...     result = scorer.score(signals, gram_v)
    ...     action = rc.update(result["state"])
    """

    def __init__(
        self,
        scorer: "ToxicityScorer",
        config: Optional[RiskControllerConfig] = None,
    ):
        from modules.dictionary.toxicity_scorer import ToxicityScorer

        self.scorer = scorer
        self.config = config or RiskControllerConfig()

        self.fsm = StateMachine(StateMachineConfig(
            recovery_cooldown=self.config.recovery_cooldown,
            watch_max_consecutive=self.config.watch_max_consecutive,
        ))

        # Track decisions
        self.actions: list[dict] = []
        self.transitions: list[dict] = []

    def update(self, signals: dict, gram_v: float = 0.0) -> dict:
        """
        Full pipeline: score -> FSM transition -> action.

        Returns dict with: score, warn, hard, state, action, transition
        """
        # Score
        result = self.scorer.score(signals, gram_v)

        # FSM transition
        transition = self.fsm.transition(result["state"])
        self.transitions.append(transition)

        # Combine
        action = transition["action"]
        self.actions.append(action)

        return {
            "score": result["score"],
            "warn_trigger": result["warn_trigger"],
            "hard_trigger": result["hard_trigger"],
            "precursor_state": result["state"],
            "fsm_state": transition["state_after"],
            "action": action,
            "transition_reason": transition["reason"],
            "feature_breakdown": result["feature_breakdown"],
        }

    def summary(self) -> dict:
        """Aggregate controller statistics."""
        return {
            "scorer": self.scorer.summary(),
            "fsm": self.fsm.summary(),
            "n_actions": len(self.actions),
        }

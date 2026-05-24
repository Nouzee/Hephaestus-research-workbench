"""
State Machine — 4-State Risk Control FSM

NORMAL     — full quoting, standard spread
WATCH      — reduced size, wider spread
HARD_ALERT — withdraw all quotes, flat position
RECOVERY   — cautious re-entry, small size only

Transitions are based on multi-signal precursor states from ToxicityScorer.
Spread shock alone NEVER triggers re-entry from RECOVERY.

Design principle:
  - Escalation is fast (NORMAL -> HARD_ALERT in 1-2 steps)
  - De-escalation is slow (RECOVERY -> NORMAL requires sustained calm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class StateMachineConfig:
    """Configuration for the 4-state risk FSM."""

    # Recovery: how many consecutive NORMAL signals before leaving RECOVERY
    recovery_cooldown: int = 10

    # Watch: max consecutive WATCH signals before auto-escalating to HARD
    watch_max_consecutive: int = 5

    # ── Policy: per-state action parameters ──
    # NORMAL
    normal_size: float = 1.0
    normal_spread: float = 1.0

    # WATCH — light defense: widen spread, keep most size
    watch_size: float = 0.95
    watch_spread: float = 1.20
    watch_skew: float = 0.0      # inventory skew (not yet implemented)

    # HARD_ALERT — full withdraw
    hard_size: float = 0.0
    hard_spread: float = 0.0     # not quoting
    hard_duration: int = 2        # min windows to stay in HARD

    # RECOVERY — cautious re-entry
    recovery_size: float = 0.5
    recovery_spread: float = 1.35

    seed: int = 42


# ===========================================================================
# Core: State Machine
# ===========================================================================

class StateMachine:
    """
    4-State risk control FSM.

    States:
        NORMAL      (0)
        WATCH       (1)
        HARD_ALERT  (2)
        RECOVERY    (3)

    Usage
    -----
    >>> fsm = StateMachine()
    >>> for precursor_state in stream:
    ...     action = fsm.transition(precursor_state)
    ...     apply_action(action)
    """

    STATES = ["NORMAL", "WATCH", "HARD_ALERT", "RECOVERY"]
    STATE_IDX = {s: i for i, s in enumerate(STATES)}

    def __init__(self, config: Optional[StateMachineConfig] = None):
        self.config = config or StateMachineConfig()
        self.current_state = "NORMAL"
        self.state_history: list[str] = ["NORMAL"]

        # Counters
        self.consecutive_normal = 0
        self.consecutive_watch = 0

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------

    def transition(self, precursor_state: str) -> dict:
        """
        Evaluate transition based on precursor scorer output.

        Parameters
        ----------
        precursor_state : str — "NORMAL", "WATCH", "HARD_ALERT", or "RECOVERY"
                           from ToxicityScorer.score()

        Returns
        -------
        dict with keys: action, state_before, state_after, reason
        """
        cfg = self.config
        state_before = self.current_state

        # ── NORMAL state ──────────────────────────────────────────
        if self.current_state == "NORMAL":
            if precursor_state == "HARD_ALERT":
                self.current_state = "HARD_ALERT"
                reason = "hard trigger from NORMAL"
            elif precursor_state == "WATCH":
                self.consecutive_watch += 1
                if self.consecutive_watch >= cfg.watch_max_consecutive:
                    self.current_state = "HARD_ALERT"
                    reason = f"WATCH persisted {self.consecutive_watch}x -> HARD"
                else:
                    self.current_state = "WATCH"
                    reason = f"precursor WATCH (consecutive={self.consecutive_watch})"
            else:
                self.consecutive_watch = 0
                reason = "staying NORMAL"

        # ── WATCH state ───────────────────────────────────────────
        elif self.current_state == "WATCH":
            if precursor_state == "HARD_ALERT":
                self.current_state = "HARD_ALERT"
                reason = "hard trigger -> HARD_ALERT"
            elif precursor_state == "WATCH":
                self.consecutive_watch += 1
                if self.consecutive_watch >= cfg.watch_max_consecutive:
                    self.current_state = "HARD_ALERT"
                    reason = f"WATCH persisted {self.consecutive_watch}x -> HARD"
                else:
                    reason = f"staying WATCH (consecutive={self.consecutive_watch})"
            else:
                # NORMAL or RECOVERY from scorer -> go back to NORMAL
                self.current_state = "NORMAL"
                self.consecutive_watch = 0
                reason = "precursor cleared -> NORMAL"

        # ── HARD_ALERT state ──────────────────────────────────────
        elif self.current_state == "HARD_ALERT":
            if precursor_state == "HARD_ALERT":
                reason = "staying HARD_ALERT"
            elif precursor_state == "WATCH":
                self.current_state = "RECOVERY"
                reason = "HARD_ALERT downgraded -> RECOVERY (still WATCH)"
            else:
                self.current_state = "RECOVERY"
                self.consecutive_normal = 1
                reason = "HARD_ALERT -> RECOVERY (cooldown start)"

        # ── RECOVERY state ────────────────────────────────────────
        elif self.current_state == "RECOVERY":
            if precursor_state == "HARD_ALERT":
                self.current_state = "HARD_ALERT"
                self.consecutive_normal = 0
                reason = "re-escalation: RECOVERY -> HARD_ALERT"
            elif precursor_state == "NORMAL":
                self.consecutive_normal += 1
                if self.consecutive_normal >= cfg.recovery_cooldown:
                    self.current_state = "NORMAL"
                    reason = f"RECOVERY cooldown complete ({self.consecutive_normal}x NORMAL)"
                else:
                    reason = f"RECOVERY cooldown ({self.consecutive_normal}/{cfg.recovery_cooldown})"
            else:
                # WATCH or anything else: stay in RECOVERY
                self.consecutive_normal = 0
                reason = "staying RECOVERY"

        else:
            reason = "unknown state, resetting to NORMAL"
            self.current_state = "NORMAL"

        # Build action
        action = self._action_for_state(self.current_state)
        self.state_history.append(self.current_state)

        return {
            "action": action,
            "state_before": state_before,
            "state_after": self.current_state,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _action_for_state(self, state: str) -> dict:
        """Map FSM state to concrete market-making actions using policy config."""
        cfg = self.config
        actions = {
            "NORMAL": {
                "quote": True,
                "size_multiplier": cfg.normal_size,
                "spread_multiplier": cfg.normal_spread,
                "description": "full quoting, standard spread",
            },
            "WATCH": {
                "quote": True,
                "size_multiplier": cfg.watch_size,
                "spread_multiplier": cfg.watch_spread,
                "description": f"light defense: size={cfg.watch_size}x spread={cfg.watch_spread}x",
            },
            "HARD_ALERT": {
                "quote": False,
                "size_multiplier": cfg.hard_size,
                "spread_multiplier": cfg.hard_spread,
                "description": "withdraw all quotes",
            },
            "RECOVERY": {
                "quote": True,
                "size_multiplier": cfg.recovery_size,
                "spread_multiplier": cfg.recovery_spread,
                "description": f"cautious re-entry: size={cfg.recovery_size}x spread={cfg.recovery_spread}x",
            },
        }
        return actions.get(state, actions["NORMAL"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate FSM statistics."""
        if len(self.state_history) < 2:
            return {"transitions": 0}

        history = self.state_history
        n = len(history)
        transitions = sum(1 for i in range(1, n) if history[i] != history[i-1])

        state_counts = {}
        for s in self.STATES:
            state_counts[s] = history.count(s) / max(n, 1)

        return {
            "transitions": transitions,
            "state_distribution": state_counts,
            "current_state": self.current_state,
            "history_length": n,
        }

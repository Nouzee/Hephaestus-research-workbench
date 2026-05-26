"""
Stochastic Policy — a_t ~ π(a | S_t)

Probabilistic action selection. Every decision is a distribution over actions,
not a deterministic if-else rule.

  π(a | S_t) = soft decision over {WITHDRAW, MINIMAL, ACTIVE, AGGRESSIVE}
  where each action maps to (size, spread, cancel) parameters.

Replaces: regime-to-action lookup tables with proper probability distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ===========================================================================
# Action Definitions
# ===========================================================================

ACTIONS = {
    "WITHDRAW":   {"size": 0.0,  "spread": 1.0, "cancel": 1.0, "label": "flat"},
    "MINIMAL":    {"size": 0.1,  "spread": 1.5, "cancel": 0.8, "label": "minimal"},
    "DEFENSIVE":  {"size": 0.5,  "spread": 1.3, "cancel": 0.5, "label": "defensive"},
    "NORMAL":     {"size": 1.0,  "spread": 1.0, "cancel": 0.3, "label": "normal"},
    "ACTIVE":     {"size": 1.5,  "spread": 0.9, "cancel": 0.2, "label": "active"},
    "AGGRESSIVE": {"size": 2.5,  "spread": 0.8, "cancel": 0.1, "label": "aggressive"},
}

ACTION_NAMES = list(ACTIONS.keys())
N_ACTIONS = len(ACTION_NAMES)


# ===========================================================================
# Stochastic Policy
# ===========================================================================

@dataclass
class StochasticPolicy:
    """
    Probabilistic action selection: a_t ~ π(a | S_t).

    Usage
    -----
    >>> policy = StochasticPolicy()
    >>> policy.set_rule("CORE state → ACTIVE 80%, NORMAL 20%")
    >>> probs = policy.action_probs(state)  # (N_ACTIONS,) distribution
    >>> action = policy.sample(state)       # single draw
    """

    # Core state → action probability mapping
    # Keys: (regime, tox, tod) tuples. Values: (N_ACTIONS,) probability vectors.
    action_map: dict = field(default_factory=dict)

    # Default action when state not recognized
    default_probs: np.ndarray = None

    # Temperature for soft decisions (lower = more deterministic)
    temperature: float = 0.5

    # Exploration rate (ε-greedy component)
    epsilon: float = 0.05

    def __post_init__(self):
        if self.default_probs is None:
            # Default: mostly WITHDRAW
            p = np.zeros(N_ACTIONS)
            p[0] = 0.70  # WITHDRAW
            p[3] = 0.30  # NORMAL
            self.default_probs = p / p.sum()

    # ── Rule definition ──────────────────────────────────────────────

    def set_rule(
        self,
        state_key: tuple,        # (regime, tox, tod)
        action_probs: dict,      # {"ACTIVE": 0.8, "NORMAL": 0.2}
    ):
        """Define action probabilities for a specific state."""
        probs = np.zeros(N_ACTIONS)
        for action_name, prob in action_probs.items():
            if action_name in ACTION_NAMES:
                probs[ACTION_NAMES.index(action_name)] = prob
        probs = probs / probs.sum()  # normalize
        self.action_map[state_key] = probs

    def set_core_policy(self, core_states: set):
        """
        Set the SSP (Sparse State Participation) policy.

        CORE states → ACTIVE (0.7) + NORMAL (0.3)
        All others → WITHDRAW (0.95) + MINIMAL (0.05)
        """
        for state_key in core_states:
            self.set_rule(state_key, {"ACTIVE": 0.7, "NORMAL": 0.2, "AGGRESSIVE": 0.1})

        # Default is set in __post_init__ (mostly WITHDRAW)

    # ── Action probability computation ───────────────────────────────

    def action_probs(self, state) -> np.ndarray:
        """
        Compute π(a | S_t) — full action distribution.

        Parameters
        ----------
        state : StochasticState or tuple (regime, tox, tod)

        Returns
        -------
        probs : (N_ACTIONS,) probability vector summing to 1.
        """
        # Extract state key
        if hasattr(state, 'z_probs') and state.z_probs is not None:
            # Stochastic state: use soft regime + tox info
            regime = int(np.argmax(state.z_probs))
            tox = int(np.clip(state.h_mean * 7, 0, 6))  # continuous H → discrete tox
            tod = 1  # default MID
        elif isinstance(state, tuple):
            regime, tox, tod = state
        else:
            return self.default_probs.copy()

        key = (int(regime), int(tox), int(tod))

        if key in self.action_map:
            probs = self.action_map[key].copy()
        else:
            probs = self.default_probs.copy()

        # Apply temperature (soften/sharpen)
        if self.temperature != 1.0:
            log_probs = np.log(probs + 1e-12)
            probs = np.exp(log_probs / self.temperature)
            probs = probs / probs.sum()

        # ε-greedy exploration
        if self.epsilon > 0:
            uniform = np.ones(N_ACTIONS) / N_ACTIONS
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform

        return probs

    # ── Sampling ─────────────────────────────────────────────────────

    def sample(self, state) -> dict:
        """
        Sample a_t ~ π(a | S_t).

        Returns action dict with: size_multiplier, spread_multiplier,
        cancel_rate, action_label.
        """
        probs = self.action_probs(state)
        action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        action_name = ACTION_NAMES[action_idx]
        action = ACTIONS[action_name].copy()
        action["action_label"] = action_name
        return action

    def sample_batch(self, states: list, n_samples: int = 100) -> np.ndarray:
        """
        For each state, draw n_samples actions.
        Returns (len(states), n_samples, 3) array of (size, spread, cancel).
        """
        N_s = len(states)
        result = np.zeros((N_s, n_samples, 3))
        for i, state in enumerate(states):
            probs = self.action_probs(state)
            draws = np.random.choice(N_ACTIONS, size=n_samples, p=probs)
            for j, a_idx in enumerate(draws):
                a = ACTIONS[ACTION_NAMES[a_idx]]
                result[i, j, 0] = a["size"]
                result[i, j, 1] = a["spread"]
                result[i, j, 2] = a["cancel"]
        return result

    # ── Information measures ─────────────────────────────────────────

    def policy_entropy(self, state) -> float:
        """H(π(·|S_t)) — uncertainty of action selection."""
        probs = self.action_probs(state)
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def policy_kl(self, state, other_policy: "StochasticPolicy") -> float:
        """KL(π || π_other) at a given state."""
        p = self.action_probs(state)
        q = other_policy.action_probs(state)
        return float(np.sum(p * np.log(p / (q + 1e-12) + 1e-12)))

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        n_rules = len(self.action_map)
        active_actions = set()
        for probs in self.action_map.values():
            active_actions.update(
                ACTION_NAMES[i] for i in range(N_ACTIONS) if probs[i] > 0.1
            )
        return {
            "n_rules": n_rules,
            "active_actions": sorted(active_actions),
            "temperature": self.temperature,
            "epsilon": self.epsilon,
        }

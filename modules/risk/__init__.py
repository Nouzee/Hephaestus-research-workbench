"""
Risk Control Subpackage — state machine + inventory skew + controller.

  state_machine    — 4-state risk FSM (NORMAL/WATCH/HARD_ALERT/RECOVERY)
  layered_fsm      — 3-layer FSM: Structure(LF) + Pressure(MF) + Execution(HF)
  inventory_skew   — pressure-driven asymmetric quoting
  risk_controller  — orchestrator connecting scorer -> FSM -> MM actions
"""

__all__ = ["state_machine", "layered_fsm", "inventory_skew", "risk_controller", "hmm_scaler"]

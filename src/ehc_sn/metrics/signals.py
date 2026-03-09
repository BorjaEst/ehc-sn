"""Canonical signal vocabulary for diagnostic and research logging.

All signal producers (:meth:`~ehc_sn.training.act_head.ACTLossHead.compute_signals`,
:meth:`~ehc_sn.training.rl_head.RLLossHead.compute_signals`) and consumers
(:class:`~ehc_sn.callbacks.diagnostics.DiagnosticsCallback`) import from this
module rather than using string literals.  This ensures that renaming a signal
requires a single edit, and mismatches between producers and consumers fail
loudly via ``NameError`` rather than silently emitting nothing to the dashboard.

Structure
---------
``CROSS_PARADIGM_SIGNALS``
    Small set of signals that can appear in any training paradigm.  Use these
    in figure specs or callbacks that must work across models.

``ACT_SIGNALS``
    Signals specific to ACT (Adaptive Computation Time) training heads.

``RL_SIGNALS``
    Signals specific to RL (Reinforcement Learning) training heads.

``STANDARD_SIGNALS``
    The union of cross-paradigm + paradigm-specific signals that are stable
    enough to log at the ``"standard"`` diagnostic tier (T2).  All other signals
    are only logged at the ``"research"`` tier.
"""

from __future__ import annotations

# =================================================================================================
# Cross-paradigm — valid for any training regime
# =================================================================================================

STEPS_MEAN: str = "steps_mean"
"""Mean deliberation steps per slot (ACT and RL)."""

THETA_CLS_NORM: str = "theta_cls_norm"
"""L2 norm of the theta/CLS feature vector from the PFC backbone."""

CROSS_PARADIGM_SIGNALS: frozenset[str] = frozenset({STEPS_MEAN, THETA_CLS_NORM})


# =================================================================================================
# ACT-specific — produced by ACTLossHead.compute_signals()
# =================================================================================================

LOSS_Q_DONE: str = "loss_q_done"
"""Q-done binary cross-entropy loss (ACT)."""

TARGET_Q_MEAN: str = "target_q_mean"
"""TD(0) bootstrap Q-target mean (ACT)."""

TARGET_Q_STD: str = "target_q_std"
"""TD(0) bootstrap Q-target standard deviation (ACT)."""

ACT_SIGNALS: frozenset[str] = frozenset({LOSS_Q_DONE, TARGET_Q_MEAN, TARGET_Q_STD})


# =================================================================================================
# RL-specific — produced by RLLossHead.compute_signals()
# =================================================================================================

REWARD_MEAN: str = "reward_mean"
"""Mean environment reward across the batch."""

REWARD_STD: str = "reward_std"
"""Standard deviation of environment reward across the batch."""

Q_MEAN: str = "q_mean"
"""Mean Q-logit (vmPFC value estimate) across the batch."""

Q_STD: str = "q_std"
"""Standard deviation of Q-logits across the batch."""

RPE_MAGNITUDE: str = "rpe_magnitude"
"""Mean absolute reward prediction error (|reward - V(s)|)."""

ACTION_ENTROPY: str = "action_entropy"
"""Mean policy entropy over the action distribution."""

LOSS_ACTOR: str = "loss_actor"
"""Actor (policy gradient) loss sum for this step."""

LOSS_CRITIC: str = "loss_critic"
"""Critic (value MSE) loss sum for this step."""

LOSS_ENTROPY: str = "loss_entropy"
"""Entropy regularisation loss sum for this step."""

RL_SIGNALS: frozenset[str] = frozenset(
    {REWARD_MEAN, REWARD_STD, Q_MEAN, Q_STD, RPE_MAGNITUDE, ACTION_ENTROPY,
     LOSS_ACTOR, LOSS_CRITIC, LOSS_ENTROPY}
)  # fmt: skip


# =================================================================================================
# T2 standard set — re-used by DiagnosticsCallback
# =================================================================================================

STANDARD_SIGNALS: frozenset[str] = CROSS_PARADIGM_SIGNALS | ACT_SIGNALS | RL_SIGNALS
"""All signals that are logged at the ``"standard"`` diagnostic tier.

A :class:`~ehc_sn.callbacks.diagnostics.DiagnosticsCallback` configured with
``diagnostic_level="standard"`` will only log signals whose keys appear in this
set.  All other signals require ``diagnostic_level="research"``.
"""

# =================================================================================================
__all__ = [
    # Cross-paradigm
    "STEPS_MEAN", "THETA_CLS_NORM", "CROSS_PARADIGM_SIGNALS",
    # ACT
    "LOSS_Q_DONE", "TARGET_Q_MEAN", "TARGET_Q_STD", "ACT_SIGNALS",
    # RL
    "REWARD_MEAN", "REWARD_STD", "Q_MEAN", "Q_STD", "RPE_MAGNITUDE", "ACTION_ENTROPY",
    "LOSS_ACTOR", "LOSS_CRITIC", "LOSS_ENTROPY", "RL_SIGNALS",
    # Aggregate
    "STANDARD_SIGNALS",
]  # fmt: skip

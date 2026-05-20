"""Canonical signal vocabulary for diagnostic and research logging.

All signal producers (:meth:`~ehc_sn.objectives.act.ACTObjective.compute_signals`,
:meth:`~ehc_sn.objectives.hybrid_rl.HybridRLObjective.compute_step`,
:meth:`~ehc_sn.objectives.tem.TEMObjective.compute_signals`) and consumers
(:class:`~ehc_sn.callbacks.diagnostics.DiagnosticsCallback`) import from this
module rather than using string literals. This ensures that renaming a signal
requires a single edit, and mismatches between producers and consumers fail
loudly via ``NameError`` rather than silently emitting nothing to the dashboard.

Structure
---------
``CROSS_PARADIGM_SIGNALS``
    Small set of signals that can appear in any training paradigm.  Use these
    in figure specs or callbacks that must work across models.

``ACT_SIGNALS``
    Signals specific to ACT (Adaptive Computation Time) training objectives.

``RL_SIGNALS``
    Signals specific to RL (Reinforcement Learning) training objectives.

``VAR_SIGNALS``
    Latent variational signals shared across TEM-family objectives (base set for
    :data:`TEM_SIGNALS`).

``TEM_SIGNALS``
    Signals specific to TEM variational training objectives.

``STANDARD_SIGNALS``
    The union of cross-paradigm + paradigm-specific signals that are stable
    enough to log at the ``"standard"`` diagnostic tier (T2).  All other signals
    are only logged at the ``"research"`` tier.
"""

from __future__ import annotations

# =============================================================================
# Cross-paradigm — valid for any training regime
# =============================================================================

STEPS_MEAN: str = "steps_mean"
"""Mean deliberation steps per slot."""

THETA_CLS_NORM: str = "theta_cls_norm"
"""L2 norm of the theta/CLS feature vector from the PFC backbone."""

CROSS_PARADIGM_SIGNALS: frozenset[str] = frozenset({STEPS_MEAN, THETA_CLS_NORM})


# =============================================================================
# ACT-specific — produced by ACTObjective.compute_signals()
# =============================================================================

LOSS_Q_DONE: str = "loss_q_done"
"""Q-done binary cross-entropy loss (ACT)."""

TARGET_Q_MEAN: str = "target_q_mean"
"""TD(0) bootstrap Q-target mean (ACT)."""

TARGET_Q_STD: str = "target_q_std"
"""TD(0) bootstrap Q-target standard deviation (ACT)."""

ACT_SIGNALS: frozenset[str] = frozenset(
    {LOSS_Q_DONE, TARGET_Q_MEAN, TARGET_Q_STD}
)


# =============================================================================
# RL-specific — produced by HybridRLObjective.compute_step()
# =============================================================================


REWARD_MEAN: str = "reward_mean"
"""Mean environment reward across the batch."""

REWARD_STD: str = "reward_std"
"""Standard deviation of environment reward across the batch."""

Q_MEAN: str = "q_mean"
"""Mean Q-value across the batch."""

Q_STD: str = "q_std"
"""Standard deviation of Q-values across the batch."""

RPE_MAGNITUDE: str = "rpe_magnitude"
"""Mean absolute reward prediction error (|reward - V(s)|)."""

LOSS_STATE_VALUE: str = "loss_state_value"
"""State-value regression loss sum for this step."""

LOSS_Q_VALUE: str = "loss_q_value"
"""Q-value regression loss sum for this step."""

RL_SIGNALS: frozenset[str] = frozenset(
    {REWARD_MEAN, REWARD_STD, Q_MEAN, Q_STD, RPE_MAGNITUDE,
     LOSS_STATE_VALUE, LOSS_Q_VALUE}
)  # fmt: skip


# =============================================================================
# Latent variational signals — shared base for TEM-family objectives
# =============================================================================

LOSS_TOTAL: str = "loss_total"
"""Total VAR loss sum for this step."""

LOSS_OBS_NLL: str = "loss_obs_nll"
"""Observation negative log-likelihood loss sum (VAR)."""

LOSS_LATENT: str = "loss_latent"
"""Latent consistency loss sum (VAR)."""

LOSS_REG: str = "loss_reg"
"""Latent regularization loss sum (VAR)."""

LATENT_POST_NORM: str = "latent_post_norm"
"""Mean activation norm of posterior latent block(s)."""

LATENT_PRIOR_NORM: str = "latent_prior_norm"
"""Mean activation norm of prior latent block(s)."""

VAR_SIGNALS: frozenset[str] = frozenset(
    {
        LOSS_TOTAL,
        LOSS_OBS_NLL,
        LOSS_LATENT,
        LOSS_REG,
        LATENT_POST_NORM,
        LATENT_PRIOR_NORM,
    }
)


# =============================================================================
# TEM-specific — produced by TEMObjective.compute_signals() (canonical: TEMObjective)
# =============================================================================

LOSS_GRID_KL: str = "loss_grid_kl"
"""Grid latent consistency loss sum (TEM)."""

LOSS_PLACE_CONSISTENCY: str = "loss_place_consistency"
"""Place consistency loss sum (TEM)."""

LOSS_OBS_INFER: str = "loss_obs_infer"
"""Objective-scope inference-pathway observation NLL (TEM).

Weighted by ``c_obs``, revisit-masked, and normalised by ``protocol_count``.
Equals ``loss_obs_nll`` minus the retrieved and ancestral components.
"""

LOSS_OBS_RETRIEVED: str = "loss_obs_retrieved"
"""Objective-scope retrieved-pathway observation NLL (TEM).

Weighted by ``c_obs``, revisit-masked, and normalised by ``protocol_count``.
Equals ``loss_obs_nll`` minus the inference and ancestral components.
"""

LOSS_OBS_ANCESTRAL: str = "loss_obs_ancestral"
"""Objective-scope ancestral-pathway observation NLL (TEM).

Weighted by ``c_obs``, revisit-masked, and normalised by ``protocol_count``.
Equals ``loss_obs_nll`` minus the inference and retrieved components.
"""

LOSS_PLACE_TRANSITION: str = "loss_place_transition"
"""Objective-scope place-consistency transition contribution (TEM).

Weighted by ``c_place``, revisit-masked, and normalised by ``protocol_count``.
"""

LOSS_PLACE_SENSORY: str = "loss_place_sensory"
"""Objective-scope place-consistency sensory-cued contribution (TEM).

Weighted by ``c_place``, revisit-masked, and normalised by ``protocol_count``.
"""

GRID_POST_NORM: str = "grid_post_norm"
"""Mean activation norm of inferred grid-code blocks (TEM)."""

GRID_PRIOR_NORM: str = "grid_prior_norm"
"""Mean activation norm of generated grid-code blocks (TEM)."""

PLACE_POST_NORM: str = "place_post_norm"
"""Mean activation norm of inferred place-code blocks (TEM)."""

PLACE_PRIOR_NORM: str = "place_prior_norm"
"""Mean activation norm of generated place-code blocks (TEM)."""

TEM_SIGNALS: frozenset[str] = VAR_SIGNALS | frozenset(
    {
        LOSS_GRID_KL, LOSS_PLACE_CONSISTENCY, LOSS_OBS_INFER, LOSS_OBS_RETRIEVED, LOSS_OBS_ANCESTRAL,
        LOSS_PLACE_TRANSITION, LOSS_PLACE_SENSORY,
        GRID_POST_NORM, GRID_PRIOR_NORM, PLACE_POST_NORM,
        PLACE_PRIOR_NORM,
    }
)  # fmt: skip


# =============================================================================
# T2 standard set — re-used by DiagnosticsCallback
# =============================================================================

STANDARD_SIGNALS: frozenset[str] = (
    CROSS_PARADIGM_SIGNALS
    | ACT_SIGNALS
    | RL_SIGNALS
    | VAR_SIGNALS
    | TEM_SIGNALS
)
"""All signals that are logged at the ``"standard"`` diagnostic tier.

A :class:`~ehc_sn.callbacks.diagnostics.DiagnosticsCallback` configured with
``diagnostic_level="standard"`` will only log signals whose keys appear in this
set.  All other signals require ``diagnostic_level="research"``.
"""

# =============================================================================
__all__ = [
    # Cross-paradigm
    "STEPS_MEAN", "THETA_CLS_NORM", "CROSS_PARADIGM_SIGNALS",
    # ACT
    "LOSS_Q_DONE", "TARGET_Q_MEAN", "TARGET_Q_STD", "ACT_SIGNALS",
    # RL
    "REWARD_MEAN", "REWARD_STD", "Q_MEAN", "Q_STD", "RPE_MAGNITUDE",
    "LOSS_STATE_VALUE", "LOSS_Q_VALUE", "RL_SIGNALS",
    # VAR
    "LOSS_TOTAL", "LOSS_OBS_NLL", "LOSS_LATENT", "LOSS_REG", "LATENT_POST_NORM",
    "LATENT_PRIOR_NORM", "VAR_SIGNALS",
    # TEM
    "LOSS_GRID_KL", "LOSS_PLACE_CONSISTENCY", "LOSS_OBS_INFER", "LOSS_OBS_RETRIEVED",
    "LOSS_OBS_ANCESTRAL", "LOSS_PLACE_TRANSITION", "LOSS_PLACE_SENSORY", "GRID_POST_NORM",
    "GRID_PRIOR_NORM", "PLACE_POST_NORM", "PLACE_PRIOR_NORM", "TEM_SIGNALS",
    # Aggregate
    "STANDARD_SIGNALS",
]  # fmt: skip

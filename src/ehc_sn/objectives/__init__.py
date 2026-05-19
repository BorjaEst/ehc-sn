"""Canonical public surface for rollout-scoring objective modules.

For the ACT and TEM families the canonical names follow the ``*Objective*``
vocabulary; the legacy ``*Objective*`` names are preserved as backward-compatible
aliases. The hybrid RL family uses ``*Objective*`` names directly — it is a
learner-owned batch-loss path, not a rollout-scoring objective.

Prefer ``from ehc_sn.objectives import ...`` over any sub-module import.
"""

from ehc_sn.objectives._base import BaseObjective
from ehc_sn.objectives._token import (
    IGNORE_LABEL_ID,
    AccuracyStats,
    TokenSupervisionBinding,
    build_token_step_metrics,
    compute_accuracy_stats,
    compute_lm_loss_sum,
)
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalObjectiveBase,
    VariationalObjectiveStep,
    build_variational_step_metrics,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.objectives.act import (
    ACTLosses,
    ACTObjective,
    ACTObjectiveBinding,
    ACTObjectiveConfig,
    ACTObjectiveStep,
    ACTStepOutput,
)
from ehc_sn.objectives.hybrid_rl import (
    HybridRLLossConfig,
    HybridRLLosses,
    HybridRLObjective,
    HybridRLObjectiveStep,
)
from ehc_sn.objectives.tem import (
    TEMLosses,
    TEMObjective,
    TEMObjectiveBinding,
    TEMObjectiveConfig,
    TEMObjectiveStep,
    TEMStepOutput,
)

# =============================================================================
__all__ = [
    # base
    "BaseObjective",
    # token family — implementation / compat
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "TokenSupervisionBinding",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    "compute_lm_loss_sum",
    # variational family — canonical
    "VariationalObjectiveBase",
    # variational family — implementation / compat
    "VariationalLosses",
    "VariationalObjectiveStep",
    "build_variational_step_metrics",
    "get_reg_term",
    "require_latent_relation",
    # act — canonical
    "ACTObjectiveConfig",
    "ACTObjective",
    "ACTObjectiveStep",
    "ACTLosses",
    "ACTObjectiveBinding",
    "ACTStepOutput",
    # hybrid rl — batch-loss path (no *Objective* aliases; not a rollout scorer)
    "HybridRLLossConfig",
    "HybridRLObjective",
    "HybridRLLosses",
    "HybridRLObjectiveStep",
    # tem — canonical
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    "TEMLosses",
    "TEMObjectiveBinding",
    "TEMStepOutput",
]

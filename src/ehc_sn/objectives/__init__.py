"""Canonical public surface for rollout-scoring objective modules.

For the ACT and TEM families the canonical names follow the ``*Objective*``
vocabulary; the legacy ``*LossHead*`` names are preserved as backward-compatible
aliases. The hybrid RL family uses ``*LossHead*`` names directly — it is a
learner-owned batch-loss path, not a rollout-scoring objective.

Prefer ``from ehc_sn.objectives import ...`` over any sub-module import.
"""

from ehc_sn.objectives._base import BaseObjective
from ehc_sn.objectives._token import (
    IGNORE_LABEL_ID,
    AccuracyStats,
    TokenLosses,
    TokenObjectiveBase,
    TokenSupervisionBinding,
    build_token_step_metrics,
    compute_accuracy_stats,
    compute_lm_loss_sum,
)
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalLossStep,
    VariationalObjectiveBase,
    build_variational_step_metrics,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.objectives.act import ACTObjective, ACTObjectiveConfig, ACTObjectiveStep
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLLosses, HybridRLLossHead, HybridRLLossStep
from ehc_sn.objectives.tem import TEMLosses, TEMObjective, TEMObjectiveConfig, TEMObjectiveStep

__all__ = [
    # base
    "BaseObjective",
    # token family — canonical
    "TokenObjectiveBase",
    # token family — implementation / compat
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "TokenLosses",
    "TokenSupervisionBinding",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    "compute_lm_loss_sum",
    # variational family — canonical
    "VariationalObjectiveBase",
    # variational family — implementation / compat
    "VariationalLosses",
    "VariationalLossStep",
    "build_variational_step_metrics",
    "get_reg_term",
    "require_latent_relation",
    # act — canonical
    "ACTObjectiveConfig",
    "ACTObjective",
    "ACTObjectiveStep",
    # hybrid rl — batch-loss path (no *Objective* aliases; not a rollout scorer)
    "HybridRLLossConfig",
    "HybridRLLossHead",
    "HybridRLLosses",
    "HybridRLLossStep",
    # tem — canonical
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    "TEMLosses",
]

"""Canonical public surface for rollout-scoring objective modules.

All objective-scoring implementations reside in the submodules of this package.
``ehc_sn.heads`` is a deprecated compatibility alias that re-exports from here;
prefer ``from ehc_sn.objectives import ...`` for all new code.
"""

from ehc_sn.objectives._base import BaseObjective
from ehc_sn.objectives._token import (
    IGNORE_LABEL_ID,
    AccuracyStats,
    TokenLosses,
    TokenLossHeadBase,
    TokenSupervisionBinding,
    build_token_step_metrics,
    compute_accuracy_stats,
    compute_lm_loss_sum,
)
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalLossHeadBase,
    VariationalLossStep,
    build_variational_step_metrics,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.objectives.act import ACTLossConfig, ACTLossHead, ACTLossStep, ACTTaskBinding
from ehc_sn.objectives.rl import RLLossConfig, RLLossHead, RLLossStep, RLObjectiveBinding
from ehc_sn.objectives.tem import TEMLossConfig, TEMLosses, TEMLossHead, TEMLossStep, TEMSupervisionBinding
from ehc_sn.objectives.var import LatentCode, VARLossConfig, VARLosses, VARLossHead, VARLossStep

__all__ = [
    # base
    "BaseObjective",
    # token family
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "TokenLossHeadBase",
    "TokenLosses",
    "TokenSupervisionBinding",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    "compute_lm_loss_sum",
    # variational family
    "VariationalLossHeadBase",
    "VariationalLosses",
    "VariationalLossStep",
    "build_variational_step_metrics",
    "get_reg_term",
    "require_latent_relation",
    # act
    "ACTLossConfig",
    "ACTLossHead",
    "ACTLossStep",
    "ACTTaskBinding",
    # rl
    "RLLossConfig",
    "RLLossHead",
    "RLLossStep",
    "RLObjectiveBinding",
    # tem
    "TEMLossConfig",
    "TEMLossHead",
    "TEMLosses",
    "TEMLossStep",
    "TEMSupervisionBinding",
    # var
    "LatentCode",
    "VARLossConfig",
    "VARLossHead",
    "VARLosses",
    "VARLossStep",
]

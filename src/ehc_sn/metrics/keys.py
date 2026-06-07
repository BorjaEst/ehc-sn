"""Stable internal keys for algorithm-specific ratio metrics."""

from __future__ import annotations

# =============================================================================
LOSS_TOKEN: str = "loss_token"

ACT_LOSS_Q_DONE: str = "act_loss_q_done"
ACT_LOSS_Q_CONTINUE: str = "loss_q_continue"

RL_LOSS_STATE_VALUE: str = "rl_loss_state_value"
RL_LOSS_Q_VALUE: str = "rl_loss_q_value"

TEM_LOSS_OBS_NLL: str = "loss_obs_nll_revisit"
TEM_LOSS_GRID_KL: str = "loss_grid_kl_revisit"
TEM_LOSS_PLACE_TRANSITION: str = "loss_place_transition_revisit"
TEM_LOSS_PLACE_SENSORY: str = "loss_place_sensory_revisit"
TEM_LOSS_PLACE_CONSISTENCY: str = "loss_place_consistency_revisit"
TEM_LOSS_REG: str = "loss_reg_revisit"
TEM_ACC_OBS_INFERENCE_REVISIT: str = "accuracy_obs_inference_revisit"
TEM_ACC_OBS_INFERENCE_ALL: str = "accuracy_obs_inference_all"
TEM_ACC_OBS_RETRIEVED_REVISIT: str = "accuracy_obs_retrieved_revisit"
TEM_ACC_OBS_RETRIEVED_ALL: str = "accuracy_obs_retrieved_all"
TEM_ACC_OBS_ANCESTRAL_REVISIT: str = "accuracy_obs_ancestral_revisit"
TEM_ACC_OBS_ANCESTRAL_ALL: str = "accuracy_obs_ancestral_all"

TEM_LOSS_OBS_INFERENCE: str = "loss_obs_inference_revisit"
TEM_LOSS_OBS_RETRIEVED: str = "loss_obs_retrieved_revisit"
TEM_LOSS_OBS_ANCESTRAL: str = "loss_obs_ancestral_revisit"

EHC_LOSS_OBS_NLL_REVISIT: str = "ehc_loss_obs_nll_revisit"
EHC_LOSS_OBS_NLL_ALL: str = "ehc_loss_obs_nll_all"
EHC_LOSS_GRID_KL_REVISIT: str = "ehc_loss_grid_kl_revisit"
EHC_LOSS_GRID_KL_ALL: str = "ehc_loss_grid_kl_all"
EHC_LOSS_PLACE_CONSISTENCY_REVISIT: str = "ehc_loss_place_consistency_revisit"
EHC_LOSS_PLACE_CONSISTENCY_ALL: str = "ehc_loss_place_consistency_all"
EHC_LOSS_REG_REVISIT: str = "ehc_loss_reg_revisit"
EHC_LOSS_REG_ALL: str = "ehc_loss_reg_all"
EHC_ACC_OBS_INFERENCE_REVISIT: str = "ehc_accuracy_obs_inference_revisit"
EHC_ACC_OBS_INFERENCE_ALL: str = "ehc_accuracy_obs_inference_all"
EHC_ACC_OBS_RETRIEVED_REVISIT: str = "ehc_accuracy_obs_retrieved_revisit"
EHC_ACC_OBS_RETRIEVED_ALL: str = "ehc_accuracy_obs_retrieved_all"
EHC_ACC_OBS_ANCESTRAL_REVISIT: str = "ehc_accuracy_obs_ancestral_revisit"
EHC_ACC_OBS_ANCESTRAL_ALL: str = "ehc_accuracy_obs_ancestral_all"


# =============================================================================
def extra_ratio_paths(  # -----------------------------------------------------
    key: str,
) -> tuple[str, str]:
    """Return numerator/denominator paths for a keyed extra ratio metric."""
    base_path = f"extras.{key}"
    return f"{base_path}.numerator_sum", f"{base_path}.denominator_sum"


# =============================================================================
__all__ = [
    "ACT_LOSS_Q_DONE",
    "ACT_LOSS_Q_CONTINUE",
    "LOSS_TOKEN",
    "RL_LOSS_Q_VALUE",
    "RL_LOSS_STATE_VALUE",
    "EHC_ACC_OBS_ANCESTRAL_ALL",
    "EHC_ACC_OBS_ANCESTRAL_REVISIT",
    "EHC_ACC_OBS_INFERENCE_ALL",
    "EHC_ACC_OBS_INFERENCE_REVISIT",
    "EHC_ACC_OBS_RETRIEVED_ALL",
    "EHC_ACC_OBS_RETRIEVED_REVISIT",
    "EHC_LOSS_GRID_KL_ALL",
    "EHC_LOSS_GRID_KL_REVISIT",
    "EHC_LOSS_OBS_NLL_ALL",
    "EHC_LOSS_OBS_NLL_REVISIT",
    "EHC_LOSS_PLACE_CONSISTENCY_ALL",
    "EHC_LOSS_PLACE_CONSISTENCY_REVISIT",
    "EHC_LOSS_REG_ALL",
    "EHC_LOSS_REG_REVISIT",
    "TEM_ACC_OBS_ANCESTRAL_ALL",
    "TEM_ACC_OBS_ANCESTRAL_REVISIT",
    "TEM_ACC_OBS_INFERENCE_ALL",
    "TEM_ACC_OBS_INFERENCE_REVISIT",
    "TEM_ACC_OBS_RETRIEVED_ALL",
    "TEM_ACC_OBS_RETRIEVED_REVISIT",
    "TEM_LOSS_GRID_KL",
    "TEM_LOSS_OBS_NLL",
    "TEM_LOSS_PLACE_CONSISTENCY",
    "TEM_LOSS_REG",
    "TEM_LOSS_OBS_INFERENCE",
    "TEM_LOSS_OBS_RETRIEVED",
    "TEM_LOSS_OBS_ANCESTRAL",
    "extra_ratio_paths",
]

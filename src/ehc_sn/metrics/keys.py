"""Stable internal keys for algorithm-specific ratio metrics."""

from __future__ import annotations

# =============================================================================
LOSS_TOKEN: str = "loss_token"

ACT_LOSS_Q_DONE: str = "loss_q_done"
ACT_LOSS_Q_CONTINUE: str = "loss_q_continue"

RL_LOSS_STATE_VALUE: str = "loss_state_value"
RL_LOSS_Q_VALUE: str = "loss_q_value"

TEM_ACC_OBS_POST_REVISIT: str = "accuracy_obs_post_revisit"
TEM_ACC_OBS_POST_ALL: str = "accuracy_obs_post_all"
TEM_ACC_OBS_RECALL_REVISIT: str = "accuracy_obs_recall_revisit"
TEM_ACC_OBS_RECALL_ALL: str = "accuracy_obs_recall_all"
TEM_ACC_OBS_PATH_REVISIT: str = "accuracy_obs_path_revisit"
TEM_ACC_OBS_PATH_ALL: str = "accuracy_obs_path_all"
TEM_LOSS_OBS_NLL: str = "loss_obs_nll"
TEM_LOSS_GRID_KL: str = "loss_grid_kl"
TEM_LOSS_PLACE_TRANSITION: str = "loss_place_transition"
TEM_LOSS_PLACE_SENSORY: str = "loss_place_sensory"
TEM_LOSS_PLACE_CONSISTENCY: str = "loss_place_consistency"
TEM_LOSS_REG: str = "loss_reg_revisit"
TEM_LOSS_OBS_POST: str = "loss_obs_post"
TEM_LOSS_OBS_RECALL: str = "loss_obs_recall"
TEM_LOSS_OBS_PATH: str = "loss_obs_path"


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
    "TEM_ACC_OBS_PATH_ALL",
    "TEM_ACC_OBS_PATH_REVISIT",
    "TEM_ACC_OBS_POST_ALL",
    "TEM_ACC_OBS_POST_REVISIT",
    "TEM_ACC_OBS_RECALL_ALL",
    "TEM_ACC_OBS_RECALL_REVISIT",
    "TEM_LOSS_GRID_KL",
    "TEM_LOSS_OBS_NLL",
    "TEM_LOSS_PLACE_CONSISTENCY",
    "TEM_LOSS_REG",
    "TEM_LOSS_OBS_POST",
    "TEM_LOSS_OBS_RECALL",
    "TEM_LOSS_OBS_PATH",
    "extra_ratio_paths",
]

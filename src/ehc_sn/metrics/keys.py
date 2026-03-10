"""Stable internal keys for algorithm-specific ratio metrics."""

from __future__ import annotations

# =================================================================================================
LOSS_LM: str = "loss_lm"

ACT_LOSS_Q_DONE: str = "loss_q_done"
ACT_LOSS_Q_CONTINUE: str = "loss_q_continue"

RL_LOSS_ACTOR: str = "loss_actor"
RL_LOSS_CRITIC: str = "loss_critic"
RL_LOSS_ENTROPY: str = "loss_entropy"
RL_LOSS_Q_VALUE: str = "loss_q_value"

VAR_LOSS_OBS_NLL: str = "loss_obs_nll"
VAR_LOSS_LATENT: str = "loss_latent"
VAR_LOSS_REG: str = "loss_reg"

TEM_LOSS_OBS_NLL: str = "loss_obs_nll"
TEM_LOSS_GRID_KL: str = "loss_grid_kl"
TEM_LOSS_PLACE_CONSISTENCY: str = "loss_place_consistency"
TEM_LOSS_REG: str = "loss_reg"


# =================================================================================================
def extra_ratio_paths(  # -------------------------------------------------------------------------
    key: str,
) -> tuple[str, str]:  # fmt: skip
    """Return numerator/denominator paths for a keyed extra ratio metric."""
    base_path = f"extras.{key}"
    return f"{base_path}.numerator_sum", f"{base_path}.denominator_sum"


# =================================================================================================
__all__ = [
    "ACT_LOSS_Q_CONTINUE", "ACT_LOSS_Q_DONE", "LOSS_LM", "RL_LOSS_ACTOR", "RL_LOSS_CRITIC",
    "RL_LOSS_ENTROPY", "RL_LOSS_Q_VALUE", "TEM_LOSS_GRID_KL", "TEM_LOSS_OBS_NLL",
    "TEM_LOSS_PLACE_CONSISTENCY", "TEM_LOSS_REG", "VAR_LOSS_LATENT", "VAR_LOSS_OBS_NLL",
    "VAR_LOSS_REG", "extra_ratio_paths",
]  # fmt: skip

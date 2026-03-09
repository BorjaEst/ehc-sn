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
    "RL_LOSS_ENTROPY", "RL_LOSS_Q_VALUE", "extra_ratio_paths",
]  # fmt: skip

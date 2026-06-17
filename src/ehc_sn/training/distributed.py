"""Distributed-training helpers for experiment entrypoints.

These helpers keep launcher policy out of experiment scripts while preserving
cluster behavior for true multi-rank runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from lightning.pytorch.strategies import DDPStrategy
from torch import Tensor


# =============================================================================
@dataclass(frozen=True)
class SumOverBatch:
    """Loss reduced by sum over the local batch.

    ``normalize_loss_for_backward`` accepts only this type, enforcing
    the sum-reduction contract at the call site. A mean-reduced loss
    passed here produces incorrectly scaled gradients under DDP.

    The caller (training_step) is responsible for wrapping the objective's
    summed loss in this type before passing to ``normalize_loss_for_backward``.
    """

    value: Tensor


# =============================================================================
def normalize_loss_for_backward(  # -------------------------------------------
    loss: SumOverBatch,
    local_bs: int,
) -> Tensor:
    """Normalize a sum-reduced loss by the local batch size for DDP.

    In distributed training, each rank computes gradients on its local
    mini-batch. Dividing the summed loss by the local batch size converts
    it to a per-example mean before ``manual_backward()``. DDP then
    averages gradients across ranks, yielding the correct effective batch
    size of ``local_bs * world_size``.

    Args:
        loss: Sum-reduced loss over the local batch.
        local_bs: Effective local batch size on the current rank.

    Returns:
        Mean loss per example, ready for ``backward()``.
    """
    if local_bs <= 0:
        raise ValueError(f"local_bs must be positive, got {local_bs}.")
    return loss.value / float(local_bs)


# =============================================================================
def resolve_effective_world_size(  # ------------------------------------------
    trainer_strategy: str,
    trainer_devices: int,
    trainer_num_nodes: int,
) -> int:
    """Resolve the effective distributed world size.

    Args:
        trainer_strategy: Requested Lightning strategy.
        trainer_devices: Number of devices per node.
        trainer_num_nodes: Number of nodes.

    Returns:
        Effective world size used for batch-size validation and strategy resolution.
    """
    configured_world_size = max(
        int(trainer_devices) * int(trainer_num_nodes), 1
    )

    if os.environ.get("SLURM_JOB_ID"):
        return max(
            int(os.environ.get("SLURM_NTASKS", str(configured_world_size))), 1
        )
    if trainer_strategy == "ddp":
        return configured_world_size
    return 1


# =============================================================================
def validate_num_slots_divisibility(  # ---------------------------------------
    num_slots: int,
    world_size: int,
) -> None:
    """Validate that num_slots is divisible by the effective world size."""
    if world_size <= 0:
        raise ValueError(
            "World size must be a positive integer.",
        )
    if num_slots % world_size != 0:
        raise ValueError(
            "num_slots must be divisible by world_size. "
            f"Got num_slots={num_slots}, "
            f"world_size={world_size}.",
        )


# =============================================================================
def resolve_trainer_strategy(  # ----------------------------------------------
    trainer_strategy: str,
    world_size: int,
    *,
    find_unused_parameters: bool = False,
) -> str | DDPStrategy:
    """Return the effective Lightning Trainer strategy.

    Requested DDP is downgraded to ``auto`` when the effective world size is 1,
    avoiding unnecessary process relaunch during local debugging.
    """
    if trainer_strategy != "ddp":
        return trainer_strategy
    if world_size <= 1:
        return "auto"
    return DDPStrategy(find_unused_parameters=find_unused_parameters)


# =============================================================================
__all__ = [
    "SumOverBatch",
    "normalize_loss_for_backward",
    "resolve_effective_world_size",
    "resolve_trainer_strategy",
    "validate_num_slots_divisibility",
]

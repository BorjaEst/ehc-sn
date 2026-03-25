"""Distributed-training helpers for experiment entrypoints.

These helpers keep launcher policy out of experiment scripts while preserving
cluster behavior for true multi-rank runs.
"""

from __future__ import annotations

import os

from lightning.pytorch.strategies import DDPStrategy


# =================================================================================================
def resolve_effective_world_size(  # --------------------------------------------------------------
    trainer_strategy: str,
    trainer_devices: int,
    trainer_num_nodes: int,
) -> int:  # fmt: skip
    """Resolve the effective distributed world size.

    Args:
        trainer_strategy: Requested Lightning strategy.
        trainer_devices: Number of devices per node.
        trainer_num_nodes: Number of nodes.

    Returns:
        Effective world size used for batch-size validation and strategy resolution.
    """
    configured_world_size = max(int(trainer_devices) * int(trainer_num_nodes), 1)

    if os.environ.get("SLURM_JOB_ID"):
        return max(int(os.environ.get("SLURM_NTASKS", str(configured_world_size))), 1)
    if trainer_strategy == "ddp":
        return configured_world_size
    return 1


# =================================================================================================
def validate_batch_size_divisibility(  # ----------------------------------------------------------
    global_batch_size: int,
    world_size: int,
) -> None:  # fmt: skip
    """Validate that the global batch size is divisible by the effective world size."""
    if world_size <= 0:
        raise ValueError("World size must be a positive integer.")
    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size must be divisible by world_size. Got global_batch_size={global_batch_size}, world_size={world_size}.")  # fmt: skip


# =================================================================================================
def resolve_trainer_strategy(  # ------------------------------------------------------------------
    trainer_strategy: str,
    world_size: int,
    *,
    find_unused_parameters: bool = False,
) -> str | DDPStrategy:  # fmt: skip
    """Return the effective Lightning Trainer strategy.

    Requested DDP is downgraded to ``auto`` when the effective world size is 1,
    avoiding unnecessary process relaunch during local debugging.
    """
    if trainer_strategy != "ddp":
        return trainer_strategy
    if world_size <= 1:
        return "auto"
    return DDPStrategy(find_unused_parameters=find_unused_parameters)


# =================================================================================================
__all__ = [
    "resolve_effective_world_size",
    "resolve_trainer_strategy",
    "validate_batch_size_divisibility",
]

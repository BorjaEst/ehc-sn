from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.optimizer import Optimizer


# =================================================================================================
class SchedulerConfig(BaseModel, extra="forbid"):
    warmup_steps: int = Field(
        default=0,
        description="Number of warmup steps for learning rate scheduling.",
    )
    min_ratio: float = Field(
        default=0.0,
        description="Minimum learning rate ratio for cosine decay. The learning rate will decay to `base_lr * min_ratio` at the end of training.",
    )


# =================================================================================================
# class SequentialLR(SequentialLR):
#     """ """

#     def __init__(  # ------------------------------------------------------------------------------
#         self, optimizer: Optimizer, total_steps: int, config: Optional[SchedulerConfig] = None,
#     ) -> None:  # fmt: skip
#         """ """
#         config = config or SchedulerConfig()
#         super().__init__(
#             optimizer,
#             schedulers=[LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)],
#             milestones=[total_steps],
#         )


# =================================================================================================
class CosineAnnealingLRWithWarmup(SequentialLR):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, optimizer: Optimizer, total_steps: int, config: Optional[SchedulerConfig] = None,
    ) -> None:  # fmt: skip
        """ """
        config = config or SchedulerConfig()
        warmup_steps = config.warmup_steps
        cosine_steps = total_steps - warmup_steps
        min_ratio = config.min_ratio
        lr = optimizer.param_groups[0]["lr"]

        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr * min_ratio)
        super().__init__(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# =================================================================================================
__all__ = [
    "SchedulerConfig", "SequentialLR", "CosineAnnealingLR", "LinearLR", "CosineAnnealingLRWithWarmup",
]  # fmt: skip

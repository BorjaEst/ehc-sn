from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from adam_atan2_pytorch import AdamAtan2
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Adam as TorchAdam
from torch.optim.optimizer import Optimizer, ParamsT

__all__ = ["AdamATan2Config", "AdamATan2"]


class AdamATan2Config(BaseModel, extra="forbid"):
    lr: float = Field(
        default=1e-4,
        description="Base learning rate for the main optimizer (e.g. Adam). The learning rate for the puzzle embedding optimizer is set by `puzzle_emb_lr`.",
    )
    weight_decay: float = Field(
        default=1e-2,
        description="Weight decay for the main optimizer (e.g. Adam). The weight decay for the puzzle embedding optimizer is set by `emb_weight_decay`.",
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.98),
        description="Betas for Adam optimizer. The betas for the puzzle embedding optimizer are not set by default since Adam is not used for the puzzle embedding optimizer.",
    )


class AdamATan2(AdamAtan2):
    def __init__(self, params: ParamsT, config: Optional[AdamATan2Config] = None):
        config = config or AdamATan2Config()
        super().__init__(params, **config.model_dump())


class AdamConfig(BaseModel, extra="forbid"):
    lr: float = Field(
        default=1e-4,
        description="Learning rate for the Adam optimizer.",
    )
    weight_decay: float = Field(
        default=1e-2,
        description="Weight decay for the Adam optimizer.",
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.98),
        description="Betas for the Adam optimizer.",
    )


class Adam(TorchAdam):
    def __init__(self, params: ParamsT, config: Optional[AdamConfig] = None):
        config = config or AdamConfig()
        super().__init__(params, **config.model_dump())

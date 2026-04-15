"""HRM v1 with a canonical named-workspace core and a legacy batch bridge."""

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.types import Batch
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class ModelSettingsV1(BaseModel, extra="forbid"):
    """Model-level settings composing a PFC module with embedding/LM-head parameters."""

    # TODO: implement


# =================================================================================================
@dataclass
class HRMStateV1(DetachMixin):
    """Container for the full recurrent HRM state."""

    pfc: PFCState


# =================================================================================================
class HRModelV1(nn.Module):
    """Pure HRM v1 architecture with a legacy batch-compatible wrapper."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        config: ModelSettingsV1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """ """
        # TODO: implement the component modules instantiation

    @property
    def config(self) -> ModelSettingsV1:
        """ """
        return self._config

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:
        """Initialize parameters and buffers."""
        # TODO: implement

    def init_state(  # ---------------------------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Create a fresh recurrent state (``ACTRolloutBackbone`` protocol)."""
        # TODO: implement

    def reset_state(  # --------------------------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Selectively reset rows of the recurrent state (``ACTRolloutBackbone`` protocol)."""
        # TODO: implement

    def forward(  # -------------------------------------------------------------------------------
        self,
        input_a: Tensor,
        input_b: Tensor,
        # TODO: define inputs according to model, task agnostic
        state: Optional[HRMStateV1] = None,
    ) -> tuple[HRMStateV1, tuple[Tensor, Tensor], Tensor]:
        """Legacy ACT-compatible wrapper over the canonical named-workspace HRM core."""
        # TODO: implement


# =================================================================================================
__all__ = ["HRModelV1", "HRMStateV1", "ModelSettingsV1"]

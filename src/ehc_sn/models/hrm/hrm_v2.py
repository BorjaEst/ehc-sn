"""HRM v2 core contracts over a task-agnostic recurrent substrate."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.pfc import FixedSlot, PFCModel, PFCSettings, PFCState, SlotFamily, WorkspaceLayout, WorkspaceSchema
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ModelSettingsV2(BaseModel, extra="forbid"):
    """Model-level settings for HRM v2.

    This settings object composes:
        - PFC settings (recurrent reasoning core)
        - STR settings (actor-critic / reward head)
        - schema-slot width and count inherited from PFC settings

    Notes:
        HRM v2 currently requires RoPE positional encodings inside the PFC modules
        for legacy parity and to match the environment tokenization.
    """

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelSettingsV2":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)

    pfc: PFCSettings = Field(..., description="Settings for the core PFC model architecture.")
    str: STRSettings = Field(..., description="Settings for the STR actor-critic architecture.")

    @property
    def num_schema_slots(self) -> int:
        """Return the number of schema slots owned by the HRM core."""
        return self.pfc.seq_length

    @property
    def schema_layout(self) -> WorkspaceLayout:
        """Return the body-only schema layout."""
        return WorkspaceLayout.from_schema(WorkspaceSchema(fixed=(), families=(SlotFamily("schema", self.num_schema_slots),)))

    @property
    def workspace_layout(self) -> WorkspaceLayout:
        """Return the full workspace layout."""
        return WorkspaceLayout.from_schema(
            WorkspaceSchema(
                fixed=(FixedSlot("controller"),),
                families=(SlotFamily("schema", self.num_schema_slots),),
            )
        )


# =============================================================================
@dataclass(frozen=True)
class HRMInputV2:
    """Task-agnostic schema payload consumed by the HRM v2 core."""

    schema_tokens: Tensor
    prefix_bias: Tensor | None = None


# =============================================================================
@dataclass
class HRMStateV2(DetachMixin):
    """Recurrent state carried across steps for HRM v2."""

    pfc: PFCState  # Prefrontal Cortex state
    str: STRState  # STR actor-critic state


# =============================================================================
@dataclass(frozen=True)
class HRMOutputV2:
    """Architecture-native HRM v2 output."""

    theta_summary: Tensor
    schema_slots: Tensor
    q_logits: Tensor
    state_value: Tensor

    @property
    def r_logits(self) -> Tensor:
        """Backward-compatible alias for the critic state value."""
        return self.state_value


# =============================================================================
class HRModelV2(nn.Module):
    """Core HRM v2 model.

    The model consists of:
        - PFC recurrent reasoning module over schema-slot tokens
        - STR actor-critic module consuming the CLS summary and PFC Q logits

    The forward pass returns:
        - architecture-native output bundle with schema slots, policy logits, and value
        - updated recurrent state
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ModelSettingsV2,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Critic over PFC summary + q values
        self.reset_parameters()

    @property
    def config(self) -> ModelSettingsV2:
        """Return the parsed model settings used to build this module."""
        return self._config

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Reset all learnable parameters owned by the core."""
        self.pfc.reset_parameters()
        # self.str.reset_parameters()

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV2:
        """Allocate a fresh recurrent state for the given batch size."""
        return HRMStateV2(
            pfc=self.pfc.init_state(batch_size, workspace_layout=self.config.workspace_layout),
            str=self.str.init_state(batch_size),
        )

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV2,
    ) -> HRMStateV2:
        """Selectively reset recurrent state rows according to the given boolean mask."""
        return HRMStateV2(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
        )

    def step(  # --------------------------------------------------------------
        self,
        payload: HRMInputV2,
        state: Optional[HRMStateV2] = None,
    ) -> tuple[HRMOutputV2, HRMStateV2]:
        """Run one model step over task-agnostic schema tokens.

        Args:
            payload: Schema-slot tokens with shape ``(B, S, D)`` plus optional prefix bias.
            state: Optional recurrent state to carry across steps. If ``None``, a
                fresh state is created.

        Returns:
            ``(output, next_state)`` where ``output`` exposes controller-facing
            architecture-native readouts for the current step.
        """
        if state is None:
            state = self.init_state(payload.batch_size)
        else:
            state = state.detach()

        # Step the PFC core with the schema tokens bound to the workspace layout
        state.pfc, q_values = self.pfc.step(
            self.config.schema_layout.bind(payload.schema_tokens),
            state=state.pfc,
            prefix_bias=payload.prefix_bias,
        )

        # Step the STR actor-critic module with the PFC summary and Q values as input
        state.str, state_value = self.str(
            state.pfc.workspace.slot("controller"),
            q_values,
            state=state.str,
        )

        # Extract architecture-native readouts for the current step
        output = HRMOutputV2(
            theta_summary=state.pfc.workspace.slot("controller"),
            schema_slots=state.pfc.workspace.family("schema"),
            q_logits=q_values,
            state_value=state_value.unsqueeze(-1),
        )
        return output, state

    def forward(  # -----------------------------------------------------------
        self,
        payload: HRMInputV2,
        state: Optional[HRMStateV2] = None,
    ) -> tuple[HRMOutputV2, HRMStateV2]:
        """Compatibility wrapper over :meth:`step` for module-call users."""
        return self.step(payload, state=state)


# =============================================================================
__all__ = ["ModelSettingsV2", "HRMInputV2", "HRMOutputV2", "HRMStateV2", "HRModelV2"]

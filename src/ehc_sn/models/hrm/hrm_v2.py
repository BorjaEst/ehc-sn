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

from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState, WorkspaceSpec
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
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

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the core PFC model architecture.",
    )
    str: STRSettings = Field(
        ...,
        description="Settings for the STR actor-critic architecture.",
    )
    schema_slot_prefix: str = Field(
        default="schema",
        min_length=1,
        description="Prefix used for the internal schema-slot ontology.",
    )

    @property
    def num_schema_slots(self) -> int:
        """Return the number of schema slots owned by the HRM core."""
        return self.pfc.seq_length

    @property
    def schema_slot_names(self) -> tuple[str, ...]:
        """Return the canonical schema-slot names for the recurrent substrate."""
        return tuple(f"{self.schema_slot_prefix}_{index}" for index in range(self.num_schema_slots))

    @property
    def workspace_spec(self) -> WorkspaceSpec:
        """Return the schema-only workspace layout consumed by the core."""
        return WorkspaceSpec(names=self.schema_slot_names)


# =================================================================================================
@dataclass(frozen=True)
class HRMInputV2:
    """Task-agnostic schema payload consumed by the HRM v2 core."""

    schema_tokens: Tensor
    prefix_bias: Tensor | None = None


# =================================================================================================
@dataclass
class HRMStateV2(DetachMixin):
    """Recurrent state carried across steps for HRM v2.

    Attributes:
        pfc: PFC recurrent state.
        str: STR recurrent state.
    """

    pfc: PFCState  # Prefrontal Cortex state, containing working memory and reasoning module states.
    str: STRState  # STR actor-critic state, containing any recurrent state for the STR module (if needed).


# =================================================================================================
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


# =================================================================================================
class HRModelV2(nn.Module):
    """Core HRM v2 model.

    The model consists of:
        - PFC recurrent reasoning module over schema-slot tokens
        - STR actor-critic module consuming the CLS summary and PFC Q logits

    The forward pass returns:
        - updated recurrent state
        - architecture-native output bundle with schema slots, policy logits, and value
    """

    def __init__(  # ------------------------------------------------------------------------------
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

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:
        """Reset all learnable parameters owned by the core."""
        self.pfc.reset_parameters()
        self.str.reset_parameters()

    def init_state(  # ---------------------------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV2:
        """Allocate a fresh recurrent state for the given batch size."""
        return HRMStateV2(
            pfc=self.pfc.init_state(batch_size, self.config.workspace_spec),
            str=self.str.init_state(batch_size),
        )

    def reset_state(  # --------------------------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV2,
    ) -> HRMStateV2:
        """Selectively reset recurrent state rows according to the given boolean mask."""
        return HRMStateV2(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
        )

    def step(  # ----------------------------------------------------------------------------------
        self,
        payload: HRMInputV2,
        state: Optional[HRMStateV2] = None,
    ) -> tuple[HRMStateV2, HRMOutputV2]:
        """Run one model step over task-agnostic schema tokens.

        Args:
            payload: Schema-slot tokens with shape ``(B, S, D)`` plus optional prefix bias.
            state: Optional recurrent state to carry across steps. If ``None``, a
                fresh state is created.

        Returns:
            ``(next_state, output)`` where ``output`` exposes controller-facing
            architecture-native readouts for the current step.
        """

        if state is None:
            state = self.init_state(batch_size=int(payload.schema_tokens.shape[0]))
        else:
            state = state.clone()

        # TODO: fix to `next_state.pfc, ... = self.pfc(..., state=state)`
        pfc_output = self.pfc.step_tokens(
            payload.schema_tokens,
            state=state.pfc,
            workspace_spec=self.config.workspace_spec,
            prefix_bias=payload.prefix_bias,
        )
        state.pfc = pfc_output.state
        state.str, state_value = self.str(
            pfc_output.summary.detach(),
            pfc_output.q_values,
            state=state.str,
        )
        output = HRMOutputV2(
            theta_summary=pfc_output.summary,
            schema_slots=pfc_output.workspace.tokens,
            q_logits=pfc_output.q_values,
            state_value=state_value.unsqueeze(-1),
        )
        return state, output

    def forward(  # -------------------------------------------------------------------------------
        self,
        payload: HRMInputV2,
        state: Optional[HRMStateV2] = None,
    ) -> tuple[HRMStateV2, HRMOutputV2]:
        """Compatibility wrapper over :meth:`step` for module-call users."""
        return self.step(payload, state=state)


# =================================================================================================
__all__ = ["Batch", "HRMInputV2", "HRMOutputV2", "HRMStateV2", "HRModelV2", "ModelSettingsV2"]

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

from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class ModelSettings_V2(BaseModel, extra="forbid"):
    """Model-level settings for HRM v2.

    This settings object composes:
        - PFC settings (recurrent reasoning core)
        - STR settings (actor-critic / reward head)
        - schema-slot width and count inherited from PFC settings

    Notes:
        HRM v2 currently requires RoPE positional encodings inside the PFC modules
        for legacy parity and to match the environment tokenization.
    """

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the core PFC model architecture.",
    )
    str: STRSettings = Field(
        ...,
        description="Settings for the STR actor-critic architecture.",
    )

    @field_validator("pfc", mode="after")
    def validate_pfc(cls, v: PFCSettings) -> PFCSettings:
        """Ensure that the PFC settings have a valid reasoning module configuration."""
        if v.cortex.pos_encodings != "rope":
            raise ValueError("PFC reasoning modules must use RoPE positional encodings")
        return v

    @property
    def seq_length(self) -> int:
        """Convenience property to access sequence length from the PFC settings."""
        return self.pfc.seq_length

    @property
    def num_schema_slots(self) -> int:
        """Return the number of schema slots owned by the HRM core."""
        return self.pfc.seq_length

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the PFC settings."""
        return self.pfc.reasoning_h.cortex.embedding_dim

    @classmethod
    def from_config(cls, path: Path) -> "ModelSettings_V2":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)


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
        self, config: ModelSettings_V2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Critic over PFC summary + q values

    @property
    def config(self) -> ModelSettings_V2:
        """Return the parsed model settings used to build this module."""
        return self._config

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMStateV2:  # fmt: skip
        """Create a fresh recurrent state.

        Args:
            batch_size: Number of parallel environments / sequences.

        Returns:
            A new :class:`HRMStateV2` with initialized PFC and STR states.
        """
        return HRMStateV2(
            pfc=self.pfc.init_state(batch_size),
            str=self.str.init_state(batch_size),
        )

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMStateV2,
    ) -> HRMStateV2:  # fmt: skip
        """Selectively reset rows of the recurrent state.

        Args:
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating which
                batch rows should be reset.
            state: Current recurrent state.

        Returns:
            New state with flagged rows reset for both PFC and STR.
        """
        return HRMStateV2(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
        )

    def step(  # ----------------------------------------------------------------------------------
        self, payload: HRMInputV2, state: Optional[HRMStateV2] = None,
    ) -> tuple[HRMStateV2, HRMOutputV2]:  # fmt: skip
        """Run one model step over task-agnostic schema tokens.

        Args:
            payload: Schema-slot tokens with shape ``(B, S, D)`` plus optional prefix bias.
            state: Optional recurrent state to carry across steps. If ``None``, a
                fresh state is created.

        Returns:
            ``(next_state, output)`` where ``output`` exposes controller-facing
            architecture-native readouts for the current step.
        """
        state = state or self.init_state(batch_size=int(payload.schema_tokens.shape[0]))

        state_pfc, z_H, q_logits = self.pfc(payload.schema_tokens, state=state.pfc, prefix_bias=payload.prefix_bias)
        theta_summary = z_H[:, 0]
        schema_slots = z_H[:, 1:]
        state_str, state_value = self.str(theta_summary.detach(), q_logits, state.str)

        next_state = HRMStateV2(pfc=state_pfc, str=state_str)
        output = HRMOutputV2(
            theta_summary=theta_summary,
            schema_slots=schema_slots,
            q_logits=q_logits,
            state_value=state_value.unsqueeze(-1),
        )
        return next_state, output

    def forward(  # -------------------------------------------------------------------------------
        self, payload: HRMInputV2, state: Optional[HRMStateV2] = None,
    ) -> tuple[HRMStateV2, HRMOutputV2]:  # fmt: skip
        """Compatibility wrapper over :meth:`step` for module-call users."""
        return self.step(payload, state=state)


# =================================================================================================
__all__ = ["Batch", "HRMInputV2", "HRMOutputV2", "HRMStateV2", "HRModelV2", "ModelSettings_V2"]

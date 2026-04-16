"""HRM v1 core contracts over a named working-memory substrate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.pfc import PFCSettings, PFCState, WorkspaceSpec
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ModelSettingsV1(BaseModel, extra="forbid"):
    """HRM v1 core settings.

    No task vocabulary, token embedding, positional encoding, or LM head fields
    belong here. Those remain adapter-owned.
    """

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the recurrent PFC core.",
    )
    controller_slot_name: str = Field(
        default="controller",
        min_length=1,
        description="Public slot name used when exposing the explicit controller slot.",
    )
    schema_slot_prefix: str = Field(
        default="schema",
        min_length=1,
        description="Prefix used for the exchangeable schema slots.",
    )

    @property
    def num_schema_slots(self) -> int:
        """Return the number of schema slots owned by the HRM core."""
        return self.pfc.seq_length

    @property
    def schema_slot_names(self) -> tuple[str, ...]:
        """Return the canonical exchangeable schema slot names."""
        return tuple(f"{self.schema_slot_prefix}_{index}" for index in range(self.num_schema_slots))

    @property
    def workspace_spec(self) -> WorkspaceSpec:
        """Return the schema-only workspace layout consumed by the core."""
        return WorkspaceSpec(self.schema_slot_names)


# =============================================================================
@dataclass(frozen=True)
class HRMInputV1:
    """Task-agnostic schema payload consumed by the HRM core.

    Attributes:
        schema_tokens: Schema slot tokens with shape ``(B, N, D)``.
        prefix_bias: Optional controller-prefix bias with shape ``(B, D)``.
    """

    schema_tokens: Tensor
    prefix_bias: Tensor | None = None

    @property
    def batch_size(self) -> int:
        """Return the leading batch size."""
        return int(self.schema_tokens.shape[0])

    @property
    def num_schema_slots(self) -> int:
        """Return the declared number of schema slots."""
        return int(self.schema_tokens.shape[1])

    @property
    def hidden_size(self) -> int:
        """Return the shared schema-token width."""
        return int(self.schema_tokens.shape[2])


# =============================================================================
@dataclass
class HRMStateV1(DetachMixin):
    """Recurrent carry owned by the HRM core."""

    pfc: PFCState


# =============================================================================
@dataclass(frozen=True)
class HRMOutputV1:
    """Architecture-native HRM output.

    Attributes:
        theta_summary: Controller summary vector with shape ``(B, D)``.
        someother_logits: Placeholder for other core outputs with shape ``(B, ...)``.
        q_logits: Control logits consumed by ACT-style controllers.
    """

    theta_summary: Tensor  # working-memory z_H[:, 0] from dlPFC
    someother_logits: Tensor  # working-memory z_h[:, 1:] from dlPFC
    q_logits: Tensor  # Control logits from  vmPFC


# =============================================================================
class HRModelV1(nn.Module):
    """Signature-only HRM v1 shell over the canonical recurrent core contract."""

    def __init__(  # ----------------------------------------------------------
        self,
        config: ModelSettingsV1,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Store core settings for later construction by a concrete implementation."""
        super().__init__()
        self._config = config

    @property
    def config(self) -> ModelSettingsV1:
        """Return the immutable model settings used to parameterize the core."""
        return self._config

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Reset local parameters for concrete subclasses when they are added."""
        return None

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Create a fresh recurrent state for one batch."""
        raise NotImplementedError("HRModelV1.init_state is signature-only until the core is implemented.")

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Selectively reset rows of the recurrent state."""
        raise NotImplementedError("HRModelV1.reset_state is signature-only until the core is implemented.")

    def forward(  # -----------------------------------------------------------
        self,
        payload: HRMInputV1,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMStateV1, HRMOutputV1]:
        """Compatibility wrapper over :meth:`step` for module-call users."""
        return self.step(payload, state=state)


# =============================================================================
__all__ = ["ModelSettingsV1", "HRMInputV1", "HRMOutputV1", "HRMStateV1", "HRModelV1"]

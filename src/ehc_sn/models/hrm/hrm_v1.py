"""HRM v1 core contracts over a task-agnostic PFC working-memory substrate."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.pfc import FixedSlot, PFCModel, PFCSettings, PFCState, SlotFamily, WorkspaceLayout, WorkspaceSchema
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ModelSettingsV1(BaseModel, extra="forbid"):
    """HRM v1 core settings.

    No task vocabulary, token embedding, positional encoding, or LM head fields
    belong here. Those remain adapter-owned.
    """

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelSettingsV1":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the recurrent PFC core.",
    )

    @property
    def num_schema_slots(self) -> int:
        """Return the number of schema slots owned by the HRM core."""
        return self.pfc.seq_length

    @property
    def schema_layout(self) -> WorkspaceLayout:
        """Return the body-only schema layout (no controller) for use with :meth:`WorkspaceLayout.bind`."""
        return WorkspaceLayout.from_schema(WorkspaceSchema(fixed=(), families=(SlotFamily("schema", self.num_schema_slots),)))

    @property
    def workspace_layout(self) -> WorkspaceLayout:
        """Return the full workspace layout (controller at position 0 + schema family) for :meth:`~ehc_sn.modules.pfc.PFCModel.init_state`."""
        return WorkspaceLayout.from_schema(
            WorkspaceSchema(
                fixed=(FixedSlot("controller"),),
                families=(SlotFamily("schema", self.num_schema_slots),),
            )
        )


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
        schema_slots: Schema-slot bank with shape ``(B, N, D)``.
        q_logits: Control logits consumed by ACT-style controllers.
    """

    theta_summary: Tensor
    schema_slots: Tensor
    q_logits: Tensor


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
        """Construct the HRM v1 core around the reusable PFC reasoning module."""
        super().__init__()
        self._config = config
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)
        self.reset_parameters()

    @property
    def config(self) -> ModelSettingsV1:
        """Return the immutable model settings used to parameterize the core."""
        return self._config

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Reset the reusable PFC core surfaces owned by HRM v1."""
        self.pfc.reset_parameters()

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Create a fresh recurrent state for one batch."""
        return HRMStateV1(
            pfc=self.pfc.init_state(batch_size, workspace_layout=self.config.workspace_layout),
        )

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Selectively reset rows of the recurrent state."""
        return HRMStateV1(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
        )

    def step(  # --------------------------------------------------------------
        self,
        payload: HRMInputV1,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMOutputV1, HRMStateV1]:
        """Run one HRM core step over schema-slot tokens and return state plus readouts.

        Args:
            payload: Task-agnostic schema-slot payload with shape ``(B, N, D)``.
            state: Optional recurrent carry from the previous step.

        Returns:
            ``(next_state, output)`` where ``next_state`` owns the recurrent slot
            substrate and ``output`` exposes the architecture-native readouts for
            the current step.
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

        # Extract architecture-native readouts for the current step
        output = HRMOutputV1(
            theta_summary=state.pfc.workspace.slot("controller"),
            schema_slots=state.pfc.workspace.family("schema"),
            q_logits=q_values,
        )

        return output, state

    def forward(  # -----------------------------------------------------------
        self,
        payload: HRMInputV1,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMOutputV1, HRMStateV1]:
        """Compatibility wrapper over :meth:`step` for module-call users."""
        return self.step(payload, state=state)


# =============================================================================
__all__ = ["ModelSettingsV1", "HRMInputV1", "HRMOutputV1", "HRMStateV1", "HRModelV1"]

"""Routebind HRM v1 bridge adapter.

Encodes routebind spatial-grid task inputs (cell_type, observation_id,
start/goal flags) into schema tokens via element-wise embedding composition,
runs HRM v1 deliberation, and decodes schema slots into a trajectory field,
waypoint field, next-direction logits, and next-observation logits.

The adapter follows the GoaltraceHRMV1BridgeAdapter public surface
(``init_state``, ``reset_state``, ``prepare_inputs``, ``postprocess``,
``forward``) and includes a multi-head decoder for the routebind output
contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ehc_sn.adapters.hrm._base import (
    HasSchemaSlots,
    RoutebindDecoder,
    RoutebindHRMAdapterSettings,
    RoutebindRoPEEncoder,
)
from ehc_sn.models.hrm.hrm_v1 import (
    HRMInputV1,
    HRModelV1,
    HRMStateV1,
)
from ehc_sn.tasks.routebind.contracts import (
    CELL_OBSERVATION,
    RoutebindTaskOutput,
)
from ehc_sn.tasks.routebind.runtime import extract_routebind_task_input
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class RoutebindHRMV1ControlOutput:
    """ACT control readouts emitted by the Routebind HRM v1 bridge."""

    action_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class RoutebindHRMV1BridgeOutput:
    """Controller-consumable Routebind HRM v1 bridge output bundle."""

    task: RoutebindTaskOutput
    control: RoutebindHRMV1ControlOutput


# =============================================================================
def _make_input_v1(
    schema_tokens: Tensor, prefix_bias: Tensor | None
) -> HRMInputV1:
    """Construct the model-native HRM v1 input payload."""
    return HRMInputV1(schema_tokens=schema_tokens, prefix_bias=prefix_bias)


# =============================================================================
def _build_encoder(
    model: HRModelV1,
    config: RoutebindHRMAdapterSettings,
) -> RoutebindRoPEEncoder:
    """Construct the token encoder front-end for the bridge adapter."""
    params = next(model.parameters())
    return RoutebindRoPEEncoder(
        seq_length=model.config.num_schema_slots,
        num_cell_types=config.num_cell_types,
        num_observations=config.num_observations,
        hidden_size=model.config.pfc.hidden_size,
        grid_height=config.grid_height,
        grid_width=config.grid_width,
        input_factory=_make_input_v1,
        padding_idx=config.num_observations,
        device=params.device,
        dtype=params.dtype,
    )


# =============================================================================
def _build_decoder(
    model: HRModelV1,
    config: RoutebindHRMAdapterSettings,
) -> RoutebindDecoder:
    """Construct the multi-head decoder for the bridge adapter."""
    params = next(model.parameters())
    num_slots = config.grid_height * config.grid_width
    return RoutebindDecoder(
        hidden_size=model.config.pfc.hidden_size,
        num_slots=num_slots,
        num_observations=config.num_observations,
        device=params.device,
        dtype=params.dtype,
    )


# =============================================================================
class RoutebindHRMV1BridgeAdapter(nn.Module):
    """Routebind field-prediction bridge over HRM v1 (ACT-supervised).

    Wraps HRModelV1, encodes routebind grid inputs (cell type, observation
    IDs, start/goal flags, 2-D positional coordinates) into schema tokens
    via element-wise embedding composition, runs HRM deliberation, and
    decodes schema slots into continuous trajectory/waypoint fields plus
    auxiliary next-direction and next-observation logits.

    The adapter follows the goaltrace pattern:
    - ``postprocess`` reads the pre-HRM input embeddings so per-slot identity
      survives the post-norm transformer.
    - The start-slot heads (direction, observation) use the mean-pooled
      representation across all slots so HRM self-attention can integrate
      the start position information.
    """

    def __init__(
        self,
        model: HRModelV1,
        config: RoutebindHRMAdapterSettings,
    ) -> None:
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder(model, config)
        self._decoder = _build_decoder(model, config)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize encoder and decoder with small normal weights."""
        self._encoder.reset_parameters()
        self._decoder.reset_parameters()

    @property
    def config(self) -> RoutebindHRMAdapterSettings:
        """Return the adapter settings used to build this module."""
        return self._config

    def init_state(self, batch_size: int) -> HRMStateV1:
        """Delegate to the wrapped HRM model."""
        return self.model.init_state(batch_size)

    def reset_state(self, reset_flag: Tensor, state: HRMStateV1) -> HRMStateV1:
        """Delegate to the wrapped HRM model."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(self, batch: Batch) -> HRMInputV1:
        """Extract routebind task input from a raw batch and encode into
        schema tokens.

        Maps observation ID sentinel (-1) to the padding index
        (``config.num_observations``) so non-observation cells embed to zero.

        Returns:
            HRMInputV1 with schema tokens shaped ``(B, S_model, D)``.
        """
        task_input = extract_routebind_task_input(batch)

        # Map -1 sentinel to padding index for the embedding table
        obs_id = task_input.observation_id.clone()
        obs_id[obs_id < 0] = self._config.num_observations

        # Clamp cell_type to valid range (CELL_PAD=3 maps to index 3)
        cell_type = task_input.cell_type.clamp(
            min=0, max=self._config.num_cell_types - 1
        )

        return self._encoder(
            cell_type=cell_type,
            observation_id=obs_id,
            start_flag=task_input.start_flag,
            goal_flag=task_input.goal_flag,
        )

    def postprocess(
        self,
        outputs: HRMOutputV1,
    ) -> RoutebindHRMV1BridgeOutput:
        """Decode schema slots into routebind outputs.

        Args:
            outputs: Architecture-native HRM v1 output (or substituted
                output with pre-HRM schema slots when called from ``forward``).

        Returns:
            Bridge output bundle with trajectory/waypoint fields over all
            slots and auxiliary heads decoded from the start-slot latent.
        """
        traj_field, wp_field, dir_logits, obs_logits = self._decoder(outputs)
        return RoutebindHRMV1BridgeOutput(
            task=RoutebindTaskOutput(
                trajectory_field=traj_field,
                waypoint_field=wp_field,
                next_direction_logits=dir_logits,
                next_observation_logits=obs_logits,
            ),
            control=RoutebindHRMV1ControlOutput(
                action_logits=outputs.action_logits
            ),
        )

    def forward(
        self,
        batch: Batch,
        state: HRMStateV1 | None = None,
    ) -> tuple[RoutebindHRMV1BridgeOutput, HRMStateV1]:
        """Run a forward pass on one batch of routebind samples.

        The decoder consumes ``HRMOutputV1.schema_readout`` — a residual-
        combined representation of the reasoned PFC workspace.  This ensures
        all Routebind predictions are a differentiable function of the HRM
        recurrent state and the field loss produces nonzero gradients in
        HRM parameters.
        """
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model.step(inputs, state=state)
        bridge_out = self.postprocess(outputs)
        return bridge_out, next_state


# =============================================================================
__all__ = [
    "RoutebindHRMV1BridgeAdapter",
    "RoutebindHRMV1BridgeOutput",
    "RoutebindHRMV1ControlOutput",
]

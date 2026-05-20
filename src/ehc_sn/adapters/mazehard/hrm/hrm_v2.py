"""MazeHard plus HRM v2 bridge implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor, nn

from ehc_sn.adapters.mazehard.hrm.core import (
    MazeHardHRMAdapterSettings,
    MazeHardLearnedEncoder,
    MazeHardMLPDecoder,
    MazeHardRoPEEncoder,
    build_token_decoder,
    build_token_encoder,
)
from ehc_sn.models.hrm.hrm_v2 import (
    HRMInputV2,
    HRModelV2,
    HRMOutputV2,
    HRMStateV2,
)
from ehc_sn.tasks.mazehard.runtime import extract_maze_hard_task_input
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV2PolicyOutput:
    """Value-control readouts emitted by the MazeHard HRM v2 bridge."""

    q_values: Tensor
    valid_action_mask: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV2CriticOutput:
    """Critic readouts emitted by the MazeHard HRM v2 bridge."""

    state_value: Tensor


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV2BridgeOutput:
    """Controller-consumable MazeHard HRM v2 bridge output bundle."""

    task: MazeHardTaskOutput
    policy: MazeHardHRMV2PolicyOutput
    critic: MazeHardHRMV2CriticOutput


MazeHardTokenEncoder: TypeAlias = (
    MazeHardLearnedEncoder[HRMInputV2] | MazeHardRoPEEncoder[HRMInputV2]
)


def _make_input_v2(
    schema_tokens: Tensor, prefix_bias: Tensor | None
) -> HRMInputV2:
    """Construct the model-native HRM v2 input payload."""
    return HRMInputV2(schema_tokens=schema_tokens, prefix_bias=prefix_bias)


def _build_encoder(  # ----------------------------------------------------
    model: HRModelV2,
    config: MazeHardHRMAdapterSettings,
) -> MazeHardTokenEncoder:
    """Construct the token encoder front-end for the bridge adapter based on config."""
    params = next(model.parameters())
    return build_token_encoder(
        seq_length=model.config.num_schema_slots,
        vocab_size=config.vocab_size,
        hidden_size=model.config.pfc.hidden_size,
        encoder_kind=config.encoder_kind,
        input_factory=_make_input_v2,
        device=params.device,
        dtype=params.dtype,
    )


MazeHardTokenDecoder: TypeAlias = MazeHardMLPDecoder


def _build_decoder(  # --------------------------------------------------------
    model: HRModelV2,
    config: MazeHardHRMAdapterSettings,
) -> MazeHardTokenDecoder:
    """Construct the token decoder head for the bridge adapter."""
    params = next(model.parameters())
    return build_token_decoder(
        hidden_size=model.config.pfc.hidden_size,
        vocab_size=config.vocab_size,
        device=params.device,
        dtype=params.dtype,
    )


# =============================================================================
class MazeHardHRMV2BridgeAdapter(nn.Module):
    """MazeHard plus HRM v2 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV2,
        config: MazeHardHRMAdapterSettings | None = None,
    ) -> None:
        """Initialize the HRM v2 bridge adapter with its component modules."""
        super().__init__()
        self._config = config or MazeHardHRMAdapterSettings()
        self.model = model
        self._encoder = _build_encoder(model, self._config)
        self._decoder = _build_decoder(model, self._config)

    @property
    def config(self) -> MazeHardHRMAdapterSettings:
        """Return the immutable adapter settings used to configure the bridge."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV2:
        """Create a fresh HRM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size)

    def reset_state(  # --------------------------------------------------------
        self,
        reset_flag: torch.Tensor,
        state: HRMStateV2,
    ) -> HRMStateV2:
        """Reset halted rows of the HRM recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> HRMInputV2:
        """Prepare the HRM-core payload from one generic rollout batch."""
        task_input = extract_maze_hard_task_input(batch)
        return self._encoder(task_input)

    def postprocess(  # -------------------------------------------------------
        self,
        outputs: HRMOutputV2,
    ) -> MazeHardHRMV2BridgeOutput:
        """Split one HRM step output into task, policy, and critic surfaces."""
        return MazeHardHRMV2BridgeOutput(
            task=self._decoder(outputs),
            policy=MazeHardHRMV2PolicyOutput(q_values=outputs.q_values),
            critic=MazeHardHRMV2CriticOutput(state_value=outputs.state_value),
        )

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: HRMStateV2 | None = None,
    ) -> tuple[MazeHardHRMV2BridgeOutput, HRMStateV2]:
        """Run a forward pass of the HRM v2 bridge adapter on a MazeHard task batch."""
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model(inputs, state=state)
        outputs = self.postprocess(outputs)
        return outputs, next_state


# =============================================================================
__all__ = [
    "MazeHardHRMV2BridgeOutput",
    "MazeHardHRMV2CriticOutput",
    "MazeHardHRMAdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardHRMV2PolicyOutput",
    "MazeHardRoPEEncoder",
    "MazeHardTokenEncoder",
    "MazeHardMLPDecoder",
    "MazeHardTokenDecoder",
    "MazeHardHRMV2BridgeAdapter",
]

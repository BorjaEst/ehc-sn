"""MazeHard plus HRM v1 bridge implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torch import Tensor, nn

from ehc_sn.adapters.mazehard.hrm.core import (
    MazeHardHRMAdapterSettings,
    MazeHardLearnedEncoder,
    MazeHardMLPDecoder,
    MazeHardRoPEEncoder,
    build_token_decoder,
    build_token_encoder,
)
from ehc_sn.models.hrm.hrm_v1 import HRMInputV1, HRModelV1, HRMOutputV1, HRMStateV1
from ehc_sn.tasks.mazehard.batch import extract_maze_hard_task_input
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV1ControlOutput:
    """ACT-compatible control readouts emitted by the MazeHard HRM v1 bridge."""

    q_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class MazeHardHRMV1BridgeOutput:
    """Controller-consumable MazeHard HRM v1 bridge output bundle."""

    task: MazeHardTaskOutput
    control: MazeHardHRMV1ControlOutput


MazeHardTokenEncoder: TypeAlias = MazeHardLearnedEncoder[HRMInputV1] | MazeHardRoPEEncoder[HRMInputV1]


def _make_input_v1(schema_tokens: Tensor, prefix_bias: Tensor | None) -> HRMInputV1:
    """Construct the model-native HRM v1 input payload."""
    return HRMInputV1(schema_tokens=schema_tokens, prefix_bias=prefix_bias)


def _build_encoder(  # ----------------------------------------------------
    model: HRModelV1,
    config: MazeHardHRMAdapterSettings,
) -> MazeHardTokenEncoder:
    """Construct the token encoder front-end for the bridge adapter based on config."""
    params = next(model.parameters())
    return build_token_encoder(
        seq_length=model.config.num_schema_slots,
        vocab_size=config.vocab_size,
        hidden_size=model.config.pfc.hidden_size,
        encoder_kind=config.encoder_kind,
        input_factory=_make_input_v1,
        device=params.device,
        dtype=params.dtype,
    )


MazeHardTokenDecoder: TypeAlias = MazeHardMLPDecoder


def _build_decoder(  # --------------------------------------------------------
    model: HRModelV1,
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
class MazeHardHRMV1BridgeAdapter(nn.Module):
    """MazeHard plus HRM v1 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV1,
        config: MazeHardHRMAdapterSettings,
    ) -> None:
        """Initialize the HRM v1 bridge adapter with its component modules."""
        super().__init__()
        self._config = config
        self.model = model
        self._encoder = _build_encoder(model, config)
        self._decoder = _build_decoder(model, config)

    @property
    def config(self) -> MazeHardHRMAdapterSettings:
        """Return the immutable adapter settings used to configure the bridge."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Create a fresh HRM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size)

    def reset_state(  # --------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Reset halted rows of the HRM recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> HRMInputV1:
        """Prepare the HRM-core payload from one generic rollout batch."""
        task_input = extract_maze_hard_task_input(batch)
        return self._encoder(task_input)

    def postprocess(  # -------------------------------------------------------
        self,
        outputs: HRMOutputV1,
    ) -> MazeHardHRMV1BridgeOutput:
        """Split one HRM step output into controller-consumable task and control heads."""
        return MazeHardHRMV1BridgeOutput(
            task=self._decoder(outputs),
            control=MazeHardHRMV1ControlOutput(q_logits=outputs.q_logits),
        )

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: HRMStateV1 | None = None,
    ) -> tuple[MazeHardHRMV1BridgeOutput, HRMStateV1]:
        """Run a forward pass of the HRM v1 bridge adapter on a MazeHard task batch."""
        inputs = self.prepare_inputs(batch)
        outputs, next_state = self.model(inputs, state=state)
        outputs = self.postprocess(outputs)
        return outputs, next_state


# =============================================================================
__all__ = [
    "MazeHardHRMV1BridgeOutput",
    "MazeHardHRMV1ControlOutput",
    "MazeHardHRMAdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardTokenEncoder",
    "MazeHardMLPDecoder",
    "MazeHardTokenDecoder",
    "MazeHardHRMV1BridgeAdapter",
]

"""MazeHard plus HRM v1 bridge and shared composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.mazehard.decoders import MazeHardDecoder
from ehc_sn.adapters.mazehard.encoders import MazeHardEncoder
from ehc_sn.models.hrm.hrm_v1 import HRMInputV1, HRModelV1, HRMOutputV1, HRMStateV1
from ehc_sn.tasks.maze_hard.contracts import MazeHardTaskInput, MazeHardTaskOutput
from ehc_sn.tasks.maze_hard.runtime import extract_maze_hard_task_input
from ehc_sn.types import Batch


# =============================================================================
class MazeHardHRMV1AdapterSettings(BaseModel, extra="forbid"):
    """Task-side MazeHard settings required to bind the HRM v1 core."""

    encoder_kind: Literal["learned", "rope"] = Field(
        default="rope",
        description="Positional front-end used by the MazeHard token encoder.",
    )
    vocab_size: int = Field(
        ...,
        ge=1,
        description="MazeHard token vocabulary size used by encoder and decoder heads.",
    )


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


# =============================================================================
class MazeHardLearnedEncoder(nn.Module, MazeHardEncoder):
    """Encoder for MazeHard token inputs using learned positional embeddings."""

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Construct the learned encoder with token and positional embedding tables."""
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(seq_length, hidden_size, device=device, dtype=dtype)
        self.embedding_scale = 0.707106781 * (hidden_size**0.5)  # Scale factor to maintain variance

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskInput,
    ) -> HRMInputV1:
        """Encode MazeHard tokens with learned positional embeddings."""
        token_embeddings = self.embed_tokens(batch.input_ids.to(dtype=torch.int32))

        # Learned mode: positional embeddings are added to token embeddings
        positions = torch.arange(self.embed_pos.num_embeddings, device=batch.input_ids.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)

        # Scale the combined embeddings to maintain variance, then return the HRM input bundle.
        return HRMInputV1(
            schema_tokens=self.embedding_scale * (token_embeddings + pos_embeddings),
            prefix_bias=None,
        )


# =============================================================================
class MazeHardRoPEEncoder(nn.Module, MazeHardEncoder):
    """Encoder for MazeHard token inputs using a RoPE-compatible front-end."""

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,  # Included for config consistency
        vocab_size: int,
        hidden_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Construct the RoPE encoder with token embedding table only."""
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.embed_pos = None  # Not used in RoPE mode
        self.embedding_scale = hidden_size**0.5  # Scale factor for RoPE mode

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskInput,
    ) -> HRMInputV1:
        """Encode MazeHard tokens without a learned positional table."""
        token_embeddings = self.embed_tokens(batch.input_ids.to(dtype=torch.int32))

        # RoPE mode: positions are encoded in QK rotation - scale by sqrt only.
        return HRMInputV1(
            schema_tokens=self.embedding_scale * token_embeddings,
            prefix_bias=None,
        )


# =============================================================================
MazeHardTokenEncoder: TypeAlias = MazeHardLearnedEncoder | MazeHardRoPEEncoder


def _build_encoder(  # ----------------------------------------------------
    model: HRModelV1,
    config: MazeHardHRMV1AdapterSettings,
) -> MazeHardTokenEncoder:
    """Construct the token encoder front-end for the bridge adapter based on config."""
    match config.encoder_kind:
        case "learned":
            encoder_cls = MazeHardLearnedEncoder
        case "rope":
            encoder_cls = MazeHardRoPEEncoder
        case _:
            raise ValueError(f"Unsupported encoder kind: {config.encoder_kind}")

    # Build the encoder with the appropriate config parameters and device/dtype
    return encoder_cls(
        seq_length=model.config.num_schema_slots,
        vocab_size=config.vocab_size,
        hidden_size=model.config.pfc.hidden_size,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )


# =============================================================================
class MazeHardMLPDecoder(nn.Module, MazeHardDecoder):
    """Decoder mapping HRM schema-slot features to MazeHard task logits."""

    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Construct the token decoder with a linear head mapping to MazeHard vocab size."""
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(  # -----------------------------------------------------------
        self,
        outputs: HRMOutputV1,
    ) -> MazeHardTaskOutput:
        """Decode schema-slot activations into MazeHard token logits."""
        return MazeHardTaskOutput(
            task_logits=self.lm_head(outputs.schema_slots),
        )


# =============================================================================
MazeHardTokenDecoder: TypeAlias = MazeHardMLPDecoder


def _build_decoder(  # --------------------------------------------------------
    model: HRModelV1,
    config: MazeHardHRMV1AdapterSettings,
) -> MazeHardTokenDecoder:
    """Construct the token decoder head for the bridge adapter."""
    return MazeHardTokenDecoder(
        hidden_size=model.config.pfc.hidden_size,
        vocab_size=config.vocab_size,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )


# =============================================================================
class MazeHardHRMV1BridgeAdapter(nn.Module):
    """MazeHard plus HRM v1 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV1,
        config: MazeHardHRMV1AdapterSettings,
    ) -> None:
        """Initialize the HRM v1 bridge adapter with its component modules."""
        super().__init__()
        self._config = config
        self.model = model
        self.encoder = _build_encoder(model, config)
        self.decoder = _build_decoder(model, config)

    @property
    def config(self) -> MazeHardHRMV1AdapterSettings:
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
        return self.encoder(task_input)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        outputs: HRMOutputV1,
    ) -> MazeHardHRMV1BridgeOutput:
        """Split one HRM step output into controller-consumable task and control heads."""
        return MazeHardHRMV1BridgeOutput(
            task=self.decoder(outputs),
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
        outputs = self.prepare_outputs(outputs)
        return outputs, next_state


# =============================================================================
__all__ = [
    "MazeHardHRMV1AdapterSettings",
    "MazeHardHRMV1BridgeOutput",
    "MazeHardHRMV1ControlOutput",
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardTokenEncoder",
    "MazeHardMLPDecoder",
    "MazeHardTokenDecoder",
    "MazeHardHRMV1BridgeAdapter",
]

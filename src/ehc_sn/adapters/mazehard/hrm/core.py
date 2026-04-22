"""Shared MazeHard+HRM bridge family core.

Holds the task-side settings and token encoder/decoder glue shared by the
MazeHard+HRM v1 and v2 bridge adapters. Versioned bridge modules keep the
model-native input and controller-output types local.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Literal, Protocol, TypeVar

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.mazehard.decoders import MazeHardDecoder
from ehc_sn.adapters.mazehard.encoders import MazeHardEncoder
from ehc_sn.data.schema import O_ID
from ehc_sn.data.vocabulary import VOCAB_SIZE as MAZE_SEM_VOCAB_SIZE
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskInput, MazeHardTaskOutput

TInput = TypeVar("TInput")

DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE: int = max(MAZE_SEM_VOCAB_SIZE, O_ID + 1)


# =============================================================================
class MazeHardHRMAdapterSettings(BaseModel, extra="forbid"):
    """Task-side MazeHard settings shared by the HRM bridge family."""

    encoder_kind: Literal["learned", "rope"] = Field(
        default="rope",
        description="Positional front-end used by the MazeHard token encoder.",
    )
    vocab_size: int = Field(
        default=DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE,
        ge=1,
        description=(
            "MazeHard token vocabulary size used by encoder and decoder heads. "
            "Defaults to the canonical SEM vocabulary plus the solution-overlay token."
        ),
    )


# =============================================================================
class MazeHardLearnedEncoder(nn.Module, MazeHardEncoder, Generic[TInput]):
    """Encoder for MazeHard token inputs using learned positional embeddings."""

    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(seq_length, hidden_size, device=device, dtype=dtype)
        self.embedding_scale = 0.707106781 * (hidden_size**0.5)
        self._input_factory = input_factory

    def forward(
        self,
        batch: MazeHardTaskInput,
    ) -> TInput:
        """Encode MazeHard tokens with learned positional embeddings."""
        token_embeddings = self.embed_tokens(batch.input_ids.to(dtype=torch.int32))
        positions = torch.arange(self.embed_pos.num_embeddings, device=batch.input_ids.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)
        return self._input_factory(
            self.embedding_scale * (token_embeddings + pos_embeddings),
            None,
        )


# =============================================================================
class MazeHardRoPEEncoder(nn.Module, MazeHardEncoder, Generic[TInput]):
    """Encoder for MazeHard token inputs using a RoPE-compatible front-end."""

    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        input_factory: Callable[[Tensor, Tensor | None], TInput],
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        _ = seq_length
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.embedding_scale = hidden_size**0.5
        self._input_factory = input_factory

    def forward(
        self,
        batch: MazeHardTaskInput,
    ) -> TInput:
        """Encode MazeHard tokens without a learned positional table."""
        token_embeddings = self.embed_tokens(batch.input_ids.to(dtype=torch.int32))
        return self._input_factory(
            self.embedding_scale * token_embeddings,
            None,
        )


# =============================================================================
def build_token_encoder(
    *,
    seq_length: int,
    vocab_size: int,
    hidden_size: int,
    encoder_kind: Literal["learned", "rope"],
    input_factory: Callable[[Tensor, Tensor | None], TInput],
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> MazeHardLearnedEncoder[TInput] | MazeHardRoPEEncoder[TInput]:
    """Construct the MazeHard token encoder front-end for one HRM bridge."""
    match encoder_kind:
        case "learned":
            encoder_cls = MazeHardLearnedEncoder[TInput]
        case "rope":
            encoder_cls = MazeHardRoPEEncoder[TInput]
        case _:
            raise ValueError(f"Unsupported encoder kind: {encoder_kind}")

    return encoder_cls(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        input_factory=input_factory,
        device=device,
        dtype=dtype,
    )


# =============================================================================
class HasSchemaSlots(Protocol):
    """Minimal model-output surface required by the shared MazeHard decoder."""

    schema_slots: Tensor


# =============================================================================
class MazeHardMLPDecoder(nn.Module, MazeHardDecoder):
    """Decoder mapping HRM schema-slot features to MazeHard task logits."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        outputs: HasSchemaSlots,
    ) -> MazeHardTaskOutput:
        """Decode schema-slot activations into MazeHard token logits."""
        return MazeHardTaskOutput(task_logits=self.lm_head(outputs.schema_slots))


# =============================================================================
def build_token_decoder(
    *,
    hidden_size: int,
    vocab_size: int,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> MazeHardMLPDecoder:
    """Construct the MazeHard token decoder head for one HRM bridge."""
    return MazeHardMLPDecoder(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
    )


# =============================================================================
__all__ = [
    "DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE",
    "MazeHardHRMAdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardMLPDecoder",
    "HasSchemaSlots",
    "build_token_encoder",
    "build_token_decoder",
]

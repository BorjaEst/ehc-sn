"""Shared MazeHard+HRM bridge family core.

Holds the task-side settings, raw channel coercion, and token encoder/decoder
glue shared by the MazeHard+HRM v1 and v2 bridge adapters.  Versioned bridge
modules keep the model-native input and controller-output types local.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.mazehard.decoders import MazeHardDecoder
from ehc_sn.adapters.mazehard.encoders import MazeHardEncoder
from ehc_sn.adapters.mazehard.transforms import channels_to_grid
from ehc_sn.adapters.mazehard.vocabulary import VOCAB_SIZE as MAZE_SEM_VOCAB_SIZE
from ehc_sn.data.schema import validate_processed
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskInput, MazeHardTaskOutput
from ehc_sn.types import Batch

TInput = TypeVar("TInput")

O_ID: int = 5
"""Overlay token ID — solution-path cell token in the MazeHard+HRM vocabulary."""

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


_CHANNEL_SOLUTION: str = "solution"


def coerce_maze_hard_batch(raw: Mapping[str, Any]) -> Batch:
    """Convert raw MazeHard channels into canonical token and label tensors.

    Supports both single-maze arrays ``(H, W)`` and aligned stacked arrays
    ``(B, H, W)``.  Spatial dimensions are flattened while any leading batch
    dimensions are preserved.
    """
    channels = _coerce_numpy_channels(raw)
    _validate_channel_stack_shapes(channels)

    grid = channels_to_grid(channels)["grid"]
    input_ids = _flatten_spatial_to_tensor(grid, dtype=np.int64, name="grid")
    labels = input_ids.clone()

    if _CHANNEL_SOLUTION in channels:
        solution_mask = _flatten_spatial_to_tensor(
            channels[_CHANNEL_SOLUTION] > 0,
            dtype=np.bool_,
            name=_CHANNEL_SOLUTION,
        ).to(dtype=torch.bool)
        labels = torch.where(solution_mask, torch.full_like(labels, O_ID), labels)

    return {"input_ids": input_ids, "labels": labels}


def _coerce_numpy_channels(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            channels[key] = value
            continue
        if isinstance(value, Tensor):
            channels[key] = value.detach().cpu().numpy()
            continue
        raise TypeError(f"Unsupported MazeHard channel type for key {key!r}: {type(value).__name__}.")
    validate_processed(channels)
    return channels


def _validate_channel_stack_shapes(channels: dict[str, np.ndarray]) -> None:
    reference_name, reference = next(iter(channels.items()))
    mismatched = {name: value.shape for name, value in channels.items() if value.shape != reference.shape}
    if mismatched:
        detail = ", ".join(f"{name}={shape}" for name, shape in mismatched.items())
        raise ValueError(
            "MazeHard batch requires aligned raw channel shapes; "
            f"expected all channels to match {reference_name}={reference.shape}, got {detail}."
        )


def _flatten_spatial_to_tensor(array: np.ndarray, *, dtype: Any, name: str) -> Tensor:
    if array.ndim not in (2, 3):
        raise ValueError(f"MazeHard field {name!r} must have shape (H, W) or (B, H, W), got {array.shape}.")
    return torch.from_numpy(array.reshape(*array.shape[:-2], -1).astype(dtype, copy=False))


# =============================================================================
__all__ = [
    "O_ID",
    "DEFAULT_MAZE_HARD_HRM_VOCAB_SIZE",
    "coerce_maze_hard_batch",
    "MazeHardHRMAdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardMLPDecoder",
    "HasSchemaSlots",
    "build_token_encoder",
    "build_token_decoder",
]

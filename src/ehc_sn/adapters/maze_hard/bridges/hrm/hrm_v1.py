"""MazeHard plus HRM v1 bridge and shared composition helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.maze_hard.decoders import MazeHardDecoder
from ehc_sn.adapters.maze_hard.encoders import MazeHardEncoder
from ehc_sn.models.hrm.hrm_v1 import HRMInputV1, HRModelV1, HRMOutputV1, HRMStateV1
from ehc_sn.tasks.maze_hard.contracts import MazeHardTaskInput, MazeHardTaskOutput


# =============================================================================
class MazeHardHRMV1AdapterSettings(BaseModel, extra="forbid"):
    """Task-side MazeHard settings required to bind the HRM v1 core."""

    encoder_kind: Literal["learned", "rope"] = Field(
        default="learned",
        description="Positional front-end used by the MazeHard token encoder.",
    )

    vocab_size: int = Field(
        ...,
        ge=1,
        description="MazeHard token vocabulary size used by encoder and decoder heads.",
    )


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
    match config.config.encoder_kind:
        case "learned":
            encoder_cls = MazeHardLearnedEncoder
        case "rope":
            encoder_cls = MazeHardRoPEEncoder
        case _:
            raise ValueError(f"Unsupported encoder kind: {config.encoder_kind}")

    # Build the encoder with the appropriate config parameters and device/dtype
    return encoder_cls(
        seq_length=model.config.seq_length,
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

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: MazeHardTaskInput,
    ) -> HRMInputV1:
        """Prepare the HRM-core payload from the generic task batch."""
        return self.encoder(batch)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        logits: HRMOutputV1,
    ) -> MazeHardTaskOutput:
        """Split one HRM step output into task bridge heads."""
        return self.decoder(logits)

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskInput,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMStateV1, MazeHardTaskOutput]:
        """Run a forward pass of the HRM v1 bridge adapter on a MazeHard task batch."""
        inputs = self.prepare_inputs(batch)
        next_state, logits = self.model(inputs, state=state)
        outputs = self.prepare_outputs(logits)
        return next_state, outputs


# =============================================================================
__all__ = [
    "MazeHardHRMV1AdapterSettings",
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardTokenEncoder",
    "MazeHardMLPDecoder",
    "MazeHardTokenDecoder",
    "MazeHardHRMV1BridgeAdapter",
]

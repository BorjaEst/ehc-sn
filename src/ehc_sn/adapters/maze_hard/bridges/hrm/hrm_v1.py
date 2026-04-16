"""MazeHard plus HRM v1 bridge structural wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.adapters.maze_hard.decoders import MazeHardDecoder
from ehc_sn.adapters.maze_hard.encoders import MazeHardEncoder
from ehc_sn.models.hrm.hrm_v1_new import HRMInputV1, HRModelV1, HRMOutputV1, HRMStateV1
from ehc_sn.tasks.maze_hard import MazeHardTaskBatch, MazeHardTaskOutput


# =============================================================================
class MazeHardLearnedEncoder(nn.Module, MazeHardEncoder):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """ """
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(seq_length, hidden_size, device=device, dtype=dtype)
        self.embedding_scale = 0.707106781 * (hidden_size**0.5)  # Scale factor to maintain variance

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskBatch,
        *,
        device: Device | None = None,
    ):  # TODO: specify return type
        """ """
        token_embeddings = self.embed_tokens(input_ids.to(torch.int32))

        # Learned mode: add positional table, then scale to maintain variance.
        positions = torch.arange(self.config.seq_length, device=input_ids.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)
        token_embeddings = self.embedding_scale * (token_embeddings + pos_embeddings)

        return inputs


# =============================================================================
class MazeHardRoPEEncoder(nn.Module, MazeHardEncoder):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,  # Included for config consistency
        vocab_size: int,
        hidden_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """ """
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.embed_pos = None  # Not used in RoPE mode
        self.embedding_scale = 1 * (hidden_size**0.5)  # Scale factor for RoPE mode (no positional table)

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskBatch,
        *,
        device: Device | None = None,
    ):  # TODO: specify return type
        """ """
        token_embeddings = self.embed_tokens(input_ids.to(torch.int32))

        # RoPE mode: positions are encoded in QK rotation — scale by sqrt(d) only.
        token_embeddings = self.embedding_scale * token_embeddings

        return inputs


# =============================================================================
MazeHardTokenEncoder: TypeAlias = MazeHardLearnedEncoder | MazeHardRoPEEncoder


# =============================================================================
class MazeHardTokenDecoder(nn.Module, MazeHardDecoder):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """ """
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(  # -----------------------------------------------------------
        self,
        outputs: HRMOutputV1,
        *,
        device: Device | None = None,
    ) -> MazeHardTaskOutput:
        """ """
        logits = self.lm_head(outputs.someother_logits)  # Strip CLS → (B, S, vocab_size)
        return logits


# =============================================================================
class MazeHardHRMV1BridgeAdapter(nn.Module):
    """MazeHard plus HRM v1 model-task binding over the canonical HRM core."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV1,
        encoder: MazeHardLearnedEncoder | MazeHardRoPEEncoder,
        decoder: MazeHardTokenDecoder,
    ) -> None:
        """Initialize the HRM v1 bridge adapter with its component modules."""
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: MazeHardTaskBatch,
    ) -> HRMInputV1:
        """Prepare the model-facing core payload and task-side decoder context."""
        return self.encoder(batch)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        logits: HRMOutputV1,
    ) -> MazeHardTaskOutput:
        """Decode one task-owned MazeHard output from one HRM core output."""
        return self.decoder(logits)

    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskBatch,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMStateV1, MazeHardTaskOutput]:
        """Run a forward pass of the HRM v1 bridge adapter on a MazeHard task batch."""
        inputs = self.prepare_inputs(batch)
        next_state, logits = self.model(inputs, state=state)
        outputs = self.decode(logits)
        return next_state, outputs


# =============================================================================
__all__ = [
    "MazeHardLearnedEncoder",
    "MazeHardRoPEEncoder",
    "MazeHardTokenEncoder",
    "MazeHardTokenDecoder",
    "MazeHardHRMV1BridgeAdapter",
]

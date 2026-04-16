"""MazeHard task-side decoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.types import Batch


# =============================================================================
class MazeHardTokenEncoder(ABC):
    """Abstract base class for MazeHard token encoders."""

    @abstractmethod
    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        *,
        device: Device | None = None,
    ):  # TODO: specify return type
        """Encode a MazeHard task batch into token embeddings suitable for HRM input."""


# =============================================================================
class MazeHardLearnedEncoder(nn.Module):
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
        batch: Batch,
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
class MazeHardRoPEEncoder(nn.Module):
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
        batch: Batch,
        *,
        device: Device | None = None,
    ):  # TODO: specify return type
        """ """
        token_embeddings = self.embed_tokens(input_ids.to(torch.int32))

        # RoPE mode: positions are encoded in QK rotation — scale by sqrt(d) only.
        token_embeddings = self.embedding_scale * token_embeddings

        return inputs


# =============================================================================
__all__ = ["MazeHardTokenEncoder", "MazeHardLearnedEncoder", "MazeHardRoPEEncoder"]

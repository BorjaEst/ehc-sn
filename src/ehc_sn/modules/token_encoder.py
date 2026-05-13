"""Generic token-sequence conditioning modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.transformer import TransformerSequenceSummary, TransformerSequenceSummaryConfig
from ehc_sn.utils import trunc_normal_init_


class TokenSummarizer(nn.Module):
    """Embed token ids and summarize them into one batch-aligned vector.

    This module keeps token-id validation, token embedding, and CLS-style
    transformer summarization together behind a model-agnostic interface.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        summary_config: TransformerSequenceSummaryConfig,
        padding_idx: int = 0,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        if summary_config.hidden_size != hidden_size:
            raise ValueError("summary_config.hidden_size must match hidden_size, got " f"{summary_config.hidden_size} and {hidden_size}.")

        self._hidden_size = hidden_size
        self._embedding_scale = math.sqrt(float(hidden_size))
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.summary = TransformerSequenceSummary(summary_config, device=device, dtype=dtype)

    @property
    def hidden_size(self) -> int:
        """Return the summary width produced by this conditioner."""
        return self._hidden_size

    @property
    def summary_dtype(self) -> torch.dtype:
        """Return the dtype used by the summary output."""
        return self.summary.cls_token.dtype

    @property
    def padding_idx(self) -> int | None:
        """Return the embedding padding token id, if configured."""
        return self.token_embedding.padding_idx

    def reset_parameters(self, *, init_std: float) -> None:
        """Reset the local embedding and summary parameters."""
        self.summary.reset_parameters()
        trunc_normal_init_(self.token_embedding.weight, std=init_std)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()

    def forward(self, input_ids: Tensor, *, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Return one summary vector per batch row.

        Args:
            input_ids: Integer token ids with shape ``(B, S)``.
            attn_mask: Optional boolean token mask with shape ``(B, S)`` where
                ``True`` marks tokens that remain visible to attention.

        Returns:
            Tensor of shape ``(B, D)``.
        """
        device = self.summary.cls_token.device
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B, S), got {tuple(input_ids.shape)}.")

        batch_size = int(input_ids.shape[0])
        if attn_mask is not None:
            if attn_mask.shape != input_ids.shape:
                raise ValueError(f"attn_mask must have shape {tuple(input_ids.shape)}, got {tuple(attn_mask.shape)}.")
            attn_mask = attn_mask.to(device=device, dtype=torch.bool)

        input_ids = input_ids.to(device=device, dtype=torch.int64)
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= self.token_embedding.num_embeddings:
            raise ValueError(
                "input_ids must lie within " f"[0, {self.token_embedding.num_embeddings - 1}], got min={min_id}, max={max_id}."
            )
        token_embeddings = self.token_embedding(input_ids.to(torch.int32))
        return self.summary(self._embedding_scale * token_embeddings, attn_mask=attn_mask)


__all__ = ["TokenSummarizer"]

"""Attention building block.

This module implements a standard multi-head attention (MHA) layer with optional:

- GQA-style head sharing (a.k.a. grouped-query attention) via `num_kv_heads`.
- PyTorch SDPA (scaled dot-product attention) backend.

The implementation is intentionally small and shape-explicit since it is a core
primitive used throughout the model.
"""

import math
from typing import Optional

import torch.nn.functional as F
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_


# =================================================================================================
class AttentionConfig(BaseModel, extra="forbid"):
    """Configuration for the :class:`Attention` module.

    Notes:
        - `embedding_dim` must be divisible by `num_heads`.
        - `num_kv_heads` enables grouped-query attention when it is smaller than
          `num_heads` (keys/values are shared across multiple query heads).
    """

    embedding_dim: int = Field(
        ...,
        ge=32,
        frozen=True,
        description="Hidden size of the attention module.",
    )
    num_heads: int = Field(
        ...,
        ge=1,
        frozen=True,
        description="Number of attention heads.",
    )

    @field_validator("embedding_dim")
    @classmethod
    def _check_embed_dim(cls, v: int, info: ValidationInfo) -> int:
        num_heads = info.data.get("num_heads")
        if num_heads is not None and v % num_heads != 0:
            raise ValueError(f"embedding_dim ({v}) must be divisible by num_heads ({num_heads}).")
        return v

    num_kv_heads: Optional[int] = Field(
        default=None,
        frozen=True,
        validate_default=True,
        description="Key/value heads for grouped-query.",
    )

    @field_validator("num_kv_heads", mode="before")
    @classmethod
    def _resolve_num_kv_heads(cls, v: int | None, info: ValidationInfo) -> int | None:
        if v is not None:
            return v
        num_heads = info.data.get("num_heads")
        return num_heads if num_heads is not None else v

    @field_validator("num_kv_heads", mode="after")
    @classmethod
    def _validate_num_kv_heads(cls, v: int | None, info: ValidationInfo) -> int | None:
        if v is None:
            return v  # shouldn’t happen if num_heads was present, but keeps this validator total
        num_heads = info.data.get("num_heads")
        if num_heads is None:
            return v
        if v > num_heads:
            raise ValueError(f"num_kv_heads ({v}) must be <= num_heads ({num_heads}).")
        if num_heads % v != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({v}).")
        return v

    @field_validator("num_kv_heads", mode="before")
    @classmethod
    def _set_num_kv_heads(cls, v, info: ValidationInfo):
        return v if v is not None else info.data.get("num_heads")

    is_causal: bool = Field(default=False, description="Apply causal masking in attention.")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.embedding_dim // self.num_heads

    @property
    def output_size(self) -> int:
        """Output dimension of the attention module."""
        return self.head_dim * self.num_heads


# =================================================================================================
class Attention(nn.Module):
    """Multi-head attention with grouped-query attention support.

    Inputs and outputs use the common Transformer layout `[batch, seq, embed]`.
    Internally, tensors are reshaped to per-head form and fed through
    `torch.nn.functional.scaled_dot_product_attention`.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: AttentionConfig, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        self._config = config
        super().__init__()

        self._qkv_head_count = config.num_heads + 2 * (config.num_kv_heads or config.num_heads)
        self._qkv_proj_out_dim = self._qkv_head_count * config.head_dim
        self.in_proj = nn.Linear(
            config.embedding_dim, self._qkv_proj_out_dim, bias=False, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            config.output_size, config.embedding_dim, bias=False, device=device, dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:  # -------------------------------------------------------
        """Initialize projection weights with truncated normal matching legacy ``CastedLinear``.

        Std formulas (fan_in = input dimension of each projection):
            - ``in_proj``:  ``std = 1 / sqrt(embedding_dim)``
            - ``out_proj``: ``std = 1 / sqrt(output_size)``
        """
        trunc_normal_init_(self.in_proj.weight, std=1.0 / math.sqrt(self._config.embedding_dim))
        trunc_normal_init_(self.out_proj.weight, std=1.0 / math.sqrt(self._config.output_size))

    @property
    def config(self) -> AttentionConfig:
        return self._config

    def _compute_qkv(  # --------------------------------------------------------------------------
        self, x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
        """Project and split inputs into q, k, and v tensors.

        Args:
            x: Hidden states of shape `[batch, seq_len, embedding_dim]`.

        Returns:
            Tuple `(q, k, v)` with shapes:

            - `q`: `[batch, seq_len, num_heads, head_dim]`
            - `k`: `[batch, seq_len, num_kv_heads, head_dim]`
            - `v`: `[batch, seq_len, num_kv_heads, head_dim]`
        """
        config, qkv_head_count = self.config, self._qkv_head_count
        batch_size, seq_len, _ = x.shape

        # Project once, then slice into (q, k, v) head groups.
        qkv = self.in_proj(x)

        # Reshape to per-head representation.
        qkv = qkv.view(batch_size, seq_len, qkv_head_count, config.head_dim)
        q = qkv[:, :, : config.num_heads]
        k = qkv[:, :, config.num_heads : config.num_heads + config.num_kv_heads]  # type: ignore[assignment]
        v = qkv[:, :, config.num_heads + config.num_kv_heads :]  # type: ignore[assignment]
        return q, k, v

    def _expand_kv_heads(  # ----------------------------------------------------------------------
        self, k: Tensor, v: Tensor,
    ) -> tuple[Tensor, Tensor]:  # fmt: skip
        """Repeat k/v heads for grouped-query attention when needed.

        SDPA expects matching head counts for Q/K/V. When `num_kv_heads < num_heads`,
        we repeat keys/values across query heads.
        """
        config = self.config
        if config.num_kv_heads == config.num_heads:
            return k, v
        repeat = config.num_heads // config.num_kv_heads  # type: ignore[assignment]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        return k, v

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor, *, attn_mask: Optional[Tensor] = None,
    ) -> Tensor:  # fmt: skip
        """Compute attention outputs for a batch of sequences.

        Args:
            x: Hidden states of shape `[batch, seq_len, embedding_dim]`.
            attn_mask: Optional attention mask passed through to PyTorch SDPA.
                Shape and dtype semantics follow `scaled_dot_product_attention`.

        Returns:
            Attention outputs of shape `[batch, seq_len, embedding_dim]`.
        """
        config = self.config
        batch_size, seq_len, _ = x.shape
        q, k, v = self._compute_qkv(x)

        # SDPA expects `[batch, heads, seq, head_dim]`.
        q = q.transpose(1, 2)  # [bs, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k, v = self._expand_kv_heads(k, v)

        # `is_causal` is handled by SDPA; `attn_mask` is optional and may embed_inputs
        # padding, block masks, etc.
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=config.is_causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, config.output_size)
        return self.out_proj(attn)

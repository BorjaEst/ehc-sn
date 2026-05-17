from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.attention import Attention, AttentionConfig
from ehc_sn.modules.mlp import MLPConfig, SwiGLU
from ehc_sn.utils.norms import rms_norm


# =============================================================================
class TransformerBlockConfig(BaseModel, extra="forbid"):
    """Configuration for a single Transformer-style block.

    Notes:
        - The hidden size is the attention embedding dimension.
        - This config is used as a template and may be repeated to build a
          stack of identical blocks.
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

    is_causal: bool = Field(
        default=False,
        description="Whether to apply causal masking in attention.",
    )

    pos_encodings: Literal["learned", "rope"] = Field(
        default="learned",
        description=(
            "Positional encoding strategy. 'learned': add learned position embeddings once at the "
            "input (current default). 'rope': apply rotary embeddings inside every attention "
            "operation — required for multi-step recurrent reasoning (legacy parity)."
        ),
    )
    rope_theta: float = Field(
        default=10000.0,
        description="Base period for RoPE frequency bands. Ignored when pos_encodings='learned'.",
    )

    @property
    def attention(self) -> AttentionConfig:
        """Convenience property to access the attention config."""
        return AttentionConfig.model_validate(self, from_attributes=True)

    expansion: float = Field(
        default=4.0,
        gt=1.0,
        description="Expansion factor for the MLP layers in the transformer blocks.",
    )

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the attention config."""
        return self.embedding_dim

    @property
    def mlp(self) -> MLPConfig:
        """Convenience property to construct the MLP config from the block config."""
        return MLPConfig.model_validate(self, from_attributes=True)

    rms_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon value for RMS normalization layers.",
    )


# =============================================================================
class TransformerBlock(nn.Module):
    """A minimal Transformer block with RMSNorm residuals.

    Structure:
    1) Self-attention
    2) Residual + RMSNorm
    3) SwiGLU MLP
    4) Residual + RMSNorm

    Notes:
        - This block is intentionally small and dependency-light (vanilla PyTorch).
        - RMSNorm is applied *after* the residual add ("post-norm" style).
    """

    def __init__(  # -----------------------------------------------------------
        self,
        config: TransformerBlockConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self._config = config

        self.self_attn = Attention(config.attention, device=device, dtype=dtype)
        self.mlp = SwiGLU(config.mlp, device=device, dtype=dtype)

    @property
    def config(self) -> TransformerBlockConfig:
        """Return the parsed block configuration."""
        return self._config

    def forward(  # ------------------------------------------------------------
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply self-attention and MLP sublayers with post-norm residuals."""
        attention = self.self_attn(x, attn_mask=attn_mask)
        x = rms_norm(x + attention, variance_epsilon=self.config.rms_norm_eps)
        x = rms_norm(x + self.mlp(x), variance_epsilon=self.config.rms_norm_eps)
        return x


# =============================================================================
class TransformerSequenceSummaryConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`TransformerSequenceSummary`.

    This config keeps the reusable summary stack separate from any model-local
    token-id embedding or positional front-end.
    """

    block: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block configuration reused across the summary stack.",
    )
    n_layers: int = Field(
        default=1,
        ge=1,
        description="Number of transformer blocks in the summary stack.",
    )

    @property
    def hidden_size(self) -> int:
        """Hidden size consumed and produced by the summary module."""
        return self.block.hidden_size

    @property
    def layers(self) -> list[TransformerBlockConfig]:
        """Concrete block configs used to instantiate the summary stack."""
        return [self.block for _ in range(self.n_layers)]


# =============================================================================
class TransformerSequenceSummary(nn.Module):
    """Summarize embedded sequences into one CLS-pooled hidden vector.

    Inputs are already-embedded hidden states of shape ``(B, S, D)``. Any
    token-id embedding, padding behavior, or positional front-end belongs to
    the caller rather than the shared transformer layer.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: TransformerSequenceSummaryConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self._config = config
        hidden_size = config.hidden_size
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(layer, device=device, dtype=dtype)
                for layer in config.layers
            ]
        )
        self.reset_parameters()

    @property
    def config(self) -> TransformerSequenceSummaryConfig:
        """Return the parsed summary-module configuration."""
        return self._config

    def reset_parameters(self) -> None:
        """Initialize the CLS summary token."""
        self.cls_token.data.zero_()

    @staticmethod
    def _prepare_attn_mask(  # ------------------------------------------------
        attn_mask: Optional[Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Convert a token mask for non-CLS inputs into an SDPA-ready mask."""
        if attn_mask is None:
            return None
        if attn_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attn_mask must have shape ({batch_size}, {seq_len}), got {tuple(attn_mask.shape)}."
            )
        attn_mask = attn_mask.to(device=device, dtype=torch.bool)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        return torch.cat([cls_mask, attn_mask], dim=1).unsqueeze(1).unsqueeze(1)

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return one pooled summary vector per embedded sequence.

        Args:
            x: Embedded inputs of shape ``(B, S, D)``.
            attn_mask: Optional boolean token mask with shape ``(B, S)`` for
                the non-CLS inputs. ``True`` marks tokens visible to attention.

        Returns:
            Tensor of shape ``(B, D)`` containing the pooled CLS summaries.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (B, S, D), got {tuple(x.shape)}."
            )
        if int(x.shape[-1]) != self.config.hidden_size:
            raise ValueError(
                "x last dimension must match the summary hidden size, got "
                f"{int(x.shape[-1])} and {self.config.hidden_size}."
            )

        prepared_mask = self._prepare_attn_mask(
            attn_mask,
            batch_size=int(x.shape[0]),
            seq_len=int(x.shape[1]),
            device=x.device,
        )
        hidden = torch.cat(
            [self.cls_token.expand(x.shape[0], -1, -1), x], dim=1
        )
        for block in self.blocks:
            hidden = block(hidden, attn_mask=prepared_mask)
        return hidden[:, 0]


# =============================================================================
class TransformerStack(nn.Module):
    """
    A sequential stack of Transformer blocks that integrates an
    injected input into a running state via a residual connection.
    """

    def __init__(  # -----------------------------------------------------------
        self,
        layers: list[TransformerBlockConfig],
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Initialize the stack with a list of block configurations."""
        super().__init__()

        # List comprehension to instantiate blocks from configurations
        modules = [
            TransformerBlock(config, device=device, dtype=dtype)
            for config in layers
        ]
        # ModuleList is required so PyTorch tracks the parameters of each layer
        self.layers = nn.ModuleList(modules)

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
        residual: Tensor,
    ) -> Tensor:
        """
        Applies a residual injection followed by sequential transformer transformations.

        Args:
            x: The primary hidden state or 'main path' tensor.
            residual: The injected input or 'update' to be added to x.

        Returns:
            The transformed tensor after passing through all blocks.
        """
        # Residual-focused injection: combining the update with the state
        x = x + residual

        # Sequential processing through the block stack
        for layer in self.layers:
            x = layer(x)

        return x


# =============================================================================
__all__ = [
    "TransformerBlockConfig",
    "TransformerBlock",
    "TransformerSequenceSummaryConfig",
    "TransformerSequenceSummary",
    "TransformerStack",
]

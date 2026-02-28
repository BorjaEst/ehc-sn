from typing import Optional

from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.modules.attention import Attention, AttentionConfig
from ehc_sn.modules.mlp import MLPConfig, SwiGLU
from ehc_sn.types import Device, Dtype
from ehc_sn.utils.norms import rms_norm


# =================================================================================================
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


# =================================================================================================
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

    def __init__(  # ------------------------------------------------------------------------------
        self, config: TransformerBlockConfig, *, 
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.self_attn = Attention(config.attention, device=device, dtype=dtype)
        self.mlp = SwiGLU(config.mlp, device=device, dtype=dtype)

    @property
    def config(self) -> TransformerBlockConfig:
        """ """
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor,
    ) -> Tensor:  # fmt: skip
        """ """
        attention = self.self_attn(x)
        x = rms_norm(x + attention, variance_epsilon=self.config.rms_norm_eps)
        x = rms_norm(x + self.mlp(x), variance_epsilon=self.config.rms_norm_eps)
        return x


class TransformerStack(nn.Module):
    """
    A sequential stack of Transformer blocks that integrates an
    injected input into a running state via a residual connection.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, layers: list[TransformerBlockConfig], *,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()

        # List comprehension to instantiate blocks from configurations
        modules = [TransformerBlock(config, device=device, dtype=dtype) for config in layers]
        # ModuleList is required so PyTorch tracks the parameters of each layer
        self.layers = nn.ModuleList(modules)

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor, residual: Tensor,
    ) -> Tensor:  # fmt: skip
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

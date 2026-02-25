"""Hierarchical Reasoning Model (HRM) building blocks.

This module contains the core HRM architecture used in this repository:

- :class:`TransformerBlock`: a minimal Transformer-style block
    (self-attention + SwiGLU MLP) with RMSNorm residuals.
- :class:`ReasoningModule`: a stack of blocks that applies an additive
    *input injection* before each layer stack.
- :class:`HRModel`: a two-level recurrent model with a high-level state $z_H$
    and a low-level state $z_L$ that are updated in alternating cycles.

The model is implemented as a plain :class:`torch.nn.Module` so it can be
composed into training/evaluation code (e.g., Lightning) without introducing
framework-specific side effects.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.modules.attention import Attention, AttentionConfig
from hrm_sn.modules.mlp import MLPConfig, SwiGLU
from hrm_sn.types import Device, Dtype
from hrm_sn.utils import trunc_normal_init_
from hrm_sn.utils.norms import rms_norm


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
        self, config: TransformerBlockConfig, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.self_attn = Attention(config.attention)
        self.mlp = SwiGLU(config.mlp)
        self.norm_eps = config.rms_norm_eps

    @property
    def config(self) -> TransformerBlockConfig:
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor
    ) -> Tensor:  # fmt: skip
        attention = self.self_attn(x)
        x = rms_norm(x + attention, variance_epsilon=self.norm_eps)
        x = rms_norm(x + self.mlp(x), variance_epsilon=self.norm_eps)
        return x


# =================================================================================================
class ReasoningModule(nn.Module):
    """A stack of :class:`TransformerBlock` layers with additive input injection.

    The HRM alternates between high-level and low-level reasoning modules.
    Each module receives the current state tensor and an *injection* tensor that
    anchors computation to the external inputs and/or the other level's state.

    Expected shapes:
        - ``x``: ``[batch, seq_length, hidden_size]``
        - ``input_injection``: broadcastable to ``x`` (typically the same shape).
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, layers: List[TransformerBlockConfig], device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()

        modules = [TransformerBlock(config) for config in layers]
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, x: Tensor, input_injection: Tensor) -> Tensor:
        x = x + input_injection
        for layer in self.layers:
            x = layer(x=x)
        return x


# =================================================================================================
class ReasoningConfig(BaseModel, extra="forbid"):
    """ """

    layers: int = Field(
        default=4,
        ge=1,
        description="Number of layers in each reasoning module (high-level and low-level).",
    )
    cycles: int = Field(
        default=2,
        ge=1,
        description="Number of cycles to alternate between high-level and low-level reasoning modules.",
    )


# =================================================================================================
class HRMConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`HRModel`.

    This config bundles embedding settings, sequence constraints, and the base
    Transformer block hyperparameters used to build both reasoning modules.
    """

    # Model parameters for features
    vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for token embeddings and LM head.",
    )

    seq_length: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
    )

    # Model parameters for the core HRM architecture
    transformer_block: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block configuration.",
    )

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the transformer block config."""
        return self.transformer_block.hidden_size

    @property
    def embedding_scale(self) -> float:
        """Convenience property for scaling embeddings to maintain variance."""
        # scale by 1/sqrt(2) to maintain forward variance
        return 0.707106781 * math.sqrt(self.transformer_block.embedding_dim)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization."""
        return 1.0 / math.sqrt(self.transformer_block.embedding_dim)

    # Reasoning module configs
    reasoning_h: ReasoningConfig = Field(
        default_factory=ReasoningConfig,
        description="Configuration for the high-level reasoning module.",
    )

    reasoning_l: ReasoningConfig = Field(
        default_factory=ReasoningConfig,
        description="Configuration for the low-level reasoning module.",
    )

    @property
    def h_layers(self) -> List[TransformerBlockConfig]:
        """Convenience property to construct the list of transformer block configs for the high-level reasoning module."""
        return [self.transformer_block for _ in range(self.reasoning_h.layers)]

    @property
    def l_layers(self) -> List[TransformerBlockConfig]:
        """Convenience property to construct the list of transformer block configs for the low-level reasoning module."""
        return [self.transformer_block for _ in range(self.reasoning_l.layers)]


# =================================================================================================
@dataclass
class HRMState:
    """Recurrent state carried between forward passes.

    Attributes:
        z_H: High-level state of shape ``[batch, seq_length, hidden_size]``.
        z_L: Low-level state of shape ``[batch, seq_length, hidden_size]``.

    Notes:
        The state returned by :meth:`HRModel.forward` is detached so subsequent
        steps do not backpropagate through time.
    """

    z_H: Tensor  # Higher-level state tensor of shape [batch, seq_length, hidden_size].
    z_L: Tensor  # Lower-level state tensor of shape [batch, seq_length, hidden_size].


# =================================================================================================
class HRModel(nn.Module):
    """Hierarchical Reasoning Model (HRM).

    The model maintains two latent sequences, a high-level state ``z_H`` and a
    low-level state ``z_L``. Computation proceeds by alternating *cycles* of
    updates between these two levels.

    Implementation detail:
        To reduce memory usage, most iterative updates are executed under
        ``torch.no_grad()``. Only the final update at each level is executed
        with gradients ("one-step grad"). This is a deliberate trade-off between
        compute, memory, and training signal.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HRMConfig, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(config.seq_length, config.hidden_size, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

        # Reasoning Layers
        self.high_level = ReasoningModule(config.h_layers)
        self.low_level = ReasoningModule(config.l_layers)

        # Fixed per-model reset vectors for recurrent state.
        self.register_buffer("high_reset_vector", torch.empty((config.hidden_size,)), persistent=True)
        self.register_buffer("low_reset_vector", torch.empty((config.hidden_size,)), persistent=True)
        self.high_reset_vector = cast(Tensor, self.high_reset_vector)
        self.low_reset_vector = cast(Tensor, self.low_reset_vector)

        self.reset_parameters()

    @property
    def config(self) -> HRMConfig:
        """Convenience property to access the model configuration."""
        return self._config

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize parameters and buffers.

        Matches legacy ``CastedEmbedding`` / ``CastedLinear`` initialization so that
        the input embedding magnitude ``||x||`` and recurrent state magnitude ``||z_H||``
        are comparable (~1:1 ratio at init), which is required for multi-step reasoning
        dynamics to emerge during training.

        Std formulas (truncated normal, legacy parity):
            - Embeddings: ``std = 1 / sqrt(hidden_size)``  (= ``config.init_std``)
            - lm_head:    ``std = 1 / sqrt(hidden_size)``  (fan_in = hidden_size)
            - Reset vecs: ``std = 1``
        """
        init_std = self.config.init_std  # 1 / sqrt(hidden_size)
        trunc_normal_init_(self.embed_tokens.weight, std=init_std)
        trunc_normal_init_(self.embed_pos.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)
        trunc_normal_init_(self.high_reset_vector, std=1)
        trunc_normal_init_(self.low_reset_vector, std=1)

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """Allocate a new state tensor with the right shape/dtype/device.

        The returned state is initialized via :meth:`reset_state` using the
        model-owned initial buffers (``high_init``/``low_init``). These are
        stored in the module state but are not trainable parameters.
        """
        config = self.config
        new_state = HRMState(
            z_H=self.high_reset_vector.new_empty(batch_size, config.seq_length, config.hidden_size),
            z_L=self.low_reset_vector.new_empty(batch_size, config.seq_length, config.hidden_size),
        )
        reset_flag = torch.ones(batch_size, dtype=torch.bool, device=new_state.z_H.device)
        return self.reset_state(reset_flag, state=new_state)

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """Reset selected batch elements of the state to initial states.

        Args:
            reset_flag: Boolean-ish tensor of shape ``[batch]`` (or broadcastable
                to it). True entries reset the corresponding state sequences.
            state: Current state.
        """
        batch_size = state.z_H.shape[0]
        if reset_flag.numel() not in {1, batch_size}:
            raise ValueError(f"reset_flag must be broadcastable to [batch], got numel={reset_flag.numel()} for batch_size={batch_size}.")

        reset_flag = reset_flag.to(torch.bool)
        init_H = self.high_reset_vector.view(1, 1, -1).expand_as(state.z_H)
        init_L = self.low_reset_vector.view(1, 1, -1).expand_as(state.z_L)
        mask = reset_flag.view(-1, 1, 1)

        return HRMState(
            z_H=torch.where(mask, init_H, state.z_H),
            z_L=torch.where(mask, init_L, state.z_L),
        )

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Tensor, 
        state: Optional[HRMState] = None,
    ) -> Tuple[HRMState, Tensor, Tensor]:  # fmt: skip
        """Run a forward pass.

        Args:
            inputs: Token ids with shape ``[batch, seq_length]``.
            state: Optional previous :class:`HRMState`. If omitted, an empty
                state is allocated.

        Returns:
            ``(new_state, lm_logits, features)`` where:
            - ``new_state`` contains detached states to carry to the next step.
            - ``lm_logits`` has shape ``[batch, seq_length, vocab_size]``.
            - ``features`` is a differentiable tensor of shape ``[batch, hidden_size]``.
        """
        config = self.config  # convenience alias
        state = state or self.init_state(batch_size=inputs.shape[0])
        x = self.embed_inputs(inputs)

        # Forward iterations without grad for memory efficiency.
        # The final update at each level is executed with gradients below.
        with torch.no_grad():
            self.run_high_cycles(x, state, n_cycles=config.reasoning_h.cycles - 1)
            self.run_low_cycles(x, state, n_cycles=config.reasoning_l.cycles - 1)

        # One-step grad: provide a training signal while keeping memory bounded.
        z_L = state.z_L = self.low_level(state.z_L, state.z_H + x)
        z_H = state.z_H = self.high_level(state.z_H, state.z_L)

        # Carry is detached so the next step does not backprop through time.
        new_state = HRMState(z_H=z_H.detach(), z_L=z_L.detach())
        # Language-modeling head predicts a token distribution at each position.
        output = self.lm_head(z_H)

        return new_state, output, z_H[:, 0]

    def run_high_cycles(  # -----------------------------------------------------------------------
        self, x: Tensor, state: HRMState,
        n_cycles: Optional[int] = None,
    ) -> Tensor:  # fmt: skip
        """Iterate high-level cycles, interleaving low-level updates."""
        cycles = self.config.reasoning_h.cycles if n_cycles is None else n_cycles
        for _ in range(cycles):
            state.z_L = self.run_low_cycles(x, state)
            state.z_H = self.high_level(state.z_H, state.z_L)
        return state.z_H

    def run_low_cycles(  # ------------------------------------------------------------------------
        self, x: Tensor, state: HRMState,
        n_cycles: Optional[int] = None,
    ) -> Tensor:  # fmt: skip
        """Iterate low-level cycles (conditioned on high-level state and inputs)."""
        cycles = self.config.reasoning_l.cycles if n_cycles is None else n_cycles
        for _ in range(cycles):
            state.z_L = self.low_level(state.z_L, state.z_H + x)
        return state.z_L

    def embed_inputs(  # --------------------------------------------------------------------------
        self, input: Tensor,
    ) -> Tensor:  # fmt: skip
        """Embed token ids and add positional embeddings.

        Args:
            input: Integer token ids with shape ``[batch, seq_length]``.

        Returns:
            Embedded inputs with shape ``[batch, seq_length, hidden_size]``.

        Raises:
            ValueError: If input dimensionality is not 2D or if ``seq_length``
                exceeds the configured maximum.
        """
        if input.ndim != 2:
            raise ValueError(f"Expected inputs with shape [batch, seq_length], got {tuple(input.shape)}")

        seq_length = input.shape[1]
        if seq_length != self.config.seq_length:
            raise ValueError(f"Input must match configured seq_length ({self.config.seq_length}); got {seq_length}.")

        token_embeddings = self.embed_tokens(input.to(torch.int32))
        positions = torch.arange(seq_length, device=input.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)

        # Scale embeddings to keep activations in a reasonable range.
        return self.config.embedding_scale * (token_embeddings + pos_embeddings)

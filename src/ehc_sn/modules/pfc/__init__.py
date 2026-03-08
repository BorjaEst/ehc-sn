import itertools
import math
from dataclasses import dataclass
from itertools import islice
from typing import List, Literal, Optional, Tuple, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.pfc import reasoning as r
from ehc_sn.modules.pfc.reasoning import HighLvRModule, LowLvRModule, ReasoningSettings, WorkingMemory
from ehc_sn.modules.pfc.values import QEstimatorSettings, QValueEstimator
from ehc_sn.modules.transformer import TransformerBlockConfig, TransformerStack
from ehc_sn.types import Activation, Device, Dtype, Matrix, MemoryState, MultiScaleCode
from ehc_sn.utils import trunc_normal_init_


# =================================================================================================
class PFCSettings(BaseModel, extra="forbid"):
    """Configuration for :class:`PFCModel`.

    The PFC module is a two-timescale recurrent reasoning system (high/low level)
    with an auxiliary value head (vmPFC analogue).

    Attributes:
        seq_length: Number of tokens per example (without the CLS prefix).
        value_head: Settings for the auxiliary Q/value estimator.
        cortex: Base transformer block configuration reused across reasoning modules.
        layers_h/cycles_h: Depth and recurrence cycles for the high-level module.
        layers_l/cycles_l: Depth and recurrence cycles for the low-level module.
    """

    # Model parameters for features
    seq_length: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
    )
    value_head: QEstimatorSettings = Field(
        ...,
        description="Configuration for the Q-value estimator (vmPFC analogue).",
    )
    cortex: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block config for reasoning modules.",
    )

    @property
    def hidden_size(self) -> int:
        """Convenience property to access the hidden size from the cortex config."""
        return self.cortex.embedding_dim

    # High-level reasoning module parameters (anterior dlPFC analogue)
    layers_h: int = Field(
        default=4,
        ge=1,
        description="Number of layers in the high-level reasoning module.",
    )
    cycles_h: int = Field(
        default=2,
        ge=1,
        description="Number of cycles to reason in the high-level reasoning module.",
    )

    @property
    def reasoning_h(self) -> ReasoningSettings:
        """Convenience property to access the high-level reasoning config."""
        return ReasoningSettings(cortex=self.cortex, n_layers=self.layers_h, n_cycles=self.cycles_h)

    # Low-level reasoning module parameters (posterior dlPFC analogue)
    layers_l: int = Field(
        default=4,
        ge=1,
        description="Number of layers in the low-level reasoning module.",
    )
    cycles_l: int = Field(
        default=2,
        ge=1,
        description="Number of cycles to reason in the low-level reasoning module.",
    )

    @property
    def reasoning_l(self) -> ReasoningSettings:
        """Convenience property to access the low-level reasoning config."""
        return ReasoningSettings(cortex=self.cortex, n_layers=self.layers_l, n_cycles=self.cycles_l)


# =================================================================================================
@dataclass
class PFCState:
    """Recurrent state for :class:`PFCModel`.

    The state is represented as a :class:`~ehc_sn.modules.pfc.reasoning.WorkingMemory`
    holding the high-level and low-level activation tensors.
    """

    # Working memory state of the PFC, containing the theta and gamma cell activations.
    memory: WorkingMemory

    @property
    def theta_cells(self) -> Tensor:
        """Return the theta cells from the low-level state.
        Biologically ($4-8$ Hz): These represent the Sequence and Context.
        Theta acts as the "metronome" that organizes the Gamma bursts into a logical order.
        """
        return self.memory.z_H

    @property
    def gamma_cells(self) -> Tensor:
        """Return the gamma cells from the high-level state.
        Biologically ($>30$ Hz): These represent the Active Content.
        If you are holding a specific rule in your head that rule is "loaded" into Gamma bursts.
        """
        return self.memory.z_L

    def detach(self) -> "PFCState":
        """Return a detached copy of the state."""
        return PFCState(self.memory.detach())


# =================================================================================================
class PFCModel(nn.Module):
    """Prefrontal Cortex (PFC) reasoning module.

    This module implements a two-level recurrent reasoning process:
        - high level (theta-like) state update
        - low level (gamma-like) state update

    It also includes an auxiliary estimator (vmPFC analogue) that predicts values
    from the current working memory.

    Notes:
        The forward pass intentionally detaches the returned carry state while
        keeping the main outputs differentiable.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: PFCSettings, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.high_level = HighLvRModule(config.reasoning_h, device=device, dtype=dtype)
        self.low_level = LowLvRModule(config.reasoning_l, device=device, dtype=dtype)
        self.estimator = QValueEstimator(config.value_head, device=device, dtype=dtype)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size, device=device, dtype=dtype))
        self.optimizer = None  # Placeholder for optimizer future dACC reward-based updates
        self.reset_parameters()

    @property
    def config(self) -> PFCSettings:
        """PFC module settings."""
        return self._config

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize parameters and buffers."""
        trunc_normal_init_(self.high_level.reset_vector, std=1)
        trunc_normal_init_(self.low_level.reset_vector, std=1)
        self.cls_token.data.zero_()  # Legacy parity: puzzle prefix initialized to zero

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int, 
    ) -> PFCState:  # fmt: skip
        """Create a fresh PFC recurrent state.

        Args:
            batch_size: Number of parallel sequences.

        Returns:
            Initialized :class:`PFCState`.
        """
        # seq_length + 1: CLS prefix occupies position 0; cell tokens fill positions 1..S.
        memory = r.init_memory(batch_size, self.config.seq_length + 1, self.high_level, self.low_level)
        return PFCState(memory=memory)

    def reset_state(  # --------------------------------------------------------------------------
        self, state: PFCState, reset_flag: Tensor,
    ) -> PFCState:  # fmt: skip
        """Selectively reset rows of the PFC state.

        Args:
            state: Current state.
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating which
                rows should be reset.

        Returns:
            New state with flagged rows reset.
        """
        memory = r.reset_memory(state.memory, reset_flag, self.high_level, self.low_level)
        return PFCState(memory=memory)

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor, state: Optional[PFCState] = None,
    ) -> Tuple[PFCState, Tensor, Tensor]:  # fmt: skip
        """Run recurrent reasoning and compute auxiliary value estimates.

        Args:
            x: Embedded inputs of shape ``(B, S, D)`` (cell tokens only; no CLS).
            state: Optional carry state. If ``None``, a fresh state is created.

        Returns:
            ``(new_state, z_H, q_estimation)`` where:
                - ``new_state`` is detached carry state
                - ``z_H`` is the high-level activation tensor (includes CLS)
                - ``q_estimation`` are value/Q predictions from the estimator
        """
        state = state or self.init_state(batch_size=x.shape[0])
        total_steps = self.config.reasoning_h.n_cycles * (self.config.reasoning_l.n_cycles + 1)

        # Prepend CLS to cell embeddings: (B, S, D) → (B, S+1, D).
        # Reasoning modules are CLS-agnostic; they see a uniform sequence.
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # Forward iterations without grad for memory efficiency.
        # The final update at each level is executed with gradients below.
        memory_gen = r.reasoning_gen(x, state.memory, self.high_level, self.low_level)
        with torch.no_grad():
            for _ in range(total_steps - 2):  # Last 2 steps need gradients
                memory = next(memory_gen)

        # One-step grad: provide a training signal while keeping memory bounded.
        memory = next(memory_gen)  # N-2 step to update low-level state with gradients
        memory = next(memory_gen)  # N-1 step to update high-level state with gradients

        # Estimate Q-values from the updated state.
        q_estimation = self.estimator(memory.z_H, memory.z_L)

        # Important: detach only the carry, not the outputs used for supervised learning.
        # Downstream heads (LM head, value head) must see tensors that keep gradients.
        return PFCState(memory=memory.detach()), memory.z_H, q_estimation

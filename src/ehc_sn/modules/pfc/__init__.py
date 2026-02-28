import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.transformer import TransformerBlockConfig, TransformerStack
from ehc_sn.types import Device, Dtype, Matrix, MemoryState, MultiScaleCode


# =================================================================================================
class ReasoningSettings(BaseModel, extra="forbid"):
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
class PFCSettings(BaseModel, extra="forbid"):
    """ """

    # Model parameters for features
    seq_length: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
    )

    # Model parameters for the core HRM architecture
    cortex: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block configuration.",
    )

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the transformer block config."""
        return self.cortex.hidden_size

    @property
    def embedding_scale(self) -> float:
        """Convenience property for scaling embeddings to maintain variance."""
        # scale by 1/sqrt(2) to maintain forward variance
        return 0.707106781 * math.sqrt(self.cortex.embedding_dim)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization."""
        return 1.0 / math.sqrt(self.cortex.embedding_dim)

    # Reasoning module configs
    reasoning_h: ReasoningSettings = Field(
        default_factory=ReasoningSettings,
        description="Configuration for the high-level reasoning module.",
    )

    reasoning_l: ReasoningSettings = Field(
        default_factory=ReasoningSettings,
        description="Configuration for the low-level reasoning module.",
    )

    @property
    def h_layers(self) -> List[TransformerBlockConfig]:
        """Convenience property to construct the list of transformer block configs for the high-level reasoning module."""
        return [self.cortex for _ in range(self.reasoning_h.layers)]

    @property
    def l_layers(self) -> List[TransformerBlockConfig]:
        """Convenience property to construct the list of transformer block configs for the low-level reasoning module."""
        return [self.cortex for _ in range(self.reasoning_l.layers)]


# =================================================================================================
@dataclass
class PFCState:
    """ """

    z_H: Tensor  # Higher-level state tensor of shape [batch, seq_length, hidden_size].

    @property
    def theta_cells(self) -> Tensor:
        """Return the theta cells from the low-level state.
        Biologically ($4-8$ Hz): These represent the Sequence and Context.
        Theta acts as the "metronome" that organizes the Gamma bursts into a logical order.
        """
        return self.z_H

    z_L: Tensor  # Lower-level state tensor of shape [batch, seq_length, hidden_size].

    @property
    def gamma_cells(self) -> Tensor:
        """Return the gamma cells from the high-level state.
        Biologically ($>30$ Hz): These represent the Active Content.
        If you are holding a specific rule in your head that rule is "loaded" into Gamma bursts.
        """
        return self.z_L

    def detach(self) -> "PFCState":
        """Return a detached copy of the state."""
        return PFCState(self.z_H.detach(), self.z_L.detach())


# =================================================================================================
class PFCModel(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: PFCSettings, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.estimator = None
        self.memory_system = None
        self.optimizer = None  # Placeholder for optimizer future dACC reward-based updates

        # Legacy code to be replaced later
        self.high_level = TransformerStack(config.h_layers, device=device, dtype=dtype)
        self.low_level = TransformerStack(config.l_layers, device=device, dtype=dtype)
        self.register_buffer("high_reset_vector", torch.empty((config.hidden_size,)), persistent=True)
        self.register_buffer("low_reset_vector", torch.empty((config.hidden_size,)), persistent=True)
        self.high_reset_vector = cast(Tensor, self.high_reset_vector)
        self.low_reset_vector = cast(Tensor, self.low_reset_vector)

    @property
    def config(self) -> PFCSettings:
        """PFC module settings."""
        return self._config

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int, state: Optional[PFCState] = None, *,
        reset_flag: Optional[Tensor] = None,
    ) -> PFCState:  # fmt: skip
        """Initialize or selectively reset PFC state.

        Args:
            batch_size: Number of examples in the batch.
            state: Existing state to selectively reset. If ``None``, creates a
                fresh state filled with the learned reset vectors.
            reset_flag: Boolean mask of shape ``(B,)``. ``True`` = reset that
                row. Defaults to all-``True`` (full reset).
        """
        device = self.high_reset_vector.device
        if reset_flag is None:
            reset_flag = torch.ones(batch_size, dtype=torch.bool, device=device)

        init_H = self.high_reset_vector.view(1, 1, -1).expand(batch_size, self.config.seq_length, -1)
        init_L = self.low_reset_vector.view(1, 1, -1).expand(batch_size, self.config.seq_length, -1)

        if state is None:
            return PFCState(z_H=init_H.clone(), z_L=init_L.clone())

        mask = reset_flag.view(-1, 1, 1)
        return PFCState(z_H=torch.where(mask, init_H, state.z_H), z_L=torch.where(mask, init_L, state.z_L))

    def init_memory(self, *, batch_size: int, device: torch.device) -> List[Tensor]:
        raise NotImplementedError("PRC working-memory initialization not implemented.")

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor, state: Optional[PFCState] = None,
    ) -> Tuple[PFCState, Tensor]:  # fmt: skip
        """ """
        state = state or self.init_state(batch_size=x.shape[0])

        # Forward iterations without grad for memory efficiency.
        # The final update at each level is executed with gradients below.
        with torch.no_grad():
            self.run_high_cycles(x, state, n_cycles=self.config.reasoning_h.cycles - 1)
            self.run_low_cycles(x, state, n_cycles=self.config.reasoning_l.cycles - 1)

        # One-step grad: provide a training signal while keeping memory bounded.
        z_L = state.z_L = self.low_level(state.z_L, state.z_H + x)
        z_H = state.z_H = self.high_level(state.z_H, state.z_L)

        # Carry is detached so the next step does not backprop through time.
        new_state = PFCState(z_H=z_H.detach(), z_L=z_L.detach())
        return new_state, z_H

    # Legacy code to be replaced later with more biologically-plausible iterative updates and memory interactions.

    def run_high_cycles(  # -----------------------------------------------------------------------
        self, x: Tensor, state: PFCState,
        n_cycles: Optional[int] = None,
    ) -> Tensor:  # fmt: skip
        """Iterate high-level cycles, interleaving low-level updates."""
        cycles = self.config.reasoning_h.cycles if n_cycles is None else n_cycles
        for _ in range(cycles):
            state.z_L = self.run_low_cycles(x, state)
            state.z_H = self.high_level(state.z_H, state.z_L)
        return state.z_H

    def run_low_cycles(  # ------------------------------------------------------------------------
        self, x: Tensor, state: PFCState,
        n_cycles: Optional[int] = None,
    ) -> Tensor:  # fmt: skip
        """Iterate low-level cycles (conditioned on high-level state and inputs)."""
        cycles = self.config.reasoning_l.cycles if n_cycles is None else n_cycles
        for _ in range(cycles):
            state.z_L = self.low_level(state.z_L, state.z_H + x)
        return state.z_L

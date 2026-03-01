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
from ehc_sn.types import Device, Dtype


# =================================================================================================
class PFCSettings(BaseModel, extra="forbid"):
    """ """

    # Model parameters for features
    seq_length: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
    )

    # Reasoning module configs
    reasoning_h: ReasoningSettings = Field(
        default_factory=ReasoningSettings,
        description="Configuration for the high-level reasoning module (anterior dlPFC).",
    )
    reasoning_l: ReasoningSettings = Field(
        default_factory=ReasoningSettings,
        description="Configuration for the low-level reasoning module (posterior dlPFC).",
    )


# =================================================================================================
@dataclass
class PFCState:
    """ """

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
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: PFCSettings, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.estimator = None  # Placeholder for future vmPFC q-value estimator
        self.high_level = HighLvRModule(config.reasoning_h, device=device, dtype=dtype)
        self.low_level = LowLvRModule(config.reasoning_l, device=device, dtype=dtype)
        self.optimizer = None  # Placeholder for optimizer future dACC reward-based updates

    @property
    def config(self) -> PFCSettings:
        """PFC module settings."""
        return self._config

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int, 
    ) -> PFCState:  # fmt: skip
        """ """
        memory = r.init_memory(batch_size, self.config.seq_length, self.high_level, self.low_level)
        return PFCState(memory=memory)

    def reset_state(  # --------------------------------------------------------------------------
        self, state: PFCState, reset_flag: Tensor,
    ) -> PFCState:  # fmt: skip
        """ """
        memory = r.reset_memory(state.memory, reset_flag, self.high_level, self.low_level)
        return PFCState(memory=memory)

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor, state: Optional[PFCState] = None,
    ) -> Tuple[PFCState, Tensor]:  # fmt: skip
        """ """
        state = state or self.init_state(batch_size=x.shape[0])
        total_steps = self.config.reasoning_h.n_cycles * (self.config.reasoning_l.n_cycles + 1)

        # Forward iterations without grad for memory efficiency.
        # The final update at each level is executed with gradients below.
        memory_gen = r.reasoning_gen(x, state.memory, self.high_level, self.low_level)
        with torch.no_grad():
            for _ in range(total_steps - 2):  # Last 2 steps need gradients
                memory = next(memory_gen)

        # One-step grad: provide a training signal while keeping memory bounded.
        memory = next(memory_gen)  # N-2 step to update low-level state with gradients
        memory = next(memory_gen)  # N-1 step to update high-level state with gradients

        # Return the final state and the high-level state (theta cells) for downstream use.
        return PFCState(memory=memory.detach()), memory.z_H

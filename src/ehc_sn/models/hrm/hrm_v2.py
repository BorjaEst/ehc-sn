""" """

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import torch
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]


# =================================================================================================
class ModelSettings_V2(BaseModel, extra="forbid"):
    """Model-level settings for HRM v2.

    This settings object composes:
        - PFC settings (recurrent reasoning core)
        - STR settings (actor-critic / reward head)
        - token vocabulary size

    Notes:
        HRM v2 currently requires RoPE positional encodings inside the PFC modules
        for legacy parity and to match the environment tokenization.
    """

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the core PFC model architecture.",
    )
    str: STRSettings = Field(
        ...,
        description="Settings for the STR actor-critic architecture.",
    )

    @field_validator("pfc", mode="after")
    def validate_pfc(cls, v: PFCSettings) -> PFCSettings:
        """Ensure that the PFC settings have a valid reasoning module configuration."""
        if v.cortex.pos_encodings != "rope":
            raise ValueError("PFC reasoning modules must use RoPE positional encodings")
        return v

    vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for token embeddings and LM head.",
    )

    @property
    def seq_length(self) -> int:
        """Convenience property to access sequence length from the PFC settings."""
        return self.pfc.seq_length

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the PFC settings."""
        return self.pfc.reasoning_h.cortex.embedding_dim

    @property
    def embedding_scale(self) -> float:
        """Base embedding scale applied to token embeddings."""
        return math.sqrt(self.hidden_size)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization."""
        return 1.0 / math.sqrt(self.hidden_size)

    @classmethod
    def from_config(cls, path: Path) -> "ModelSettings_V2":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)


# =================================================================================================
@dataclass
class HRMState(DetachMixin):
    """Recurrent state carried across steps for HRM v2.

    Attributes:
        pfc: PFC recurrent state.
        str: STR recurrent state.
    """

    pfc: PFCState  # Prefrontal Cortex state, containing working memory and reasoning module states.
    str: STRState  # STR actor-critic state, containing any recurrent state for the STR module (if needed).


# =================================================================================================
class HRModelV2(nn.Module):
    """Core HRM v2 model.

    The model consists of:
        - token embedding table
        - PFC recurrent reasoning module producing per-token logits and a CLS summary
        - STR actor-critic module consuming the CLS summary and PFC Q logits
        - language-model head predicting per-token labels

    The forward pass returns:
        - updated recurrent state
        - tuple of logits ``(token_logits, q_logits, r_logits)``
        - CLS feature vector (used by the controller / tracing)
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Reward estimator
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V2:
        """Return the parsed model settings used to build this module."""
        return self._config

    def reset_parameters(self) -> None:  # -------------------------------------------------------
        """Initialize parameters.

        Uses truncated normal initialization with ``std = 1/sqrt(hidden_size)`` for
        token embeddings and the LM head to keep initial activation scales stable.
        """
        init_std = self.config.init_std
        trunc_normal_init_(self.embed_tokens.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """Create a fresh recurrent state.

        Args:
            batch_size: Number of parallel environments / sequences.

        Returns:
            A new :class:`HRMState` with initialized PFC and STR states.
        """
        return HRMState(
            pfc=self.pfc.init_state(batch_size),
            str=self.str.init_state(batch_size),
        )

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """Selectively reset rows of the recurrent state.

        Args:
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating which
                batch rows should be reset.
            state: Current recurrent state.

        Returns:
            New state with flagged rows reset for both PFC and STR.
        """
        return HRMState(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
        )

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, state: Optional[HRMState] = None,
    ) -> tuple[HRMState, tuple[Tensor, Tensor, Tensor], Tensor]:  # fmt: skip
        """Run one model step.

        Args:
            batch: Input batch containing at least ``"input_ids"`` of shape ``(B, S)``.
            state: Optional recurrent state to carry across steps. If ``None``, a
                fresh state is created.

        Returns:
            ``(new_state, (logits, q_logits, r_logits), theta_cls)`` where:
                - ``logits`` is ``(B, S, vocab_size)``
                - ``q_logits`` is controller-specific (produced by PFC)
                - ``r_logits`` is reward / policy output from STR
                - ``theta_cls`` is ``(B, D)`` CLS summary vector.
        """
        state = state or self.init_state(batch_size=batch["input_ids"].shape[0])
        x = self.embed_input_ids(batch["input_ids"])  # (B, S, D)

        state_pfc, z_H, q_logits = self.pfc(x, state=state.pfc)  # z_H: (B, S+1, D)
        logits = self.lm_head(z_H[:, 1:])  # strip CLS → (B, S, vocab)
        theta_cls = z_H[:, 0]  # (B, D) — theta/CLS summary
        state_str, r_logits = self.str(theta_cls.detach(), q_logits, state.str)

        new_state = HRMState(pfc=state_pfc, str=state_str)
        return new_state, (logits, q_logits, r_logits), theta_cls

    def embed_input_ids(  # -----------------------------------------------------------------------
        self, input_ids: Tensor,
    ) -> Tensor:  # fmt: skip
        """Embed token ids into a scaled representation.

        Args:
            input_ids: Token ids of shape ``(B, S)``.

        Returns:
            Embedded inputs of shape ``(B, S, D)`` scaled by ``sqrt(D)``.
        """
        token_embeddings = self.embed_tokens(input_ids.to(torch.int32))
        # Scale embeddings to keep activations in a reasonable range.
        return self.config.embedding_scale * token_embeddings


# =================================================================================================
__all__ = [
    "HRModelV2", "HRMState", "ModelSettings_V2", "Batch",
]  # fmt: skip

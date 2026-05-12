""" """

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]


# =================================================================================================
@dataclass(frozen=True)
class HRMInputV1:
    """Model-native input payload for one HRM v1 step."""

    schema_tokens: Tensor
    prefix_bias: Optional[Tensor] = None


@dataclass(frozen=True)
class HRMOutputV1:
    """Model-native output bundle for one HRM v1 step."""

    schema_slots: Tensor
    q_logits: Tensor


# =================================================================================================
class ModelSettings_V1(BaseModel, extra="forbid"):
    """Model-level settings composing a PFC module with embedding/LM-head parameters."""

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the core PFC model architecture.",
    )

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
    def num_schema_slots(self) -> int:
        """Compatibility alias for seq_length (MazeHard bridge adapter surface)."""
        return self.seq_length

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

    @property
    def pos_encodings(self) -> str:
        """Positional encoding mode shared by both H and L reasoning modules."""
        return self.pfc.reasoning_h.cortex.pos_encodings

    @classmethod
    def from_config(cls, path: Path) -> "ModelSettings_V1":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return ModelSettings_V1.model_validate(config_map)


# =================================================================================================
@dataclass
class HRMState(DetachMixin):
    """Container for the full recurrent HRM state."""

    pfc: PFCState


HRMStateV1 = HRMState


# =================================================================================================
class HRModelV1(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V1, *,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        if config.pos_encodings == "learned":
            self.embed_pos = nn.Embedding(config.seq_length, config.hidden_size, device=device, dtype=dtype)
        else:
            self.embed_pos = None  # Not used in RoPE mode

        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)  # fmt: skip
        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V1:
        """ """
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
        if self.embed_pos is not None:
            trunc_normal_init_(self.embed_pos.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)
        # self.pfc.reset_parameters()  # Already done when pfc is initialized

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """Create a fresh recurrent state (``ACTRolloutBackbone`` protocol)."""
        return HRMState(pfc=self.pfc.init_state(batch_size))

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """Selectively reset rows of the recurrent state (``ACTRolloutBackbone`` protocol)."""
        return HRMState(pfc=self.pfc.reset_state(state.pfc, reset_flag))

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: HRMInputV1, state: Optional[HRMState] = None,
    ) -> tuple[HRMOutputV1, HRMState]:  # fmt: skip
        """Forward pass through the HRM."""
        state = state or self.init_state(batch_size=inputs.schema_tokens.shape[0])
        x = inputs.schema_tokens  # (B, S, D) — already embedded by adapter encoder

        state_pfc, z_H, q_logits = self.pfc(x, state=state.pfc)  # z_H is (B, S+1, D)
        logits = self.lm_head(z_H[:, 1:])  # Strip CLS → (B, S, vocab_size)
        theta_cls = z_H[:, 0]  # (B, D) — CLS features used for ACT control and tracing

        new_state = HRMState(pfc=state_pfc)
        model_output = HRMOutputV1(schema_slots=z_H[:, 1:], q_logits=q_logits)
        return model_output, new_state

    def embed_input_ids(  # -----------------------------------------------------------------------
        self, input_ids: Tensor,
    ) -> Tensor:  # fmt: skip
        """Embed token ids into a scaled representation ready for recurrent reasoning.

        Two modes (controlled by ``config.pos_encodings``):

        **learned**: token + additive learned position embeddings, scaled by
            ``(1/\u221a2) * sqrt(d)`` to preserve unit variance at the residual stream
            (the factor compensates for summing two independently-initialized
            embeddings, each with per-dim variance \u2248 1/d).
        **rope** (legacy parity): token embeddings only, scaled by ``sqrt(d)``.
            Positional information is injected inside every attention operation via
            Rotary Position Embeddings; no additive position table is needed or used.
        """
        token_embeddings = self.embed_tokens(input_ids.to(torch.int32))

        if self.config.pos_encodings == "learned":
            # Learned mode: add positional table, then scale to maintain variance.
            positions = torch.arange(self.config.seq_length, device=input_ids.device)
            pos_embeddings = self.embed_pos(positions).unsqueeze(0)
            return 0.707106781 * self.config.embedding_scale * (token_embeddings + pos_embeddings)

        if self.config.pos_encodings == "rope":
            # RoPE mode: positions are encoded in QK rotation — scale by sqrt(d) only.
            return self.config.embedding_scale * token_embeddings

        raise ValueError(f"Unsupported pos_encodings mode: {self.config.pos_encodings}")


# =================================================================================================
__all__ = [
    "HRModelV1", "HRMState", "HRMStateV1", "ModelSettings_V1",
    "HRMInputV1", "HRMOutputV1",
    "Batch",
]  # fmt: skip

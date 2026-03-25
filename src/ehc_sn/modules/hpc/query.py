from __future__ import annotations

"""Query systems for HPC memory modules."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import Activation, FactorMemoryView, LinearMemoryView


# =================================================================================================
class AttractorSettings(BaseModel, extra="forbid"):
    """Settings for attractor dynamics modules."""

    kappa: float = Field(
        default=0.8,
        description="Hebbian retrieval decay term",
    )
    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for attractor dynamics.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for attractor dynamics.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for attractor dynamics.",
    )


# =================================================================================================
class AttractorNetwork(nn.Module):
    """Attractor retrieval dynamics over a linear memory view."""

    def __init__(self, config: AttractorSettings) -> None:
        super().__init__()
        self._config = config
        self._activation_fn = utils.activation_from_str(self._config.activation)

    @property
    def config(self) -> AttractorSettings:
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, query: Tensor, memory_view: LinearMemoryView, *, masks: Sequence[Tensor],
    ) -> Tensor:  # fmt: skip
        kappa = self.config.kappa
        state = self.activation(query)
        stage_masks = [mask.to(dtype=state.dtype) for mask in masks]
        if not stage_masks:
            raise ValueError("masks must contain at least one stage mask.")

        for mask in stage_masks:
            field = kappa * state + memory_view.apply(state)
            state = (1 - mask) * state + mask * self.activation(field)
        return state

    def activation(  # ----------------------------------------------------------------------------
        self, code: Tensor,
    ) -> Tensor:  # fmt: skip
        code = torch.clamp(code, min=self.config.clamp_min, max=self.config.clamp_max)
        return self._activation_fn(code)


# =================================================================================================
class AttentionSettings(BaseModel, extra="forbid"):
    """Settings for explicit factor-memory retrieval."""

    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature divisor applied after dot-product scaling.",
    )
    empty_retrieval: Literal["query", "zeros"] = Field(
        default="query",
        description="Fallback returned when no factor slots are populated.",
    )
    memory_count_scaling: Literal["none", "log_count"] = Field(
        default="log_count",
        description="Optional sharpening factor based on the number of populated factor slots.",
    )
    iterations: int = Field(
        default=1,
        ge=1,
        description="Number of factor retrieval iterations to apply.",
    )
    recurrence: Literal["none", "multiplicative"] = Field(
        default="none",
        description="Recurrence rule used after the first retrieval iteration.",
    )


# =================================================================================================
class FactorRetrieval(nn.Module):
    """Masked-softmax retrieval over explicit factor-memory slots."""

    def __init__(self, config: AttentionSettings) -> None:
        super().__init__()
        self._config = config

    @property
    def config(self) -> AttentionSettings:
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, query: Tensor, memory_view: FactorMemoryView,
    ) -> Tensor:  # fmt: skip
        logits = self.compute_logits(query, memory_view.keys)
        return self.recall_from_logits(
            logits,
            memory_view.values,
            valid_mask=memory_view.valid_mask,
            fallback_query=query,
        )

    def compute_logits(self, query: Tensor, memory_bank: Tensor) -> Tensor:
        query = query.to(dtype=memory_bank.dtype)
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        return torch.einsum("bs,bts->bt", query, memory_bank) / scale

    def weights_from_logits(  # -------------------------------------------------------------------
        self, logits: Tensor, *, valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:  # fmt: skip
        scaled_logits = logits * self._memory_count_multiplier(valid_mask)
        has_valid_slot = valid_mask.any(dim=1, keepdim=True)
        masked_logits = torch.where(valid_mask, scaled_logits, torch.full_like(scaled_logits, torch.finfo(scaled_logits.dtype).min))
        stabilized_logits = torch.where(has_valid_slot, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(stabilized_logits, dim=1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        return weights, has_valid_slot

    def read_values(self, weights: Tensor, values: Tensor) -> Tensor:
        return torch.einsum("bt,bts->bs", weights, values)

    def recall_from_logits(  # --------------------------------------------------------------------
        self, logits: Tensor, values: Tensor, *, valid_mask: Tensor, fallback_query: Tensor
    ) -> Tensor:  # fmt: skip
        weights, has_valid_slot = self.weights_from_logits(logits, valid_mask=valid_mask)
        recalled = self.read_values(weights, values)
        fallback = self._empty_fallback(fallback_query)
        return torch.where(has_valid_slot, recalled, fallback)

    def _memory_count_multiplier(self, valid_mask: Tensor) -> Tensor:
        if self.config.memory_count_scaling == "none":
            return torch.ones((valid_mask.shape[0], 1), dtype=torch.float, device=valid_mask.device)
        valid_count = valid_mask.sum(dim=1, keepdim=True).to(dtype=torch.float)
        return torch.maximum(torch.log1p(valid_count), torch.ones_like(valid_count))

    def _empty_fallback(self, query: Tensor) -> Tensor:
        if self.config.empty_retrieval == "zeros":
            return torch.zeros_like(query)
        return query


# =================================================================================================
__all__ = ["AttentionSettings", "AttractorNetwork", "AttractorSettings", "FactorRetrieval"]

from __future__ import annotations

"""Query systems for HPC memory modules."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc.query_policy import RetrievalEvidence
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
        """Initialize attractor retrieval dynamics from the provided settings."""
        super().__init__()
        self._config = config
        self._activation_fn = utils.activation_from_str(self._config.activation)

    @property
    def config(self) -> AttractorSettings:
        """Return attractor retrieval settings."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, query: Tensor, memory_view: LinearMemoryView, *, masks: Sequence[Tensor],
    ) -> Tensor:  # fmt: skip
        """Run staged attractor dynamics over the provided linear memory view."""
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
        """Clamp a code vector and apply the configured attractor activation."""
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
        """Initialize factor-memory retrieval from the provided settings."""
        super().__init__()
        self._config = config

    @property
    def config(self) -> AttentionSettings:
        """Return factor-retrieval settings."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, query: Tensor, memory_view: FactorMemoryView,
    ) -> Tensor:  # fmt: skip
        """Retrieve a value vector from factor memory for the provided query."""
        logits = self.compute_logits(query, memory_view.keys)
        return self.recall_from_logits(
            logits,
            memory_view.values,
            valid_mask=memory_view.valid_mask,
            fallback_query=query,
        )

    def recall_from_evidence(  # ------------------------------------------------------------------
        self, evidence: RetrievalEvidence, memory_view: FactorMemoryView,
    ) -> Tensor:  # fmt: skip
        """Execute one-shot or multi-pass retrieval from structured evidence."""
        fallback_query = evidence.fallback_query if evidence.fallback_query is not None else evidence.anchor_query
        if fallback_query is None:
            raise ValueError("Retrieval evidence must provide a fallback or anchor query.")

        read_bank = memory_view.bank(evidence.read_bank, fallback_to_default=True)
        if evidence.mode == "anchor_query":
            if evidence.anchor_query is None:
                raise ValueError("Anchor-query evidence requires `anchor_query`.")
            logits = self.compute_logits(evidence.anchor_query, read_bank.keys)
            return self.recall_from_logits(logits, read_bank.values, valid_mask=read_bank.valid_mask, fallback_query=fallback_query)

        if evidence.mode == "factor_logits":
            logits = evidence.composed_logits if evidence.composed_logits is not None else evidence.anchor_logits
            if logits is None:
                if evidence.anchor_query is None:
                    raise ValueError("Factor-logit evidence requires logits or an anchor query.")
                logits = self.compute_logits(evidence.anchor_query, read_bank.keys)
            return self.recall_from_logits(logits, read_bank.values, valid_mask=read_bank.valid_mask, fallback_query=fallback_query)

        if evidence.mode != "anchor_refine":
            raise ValueError(f"Unsupported retrieval evidence mode '{evidence.mode}'.")

        anchor_logits = evidence.anchor_logits
        if anchor_logits is None:
            if evidence.anchor_query is None:
                raise ValueError("Anchor-refine evidence requires `anchor_logits` or `anchor_query`.")
            anchor_logits = self.compute_logits(evidence.anchor_query, read_bank.keys)

        current_logits = anchor_logits
        recalled = self.recall_from_logits(current_logits, read_bank.values, valid_mask=read_bank.valid_mask, fallback_query=fallback_query)
        for _ in range(evidence.iterations - 1):
            refined_logits = anchor_logits
            for step in evidence.refinement_steps:
                source = self._resolve_refinement_source(step.source, recalled)
                bank = memory_view.bank(step.bank_name, fallback_to_default=True)
                bank_tensor = bank.keys if step.bank_field == "keys" else bank.values
                step_logits = self.compute_logits(source, bank_tensor)
                reference_logits = anchor_logits if step.reference == "anchor" else current_logits
                refined_logits = self._combine_logits(reference_logits, step_logits, mode=step.compose)
            current_logits = refined_logits
            recalled = self.recall_from_logits(
                current_logits, read_bank.values, valid_mask=read_bank.valid_mask, fallback_query=fallback_query
            )
        return recalled

    def compute_logits(self, query: Tensor, memory_bank: Tensor) -> Tensor:
        """Compute scaled query-key similarity logits for factor slots."""
        query = query.to(dtype=memory_bank.dtype)
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        return torch.einsum("bs,bts->bt", query, memory_bank) / scale

    def weights_from_logits(  # -------------------------------------------------------------------
        self, logits: Tensor, *, valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:  # fmt: skip
        """Convert logits into masked retrieval weights and valid-row indicators."""
        scaled_logits = logits * self._memory_count_multiplier(valid_mask)
        has_valid_slot = valid_mask.any(dim=1, keepdim=True)
        masked_logits = torch.where(valid_mask, scaled_logits, torch.full_like(scaled_logits, torch.finfo(scaled_logits.dtype).min))
        stabilized_logits = torch.where(has_valid_slot, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(stabilized_logits, dim=1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        return weights, has_valid_slot

    def read_values(self, weights: Tensor, values: Tensor) -> Tensor:
        """Read factor values with the provided slot weights."""
        return torch.einsum("bt,bts->bs", weights, values)

    def recall_from_logits(  # --------------------------------------------------------------------
        self, logits: Tensor, values: Tensor, *, valid_mask: Tensor, fallback_query: Tensor
    ) -> Tensor:  # fmt: skip
        """Read from factor memory using precomputed logits and fallback behavior."""
        weights, has_valid_slot = self.weights_from_logits(logits, valid_mask=valid_mask)
        recalled = self.read_values(weights, values)
        fallback = self._empty_fallback(fallback_query)
        return torch.where(has_valid_slot, recalled, fallback)

    def _memory_count_multiplier(self, valid_mask: Tensor) -> Tensor:
        """Return an optional sharpening multiplier based on populated slot count."""
        if self.config.memory_count_scaling == "none":
            return torch.ones((valid_mask.shape[0], 1), dtype=torch.float, device=valid_mask.device)
        valid_count = valid_mask.sum(dim=1, keepdim=True).to(dtype=torch.float)
        return torch.maximum(torch.log1p(valid_count), torch.ones_like(valid_count))

    def _combine_logits(
        self, reference_logits: Tensor, refinement_logits: Tensor, *, mode: Literal["additive", "multiplicative"]
    ) -> Tensor:
        """Combine anchor/current logits with one refinement term."""
        if mode == "additive":
            return reference_logits + refinement_logits
        return reference_logits * refinement_logits

    def _resolve_refinement_source(self, source: Literal["retrieved_value"], recalled: Tensor) -> Tensor:
        """Resolve the tensor used to compute refinement logits."""
        if source != "retrieved_value":
            raise ValueError(f"Unsupported refinement source '{source}'.")
        return recalled

    def _empty_fallback(self, query: Tensor) -> Tensor:
        """Return the configured value for rows with no populated memory slots."""
        if self.config.empty_retrieval == "zeros":
            return torch.zeros_like(query)
        return query


# =================================================================================================
__all__ = ["AttentionSettings", "AttractorNetwork", "AttractorSettings", "FactorRetrieval"]

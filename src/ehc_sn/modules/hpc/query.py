"""Retrieval systems for hippocampal memory modules.

This module implements the two retrieval backends used by the HPC package:
dense attractor dynamics over a linear memory operator and attention-style
retrieval over explicit factor-memory slots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc.query_policy import (
    PreparedCueRead,
    PreparedRead,
    PreparedTargetRead,
)
from ehc_sn.types import Activation, FactorMemoryView, LinearMemoryView


# =============================================================================
class AttractorReadSettings(BaseModel, extra="forbid"):
    """Static hyperparameters for attractor-style retrieval dynamics."""

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


# =============================================================================
class AttractorRead(nn.Module):
    """Attractor retrieval dynamics over a linear memory view.

    The module performs staged pattern completion by repeatedly applying the
    current state to a linear memory operator and mixing the result with
    stage-specific update masks.
    """

    def __init__(self, config: AttractorReadSettings) -> None:
        """Initialize attractor retrieval dynamics from the provided settings."""
        super().__init__()
        self._config = config
        self._activation_fn = utils.activation_from_str(self._config.activation)

    @property
    def config(self) -> AttractorReadSettings:
        """Return attractor retrieval settings."""
        return self._config

    def forward(  # -----------------------------------------------------------
        self,
        query: Tensor,
        memory_view: LinearMemoryView,
        *,
        masks: Sequence[Tensor],
    ) -> (
        Tensor
    ):  # -------------------------------------------------------------------------------
        """Run staged attractor dynamics over a linear memory view.

        Args:
            query: Flattened query tensor with shape ``(B, S)``.
            memory_view: Linear memory operator applied at each attractor step.
            masks: Stage masks broadcastable to ``(B, S)`` that determine which
                feature blocks update at each stage.

        Returns:
            The recalled flattened code with shape ``(B, S)``.
        """
        kappa = self.config.kappa
        state = self.activation(query)
        stage_masks = [mask.to(dtype=state.dtype) for mask in masks]
        if not stage_masks:
            raise ValueError("masks must contain at least one stage mask.")

        for mask in stage_masks:
            field = kappa * state + memory_view.apply(state)
            state = (1 - mask) * state + mask * self.activation(field)
        return state

    def activation(
        self,
        code: Tensor,
    ) -> (
        Tensor
    ):  # ----------------------------------------------------------------------------
        """Clamp a code tensor and apply the configured attractor activation."""
        code = torch.clamp(
            code, min=self.config.clamp_min, max=self.config.clamp_max
        )
        return self._activation_fn(code)


# =============================================================================
class FactorReadSettings(BaseModel, extra="forbid"):
    """Static hyperparameters for explicit factor-memory retrieval."""

    beta: float = Field(
        default=1.0,
        gt=0.0,
        description="Query-key sharpening factor applied after dot-product scaling.",
    )
    empty_retrieval: Literal["query", "zeros"] = Field(
        default="query",
        description="Fallback returned when no factor slots are populated.",
    )
    beta_scaling: Literal["none", "log_memory_count"] = Field(
        default="log_memory_count",
        description="Optional sharpening multiplier based on the number of populated factor slots.",
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
    score_compose: Literal["additive", "multiplicative"] = Field(
        default="multiplicative",
        description="How targeted-read score terms are composed before softmax.",
    )


# =============================================================================
class FactorRead(nn.Module):
    """Masked-softmax retrieval over explicit factor-memory slots.

    The reader supports both resolved cue reads and targeted source-to-target
    retrieval over named factor-memory banks.
    """

    def __init__(self, config: FactorReadSettings) -> None:
        """Initialize factor-memory retrieval from the provided settings."""
        super().__init__()
        self._config = config

    @property
    def config(self) -> FactorReadSettings:
        """Return factor-retrieval settings."""
        return self._config

    def forward(  # -----------------------------------------------------------
        self,
        query: Tensor,
        memory_view: FactorMemoryView,
    ) -> (
        Tensor
    ):  # -------------------------------------------------------------------------------
        """Retrieve a flattened value code from factor memory.

        Args:
            query: Flattened query tensor with shape ``(B, S)``.
            memory_view: Explicit factor-memory slots used for retrieval.

        Returns:
            A flattened recalled tensor with shape ``(B, S)``.
        """
        return self._recall_iterative_resolved(
            query,
            memory_view.keys,
            memory_view.values,
            valid_mask=memory_view.valid_mask,
        )

    def recall_from_evidence(
        self,
        evidence: PreparedRead,
        memory_view: FactorMemoryView,
    ) -> (
        Tensor
    ):  # ------------------------------------------------------------------
        """Execute retrieval from one prepared read-evidence payload."""
        if isinstance(evidence, PreparedTargetRead):
            return self._recall_iterative_targeted(evidence, memory_view)
        if not isinstance(evidence, PreparedCueRead):
            raise TypeError(
                f"Unsupported read evidence type '{type(evidence).__name__}'."
            )

        read_bank = memory_view.bank(
            evidence.read_bank, fallback_to_default=True
        )
        return self._recall_iterative_resolved(
            evidence.query,
            read_bank.keys,
            read_bank.values,
            valid_mask=read_bank.valid_mask,
        )

    def compute_logits(self, query: Tensor, memory_bank: Tensor) -> Tensor:
        """Compute scaled query-key similarity logits for factor slots."""
        query = query.to(dtype=memory_bank.dtype)
        scale = math.sqrt(max(query.shape[1], 1))
        return torch.einsum("bs,bts->bt", query, memory_bank) * (
            self.config.beta / scale
        )

    def weights_from_logits(
        self,
        logits: Tensor,
        *,
        valid_mask: Tensor,
    ) -> tuple[
        Tensor, Tensor
    ]:  # -------------------------------------------------------------------
        """Convert logits into masked retrieval weights and valid-row indicators."""
        scaled_logits = logits * self._memory_count_multiplier(valid_mask)
        has_valid_slot = valid_mask.any(dim=1, keepdim=True)
        masked_logits = torch.where(
            valid_mask,
            scaled_logits,
            torch.full_like(
                scaled_logits, torch.finfo(scaled_logits.dtype).min
            ),
        )
        stabilized_logits = torch.where(
            has_valid_slot, masked_logits, torch.zeros_like(masked_logits)
        )
        weights = torch.softmax(stabilized_logits, dim=1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        return weights, has_valid_slot

    def read_values(self, weights: Tensor, values: Tensor) -> Tensor:
        """Read factor values with the provided slot weights."""
        return torch.einsum("bt,bts->bs", weights, values)

    def recall_from_logits(
        self,
        logits: Tensor,
        values: Tensor,
        *,
        valid_mask: Tensor,
        fallback_query: Tensor,
    ) -> (
        Tensor
    ):  # --------------------------------------------------------------------
        """Read from factor memory using precomputed logits and fallback behavior."""
        weights, has_valid_slot = self.weights_from_logits(
            logits, valid_mask=valid_mask
        )
        recalled = self.read_values(weights, values)
        fallback = self._empty_fallback(fallback_query)
        return torch.where(has_valid_slot, recalled, fallback)

    def _recall_iterative_resolved(
        self,
        anchor_query: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        valid_mask: Tensor,
    ) -> Tensor:  # ------------------------------------------------------------
        """Execute iterative retrieval for one resolved cue query."""
        logits = self.compute_logits(anchor_query, keys)
        recalled = self.recall_from_logits(
            logits, values, valid_mask=valid_mask, fallback_query=anchor_query
        )
        for _ in range(self.config.iterations - 1):
            recurrent_query = self._recurrent_query(anchor_query, recalled)
            logits = self.compute_logits(recurrent_query, keys)
            recalled = self.recall_from_logits(
                logits,
                values,
                valid_mask=valid_mask,
                fallback_query=anchor_query,
            )
        return recalled

    def _recall_iterative_targeted(
        self, evidence: PreparedTargetRead, memory_view: FactorMemoryView
    ) -> Tensor:
        """Execute iterative targeted retrieval from source cues into one bank."""
        target_bank = memory_view.bank(
            evidence.read_bank, fallback_to_default=True
        )
        source_logits = self.compose_source_logits(
            source_queries=evidence.source_queries,
            memory_view=memory_view,
            target=evidence.target,
            target_shape=target_bank.valid_mask.shape,
        )
        first_logits = source_logits
        if evidence.initial_target_query is not None:
            initial_target_logits = self.compute_logits(
                evidence.initial_target_query, target_bank.keys
            )
            self._validate_logit_shape(
                initial_target_logits,
                target_bank.valid_mask.shape,
                label="initial target",
            )
            first_logits = self._compose_logits(
                [source_logits, initial_target_logits]
            )

        recalled = self.recall_from_logits(
            first_logits,
            target_bank.values,
            valid_mask=target_bank.valid_mask,
            fallback_query=evidence.fallback_query,
        )
        for _ in range(self.config.iterations - 1):
            target_logits = self.compute_logits(recalled, target_bank.keys)
            self._validate_logit_shape(
                target_logits,
                target_bank.valid_mask.shape,
                label="recurrent target",
            )
            recalled = self.recall_from_logits(
                self._compose_logits([source_logits, target_logits]),
                target_bank.values,
                valid_mask=target_bank.valid_mask,
                fallback_query=evidence.fallback_query,
            )
        return recalled

    def compose_source_logits(
        self,
        *,
        source_queries: dict[str, Tensor],
        memory_view: FactorMemoryView,
        target: str,
        target_shape: torch.Size,
    ) -> (
        Tensor
    ):  # -----------------------------------------------------------------
        """Return composed source logits from all non-target cue families."""
        source_logits: list[Tensor] = []
        for family, query in source_queries.items():
            if family == target:
                continue
            bank = memory_view.bank(family, fallback_to_default=True)
            logits = self.compute_logits(query, bank.keys)
            self._validate_logit_shape(
                logits, target_shape, label=f"source family {family!r}"
            )
            source_logits.append(logits)

        if not source_logits:
            raise ValueError(
                "Targeted retrieval requires at least one non-target source query."
            )
        return self._compose_logits(source_logits)

    def _compose_logits(self, score_terms: list[Tensor]) -> Tensor:
        """Compose score terms for targeted retrieval."""
        if len(score_terms) == 1:
            return score_terms[0]
        if self.config.score_compose == "additive":
            return sum((term for term in score_terms[1:]), score_terms[0])

        composed = score_terms[0]
        for term in score_terms[1:]:
            composed = composed * term
        return composed

    def _memory_count_multiplier(self, valid_mask: Tensor) -> Tensor:
        """Return the optional sharpening multiplier based on populated slot count."""
        if self.config.beta_scaling == "none":
            return torch.ones(
                (valid_mask.shape[0], 1),
                dtype=torch.float,
                device=valid_mask.device,
            )
        valid_count = valid_mask.sum(dim=1, keepdim=True).to(dtype=torch.float)
        safe_count = torch.clamp(valid_count, min=1.0)
        return torch.maximum(
            torch.log(safe_count), torch.ones_like(valid_count)
        )

    def _validate_logit_shape(
        self, logits: Tensor, target_shape: torch.Size, *, label: str
    ) -> None:
        """Validate that one logit tensor matches the target-bank slot axis."""
        expected_shape = tuple(int(dim) for dim in target_shape)
        if tuple(int(dim) for dim in logits.shape) != expected_shape:
            raise ValueError(
                f"{label} logits must match target-bank slot shape {expected_shape}, got {tuple(logits.shape)}."
            )

    def _recurrent_query(
        self, anchor_query: Tensor, recalled: Tensor
    ) -> Tensor:
        """Return the query used for the next retrieval iteration."""
        if self.config.recurrence == "none":
            return anchor_query
        return anchor_query * recalled

    def _empty_fallback(self, query: Tensor) -> Tensor:
        """Return the configured value for rows with no populated memory slots."""
        if self.config.empty_retrieval == "zeros":
            return torch.zeros_like(query)
        return query


# =============================================================================
__all__ = [
    "AttractorRead",
    "AttractorReadSettings",
    "FactorRead",
    "FactorReadSettings",
]

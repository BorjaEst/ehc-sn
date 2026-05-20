"""Shared variational-family objective abstractions.

This module defines the family layer used by objectives whose main public loss
contract is observation likelihood plus one or more named latent relations and
optional regularization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from torch import Tensor

import ehc_sn.loss.consistency as consistency_module
import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.metrics.step_metrics import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.rollouts.runtime import StepRecord
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
@dataclass(frozen=True)
class VariationalLosses(DetachMixin, ABC):
    """Shared parent loss contract for variational-family objectives."""

    @property
    @abstractmethod
    def loss_obs_nll_sum(self) -> Tensor:
        """Aggregate observation loss sum for the current step."""
        ...

    @property
    @abstractmethod
    def loss_latent_sum(self) -> Tensor:
        """Aggregate latent loss sum for the current step."""
        ...

    @property
    @abstractmethod
    def loss_reg_sum(self) -> Tensor:
        """Aggregate regularization loss sum for the current step."""
        ...

    @property
    def total(self) -> Tensor:
        """Return the total scalar loss for the current step."""
        return self.loss_obs_nll_sum + self.loss_latent_sum + self.loss_reg_sum


# =============================================================================
@dataclass(frozen=True)
class VariationalObjectiveStep:
    """A single rollout/loss step produced by a variational-family head."""

    losses: VariationalLosses
    metrics: StepMetrics
    outputs: Optional[Any] = None
    signals: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =============================================================================
@dataclass(frozen=True)
class VariationalTerms:
    """ """


# =============================================================================
@dataclass(frozen=True)
class VariationalContext:
    """Shared context for variational-family loss computation and metric extraction."""

    targets: Any
    protocol_mask: Tensor
    latent_relations: dict[str, LatentRelation]
    reg_terms: Optional[dict[str, LatentCode]] = None


# =============================================================================
@dataclass(frozen=True)
class VariationalStepOutput:
    """Raw execution output produced by a single variational-family step."""


# =============================================================================
class VariationalObjectiveBase[ConfigT](BaseObjective[ConfigT]):
    """Base class for variational-family rollout heads.

    Family-level output contracts should expose named semantic latent relations.
    Each relation is binary and compares two semantic latent-code sides. Either
    side may itself be multi-block via :class:`LatentCode`.
    """

    @property
    def obs_loss_fn(self) -> Any:
        """Return the configured observation loss primitive."""
        return getattr(cross_entropy_module, self.config.observation_loss)

    @property
    def latent_term_fn(self) -> Any:
        """Return the configured latent consistency primitive."""
        return getattr(consistency_module, self.config.latent_loss)

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> Any:
        """Score one executed variational-family step."""
        step_output = record.outputs  # original controller output
        outputs = getattr(step_output, "backbone_output", step_output)

        # --- single scored-step computation ---
        context = self.build_context(record, outputs, **options)
        terms = self.compute_terms(outputs, context, **options)
        losses = self.compute_losses(terms, context, **options)

        # --- metrics from precomputed losses/terms ---
        metrics = self.evaluate_metrics(
            record, outputs, context, terms, losses, **options
        )

        # --- signals from precomputed losses/context (no re-extraction) ---
        signals = self.compute_signals(
            record, outputs, context, terms, losses, **options
        )

        return self.build_output(losses, metrics, signals, outputs)

    def build_context(  # -----------------------------------------------------
        self,
        record: StepRecord,
        outputs: VariationalStepOutput,
        **options: Any,
    ) -> VariationalContext:
        """Extract a variational context from the current step."""
        raise NotImplementedError

    def compute_terms(  # -----------------------------------------------------
        self,
        outputs: VariationalStepOutput,
        context: VariationalContext,
        **options: Any,
    ) -> VariationalTerms:
        """Extract variational-family loss terms from the current step."""
        raise NotImplementedError

    def compute_losses(  # ----------------------------------------------------
        self,
        terms: VariationalTerms,
        context: VariationalContext,
        **options: Any,
    ) -> VariationalLosses:
        """Return variational-family losses for the current step."""
        raise NotImplementedError

    def evaluate_metrics(  # --------------------------------------------------
        self,
        record: StepRecord,
        outputs: VariationalStepOutput,
        context: VariationalContext,
        terms: VariationalTerms,
        losses: VariationalLosses,
        **options: Any,
    ) -> StepMetrics:
        """Return variational-family metrics for the current step."""
        raise NotImplementedError

    def compute_signals(  # ---------------------------------------------------
        self,
        record: StepRecord,
        outputs: VariationalStepOutput,
        context: VariationalContext,
        terms: VariationalTerms,
        losses: VariationalLosses,
        **options: Any,
    ) -> dict[str, Tensor]:
        """Return detached generic variational-family diagnostic signals."""
        raise NotImplementedError

    def build_output(  # ------------------------------------------------------
        self,
        losses: VariationalLosses,
        metrics: StepMetrics,
        signals: dict[str, Tensor],
        outputs: VariationalStepOutput,
    ) -> Any:
        """Wrap losses, metrics, and signals into the concrete step-output type."""
        raise NotImplementedError


# =============================================================================
def require_latent_relation(  # -----------------------------------------------
    relations: dict[str, LatentRelation],
    key: str,
) -> LatentRelation:
    """Return the relation for *key*, raising ``KeyError`` with context if absent."""
    try:
        return relations[key]
    except KeyError:
        available = ", ".join(sorted(relations.keys()))
        raise KeyError(
            f"Latent relation {key!r} not found. Available: {available}"
        ) from None


# =============================================================================
def get_reg_term(  # ----------------------------------------------------------
    reg_terms: dict[str, LatentCode] | None,
    latent_relations: dict[str, LatentRelation],
    *,
    key: str,
    fallback: str,
) -> LatentCode:
    """Return the explicit reg code for *key* or fall back to the lhs of *fallback*."""
    if reg_terms is not None and key in reg_terms:
        return reg_terms[key]
    fallback_relation = require_latent_relation(latent_relations, fallback)
    return fallback_relation.lhs


# =============================================================================
def build_variational_step_metrics(  # ----------------------------------------
    extras: Mapping[str, RatioStat],
) -> StepMetrics:
    """Build a :class:`StepMetrics` from a variational-family extras mapping."""
    first = next(iter(extras.values()), None)
    zero = (
        first.numerator_sum.new_zeros(())
        if first is not None
        else torch.zeros(())
    )
    return StepMetrics(
        episode=RolloutAgg(zero, zero, zero, zero, zero),
        episode_tokens=TokenAgg(zero, zero),
        step=TransitionAgg(zero, zero, zero, zero, zero),
        step_tokens=TokenAgg(zero, zero),
        extras=extras,
    )


# =============================================================================
__all__ = [
    "VariationalLosses",
    "VariationalObjectiveBase",
    "VariationalObjectiveStep",
    "VariationalContext",
    "build_variational_step_metrics",
    "get_reg_term",
    "require_latent_relation",
]

"""Halt-target construction policies for ACT deliberation.

Each class in this module implements one policy for converting task evidence
into a halt-target tensor consumed by :class:`~ehc_sn.objectives.control.halt.HaltClassificationObjective`.

These are plain classes (not ``nn.Module``) — they own no parameters, buffers,
or configurable submodules.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.deliberation.act import (
    ACTController,
    collapse_act_halt_continue_logits,
)
from ehc_sn.rollouts.runtime import StepRecord


# =============================================================================
class HaltTargetBuilder(Protocol):
    """Protocol for building halt-target tensors from task evidence.

    Conforming types return a *detached* tensor in ``[0, 1]`` indicating
    whether the current step output is acceptable (1.0 = halt).
    """

    def build(
        self,
        *,
        pred_field: Tensor,
        target_field: Tensor,
        node_mask: Tensor,
    ) -> Tensor: ...


# =============================================================================
class ExactMatchHaltTarget:
    """Halt target = 1.0 when all supervised tokens are predicted correctly.

    Accepts ``seq_is_correct`` directly as a :class:`Tensor`.
    """

    @staticmethod
    def build(
        *,
        seq_is_correct: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Build halt-target tensor from per-sequence correctness.

        Args:
            seq_is_correct: Bool tensor, shape ``(B,)``, True for
                exact-match sequences.
            **kwargs: Ignored (protocol compat with field builders).

        Returns:
            Float tensor, shape ``(B,)``, 1.0 for exact-match sequences,
            0.0 otherwise.
        """
        return seq_is_correct.to(dtype=torch.float32)


# =============================================================================
class FieldQualityHaltTargetConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`FieldQualityHaltTarget`.

    The halt target is computed as::

        target = sigmoid((acceptable_error - mse) / temperature)

    Attributes:
        acceptable_error: MSE threshold around which the sigmoid midpoint
            is centered.  Default ``0.1`` matches the original hardcoded
            ``offset = 1.0`` with ``sharpness = 10.0``.
        temperature: Softness of the sigmoid transition.  Smaller values
            produce a sharper decision boundary.
    """

    acceptable_error: float = Field(
        default=0.1,
        gt=0.0,
        description="MSE at sigmoid midpoint ``(target = 0.5)``.",
    )
    temperature: float = Field(
        default=0.1,
        gt=0.0,
        description="Sigmoid temperature; smaller = sharper boundary.",
    )


# =============================================================================
class FieldQualityHaltTarget:
    """Halt target derived from continuous-field prediction quality.

    Uses sigmoid-squashed negative MSE::

        target = sigmoid((acceptable_error - mse) / temperature)

    where ``acceptable_error`` and ``temperature`` are set via
    :class:`FieldQualityHaltTargetConfig`.
    """

    def __init__(
        self,
        config: FieldQualityHaltTargetConfig | None = None,
    ) -> None:
        """Create a field-quality halt target builder.

        Args:
            config: Configuration, or ``None`` for defaults.
        """
        if config is None:
            config = FieldQualityHaltTargetConfig()
        self._acceptable_error = config.acceptable_error
        self._temperature = config.temperature

    def build(
        self,
        *,
        pred_field: Tensor,
        target_field: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Build halt-target tensor from field prediction quality.

        Args:
            pred_field: Predicted firing field, shape ``(B, N)``.
            target_field: Target firing field, shape ``(B, N)``.
            node_mask: Valid-node mask, shape ``(B, N)`` bool.

        Returns:
            Float tensor, shape ``(B,)`` in ``[0, 1]``, higher for
            better field predictions.  Always detached — no gradients
            flow through the target.
        """
        pred_field = pred_field.detach()
        target_field = target_field.detach()
        node_mask = node_mask.detach()
        diff = pred_field - target_field
        squared = diff**2
        valid_count = node_mask.sum(dim=1).clamp(min=1)
        mse = (squared * node_mask).sum(dim=1) / valid_count
        return torch.sigmoid(
            (self._acceptable_error - mse) / self._temperature
        ).detach()


# =============================================================================
class ACTTDBootstrapTarget:
    """TD bootstrap target for ACT deliberation.

    Stepping the backbone (online or target-network) on the next-step
    observations to produce ``halt_logit`` and ``continue_logit``,
    then returns ``sigmoid(max(halt, continue))``, gated at the last step.
    """

    @staticmethod
    def build(
        controller: ACTController,
        record: StepRecord,
        *,
        target_backbone: Any | None = None,
    ) -> Tensor:
        """Compute the TD bootstrap target from executed carry state.

        When *target_backbone* is provided, a fresh target model state is
        derived by cloning the online model state from the record, resetting it
        with the same post-rollout halted mask, and stepping the target backbone
        on the *same executed inputs* (``record.carry.data``) that the online
        backbone consumed.

        When *target_backbone* is *None*, falls back to the online controller's
        backbone (current self-bootstrap behaviour).

        Args:
            controller: The online ACT controller (used as fallback when
                ``target_backbone`` is *None*).
            record: The executed step record containing carry/model_state.
            target_backbone: Optional frozen lagged backbone.  Must expose
                ``init_state(batch_size)``, ``reset_state(flag, state)``, and a
                forward call ``(data, state) -> (output, next_state)``.  If
                *None*, uses ``controller.backbone``.

        Returns:
            Sigmoid-squashed TD bootstrap target tensor of shape ``(B,)``.
        """
        data = record.carry.data
        model_state = record.carry.model_state
        steps = record.carry.steps
        if data is None or model_state is None or steps is None:
            raise ValueError(
                "ACT TD target requires carry.data, carry.model_state, and "
                "carry.steps."
            )

        if target_backbone is not None:
            with torch.no_grad():
                batch_size = int(
                    next(iter(data.values())).shape[0]
                    if isinstance(data, dict)
                    else model_state.steps.shape[0]
                )
                online_halted = record.carry.halted
                target_state = model_state.detach()
                target_state = target_backbone.target.reset_state(
                    online_halted, target_state
                )
                backbone_output, _ = target_backbone(data, target_state)
                next_q = backbone_output.control.action_logits
        else:
            with torch.no_grad():
                backbone_output, _ = controller.backbone(data, model_state)
                next_q = backbone_output.control.action_logits

        done_action = controller.config.done_action
        max_halt_steps = controller.config.max_halt_steps
        scores = collapse_act_halt_continue_logits(
            next_q, done_action=done_action
        )
        is_last_step = steps >= max_halt_steps
        target_q = torch.where(
            is_last_step,
            scores.halt_logit,
            torch.maximum(scores.halt_logit, scores.continue_logit),
        )
        return torch.sigmoid(target_q)


# =============================================================================
__all__ = [
    "ACTTDBootstrapTarget",
    "ExactMatchHaltTarget",
    "FieldQualityHaltTarget",
    "FieldQualityHaltTargetConfig",
    "HaltTargetBuilder",
]

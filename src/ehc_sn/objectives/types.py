"""Canonical objective output contract.

:class:`ObjectiveResult` is the single return type for all objective
forward calls. It carries the scalar backprop loss, named reduced loss
components, and optional unreduced per-element terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


# =============================================================================
@dataclass(frozen=True)
class ObjectiveResult:
    """Differentiable output from a single objective forward call.

    Attributes:
        loss: Scalar tensor for ``.backward()``.
        losses: Named reduced loss components (summed or averaged over batch).
        terms: Optional unreduced per-element tensors (for metrics or
            composition).  Defaults to ``{}``.
    """

    loss: Tensor
    losses: dict[str, Tensor]
    terms: dict[str, Tensor] = field(default_factory=dict)


# =============================================================================
__all__ = ["ObjectiveResult"]

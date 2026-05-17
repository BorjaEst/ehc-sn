"""Detaching utilities for dataclass-based state containers.

This module provides two building blocks that help reduce boilerplate in
dataclass-heavy PyTorch codebases:

- :func:`detach_any`: recursively detaches tensors contained in common Python
    containers (``dict``, ``list``, ``tuple``, ``set``) and in nested dataclasses.
- :class:`DetachMixin`: a dataclass mixin that implements a typed ``detach()``
    method returning ``Self``.

Typical use case
----------------

Many of the project state containers (e.g. rollout/loop carry states) are
dataclasses holding tensors, nested state objects, and dict buffers.
Implementing ``detach()`` field-by-field is repetitive and easy to get wrong.
Instead, inherit from :class:`DetachMixin`.

Safety notes
------------

- This is intended for *state* objects, not modules. The helper avoids calling
    ``detach()`` on :class:`torch.nn.Module` instances.
- Do not store :class:`torch.nn.Parameter` in state dataclasses. Detaching a
    Parameter returns a Tensor and will silently drop parameter semantics.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Protocol, Self, TypeGuard, cast, runtime_checkable

import torch
from torch import nn


# =============================================================================
@runtime_checkable
class SupportsDetach(Protocol):
    """Protocol for objects that expose a ``detach()`` method.

    We use this with :func:`typing.TypeGuard` to make the "duck-typed detach"
    branch in :func:`detach_any` acceptable to type checkers.
    """

    def detach(self) -> Any:
        """Return a detached copy/variant of the object."""
        ...


# =============================================================================
def _supports_detach(  # ------------------------------------------------------
    x: object,
) -> TypeGuard[SupportsDetach]:
    """Return True if *x* looks like it supports ``detach()``.

    This is intentionally structural (duck-typing). It is used solely to narrow
    types for static analysis; runtime behaviour is handled by try/except in
    :func:`detach_any`.
    """
    return hasattr(x, "detach") and callable(getattr(x, "detach"))


# =============================================================================
def detach_any(  # ------------------------------------------------------------
    x: Any,
) -> Any:
    """Recursively detach tensors inside *x*.

    Supported inputs:

    - :class:`torch.Tensor` → returns ``x.detach()``.
    - Dataclass instances → returns ``dataclasses.replace(x, ...)`` with all
        fields processed recursively.
    - ``dict`` / ``list`` / ``tuple`` / ``set`` → container preserved with values
        processed recursively.
    - Objects that provide ``detach()`` (duck-typed) → calls ``detach()`` unless
        the object is a :class:`torch.nn.Module`.

    Anything else is returned unchanged.

    This function intentionally returns ``Any``: once recursion through nested
    containers is allowed, preserving precise static types becomes cumbersome.
    Use :class:`DetachMixin` when you want a strongly-typed ``detach()``.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()

    # Typed “duck-typing”: narrows x -> SupportsDetach for type checkers
    if _supports_detach(x) and not isinstance(x, nn.Module):
        try:
            return x.detach()
        except TypeError:
            pass

    if isinstance(x, dict):
        return {k: detach_any(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(detach_any(v) for v in x)
    if isinstance(x, list):
        return [detach_any(v) for v in x]
    if isinstance(x, set):
        return {detach_any(v) for v in x}

    if is_dataclass(x):
        updates = {f.name: detach_any(getattr(x, f.name)) for f in fields(x)}
        return replace(x, **updates)

    return x


# =============================================================================
class DetachMixin:
    """Dataclass mixin providing a typed ``detach()`` method. The method
    returns a copy of the dataclass with all tensor-like fields detached.
    """

    def detach(  # ------------------------------------------------------------
        self: Self,
    ) -> Self:
        """Return a copy of this dataclass with all tensor-like fields detached.

        Raises:
            TypeError: If the mixin is used on a non-dataclass type.
        """
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} must be a dataclass to use DetachMixin",
            )
        updates = {
            f.name: detach_any(getattr(self, f.name)) for f in fields(self)
        }
        return cast(Self, replace(self, **updates))


# =============================================================================
__all__ = ["DetachMixin", "SupportsDetach", "detach_any"]

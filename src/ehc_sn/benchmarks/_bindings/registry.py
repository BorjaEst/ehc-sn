"""Explicit registry for internal benchmark bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

BindingFactory: TypeAlias = Callable[..., Any]

_BINDINGS: dict[tuple[str, str], BindingFactory] = {}


def register_binding(*, model_family: str, capability: str, factory: BindingFactory) -> None:
    """Register one benchmark binding factory for ``(model_family, capability)``."""
    _BINDINGS[(model_family, capability)] = factory


def resolve_binding(*, model_family: str, capability: str) -> BindingFactory:
    """Return the registered binding factory for ``(model_family, capability)``."""
    key = (model_family, capability)
    try:
        return _BINDINGS[key]
    except KeyError as exc:
        raise LookupError(f"No benchmark binding is registered for model_family={model_family!r}, capability={capability!r}.") from exc


__all__ = ["BindingFactory", "register_binding", "resolve_binding"]

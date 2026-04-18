"""Deterministic bootstrap for internal benchmark-binding registration."""

from __future__ import annotations

import warnings
from importlib import import_module

_BOOTSTRAPPED = False

_BINDING_MODULES = (
    "ehc_sn.benchmarks._bindings.hrm",
    "ehc_sn.benchmarks._bindings.tem",
)


def bootstrap_bindings() -> None:
    """Import binding family registration modules exactly once.

    Import errors for individual binding families are caught and reported as
    warnings so that a broken binding family does not prevent other families
    from loading.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    for module in _BINDING_MODULES:
        try:
            import_module(module)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Benchmark binding family '{module}' could not be loaded: {exc}",
                stacklevel=2,
            )

    _BOOTSTRAPPED = True


__all__ = ["bootstrap_bindings"]

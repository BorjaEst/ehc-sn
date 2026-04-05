"""Deterministic bootstrap for internal benchmark-binding registration."""

from __future__ import annotations

from importlib import import_module

_BOOTSTRAPPED = False


def bootstrap_bindings() -> None:
    """Import binding family registration modules exactly once."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    import_module("ehc_sn.benchmarks._bindings.hrm")
    import_module("ehc_sn.benchmarks._bindings.tem")
    _BOOTSTRAPPED = True


__all__ = ["bootstrap_bindings"]

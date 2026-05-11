"""Adapter package namespace.

This package only exports task-family subpackages. Canonical public adapter
symbols live in family barrels such as ``ehc_sn.adapters.arena.tem`` and
``ehc_sn.adapters.mazehard.hrm``.
"""

from . import arena, mazehard

__all__ = ["arena", "mazehard"]

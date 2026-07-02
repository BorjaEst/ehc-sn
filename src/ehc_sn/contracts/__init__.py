# SPDX-License-Identifier: MIT
"""Cross-layer boundary contracts for controller–task and evaluation–trace interaction.

Packages in ``controllers/``, ``tasks/``, ``evaluation/``, and ``traces/`` depend
on this package; it must not import from any of them.
"""

from ehc_sn.contracts.dependencies import (
    Dependency,
    DependencyKind,
    model_view,
    record_field,
    run_metadata,
)

__all__ = [
    "Dependency",
    "DependencyKind",
    "model_view",
    "record_field",
    "run_metadata",
]

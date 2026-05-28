"""Diagnostic trace policy and bounded buffers for Lightning training surfaces.

A module declares its diagnostic trace contract through
:class:`DiagnosticTraceSpec`.  Callbacks may read or update this spec during
``setup()`` — for example, ``FigureGenerationCallback.setup()`` derives
required trace keys from configured bounded-trace figures and enables capture.
Callbacks should merge keys rather than overwrite.

This module has no ``lightning`` import dependency — it is a plain data
dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
@dataclass(frozen=True)
class DiagnosticTraceSpec:
    """Module-owned diagnostic trace contract.

    Typically set during ``__init__`` from module config with safe defaults.
    Callbacks such as ``FigureGenerationCallback.setup()`` may update this
    spec during Lightning's ``setup()`` phase to derive required trace keys
    from configured figures (merge, not overwrite).

    Attributes:
        enabled: Master switch; when ``False`` the module captures no
            diagnostic traces during validation.
        max_batches: Maximum number of validation batches for which traces
            are captured.  Must be ``>= 1`` when ``enabled=True``.
        keys: Semantic trace keys to request from the trace backend.
            Each must be a key the module's trace backend can produce.
    """

    enabled: bool = False
    max_batches: int = field(default=2, metadata={"ge": 1})
    keys: tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
__all__ = ["DiagnosticTraceSpec"]

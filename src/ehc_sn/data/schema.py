"""Generic topology metadata helpers for versioned dataset roots.

Topology-kind string constants identify the geometry of a processed root.
Family-owned substrate modules (grid2d, numberline, …) own channel constants
and per-sample validators for their respective topologies.

See ``spec/spec-data-contracts.md`` for the full topology classification.
"""

from __future__ import annotations

TOPOLOGY_KIND_GRID2D: str = "grid2d"
"""Topology kind for 2-D grid substrates (dungeongen, maze-nd)."""

TOPOLOGY_KIND_LINE1D: str = "line1d"
"""Topology kind for 1-D number-line substrates (numberline)."""

# =================================================================================================
__all__ = [
    "TOPOLOGY_KIND_GRID2D",
    "TOPOLOGY_KIND_LINE1D",
]

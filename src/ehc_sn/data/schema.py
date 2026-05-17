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

# Shared grid2d channel name constants (re-exported here for cross-module convenience)
CHANNEL_TOPOLOGY: str = "topology"
CHANNEL_OBSERVATIONS: str = "observations"
CHANNEL_MASK_VALID: str = "mask_valid"
CHANNEL_REGIONS: str = "regions"
CHANNEL_LANDMARKS: str = "landmarks"

# Dungeongen-specific channel name constants
CHANNEL_GOALS: str = "goals"
CHANNEL_START: str = "start"
CHANNEL_SOLUTION: str = "solution"

# Integer token constants
O_ID: int = 5
"""Solution-path overlay token ID.

Used by HRM tokenization to mark solution-path cells in the label grid.
Canonical vocab layout: 0-4 are the five structural SEM IDs; 5 (O_ID) is the
solution-path token. Matches ``vocab_size=6`` in HRM model configs.
"""

# =============================================================================
__all__ = [
    "TOPOLOGY_KIND_GRID2D",
    "TOPOLOGY_KIND_LINE1D",
    "CHANNEL_TOPOLOGY",
    "CHANNEL_OBSERVATIONS",
    "CHANNEL_MASK_VALID",
    "CHANNEL_REGIONS",
    "CHANNEL_LANDMARKS",
    "CHANNEL_GOALS",
    "CHANNEL_START",
    "CHANNEL_SOLUTION",
    "O_ID",
]

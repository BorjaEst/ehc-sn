"""Compatibility surface for concrete model-comparison benchmark bindings.

Internal ownership is split across protocol-specific modules under this
private package.
"""

from __future__ import annotations

from ._arena import SharedArenaReplayModelComparisonBinding
from ._mazehard import SharedMazeHardModelComparisonBinding

# =============================================================================
__all__ = [
    "SharedArenaReplayModelComparisonBinding",
    "SharedMazeHardModelComparisonBinding",
]

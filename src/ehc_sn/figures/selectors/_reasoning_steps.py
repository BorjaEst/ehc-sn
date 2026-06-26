"""Shared reasoning-step selection logic for ``prediction_reasoning_*`` figures.

Selects which deliberation steps to display in the 2×8 snapshot mosaic,
supporting variable-length traces via hybrid sampling (first 4, evenly
spaced middle, last 4).
"""

from __future__ import annotations

import numpy as np


def select_reasoning_snapshots(
    t_halt: int,
    max_snapshots: int = 16,
) -> list[int]:
    """Select step indices for the snapshot mosaic.

    * If ``t_halt + 1 <= max_snapshots`` — include all steps.
    * If ``t_halt + 1 > max_snapshots`` — hybrid sampling:
      first 4, evenly spaced middle, last 4.  The halt step is always
      included.

    Args:
        t_halt: The halt step index (inclusive).  May be the last
            available step if the trace never halted.
        max_snapshots: Maximum number of mosaic slots.

    Returns:
        Sorted list of distinct step indices.  Length is at most
        ``max_snapshots`` and at least 1.
    """
    if t_halt < 0:
        return [0]
    if t_halt + 1 <= max_snapshots:
        return list(range(t_halt + 1))

    # Hybrid sampling: first 4, evenly spaced middle, last 4
    first_n = 4
    last_n = 4
    middle_count = max_snapshots - first_n - last_n

    first = list(range(first_n))
    last = list(range(t_halt - last_n + 1, t_halt + 1))

    if middle_count > 0:
        middle_raw = np.linspace(
            first_n, t_halt - last_n, num=middle_count + 2, dtype=int
        )[1:-1]
        middle = sorted({int(idx) for idx in middle_raw})
    else:
        middle = []

    combined = sorted(set(first + middle + last))
    if t_halt not in combined:
        combined.append(t_halt)
        combined.sort()
    return combined[:max_snapshots]


__all__ = ["select_reasoning_snapshots"]

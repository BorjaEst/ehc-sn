"""Task family registry.

Canonical task families:

- :mod:`ehc_sn.tasks.arena` — stepwise teacher-forced replay for
  structural-knowledge (B1-style) benchmark claims.
- :mod:`ehc_sn.tasks.dungeon` — reward-first online control for B1-B3-style
  benchmark claims.
"""

from ehc_sn.tasks import arena, dungeon

__all__ = [
    "arena",
    "dungeon",
]

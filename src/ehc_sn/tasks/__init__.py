"""Task family registry.

Canonical task families defined by their semantic domain:

- :mod:`ehc_sn.tasks.arena` — structural navigation: agent moves through a
  maze world; task owns observation/action ontology, revisit semantics, and
  additive structural score.
- :mod:`ehc_sn.tasks.dungeon` — goal-directed navigation: agent reaches goals
  in a dungeon world; task owns observation/action ontology, episode semantics,
  and episode score.
- :mod:`ehc_sn.tasks.mazehard` — batch token prediction over full-maze token
  sequences; task owns sequence evaluation and aggregate benchmark score.

Execution-mode bindings (replay, online control, deliberation) live in
``<task>.modes.*`` sub-packages when implemented for that task family.
"""

from ehc_sn.tasks import arena, dungeon, mazehard

__all__ = [
    "arena",
    "dungeon",
    "mazehard",
]

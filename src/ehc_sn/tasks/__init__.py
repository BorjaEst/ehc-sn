"""Task family registry.

Canonical task families defined by their semantic domain:

- :mod:`ehp_sn.tasks.arena` — structural navigation: agent moves through a
  maze world; task owns observation/action ontology, revisit semantics, and
  additive structural score.
- :mod:`ehp_sn.tasks.mazehard` — batch token prediction over full-maze token
  sequences; task owns sequence evaluation and aggregate benchmark score.
- :mod:`ehp_sn.tasks.goaltrace` — goal-conditioned prospective field prediction;
  task owns field contracts, evaluation, and oracle-based corpus builder.
- :mod:`ehp_sn.tasks.routebind` — goal-conditioned spatial prospective-field
  prediction; task owns two-field output contracts, product-state oracle, and
  dual-substrate corpus builder.

Execution-binding capabilities (replay, online control, deliberation) live in
``<task>.capabilities.*`` sub-packages when implemented for that task family.
"""

from ehc_sn.tasks import (
    arena,
    goaltrace,
    mazehard,
    routebind,
    seqmaze,
)

__all__ = [
    "arena",
    "goaltrace",
    "mazehard",
    "routebind",
    "seqmaze",
]

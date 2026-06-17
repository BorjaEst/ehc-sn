"""Explicit experiment compositions — one module per (task, family, version).

Each module exports a ``build_*_model`` function that wires concrete
model, adapter, controller, objective, and metrics classes into a
Lightning regime module.  These are the canonical entry points for
training scripts and evaluation tooling.

Sub-packages:

- :mod:`~ehc_sn.experiments.arena` — Arena spatial navigation experiments.
- :mod:`~ehc_sn.experiments.mazehard` — MazeHard deliberation experiments.
- (future) :mod:`~ehc_sn.experiments.seqmaze` — SeqMaze probe experiments.
"""

__all__: list[str] = []

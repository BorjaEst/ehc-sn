"""Lightning training surfaces for EHP-SN.

Sub-packages (organized by runtime concern):

- :mod:`~ehc_sn.lightning.modules` — Lightning modules by training regime
  (variational replay, ACT, actor-critic, hybrid).
- :mod:`~ehc_sn.lightning.diagnostics` — shared diagnostic trace spec.

Related top-level modules:

- :mod:`~ehc_sn.evaluation` — shared evaluation regime contracts and runner.
- :mod:`~ehc_sn.experiments` — explicit (task, family, version) compositions
  that wire concrete components into regime modules.
- :mod:`~ehc_sn.training` — framework-agnostic training helpers.

Training surfaces are wired by entry points in ``scripts/training/`` and are not
re-exported from this package init.  Import directly from the sub-packages.
"""

__all__: list[str] = []

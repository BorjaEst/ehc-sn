"""Lightning training surfaces for EHC-SN.

Sub-packages:

- :mod:`~ehc_sn.lightning.tem` — TEM v1 and TEM v2 Lightning training surfaces.
- :mod:`~ehc_sn.lightning.ehc` — EHC v1 Lightning training surface.
- :mod:`~ehc_sn.lightning.hrm` — HRM v1 and HRM v2 Lightning training surfaces.

Related top-level package:

- :mod:`~ehc_sn.eval` — shared evaluation regime contracts and runner.

Training surfaces are wired by entry points in ``scripts/training/`` and are not
re-exported from this package init.  Import directly from the sub-packages.
"""

__all__: list[str] = []

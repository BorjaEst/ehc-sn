"""Data surface — module namespaces only.

Public sub-namespaces:

- :mod:`~ehc_sn.data.schema` — dataset schema types.
- :mod:`~ehc_sn.data.manifest` — manifest read/write helpers.
- :mod:`~ehc_sn.data.index` — dataset index types and I/O.
- :mod:`~ehc_sn.data.datasets` — PyTorch dataset wrappers.
- :mod:`~ehc_sn.data.datamodules` — Lightning datamodules.
- :mod:`~ehc_sn.data.transforms` — data transforms.
- :mod:`~ehc_sn.data.build` — versioned-root lifecycle mechanics.
- :mod:`~ehc_sn.data.substrate` — shared-substrate family builders.
"""

from ehc_sn.data import datamodules, datasets, index, lifecycle, manifest, schema, substrate, transforms

__all__ = ["schema", "manifest", "index", "datasets", "datamodules", "transforms", "lifecycle", "substrate"]

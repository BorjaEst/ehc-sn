"""Data surface — module namespaces only.

Public sub-namespaces:

- :mod:`~ehc_sn.data.schema` — topology-kind identifier constants.
- :mod:`~ehc_sn.data.manifest` — manifest read/write helpers.
- :mod:`~ehc_sn.data.index` — dataset index types and I/O.
- :mod:`~ehc_sn.data.datasets` — PyTorch dataset wrappers.
- :mod:`~ehc_sn.data.datamodules` — Lightning datamodules.
- :mod:`~ehc_sn.data.transforms` — data transforms.
- :mod:`~ehc_sn.data.build` — versioned-root build mechanics.
- :mod:`~ehc_sn.data.diagnostics` — dataset-level audits and diagnostics.
- :mod:`~ehc_sn.data.episode_sources` — demand-driven shuffled episode providers.
- :mod:`~ehc_sn.data.substrate` — shared-substrate family builders.
"""

from ehc_sn.data import (
    datamodules,
    datasets,
    diagnostics,
    episode_sources,
    index,
    lifecycle,
    manifest,
    schema,
    substrate,
    transforms,
)

__all__ = [
    "schema",
    "manifest",
    "index",
    "datasets",
    "datamodules",
    "transforms",
    "lifecycle",
    "substrate",
    "diagnostics",
    "episode_sources",
]

"""Source-spec generation for synthetic/openfield layout sources.

Public surface:

- :func:`generate_openfield_source_specs` — write per-split topology specs.
- :func:`expand_openfield_source_specs` — read specs → topology-only layouts.
"""

from ehc_sn.data.source.openfield import (
    expand_openfield_source_specs,
    generate_openfield_source_specs,
)

__all__ = [
    "expand_openfield_source_specs",
    "generate_openfield_source_specs",
]

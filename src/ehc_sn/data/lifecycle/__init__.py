"""Version-root lifecycle mechanics for canonical versioned dataset roots.

Public surface:

- :func:`extract_version` — derive version integer from path leaf.
- :func:`staging_root` — transactional materialization context manager.
- :func:`create_version_root` — create an immutable version-leaf directory.
- :func:`write_split` — stack, validate, and write one dataset split.
- :func:`write_index_at_root` — write ``index.jsonl`` at the root.
- :func:`validate_version_root` — full validation of a versioned root.
"""

from ehc_sn.data.lifecycle._validate import validate_version_root
from ehc_sn.data.lifecycle._write import create_version_root, extract_version, staging_root, write_index_at_root, write_split

__all__ = [
    "extract_version",
    "staging_root",
    "create_version_root",
    "write_split",
    "write_index_at_root",
    "validate_version_root",
]

"""Root manifest helpers for versioned dataset roots.

Every versioned dataset root under ``data/processed/`` must have a
``manifest.json`` at its root.  This module provides canonical helpers for
writing and reading that manifest.

The manifest is the authoritative descriptor for a dataset version.  It is
written once at build time and never modified afterwards.

All identity-bearing fields are explicit top-level keys.  There is no
free-form ``lineage`` blob.  Timestamps, machine-local paths, and
audit-only metadata must not appear in ``manifest.json``.

See ``spec/spec-data-contracts.md`` §4 for the full manifest field contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_FILENAME: str = "manifest.json"
"""Canonical manifest filename at each versioned dataset root."""

SCHEMA_VERSION: int = 1
"""Current manifest schema version."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_producer_revision() -> str:
    """Return the package version string from the VERSION file."""
    version_file = Path(__file__).parent.parent / "VERSION"
    try:
        return version_file.read_text().strip()
    except OSError:
        return "unknown"


def _fingerprint(stage_params: dict[str, Any]) -> str:
    """Return a stable 16-hex-char SHA-256 fingerprint of *stage_params*."""
    serialized = json.dumps(stage_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# =============================================================================
def write_manifest(
    version_root: Path,
    *,
    dataset_class: str,
    family: str,
    version: int,
    channels: list[str],
    shape: tuple[int, int],
    n_samples: dict[str, int],
    source_id: str,
    builder: str,
    seed: int,
    normalization_version: int = 1,
    shared_schema_version: int = 1,
    stage_params: dict[str, Any] | None = None,
    source_revision: str | None = None,
    # task-corpus-only fields
    task: str | None = None,
    corpus: str | None = None,
    parent_substrate: str | None = None,
    parent_family: str | None = None,
    parent_version: int | None = None,
    task_schema_version: int | None = None,
    task_protocol_version: int | None = None,
) -> None:
    """Write the authoritative ``manifest.json`` to *version_root*.

    All identity fields are explicit; there is no free-form lineage blob.
    Timestamps and machine-local paths must not be passed.

    ``producer_revision`` and ``input_fingerprint`` are computed
    automatically from the installed package version and *stage_params*.

    Args:
        version_root: Versioned dataset root directory.
        dataset_class: ``"shared_substrate"`` or ``"task_corpus"``.
        family: Shared family name (e.g. ``"maze-nd"``) or task namespace.
        version: Version integer.
        channels: Channel names present in every sample.
        shape: Common spatial grid shape ``(H, W)``.
        n_samples: Mapping from split name to sample count.
        source_id: Stable upstream source identifier.
        builder: Qualified builder function name.
        seed: Deterministic base seed.
        normalization_version: Normalization scheme version (default ``1``).
        shared_schema_version: Shared-schema version (default ``1``).
        stage_params: Deterministic build parameter bundle.  Used to compute
            ``input_fingerprint``.  Defaults to an empty dict.
        source_revision: Upstream source revision (e.g. dataset commit hash),
            when available.
        task: Owning task namespace for task corpora.
        corpus: Corpus label for task corpora.
        parent_substrate: Canonical repo-relative path to the parent shared
            substrate root for task corpora.
        parent_family: Parent shared-substrate family name.
        parent_version: Parent shared-substrate version integer.
        task_schema_version: Task channel schema version.
        task_protocol_version: Task protocol (episode/replay) version.

    Raises:
        FileExistsError: When a manifest already exists at *version_root*.
    """
    dest = version_root / MANIFEST_FILENAME
    if dest.exists():
        raise FileExistsError(
            f"Manifest already exists (dataset roots are immutable): {dest}\n"
            "Bump the version integer to create a new version."
        )

    if stage_params is None:
        stage_params = {}

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_class": dataset_class,
        "family": family,
        "version": version,
        "channels": channels,
        "shape": list(shape),
        "n_samples": n_samples,
        "source_id": source_id,
        "builder": builder,
        "seed": seed,
        "normalization_version": normalization_version,
        "shared_schema_version": shared_schema_version,
        "stage_params": stage_params,
        "producer_revision": _read_producer_revision(),
        "input_fingerprint": _fingerprint(stage_params),
    }
    if source_revision is not None:
        manifest["source_revision"] = source_revision
    if task is not None:
        manifest["task"] = task
    if corpus is not None:
        manifest["corpus"] = corpus
    if parent_substrate is not None:
        manifest["parent_substrate"] = parent_substrate
    if parent_family is not None:
        manifest["parent_family"] = parent_family
    if parent_version is not None:
        manifest["parent_version"] = parent_version
    if task_schema_version is not None:
        manifest["task_schema_version"] = task_schema_version
    if task_protocol_version is not None:
        manifest["task_protocol_version"] = task_protocol_version

    dest.write_text(json.dumps(manifest, indent=2))


# =============================================================================
def read_manifest(version_root: Path) -> dict[str, Any]:
    """Read and return the ``manifest.json`` from *version_root*.

    Args:
        version_root: Versioned dataset root directory.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: When no manifest exists at *version_root*.
    """
    path = version_root / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}.  Is '{version_root}' a valid versioned dataset root?"
        )
    return json.loads(path.read_text())


# =============================================================================
__all__ = [
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "write_manifest",
    "read_manifest",
]

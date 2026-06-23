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

# ── Manifest schema version ──────────────────────────────────────────────

MANIFEST_SCHEMA_VERSION: int = 1
"""Manifest schema version for the first-release contract.

The canonical lineage format is the ``parents`` role-addressed dict.
There is no alternative representation."""


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
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION,
    channels: list[str] | None = None,
    topology_kind: str | None = None,
    n_states: int | None = None,
    extent: list[int] | None = None,
    n_samples: dict[str, int],
    source_id: str,
    builder: str,
    seed: int,
    normalization_version: int | None = 1,
    shared_schema_version: int | None = 1,
    stage_params: dict[str, Any] | None = None,
    source_revision: str | None = None,
    # task-corpus-only fields
    task: str | None = None,
    corpus: str | None = None,
    task_schema_version: int | None = None,
    task_protocol_version: int | None = None,
    # lineage (role-addressed parent references)
    parents: dict[str, dict] | None = None,
    content_digest: str | None = None,
    **extra_fields: Any,
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
        manifest_schema_version: Manifest schema version (default ``MANIFEST_SCHEMA_VERSION``).
        channels: Channel names present in every sample.
        topology_kind: Canonical topology kind string (e.g. ``"grid2d"``, ``"line1d"``).
        n_states: Total number of states in the topology.
        extent: Topology extent list (e.g. ``[H, W]`` for grid2d, ``[N]`` for line1d).
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
        task_schema_version: Task channel schema version.
        task_protocol_version: Task protocol (episode/replay) version.
        parents: Role-addressed parent artifact references.  Keys are
            role names (e.g. ``"spatial_topology"``, ``"semantic_graph"``).
            Each value is a dict with keys ``family``, ``root``, ``version``,
            and optionally ``artifact_id``, ``split``, ``content_digest``.
        content_digest: Content digest of this artifact (``"sha256:..."``).

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
        "manifest_schema_version": manifest_schema_version,
        "dataset_class": dataset_class,
        "family": family,
        "version": version,
        "n_samples": n_samples,
        "source_id": source_id,
        "builder": builder,
        "seed": seed,
        "stage_params": stage_params,
        "producer_revision": _read_producer_revision(),
        "input_fingerprint": _fingerprint(stage_params),
    }
    if channels is not None:
        manifest["channels"] = channels
    if topology_kind is not None:
        manifest["topology_kind"] = topology_kind
    if n_states is not None:
        manifest["n_states"] = n_states
    if extent is not None:
        manifest["extent"] = list(extent)
    if normalization_version is not None:
        manifest["normalization_version"] = normalization_version
    if shared_schema_version is not None:
        manifest["shared_schema_version"] = shared_schema_version
    if source_revision is not None:
        manifest["source_revision"] = source_revision
    if task is not None:
        manifest["task"] = task
    if corpus is not None:
        manifest["corpus"] = corpus
    if task_schema_version is not None:
        manifest["task_schema_version"] = task_schema_version
    if task_protocol_version is not None:
        manifest["task_protocol_version"] = task_protocol_version
    if parents is not None:
        manifest["parents"] = parents
    if content_digest is not None:
        manifest["content_digest"] = content_digest
    manifest.update(extra_fields)

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


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "write_manifest",
    "read_manifest",
]

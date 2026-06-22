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
# Professional pattern: keep the legacy name as a public alias so old
# import paths and inline references (e.g. the JSON key "schema_version")
# continue to resolve correctly.  NumPy follows the same approach with
# np.__version__ as an alias for numpy.version.version.
# https://github.com/numpy/numpy/blob/main/numpy/__init__.py

SCHEMA_VERSION: int = 2
"""Legacy alias for MANIFEST_SCHEMA_VERSION — written as ``"schema_version"``
in every manifest to identify the manifest-schema format.  Retained as a
module-level constant for backward compatibility with inline references."""

MANIFEST_SCHEMA_VERSION: int = SCHEMA_VERSION
"""Current manifest schema version (distinct from dataset content version).
Emitted as both ``manifest_schema_version`` (v2+ field) and statically as
``schema_version`` in every manifest for backward compatibility.

v2 introduced ``parents``, ``manifest_schema_version``, and ``content_digest``.
v1 implied ``manifest_schema_version=1`` and used flat ``parent_substrate``
fields for single-parent lineage."""


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
    manifest_schema_version: int | None = MANIFEST_SCHEMA_VERSION,
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
    parent_substrate: str | None = None,
    parent_family: str | None = None,
    parent_version: int | None = None,
    task_schema_version: int | None = None,
    task_protocol_version: int | None = None,
    # multi-parent lineage (v2+)
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
            Deprecated ``_TASK_REQUIRED`` still accepted for backward-compat omission.
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
        parent_substrate: Canonical repo-relative path to the parent shared
            substrate root for task corpora.
        parent_family: Parent shared-substrate family name.
        parent_version: Parent shared-substrate version integer.
        task_schema_version: Task channel schema version.
        task_protocol_version: Task protocol (episode/replay) version.
        parents: Role-addressed parent artifact references (v2+).  Keys are
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
        # "schema_version" is the legacy key — always written for backward
        # compat so old manifest readers can still parse the version format.
        # "manifest_schema_version" is the v2+ explicit field (may be None
        # for v1 manifests that omit it; we always write it).
        "schema_version": MANIFEST_SCHEMA_VERSION,
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


# =============================================================================
def normalize_manifest_parents(manifest: dict[str, Any]) -> dict[str, dict]:
    """Return a uniform ``role -> ref`` dict from any manifest version.

    v2 manifests with ``parents`` are returned as-is.
    v1 manifests with flat ``parent_substrate``/``parent_family``/``parent_version``
    are normalized into a single ``"primary"`` role.

    Args:
        manifest: Parsed manifest dict (any schema version).

    Returns:
        Dict mapping role names to parent-ref dicts.  Each ref has at least
        ``family``, ``root``, ``version``.
    """
    parents = manifest.get("parents")
    if parents is not None:
        return dict(parents)

    # v1 fallback: synthesize "primary" role from flat fields
    primary: dict[str, Any] = {}
    if "parent_substrate" in manifest:
        primary["root"] = manifest["parent_substrate"]
    if "parent_family" in manifest:
        primary["family"] = manifest["parent_family"]
    if "parent_version" in manifest:
        primary["version"] = manifest["parent_version"]
    if primary:
        return {"primary": primary}
    return {}


__all__ = [
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",  # legacy alias, kept for backward compat
    "MANIFEST_SCHEMA_VERSION",  # canonical name
    "write_manifest",
    "read_manifest",
    "normalize_manifest_parents",
]

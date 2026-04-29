"""Shared validation helper for versioned dataset roots.

Provides :func:`validate_version_root` (full validation including path grammar)
and the private :func:`_validate_structure` (structural-only, used inside
transactional staging before the atomic rename).

Generic validation is structural only: manifest fields, canonical path grammar,
split presence, dataset.json presence, channel-file presence, sample counts,
and index counts.  Family-owned validators own channel semantics and topology
checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest

# ---------------------------------------------------------------------------
# Required manifest fields per dataset_class
# ---------------------------------------------------------------------------

_SHARED_REQUIRED: frozenset[str] = frozenset(
    {
        "schema_version",
        "dataset_class",
        "family",
        "version",
        "channels",
        "topology_kind",
        "n_states",
        "extent",
        "n_samples",
        "source_id",
        "builder",
        "seed",
        "normalization_version",
        "shared_schema_version",
        "stage_params",
        "producer_revision",
        "input_fingerprint",
    }
)

_TASK_REQUIRED: frozenset[str] = _SHARED_REQUIRED | frozenset(
    {
        "task",
        "corpus",
        "parent_substrate",
        "parent_family",
        "parent_version",
        "task_schema_version",
        "task_protocol_version",
    }
)

_KNOWN_DATASET_CLASSES: frozenset[str] = frozenset({"shared_substrate", "task_corpus"})

_FORBIDDEN_FIELDS: frozenset[str] = frozenset({"lineage", "created_at", "build_host", "build_time", "shape"})


def _validate_structure(root: Path) -> dict[str, Any]:
    """Validate structural integrity of a versioned dataset root.

    Does not check path grammar (leaf name vs. manifest version).  Safe to
    call on a temporary staging directory before the atomic rename.

    Args:
        root: Root directory to validate (may be a staging temp dir).

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: When a required file or directory is absent.
        ValueError: On any contract violation.
    """
    manifest = read_manifest(root)

    dataset_class: str = manifest.get("dataset_class", "")
    if dataset_class not in _KNOWN_DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset_class {dataset_class!r}. "
            f"Must be one of {sorted(_KNOWN_DATASET_CLASSES)}."
        )

    present_forbidden = _FORBIDDEN_FIELDS & manifest.keys()
    if present_forbidden:
        raise ValueError(
            f"Manifest contains forbidden fields: {sorted(present_forbidden)}"
        )

    required = _TASK_REQUIRED if dataset_class == "task_corpus" else _SHARED_REQUIRED
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")

    channels: list[str] = manifest["channels"]
    n_samples: dict[str, int] = manifest["n_samples"]

    if dataset_class == "task_corpus":
        ps: str = manifest["parent_substrate"]
        if ps.startswith("/"):
            raise ValueError(
                f"parent_substrate must be a repo-relative path, got absolute: {ps!r}"
            )

    index_path = root / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index: {index_path}")

    split_counts_from_index: dict[str, int] = {}
    with index_path.open() as fh:
        for line in fh:
            entry = json.loads(line)
            split = entry.get("split", "")
            split_counts_from_index[split] = split_counts_from_index.get(split, 0) + 1

    for split, n in n_samples.items():
        if split_counts_from_index.get(split, 0) != n:
            raise ValueError(
                f"index.jsonl has {split_counts_from_index.get(split, 0)} entries for split "
                f"'{split}', manifest declares {n}"
            )

    for split, n in n_samples.items():
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        dataset_json = split_dir / "dataset.json"
        if not dataset_json.exists():
            raise FileNotFoundError(f"Missing dataset.json: {dataset_json}")
        split_meta = json.loads(dataset_json.read_text())
        if split_meta.get("n_samples") != n:
            raise ValueError(
                f"{dataset_json}: n_samples={split_meta.get('n_samples')}, manifest declares {n}"
            )

        for ch in channels:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(f"Missing channel file: {ch_file}")
            arr = np.load(ch_file, mmap_mode="r")
            if arr.shape[0] != n:
                raise ValueError(f"{ch_file}: has {arr.shape[0]} samples, manifest declares {n}")

    return manifest


def _validate_path_grammar(root: Path, manifest: dict[str, Any]) -> None:
    """Validate that the path structure matches the manifest identity.

    Args:
        root: Versioned dataset root path.
        manifest: Already-parsed manifest dict.

    Raises:
        ValueError: On any path grammar violation.
    """
    version = manifest["version"]
    expected_leaf = f"v{version}"
    if root.name != expected_leaf:
        raise ValueError(
            f"Version leaf name {root.name!r} does not match manifest version "
            f"{version} (expected {expected_leaf!r})."
        )

    dataset_class: str = manifest["dataset_class"]
    if dataset_class == "shared_substrate":
        family = manifest["family"]
        if root.parent.name != family:
            raise ValueError(
                f"Shared substrate path grammar violation: "
                f"parent dir is {root.parent.name!r}, manifest family is {family!r}."
            )
    elif dataset_class == "task_corpus":
        corpus = manifest["corpus"]
        task = manifest["task"]
        if root.parent.name != corpus:
            raise ValueError(
                f"Task corpus path grammar violation: "
                f"parent dir is {root.parent.name!r}, manifest corpus is {corpus!r}."
            )
        if root.parent.parent.name != task:
            raise ValueError(
                f"Task corpus path grammar violation: "
                f"grandparent dir is {root.parent.parent.name!r}, manifest task is {task!r}."
            )
        ps: str = manifest["parent_substrate"]
        parent_family = manifest["parent_family"]
        parent_version = manifest["parent_version"]
        expected_ps = f"data/processed/{parent_family}/v{parent_version}"
        if ps != expected_ps:
            raise ValueError(
                f"parent_substrate {ps!r} does not match expected {expected_ps!r}."
            )


def validate_version_root(root: Path) -> dict[str, Any]:
    """Validate a versioned dataset root against structural and path-grammar rules.

    Args:
        root: Resolved versioned dataset root.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: When a required file is absent.
        ValueError: On any contract violation.
    """
    manifest = _validate_structure(root)
    _validate_path_grammar(root, manifest)
    return manifest

"""Read-only discovery and validation of existing eval-artifact regime directories.

This module provides lightweight artifact-manifest loading and filtering.
It does **not** load full case traces, construct models, or produce
report artifacts.  Those responsibilities belong to the builder layer.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

from ehc_sn.reporting.schema import (
    EvalArtifactReference,
    EvalArtifactSchema,
    RegimeKind,
    RegimeSelector,
    ReportSpec,
)

# ---------------------------------------------------------------------------
# Constants mirrored from ehc_sn.eval.artifacts (no import to avoid coupling)
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME: str = "manifest.json"
_SUCCESS_FILENAME: str = "_SUCCESS"
_REQUIRED_SCHEMA: EvalArtifactSchema = "ehc_sn.eval.artifact.v3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_task(
    manifest: dict[str, object],
) -> str:
    """Extract the required ``task`` identity from a v3 eval artifact manifest.

    v3 manifests are expected to carry ``task`` as a top-level key.
    A missing or non-string ``task`` is treated as a schema violation.
    """
    task = manifest.get("task")
    if task is None or not isinstance(task, str) or not task:
        raise ValueError(
            "Eval artifact manifest has missing or invalid task: "
            f"{task!r}. task must be a non-empty string."
        )
    return task


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_eval_artifact_manifest(
    path: Path,
) -> dict[str, object]:
    """Load and validate one eval-artifact regime manifest from *path*.

    *path* must point to a directory that contains ``manifest.json`` and
    ``_SUCCESS``.  The manifest's ``schema`` field must equal
    ``"ehc_sn.eval.artifact.v3"``.

    Returns
    -------
    dict
        The complete manifest payload as a dict.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or is missing ``manifest.json``.
    RuntimeError
        If ``_SUCCESS`` is missing (incomplete artifact write).
    ValueError
        If the manifest has an unsupported ``schema`` version or
        ``status != "complete"``.
    """
    path = Path(path)
    manifest_path = path / _MANIFEST_FILENAME
    success_path = path / _SUCCESS_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {_MANIFEST_FILENAME} under {path}.")
    if not success_path.exists():
        raise RuntimeError(
            f"Artifact at {path} is missing {_SUCCESS_FILENAME} sentinel "
            f"(possibly an incomplete or corrupted write)."
        )

    manifest: dict[str, object] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    schema_value = manifest.get("schema")
    if schema_value != _REQUIRED_SCHEMA:
        raise ValueError(
            f"Unsupported eval artifact schema: {schema_value!r}. "
            f"Expected {_REQUIRED_SCHEMA!r}."
        )

    status = manifest.get("status", "complete")
    if status != "complete":
        raise RuntimeError(
            f"Artifact at {path} has status={status!r} "
            f"(expected 'complete')."
        )

    return manifest


def discover_eval_artifacts(
    root: Path,
) -> list[EvalArtifactReference]:
    """Discover eval-artifact regime directories under *root*.

    Only **immediate children** of *root* are scanned.  Recursive discovery
    is intentionally not part of v1.  A child is treated as a valid regime
    artifact directory when it contains both ``manifest.json`` and
    ``_SUCCESS``.

    Children without ``manifest.json`` are silently skipped (e.g., README
    files, scratch directories).  Children with ``manifest.json`` but
    missing ``_SUCCESS`` or with an unsupported schema version raise an
    error immediately.

    Parameters
    ----------
    root:
        Flat directory of regime artifact subdirectories.

    Returns
    -------
    list of EvalArtifactReference
        Regime-level references in discovery order (sorted by directory
        name for determinism).

    Raises
    ------
    FileNotFoundError
        If *root* does not exist or is not a directory.
    RuntimeError
        If any child with ``manifest.json`` fails ``_SUCCESS`` or status
        validation.
    ValueError
        If any child has an unsupported artifact schema version.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Eval artifacts root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(
            f"Eval artifacts root is not a directory: {root}"
        )

    references: list[EvalArtifactReference] = []

    # Iterate sorted children for deterministic ordering.
    children = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    for child_dir in children:
        manifest_path = child_dir / _MANIFEST_FILENAME
        # Silently skip children that are not eval artifact directories.
        if not manifest_path.exists():
            continue
        # Silently skip report-run directories (they carry report_manifest.json).
        if (child_dir / "report_manifest.json").exists():
            continue

        manifest = load_eval_artifact_manifest(child_dir)

        task = _extract_task(manifest)
        regime_id_raw = manifest.get("regime_id")
        regime_kind_raw = manifest.get("regime_kind")

        if not isinstance(regime_id_raw, str) or not regime_id_raw:
            raise ValueError(
                f"Eval manifest in {child_dir} has missing or invalid "
                f"regime_id: {regime_id_raw!r}"
            )
        if regime_kind_raw not in ("diagnostic", "benchmark"):
            raise ValueError(
                f"Eval manifest in {child_dir} has unsupported "
                f"regime_kind: {regime_kind_raw!r}"
            )

        references.append(
            EvalArtifactReference(
                regime_id=regime_id_raw,
                regime_kind=regime_kind_raw,  # type: ignore[arg-type]
                path=child_dir.resolve(),
                task=task,
            )
        )

    return references


def select_eval_artifacts(
    artifacts: Sequence[EvalArtifactReference],
    selectors: Sequence[RegimeSelector],
) -> list[EvalArtifactReference]:
    """Filter *artifacts* by an ordered list of selectors.

    Multiple selectors are combined with **OR** semantics: an artifact
    is included if it matches *any* selector.  Results are de-duplicated
    while preserving discovery order (first match wins).

    A selector matches when **all** of its non-``None`` fields match:

    * ``task`` (if present) compares against ``artifact.task``.
    * ``regime_kind`` (if present) compares against
      ``artifact.regime_kind``.
    * ``regime_ids`` (if present) checks whether ``artifact.regime_id``
      is in the list.

    Parameters
    ----------
    artifacts:
        Regime-level references (typically from
        :func:`discover_eval_artifacts`).
    selectors:
        Ordered list of :class:`RegimeSelector` filters.

    Returns
    -------
    list of EvalArtifactReference
        Matched artifacts in first-match order.
    """
    selected: list[EvalArtifactReference] = []
    seen: set[str] = set()

    for artifact in artifacts:
        matched = False
        for sel in selectors:
            if sel.task is not None:
                if artifact.task is None:
                    raise ValueError(
                        "Cannot filter eval artifacts by task because "
                        f"artifact {artifact.path} does not declare task "
                        "identity. Extend the eval artifact manifest to "
                        "include task."
                    )
                if sel.task != artifact.task:
                    continue
            if (
                sel.regime_kind is not None
                and sel.regime_kind != artifact.regime_kind
            ):
                continue
            if sel.regime_ids is not None and artifact.regime_id not in set(
                sel.regime_ids
            ):
                continue
            matched = True
            break

        if matched and artifact.regime_id not in seen:
            seen.add(artifact.regime_id)
            selected.append(artifact)

    return selected


# ---------------------------------------------------------------------------
# Report-spec loader
# ---------------------------------------------------------------------------


def load_report_spec(path: Path) -> ReportSpec:
    """Load a :class:`ReportSpec` from a TOML file.

    Parameters
    ----------
    path:
        Path to a TOML file containing ``ReportSpec`` fields.

    Returns
    -------
    ReportSpec
        Validated report assembly specification.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    tomllib.TOMLDecodeError
        If the TOML content is malformed.
    pydantic.ValidationError
        If the content does not conform to ``ReportSpec``.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return ReportSpec.model_validate(raw)


# ============================================================================
__all__ = [
    "EvalArtifactReference",
    "EvalArtifactSchema",
    "RegimeKind",
    "RegimeSelector",
    "ReportSpec",
    "load_eval_artifact_manifest",
    "discover_eval_artifacts",
    "select_eval_artifacts",
    "load_report_spec",
]

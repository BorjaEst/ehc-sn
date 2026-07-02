"""Artifact reuse helpers — checkpoint identity and plan-digest matching.

Exact reuse is keyed on a ``plan_digest`` that covers:
- requested figure names and versions,
- aggregate spec names and versions (sorted),
- analysis spec names and versions (sorted),
- requested surface,
- capture profile identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ehc_sn.analysis.compiler import CompiledFigurePlan
from ehc_sn.evaluation.artifacts import load_evaluation_artifact_manifest

# =============================================================================
_MANIFEST_FILENAME = "manifest.json"
_SUCCESS_FILENAME = "_SUCCESS"


# =============================================================================
def compute_checkpoint_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a checkpoint file.

    Args:
        path: Path to a checkpoint file (``.ckpt`` or ``.pt``).

    Returns:
        Lowercase 64-character hex string.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)  # 64 KiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
def compute_plan_digest(
    requested_figures: frozenset[str],
    aggregate_specs: frozenset[tuple[str, int]],
    analysis_specs: frozenset[tuple[str, int]],
) -> str:
    """Compute a deterministic digest for an evaluation plan.

    Args:
        requested_figures: Sorted figure names requested.
        aggregate_specs: Set of ``(name, schema_version)`` tuples.
        analysis_specs: Set of ``(name, schema_version)`` tuples.

    Returns:
        Lowercase 64-character hex digest.
    """
    data: dict[str, Any] = {
        "figures": sorted(requested_figures),
        "aggregates": sorted(
            ({"name": n, "version": v} for n, v in aggregate_specs),
            key=lambda d: (d["name"], d["version"]),
        ),
        "analyses": sorted(
            ({"name": n, "version": v} for n, v in analysis_specs),
            key=lambda d: (d["name"], d["version"]),
        ),
    }
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
def compute_plan_digest_from_plan(
    plan: CompiledFigurePlan,
    *,
    capture_profile_name: str = "",
    capture_profile_version: int = 1,
) -> str:
    """Compute a deterministic digest from a compiled plan.

    Covers figure names and versions, aggregate/analysis specs, surface,
    and capture profile identity.

    Args:
        plan: Compiled figure plan with resolved specs.
        capture_profile_name: Name of the capture profile used.
        capture_profile_version: Version of the capture profile.

    Returns:
        Lowercase 64-character hex digest.
    """
    data: dict[str, Any] = {
        "schema": "ehp_sn.plan_digest.v1",
        "figures": sorted(
            [
                {
                    "name": f.spec_key,
                    "contract_kind": f.contract_kind,
                    "version": 1,
                }
                for f in plan.resolved_figures
            ],
            key=lambda d: d["name"],
        ),
        "aggregates": sorted(
            (
                {"name": a.name, "version": a.schema_version}
                for a in plan.aggregate_specs
            ),
            key=lambda d: (d["name"], d["version"]),
        ),
        "analyses": sorted(
            (
                {"name": a.name, "version": a.schema_version}
                for a in plan.analysis_specs
            ),
            key=lambda d: (d["name"], d["version"]),
        ),
        "surface": plan.surface.value if plan.surface is not None else None,
        "capture_profile": {
            "name": capture_profile_name,
            "version": capture_profile_version,
        },
    }
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
def find_compatible_artifact(
    output_parent: Path | str,
    plan_digest: str,
    alias: str,
    checkpoint_sha256: str,
) -> Path | None:
    """Scan *output_parent* for an existing artifact matching all criteria.

    Matching requires:
    - ``manifest["evaluation"]["plan_digest"]`` == *plan_digest*
    - ``manifest["evaluation"]["alias"]`` == *alias*
    - ``manifest["evaluation"]["checkpoint_sha256"]`` == *checkpoint_sha256*

    A missing ``alias`` or ``checkpoint_sha256`` in a candidate manifest causes
    that candidate to be skipped (not matched), which is safe — execution falls
    through to fresh evaluation.

    Args:
        output_parent: Directory whose immediate children are candidate
            artifact directories (typically the parent of ``output_dir``
            passed to ``run_offline_eval``).  Accepts ``Path`` or ``str``.
        plan_digest: The plan digest to match.
        alias: Recipe alias (e.g. ``"arena-tem-v1"``).
        checkpoint_sha256: SHA-256 hex digest of the model weights file.
    """
    output_parent = Path(output_parent)

    if not output_parent.exists():
        return None

    for child in sorted(output_parent.iterdir()):
        if not child.is_dir():
            continue
        schema_path = child / _MANIFEST_FILENAME
        success_path = child / _SUCCESS_FILENAME
        if not schema_path.exists() or not success_path.exists():
            continue
        try:
            manifest = load_evaluation_artifact_manifest(child)
            eval_block = manifest.get("evaluation", {})
            if (
                eval_block.get("plan_digest") == plan_digest
                and eval_block.get("alias") == alias
                and eval_block.get("checkpoint_sha256") == checkpoint_sha256
            ):
                return child
        except Exception:
            continue

    return None


__all__ = [
    "compute_checkpoint_sha256",
    "compute_plan_digest",
    "compute_plan_digest_from_plan",
    "find_compatible_artifact",
]

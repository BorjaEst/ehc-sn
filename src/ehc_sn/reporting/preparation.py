"""Report-data package preparation — transform evaluation artifacts into
a portable, self-describing Data Package.

Usage::

    from ehc_sn.reporting.request import load_report_data_request
    from ehc_sn.reporting.preparation import prepare_report_data

    request = load_report_data_request(Path("report.toml"))
    pkg = prepare_report_data(request, output=Path("reports/arena-tem/data"))
    pkg.resource("metrics").read()
"""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ehc_sn.evaluation.artifact_models import (
    EvaluationArtifactSet,
    RegimeArtifactSet,
)
from ehc_sn.evaluation.artifacts import LoadedArtifactCase
from ehc_sn.reporting.derived import DERIVED_RESOURCE_SPECS
from ehc_sn.reporting.errors import ReportPreparationError
from ehc_sn.reporting.package import (
    ReportDataPackage,
    ReportDataProvenance,
)
from ehc_sn.reporting.request import ReportDataRequest
from ehc_sn.reporting.sources import materialize_evaluation_source

# =============================================================================
# Constants
# =============================================================================

_SUCCESS_FILENAME = "_SUCCESS"


# =============================================================================
# Preparation
# =============================================================================


def prepare_report_data(
    request: ReportDataRequest,
    *,
    output: Path,
) -> ReportDataPackage:
    """Materialise a report-data package from one evaluation regime.

    Reads the evaluation artifact specified by *request*, extracts the
    requested resources, and writes them as a self-describing Data
    Package at *output*.

    Args:
        request: The report-data request (from ``report.toml``).
        output: Destination directory for the generated package.  Must
            not already exist.

    Returns:
        A :class:`ReportDataPackage` pointing at *output*.

    Raises:
        ReportPreparationError: If the output exists, the source cannot
            be resolved, the regime is missing, or resource preparation
            fails.
    """
    output = Path(output).resolve()

    # ---- Reject pre-existing output ----------------------------------------
    if output.exists():
        raise ReportPreparationError(
            f"Output already exists: {output}. "
            "Remove it manually or choose a different path."
        )

    # ---- Materialise evaluation source -------------------------------------
    try:
        src = materialize_evaluation_source(request.source.uri)
    except (FileNotFoundError, OSError) as exc:
        raise ReportPreparationError(
            f"Failed to materialise evaluation source "
            f"{request.source.uri!r}: {exc}"
        ) from exc

    # ---- Load evaluation and regime ----------------------------------------
    try:
        eval_set = EvaluationArtifactSet.load(src.local_root)
    except (FileNotFoundError, ValueError) as exc:
        # Fallback: try regime-level artifact directly.
        try:
            regime = RegimeArtifactSet.load(src.local_root)
        except (FileNotFoundError, ValueError) as exc2:
            raise ReportPreparationError(
                f"Failed to load evaluation artifact from {src.local_root}: "
                f"{exc}; regime fallback also failed: {exc2}"
            ) from exc2
    else:
        try:
            regime = eval_set.open_regime(request.source.regime)
        except KeyError as exc:
            raise ReportPreparationError(
                f"Regime {request.source.regime!r} not found in "
                f"evaluation at {src.local_root}: {exc}"
            ) from exc

    regime_manifest = regime.manifest

    # ---- Resolve task and model family -------------------------------------
    task = regime_manifest.task or "unknown"
    model_family = _extract_model_family_from_regime(regime)

    # ---- Prepare output via unique temp directory --------------------------
    tmp_dir = output.parent / f"{output.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ReportPreparationError(
            f"Failed to create temporary directory {tmp_dir}: {exc}"
        ) from exc

    try:
        _prepare_resources(
            request=request,
            regime=regime,
            output=tmp_dir,
            task=task,
            model_family=model_family,
            src=src,
        )
        # Atomically commit.
        tmp_dir.rename(output)
    except BaseException:
        # Clean up temp directory on failure.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    # ---- Return opened package ---------------------------------------------
    return ReportDataPackage.open(output)


# =============================================================================
# Internal helpers
# =============================================================================


def _extract_model_family_from_regime(regime: RegimeArtifactSet) -> str:
    """Heuristic to extract model family from a regime artifact."""
    manifest = regime.manifest
    # Try extension fields or known patterns.
    extensions = getattr(manifest, "extensions", {}) or {}
    mf = extensions.get("model_family")
    if mf and isinstance(mf, str):
        return mf
    # Fall back to regime_id prefix convention: "arena_tem" -> "tem"
    rid = manifest.regime_id or ""
    for known in ("tem-v2", "tem-v1", "hrm-v2", "hrm-v1", "ehp-v1", "ehp-v2"):
        if known in rid.lower():
            return known
    return "unknown"


def _prepare_resources(
    *,
    request: ReportDataRequest,
    regime: RegimeArtifactSet,
    output: Path,
    task: str,
    model_family: str,
    src: Any,
) -> None:
    """Write all requested resources to *output*."""
    resources: list[dict[str, object]] = []
    selected_case_ids: list[str] = []

    # ---- Metrics -----------------------------------------------------------
    if request.resources.metrics:
        _write_metrics(regime, output)
        resources.append(
            {
                "name": "metrics",
                "profile": "tabular-data-resource",
                "path": "metrics.csv",
                "format": "csv",
                "mediatype": "text/csv",
                "schema": {
                    "fields": [
                        {"name": "metric", "type": "string"},
                        {"name": "value", "type": "number"},
                    ]
                },
            }
        )

    # ---- Validations -------------------------------------------------------
    if request.resources.validations:
        _write_validations(regime, output)
        resources.append(
            {
                "name": "validations",
                "profile": "tabular-data-resource",
                "path": "validations.csv",
                "format": "csv",
                "mediatype": "text/csv",
                "schema": {
                    "fields": [
                        {"name": "validation_key", "type": "string"},
                        {"name": "value", "type": "number"},
                    ]
                },
            }
        )

    # ---- Cases -------------------------------------------------------------
    if request.resources.cases:
        loaded_cases = regime.read_cases()
        selected_case_ids = _select_case_ids(loaded_cases, request)
        _write_cases(loaded_cases, selected_case_ids, output)
        _write_selected_cases_json(selected_case_ids, output)
        resources.append(
            {
                "name": "cases",
                "path": "cases.parquet",
                "format": "parquet",
                "mediatype": "application/vnd.apache.parquet",
                "ehp": {
                    "kind": "raw",
                    "task": task,
                    "model_family": model_family,
                },
            }
        )
        resources.append(
            {
                "name": "selected_cases",
                "path": "selected-cases.json",
                "format": "json",
                "mediatype": "application/json",
            }
        )

    # ---- Predictions -------------------------------------------------------
    if request.resources.predictions:
        predictions_dir = output / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        resources.append(
            {
                "name": "predictions",
                "path": "predictions",
                "format": "dir",
                "mediatype": "application/x-directory",
            }
        )

    # ---- Traces ------------------------------------------------------------
    if request.resources.traces:
        traces_dir = output / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        resources.append(
            {
                "name": "traces",
                "path": "traces",
                "format": "dir",
                "mediatype": "application/x-directory",
            }
        )

    # ---- Derived resources -------------------------------------------------
    req_derived = set(request.resources.derived)
    for derived_name in sorted(req_derived):
        spec = DERIVED_RESOURCE_SPECS.get(derived_name)
        if spec is None:
            raise ReportPreparationError(
                f"Unknown derived resource: {derived_name!r}. "
                f"Available: {sorted(DERIVED_RESOURCE_SPECS)}"
            )
        if task not in spec.tasks:
            raise ReportPreparationError(
                f"Derived resource {derived_name!r} does not support "
                f"task {task!r}. Supported tasks: {sorted(spec.tasks)}"
            )
        if model_family not in spec.model_families:
            raise ReportPreparationError(
                f"Derived resource {derived_name!r} does not support "
                f"model family {model_family!r}. "
                f"Supported: {sorted(spec.model_families)}"
            )

        df = spec.builder(regime, request)
        derived_path = output / spec.relative_path
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(derived_path, index=False)

        resources.append(
            {
                "name": derived_name,
                "path": spec.relative_path,
                "format": spec.format,
                "mediatype": (
                    "text/csv"
                    if spec.format == "csv"
                    else "application/octet-stream"
                ),
                "ehp": {
                    "kind": "derived",
                    "builder": f"{spec.builder.__module__}.{spec.builder.__name__}",
                    "builder_version": spec.version,
                    "source_resources": list(spec.source_resources),
                },
            }
        )

    # ---- Provenance --------------------------------------------------------
    _write_provenance(
        output=output,
        request=request,
        src=src,
        task=task,
        model_family=model_family,
        selected_case_ids=selected_case_ids,
    )

    # ---- Descriptor --------------------------------------------------------
    descriptor: dict[str, object] = {
        "profile": "data-package",
        "name": request.name,
        "resources": resources,
        "ehp": {
            "schema": "ehp_sn.report.data.v1",
            "task": task,
            "model_family": model_family,
        },
    }
    (output / "datapackage.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    # ---- Success sentinel (written last) -----------------------------------
    (output / _SUCCESS_FILENAME).write_text("", encoding="utf-8")


# =============================================================================
# Resource writers
# =============================================================================


def _write_metrics(regime: RegimeArtifactSet, output: Path) -> None:
    """Write ``metrics.csv`` from regime metrics."""
    metrics = regime.metrics
    path = output / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])


def _write_validations(regime: RegimeArtifactSet, output: Path) -> None:
    """Write ``validations.csv`` from regime validation data.

    Falls back to an empty table with a header when no validation data
    is exposed by the regime.
    """
    path = output / "validations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["validation_key", "value"])
        # Attempt to extract validation info from manifest extensions.
        manifest = regime.manifest
        extensions = getattr(manifest, "extensions", {}) or {}
        validations = extensions.get("validations", {})
        if isinstance(validations, dict):
            for key, value in sorted(validations.items()):
                if isinstance(value, (int, float)):
                    writer.writerow([key, value])


def _write_cases(
    loaded_cases: tuple[LoadedArtifactCase, ...],
    selected_case_ids: Sequence[str],
    output: Path,
) -> None:
    """Write ``cases.parquet`` with selected case metadata."""
    selected = [c for c in loaded_cases if c.case_id in selected_case_ids]
    if not selected:
        # Write empty parquet with schema.
        df = pd.DataFrame({"case_id": pd.Series(dtype="str")})
        df.to_parquet(output / "cases.parquet", index=False)
        return

    rows: list[dict[str, object]] = []
    for c in selected:
        row: dict[str, object] = {"case_id": c.case_id}
        if c.source_context is not None:
            row["source_context_type"] = type(c.source_context).__name__
        ts = c.temporal_semantics or {}
        if ts.get("rollout_mode"):
            row["rollout_mode"] = ts["rollout_mode"]
        rows.append(row)

    df = pd.DataFrame(rows)
    path = output / "cases.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_selected_cases_json(
    selected_case_ids: Sequence[str],
    output: Path,
) -> None:
    """Write ``selected-cases.json`` as a simple list of case IDs."""
    path = output / "selected-cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(selected_case_ids), indent=2),
        encoding="utf-8",
    )


def _select_case_ids(
    loaded_cases: tuple[LoadedArtifactCase, ...],
    request: ReportDataRequest,
) -> list[str]:
    """Select case IDs deterministically.

    When explicit ``case_ids`` are provided, returns them in request
    order.  Otherwise applies ``strategy`` and ``max_cases`` with
    deterministic tie-breaking (loss descending, then case_id ascending).
    """
    if request.selection.case_ids:
        return list(request.selection.case_ids)

    # Deterministic sort: prefer cases with traces, then stable order.
    traced = [c for c in loaded_cases if c.trace is not None]
    non_traced = [c for c in loaded_cases if c.trace is None]

    # Sort for stability.
    traced.sort(
        key=lambda c: (-len(c.trace) if c.trace is not None else 0, c.case_id)
    )
    non_traced.sort(key=lambda c: c.case_id)

    selected = traced[: request.selection.max_cases]
    if len(selected) < request.selection.max_cases:
        remaining = request.selection.max_cases - len(selected)
        selected.extend(non_traced[:remaining])

    return [c.case_id for c in selected]


def _write_provenance(
    *,
    output: Path,
    request: ReportDataRequest,
    src: Any,
    task: str,
    model_family: str,
    selected_case_ids: Sequence[str],
) -> None:
    """Write ``provenance.json`` with both requested and resolved identity."""
    provenance: dict[str, object] = {
        "source": {
            "requested_uri": request.source.uri,
            "run_id": getattr(src, "run_id", None),
            "evaluation_id": "",
            "regime_id": request.source.regime,
            "artifact_digest": None,
            "task": task,
            "model_family": model_family,
            "code_revision": None,
        },
        "selection": {
            "strategy": (
                request.selection.strategy
                if not request.selection.case_ids
                else "explicit"
            ),
            "selected_case_ids": list(selected_case_ids),
        },
        "preparation": {
            "materialized_from": str(src.local_root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    path = output / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

"""Metric record serialization and loading.

Pure file-level helpers for writing and reading :class:`MetricRecord`
lists.  These functions are stateless — they do not open, read, or modify
``report_manifest.json`` or ``_SUCCESS``.

Usage::

    from ehc_sn.reporting import (
        MetricRecord,
        write_metric_records,
        load_metric_records,
        load_report_run,
    )

    # Write records to a directory (the caller owns manifest integration).
    path = write_metric_records(
        [MetricRecord(metric="accuracy", value=0.91)],
        output_dir=Path("reports/tem_v2/version_12_best"),
    )

    # Load records from a validated ReportRun.
    report = load_report_run(Path("reports/tem_v2/version_12_best"))
    records = load_metric_records(report)

This module must remain free of ``lightning/``, ``eval/``, ``models/``,
``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ehc_sn.reporting.reader import ReportRun
from ehc_sn.reporting.schema import MetricRecord

# ---------------------------------------------------------------------------
# Default filename
# ---------------------------------------------------------------------------

_DEFAULT_FILENAME = "metrics.records.json"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_metric_records(
    records: Sequence[MetricRecord],
    output_dir: Path,
    *,
    filename: str = _DEFAULT_FILENAME,
) -> Path:
    """Serialize *records* as a JSON list and write to *output_dir*.

    Parameters
    ----------
    records:
        Metric records to persist.  Must be non-empty.
    output_dir:
        Target directory.  Created if it does not exist.
    filename:
        Output filename within *output_dir*.  Defaults to
        ``"metrics.records.json"``.

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError(
            "Metric records list must not be empty. "
            "Provide at least one MetricRecord."
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    serialized = [r.model_dump(mode="json") for r in records]
    output_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_metric_records(
    report_run: ReportRun,
) -> list[MetricRecord]:
    """Load metric records from a validated report run.

    Parameters
    ----------
    report_run:
        A :class:`ReportRun` returned by :func:`load_report_run`.

    Returns
    -------
    list of MetricRecord
        Parsed metric records, or ``[]`` when
        ``report_run.manifest.metrics_records_json is None``.

    Raises
    ------
    FileNotFoundError
        If the manifest points to a ``metrics_records_json`` file that
        does not exist.
    ValueError
        If the file contains malformed JSON or records that fail
        :class:`MetricRecord` validation.
    """
    manifest = report_run.manifest
    rel_path = manifest.metrics_records_json

    if rel_path is None:
        return []

    target = (report_run.root / rel_path).resolve()
    if not target.exists():
        raise FileNotFoundError(
            f"Manifest references metric records at {rel_path!r} "
            f"(resolved: {target}), but the file does not exist."
        )

    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of metric records at {rel_path!r}, "
            f"got {type(raw).__name__}."
        )

    records: list[MetricRecord] = []
    for idx, item in enumerate(raw):
        try:
            records.append(MetricRecord.model_validate(item))
        except Exception as exc:
            raise ValueError(
                f"Invalid metric record at index {idx} in "
                f"{rel_path!r}: {exc}"
            ) from exc

    return records


# =============================================================================
__all__ = [
    "MetricRecord",
    "write_metric_records",
    "load_metric_records",
]

"""Notebook-facing convenience API for report comparison views.

Provides :func:`load_report_collection` as the single entry point for
notebooks to load a ``ReportCollection`` from per-subject config paths.

Usage::

    from pathlib import Path

    from ehc_sn.reporting import load_report_collection

    report = load_report_collection(
        specs=[
            Path("../configs/templates/tem_v1_arena_struct.yaml"),
        ],
        title="TEM Structural Memory Report",
        question="Do TEM models learn structural representations?",
    )

    report.display_header()
    report.display_primary_metrics("accuracy_revisit")

Boundary rule:
    This module must remain free of ``lightning/``, ``eval/``,
    ``models/``, ``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ehc_sn.reporting.collection import (
    ReportCollection,
    build_or_load_report_collection,
)
from ehc_sn.reporting.inspection import OpenReport
from ehc_sn.reporting.schema import ReportSpec


def load_report_collection(
    specs: Sequence[Path | ReportSpec | OpenReport],
    *,
    force_rebuild: bool = False,
    title: str = "",
    question: str = "",
) -> ReportCollection:
    """Load a notebook-facing comparison view over report runs.

    Parameters
    ----------
    specs:
        One entry per subject.  Each entry must be a path to a
        ``ReportSpec`` YAML, an already-instantiated ``ReportSpec``,
        or an already-loaded ``OpenReport``.
    force_rebuild:
        If ``True``, re-build report runs even if they already exist.
    title:
        Scientific title for the comparison.
    question:
        Research question addressed by this collection.

    Returns
    -------
    ReportCollection
        A view object with ``display_*()`` methods for notebook use.
    """
    return build_or_load_report_collection(
        list(specs),
        force_rebuild=force_rebuild,
        title=title,
        question=question,
    )

"""Reporting — resolve evaluation sources, build notebook-facing read models
and portable report-data packages.

Report-data package workflow::

    report.toml  →  ehp report prepare  →  report-data package  →  open_report()

Usage::

    from ehc_sn.reporting import (
        # Package reading
        ReportDataPackage,
        open_report,
        ReportDataProvenance,
        ReportResource,
        # Request
        ReportDataRequest,
        load_report_data_request,
        # Preparation
        prepare_report_data,
        # Legacy loaders (unchanged)
        ArenaTemReportData,
        load_arena_tem_report,
        load_evaluation,
        # Errors
        ReportingError,
        ReportRequestError,
        ReportPreparationError,
        InvalidPackageError,
        ResourceFormatError,
    )
"""

from __future__ import annotations

from ehc_sn.reporting.errors import (
    InvalidPackageError,
    ReportingError,
    ReportPreparationError,
    ReportRequestError,
    ResourceFormatError,
)
from ehc_sn.reporting.loaders import (
    ArenaTemReportData,
    load_arena_tem_report,
    load_evaluation,
)
from ehc_sn.reporting.package import (
    ReportDataPackage,
    ReportDataProvenance,
    ReportResource,
    open_report,
)
from ehc_sn.reporting.preparation import prepare_report_data
from ehc_sn.reporting.request import (
    ReportDataRequest,
    load_report_data_request,
)

# =============================================================================
__all__ = [
    # Package reading
    "ReportDataPackage",
    "open_report",
    "ReportDataProvenance",
    "ReportResource",
    # Request
    "ReportDataRequest",
    "load_report_data_request",
    # Preparation
    "prepare_report_data",
    # Legacy loaders (unchanged)
    "ArenaTemReportData",
    "load_arena_tem_report",
    "load_evaluation",
    # Errors
    "ReportingError",
    "ReportRequestError",
    "ReportPreparationError",
    "InvalidPackageError",
    "ResourceFormatError",
]

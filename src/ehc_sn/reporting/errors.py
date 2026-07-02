"""Report-data error types.

All report-data exceptions derive from :class:`ReportingError` to let
callers catch any report-data failure with a single ``except ReportingError``.
"""

from __future__ import annotations


class ReportingError(Exception):
    """Base exception for all report-data operations."""


class ReportRequestError(ReportingError):
    """Raised when a ``report.toml`` file is missing, unparseable, or
    fails structural validation (unknown schema, missing fields, extra
    fields, invalid field values)."""


class ReportPreparationError(ReportingError):
    """Raised when :func:`prepare_report_data` cannot produce a package.

    Covers: pre-existing output, unresolvable source, missing regime,
    task/model-family mismatch, incompatible derived resources, write
    failures.
    """


class InvalidPackageError(ReportingError):
    """Raised when a directory cannot be opened as a valid report-data
    package (missing required files, invalid descriptor, missing
    resources, path traversal)."""


class ResourceFormatError(ReportingError):
    """Raised when a resource file cannot be read in its declared format."""

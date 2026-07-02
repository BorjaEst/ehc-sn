"""Report-data package — read-only access to a prepared Data Package.

The package is a self-describing Frictionless Data Package directory
with a ``datapackage.json`` descriptor, ``provenance.json``, and
named resource files.

Usage::

    from ehc_sn.reporting.package import ReportDataPackage, open_report

    pkg = open_report("reports/arena-tem/data")
    metrics = pkg.resource("metrics").read()
    cases = pkg.resource("cases").read()
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ehc_sn.reporting.errors import InvalidPackageError, ResourceFormatError

# =============================================================================
# Type aliases
# =============================================================================

ReportResourceValue = pd.DataFrame | dict[str, object] | list[object]

# =============================================================================
# Descriptor types
# =============================================================================


@dataclass(frozen=True)
class DataResourceDescriptor:
    """One resource entry in ``datapackage.json``.

    Fields follow the Frictionless Data Resource specification.
    """

    name: str
    path: str
    format: str
    mediatype: str
    profile: str | None = None
    schema_: Mapping[str, object] | None = None
    ehp: Mapping[str, object] | None = None


@dataclass(frozen=True)
class DataPackageDescriptor:
    """Typed ``datapackage.json`` descriptor.

    ``profile`` must be ``"data-package"``.  EHP-specific metadata
    lives under ``ehp``.
    """

    profile: str
    name: str
    resources: tuple[DataResourceDescriptor, ...]
    ehp: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> DataPackageDescriptor:
        """Load and validate a ``datapackage.json`` from *root*.

        Args:
            root: Package root directory.

        Returns:
            A validated ``DataPackageDescriptor``.

        Raises:
            InvalidPackageError: If the descriptor is missing or invalid.
        """
        path = root / "datapackage.json"
        if not path.exists():
            raise InvalidPackageError(
                f"Missing datapackage.json in report-data package: {root}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise InvalidPackageError(
                f"Invalid datapackage.json at {path}: {exc}"
            ) from exc

        profile = raw.get("profile")
        if profile != "data-package":
            raise InvalidPackageError(
                f"Unsupported datapackage.json profile: {profile!r}. "
                f"Expected 'data-package'."
            )

        name = raw.get("name", "")
        resources_raw = raw.get("resources", [])
        if not resources_raw:
            raise InvalidPackageError(
                f"datapackage.json at {path} has an empty resources array."
            )

        resources: list[DataResourceDescriptor] = []
        seen_names: set[str] = set()
        for i, entry in enumerate(resources_raw):
            entry_name = entry.get("name", f"<resource {i}>")
            if entry_name in seen_names:
                raise InvalidPackageError(
                    f"Duplicate resource name {entry_name!r} in "
                    f"datapackage.json at {path}."
                )
            seen_names.add(entry_name)

            entry_path = entry.get("path", "")
            if not isinstance(entry_path, str):
                raise InvalidPackageError(
                    f"Resource {entry_name!r} in datapackage.json has "
                    f"non-string path: {entry_path!r}."
                )

            resources.append(
                DataResourceDescriptor(
                    name=entry_name,
                    path=entry_path,
                    format=entry.get("format", ""),
                    mediatype=entry.get("mediatype", ""),
                    profile=entry.get("profile"),
                    schema_=entry.get("schema"),
                    ehp=entry.get("ehp"),
                )
            )

        return cls(
            profile=profile,
            name=name,
            resources=tuple(resources),
            ehp=raw.get("ehp", {}),
        )


# =============================================================================
# Provenance
# =============================================================================


@dataclass(frozen=True)
class ReportDataProvenance:
    """Provenance metadata for one report-data package.

    Durable identity comes from the ``source.*`` fields.
    ``materialized_from`` and ``preparation_timestamp`` are operational
    metadata that may vary between runs.
    """

    requested_source_uri: str
    resolved_run_id: str | None
    resolved_evaluation_id: str
    resolved_regime_id: str
    artifact_digest: str | None
    task: str
    model_family: str
    code_revision: str | None
    selection_strategy: str
    selected_case_ids: tuple[str, ...]
    preparation_timestamp: str
    materialized_from: str | None = None

    @classmethod
    def load(cls, root: Path) -> ReportDataProvenance:
        """Load ``provenance.json`` from *root*.

        Args:
            root: Package root directory.

        Returns:
            A validated ``ReportDataProvenance``.

        Raises:
            InvalidPackageError: If the file is missing or unparseable.
        """
        path = root / "provenance.json"
        if not path.exists():
            raise InvalidPackageError(
                f"Missing provenance.json in report-data package: {root}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise InvalidPackageError(
                f"Invalid provenance.json at {path}: {exc}"
            ) from exc

        source = raw.get("source", {})
        preparation = raw.get("preparation", {})
        selection = raw.get("selection", {})

        return cls(
            requested_source_uri=source.get("requested_uri", ""),
            resolved_run_id=source.get("run_id"),
            resolved_evaluation_id=source.get("evaluation_id", ""),
            resolved_regime_id=source.get("regime_id", ""),
            artifact_digest=source.get("artifact_digest"),
            task=source.get("task", ""),
            model_family=source.get("model_family", ""),
            code_revision=source.get("code_revision"),
            selection_strategy=selection.get("strategy", ""),
            selected_case_ids=tuple(selection.get("selected_case_ids", [])),
            preparation_timestamp=preparation.get("timestamp", ""),
            materialized_from=preparation.get("materialized_from"),
        )


# =============================================================================
# Resource handle
# =============================================================================


@dataclass(frozen=True)
class ReportResource:
    """Handle to one named resource in a report-data package.

    Use :meth:`read` to load the resource into memory.  The returned
    type depends on ``format``: ``pd.DataFrame`` for CSV and Parquet,
    ``dict`` or ``list`` for JSON.

    Attributes:
        package_root: Resolved absolute path of the package root.
        name: Logical resource name.
        relative_path: Relative path from the package root.
        format: Resource format (``"csv"``, ``"parquet"``, ``"json"``).
        mediatype: IANA media type string.
        schema_: Optional Table Schema mapping.
    """

    package_root: Path
    name: str
    relative_path: Path
    format: str
    mediatype: str
    schema_: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate path safety on construction."""
        # Must be relative.
        if self.relative_path.is_absolute():
            raise InvalidPackageError(
                f"Resource {self.name!r} has absolute path: {self.relative_path}"
            )
        # Must not contain '..' traversal.
        if ".." in self.relative_path.parts:
            raise InvalidPackageError(
                f"Resource {self.name!r} path contains '..' traversal: "
                f"{self.relative_path}"
            )
        # Resolved path must be within package_root.
        resolved = (self.package_root / self.relative_path).resolve()
        if not str(resolved).startswith(str(self.package_root.resolve())):
            raise InvalidPackageError(
                f"Resource {self.name!r} path resolves outside package root: "
                f"{self.relative_path}"
            )

    @property
    def path(self) -> Path:
        """Absolute path to the resource file on disk."""
        return (self.package_root / self.relative_path).resolve()

    def read(self) -> ReportResourceValue:
        """Read the resource into memory.

        Returns:
            ``pd.DataFrame`` for CSV and Parquet resources;
            ``dict[str, object]`` or ``list[object]`` for JSON resources.

        Raises:
            ResourceFormatError: If the format is unsupported or the
                file cannot be parsed.
            FileNotFoundError: If the resource file is missing.
        """
        abspath = self.path
        if not abspath.exists():
            raise FileNotFoundError(
                f"Resource {self.name!r} file not found: {abspath}"
            )

        fmt = self.format.lower()

        if fmt == "csv":
            try:
                return pd.read_csv(abspath)
            except Exception as exc:
                raise ResourceFormatError(
                    f"Failed to read CSV resource {self.name!r} "
                    f"at {abspath}: {exc}"
                ) from exc

        if fmt == "parquet":
            try:
                return pd.read_parquet(abspath)
            except Exception as exc:
                raise ResourceFormatError(
                    f"Failed to read Parquet resource {self.name!r} "
                    f"at {abspath}: {exc}"
                ) from exc

        if fmt == "json":
            try:
                raw = json.loads(abspath.read_text(encoding="utf-8"))
                return raw
            except Exception as exc:
                raise ResourceFormatError(
                    f"Failed to read JSON resource {self.name!r} "
                    f"at {abspath}: {exc}"
                ) from exc

        raise ResourceFormatError(
            f"Unsupported resource format {self.format!r} for "
            f"resource {self.name!r}. Supported: csv, parquet, json."
        )


# =============================================================================
# Package
# =============================================================================


@dataclass(frozen=True)
class ReportDataPackage:
    """Read-only handle to a prepared report-data package.

    Open via :meth:`open` or the :func:`open_report` convenience.

    The package must contain ``datapackage.json``, ``provenance.json``,
    ``_SUCCESS``, and all resources declared in the descriptor.
    """

    root: Path
    descriptor: DataPackageDescriptor
    provenance: ReportDataProvenance

    @classmethod
    def open(cls, root: str | Path) -> ReportDataPackage:
        """Open a report-data package at *root*.

        Validates the presence of required metadata files and all
        declared resources.

        Args:
            root: Path to the package directory containing
                ``datapackage.json``, ``provenance.json``, and
                ``_SUCCESS``.

        Returns:
            A fully validated ``ReportDataPackage``.

        Raises:
            InvalidPackageError: If the package is incomplete or
                malformed.
        """
        root = Path(root).resolve()

        # Required sentinel and metadata files.
        if not (root / "_SUCCESS").exists():
            raise InvalidPackageError(
                f"Missing _SUCCESS sentinel in report-data package: {root}"
            )

        descriptor = DataPackageDescriptor.load(root)
        provenance = ReportDataProvenance.load(root)

        # Validate all declared resources exist.
        for res_desc in descriptor.resources:
            resource_path = root / res_desc.path
            if not resource_path.exists():
                raise InvalidPackageError(
                    f"Resource {res_desc.name!r} declared in datapackage.json "
                    f"but file not found: {resource_path}"
                )

        return cls(root=root, descriptor=descriptor, provenance=provenance)

    def resource(self, name: str) -> ReportResource:
        """Get a handle to the named resource.

        Args:
            name: Logical resource name.

        Returns:
            A :class:`ReportResource` for reading.

        Raises:
            InvalidPackageError: If the resource name is not declared
                in the descriptor.
        """
        for res_desc in self.descriptor.resources:
            if res_desc.name == name:
                return ReportResource(
                    package_root=self.root,
                    name=res_desc.name,
                    relative_path=Path(res_desc.path),
                    format=res_desc.format,
                    mediatype=res_desc.mediatype,
                    schema_=res_desc.schema_,
                )
        raise InvalidPackageError(
            f"Resource {name!r} not found in report-data package "
            f"at {self.root}. Available: "
            f"{[r.name for r in self.descriptor.resources]}"
        )


# =============================================================================
# Convenience
# =============================================================================


def open_report(path: str | Path) -> ReportDataPackage:
    """Open a report-data package by path.

    Convenience wrapper around :meth:`ReportDataPackage.open`.

    Args:
        path: Path to the package directory.

    Returns:
        A :class:`ReportDataPackage`.
    """
    return ReportDataPackage.open(path)

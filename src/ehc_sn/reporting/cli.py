"""Report-data CLI — ``ehp report`` subcommand group.

Registered from ``ehp_sn/cli.py``.

Usage::

    ehp report validate config/reporting/arena-tem-diagnostic.toml
    ehp report prepare config/reporting/arena-tem-diagnostic.toml --output reports/arena-tem/data
    ehp report inspect reports/arena-tem/data
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.reporting.errors import (
    InvalidPackageError,
    ReportPreparationError,
    ReportRequestError,
)
from ehc_sn.reporting.package import open_report
from ehc_sn.reporting.preparation import prepare_report_data
from ehc_sn.reporting.request import load_report_data_request

# =============================================================================
report_app = typer.Typer(
    name="report",
    no_args_is_help=True,
    help="Prepare, validate, and inspect report-data packages.",
)


# =============================================================================
@report_app.command("validate")
def validate_report_request(
    request_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the report request TOML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    check_source: Annotated[
        bool,
        typer.Option(
            "--check-source",
            help="Also validate source compatibility (materialises source).",
        ),
    ] = False,
) -> None:
    """Validate a ``report.toml`` request file.

    Structural validation is always performed and requires no source
    access.  Pass ``--check-source`` to also verify source compatibility.
    """
    try:
        request = load_report_data_request(request_path)
    except ReportRequestError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Request schema: {request.request_schema}")
    typer.echo(f"Name:          {request.name}")
    typer.echo(f"Source URI:    {request.source.uri}")
    typer.echo(f"Regime:        {request.source.regime}")
    typer.echo(
        f"Resources:     metrics={request.resources.metrics}, "
        f"validations={request.resources.validations}, "
        f"cases={request.resources.cases}, "
        f"predictions={request.resources.predictions}, "
        f"traces={request.resources.traces}, "
        f"derived={list(request.resources.derived)}"
    )
    typer.echo(f"Materialisation: {request.materialization.mode}")

    if not check_source:
        typer.echo(
            "Structural validation passed (use --check-source for source validation)."
        )
        return

    # Source-aware validation: attempt materialisation.
    from ehc_sn.reporting.sources import materialize_evaluation_source

    try:
        src = materialize_evaluation_source(request.source.uri)
    except (FileNotFoundError, OSError) as exc:
        typer.echo(f"ERROR: Source unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Source resolved: {src.local_root} (backend={src.backend})")
    typer.echo("Source validation passed.")


# =============================================================================
@report_app.command("prepare")
def prepare_report(
    request_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the report request TOML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "--output",
            "-o",
            help="Output directory for the generated report-data package.",
        ),
    ],
) -> None:
    """Prepare a report-data package from a request."""
    try:
        request = load_report_data_request(request_path)
    except ReportRequestError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        pkg = prepare_report_data(request, output=output)
    except ReportPreparationError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Report-data package written to: {pkg.root}")
    typer.echo(f"  Resources: {len(pkg.descriptor.resources)}")
    for r in pkg.descriptor.resources:
        typer.echo(f"    - {r.name} ({r.format}, {r.path})")


# =============================================================================
@report_app.command("inspect")
def inspect_report_package(
    package_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to a prepared report-data package directory.",
        ),
    ],
) -> None:
    """Display descriptor, provenance, and resources of a report-data package."""
    try:
        pkg = open_report(package_path)
    except (InvalidPackageError, FileNotFoundError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Package:         {pkg.root}")
    typer.echo(f"Name:            {pkg.descriptor.name}")
    typer.echo(f"Profile:         {pkg.descriptor.profile}")
    typer.echo(f"Provenance source: {pkg.provenance.requested_source_uri}")
    typer.echo(f"Provenance regime: {pkg.provenance.resolved_regime_id}")
    typer.echo(f"Provenance task:   {pkg.provenance.task}")
    typer.echo(f"Selected cases:    {len(pkg.provenance.selected_case_ids)}")
    typer.echo("")
    typer.echo("Resources:")
    for r in pkg.descriptor.resources:
        ehp_info = ""
        if r.ehp:
            kind = r.ehp.get("kind", "")
            ehp_info = f"  [ehp: {kind}]"
        typer.echo(f"  - {r.name} ({r.format}, {r.path}){ehp_info}")

"""Report-data request — editable human-maintained ``report.toml`` contract.

Usage::

    from ehc_sn.reporting.request import ReportDataRequest, load_report_data_request

    request = load_report_data_request(Path("config/reporting/arena-tem-diagnostic.toml"))
    request.source.uri
    request.selection.max_cases
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from ehc_sn.reporting.errors import ReportRequestError

# =============================================================================
# Request models
# =============================================================================


class ReportSourceRequest(BaseModel):
    """Which evaluation source to extract report data from.

    Attributes:
        uri: Local path, ``runs:/<run_id>/<path>``, or other supported
            evaluation artifact URI.
        regime: Regime ID within the evaluation (e.g. ``"diagnostic"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(
        ..., min_length=1, description="Evaluation artifact URI or local path."
    )
    regime: str = Field(
        default="diagnostic", description="Regime ID within the evaluation."
    )


class ReportCaseSelection(BaseModel):
    """Which cases from the evaluation regime to include.

    ``case_ids`` and ``strategy`` + ``max_cases`` are mutually exclusive:
    when ``case_ids`` is non-empty, ``strategy`` and ``max_cases`` are
    ignored.

    Attributes:
        strategy: Selection strategy when ``case_ids`` is empty.
        max_cases: Maximum number of cases to select.
        case_ids: Explicit case IDs to include, overriding strategy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["balanced", "first_n"] = Field(
        default="balanced",
        description="Selection strategy when ``case_ids`` is empty.",
    )
    max_cases: int = Field(
        default=8, ge=1, description="Maximum number of cases to select."
    )
    case_ids: tuple[str, ...] = Field(
        default=(),
        description="Explicit case IDs, mutually exclusive with strategy + max_cases.",
    )

    @model_validator(mode="after")
    def _check_case_ids_exclusivity(self) -> Self:
        if self.case_ids and self.max_cases < len(self.case_ids):
            raise ValueError(
                f"case_ids has {len(self.case_ids)} entries but max_cases={self.max_cases}. "
                "When case_ids is non-empty, max_cases must be >= the number of IDs."
            )
        return self


class ReportResourceSelection(BaseModel):
    """Which resource types to include in the report-data package.

    Attributes:
        metrics: Include ``metrics.csv``.
        validations: Include ``validations.csv``.
        cases: Include ``cases.parquet`` and ``selected-cases.json``.
        predictions: Include per-case prediction artifacts.
        traces: Include per-case trace artifacts.
        derived: Names of derived resources to build (e.g. ``"pathway_metrics"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: bool = Field(default=True, description="Include metrics.csv.")
    validations: bool = Field(
        default=True, description="Include validations.csv."
    )
    cases: bool = Field(
        default=True,
        description="Include cases.parquet and selected-cases.json.",
    )
    predictions: bool = Field(
        default=False, description="Include per-case prediction artifacts."
    )
    traces: bool = Field(
        default=False, description="Include per-case trace artifacts."
    )
    derived: tuple[str, ...] = Field(
        default=(),
        description="Names of derived resources to build (e.g. 'pathway_metrics').",
    )


class MaterializationPolicy(BaseModel):
    """How resource files are stored in the package.

    Attributes:
        mode: Only ``"copy"`` is supported initially. ``"reference"``
            is reserved for future use.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["copy"] = Field(
        default="copy",
        description="Materialization mode. Only 'copy' is supported.",
    )


class ReportDataRequest(BaseModel):
    """Complete report-data request — a human-editable specification of
    which scientific resources to extract from a completed evaluation.

    This model is the TOML-level contract.  It is never modified by
    automated processes.

    The request selects one evaluation regime and describes which
    resources to materialize into the output Data Package.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True
    )

    request_schema: Literal["ehp_sn.report.request.v1"] = Field(
        ..., alias="schema", description="Request schema identifier."
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name for this report-data package.",
    )
    source: ReportSourceRequest = Field(
        ..., description="Evaluation source to extract data from."
    )
    selection: ReportCaseSelection = Field(
        default_factory=ReportCaseSelection,
        description="Case selection parameters.",
    )
    resources: ReportResourceSelection = Field(
        default_factory=ReportResourceSelection,
        description="Which resource types to include.",
    )
    materialization: MaterializationPolicy = Field(
        default_factory=MaterializationPolicy,
        description="Materialization mode (only 'copy' supported).",
    )


# =============================================================================
# Loader
# =============================================================================


def load_report_data_request(path: Path) -> ReportDataRequest:
    """Load a ``report.toml`` file and return a validated
    :class:`ReportDataRequest`.

    Args:
        path: Path to the TOML file.

    Returns:
        A validated ``ReportDataRequest``.

    Raises:
        ReportRequestError: If the file is missing, unparseable, or
            fails structural validation.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise ReportRequestError(f"Report request file not found: {path}")

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ReportRequestError(
            f"Failed to parse report request TOML at {path}: {exc}"
        ) from exc

    try:
        return ReportDataRequest.model_validate(raw)
    except Exception as exc:
        raise ReportRequestError(
            f"Invalid report request at {path}: {exc}"
        ) from exc

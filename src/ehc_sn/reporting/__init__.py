"""Checkpoint identity, report schema contracts, and future report-context
utilities.

This package must remain free of ``lightning/``, ``eval/``, ``models/``,
``tasks/``, ``adapters/``, and benchmark imports.  Execution-level bindings
belong in model-aware packages at Layer 3 or above.
"""

from ehc_sn.reporting.builder import build_report_run
from ehc_sn.reporting.collection import (  # noqa: E402
    ReportCollection,
    build_or_load_report_collection,
)
from ehc_sn.reporting.figures import (
    OfflineReportFigureContextSettings,
    OfflineReportFigureEntrySettings,
    OfflineReportRenderSettings,
    ReportFigureRenderer,
    ReportFigureRendererRegistry,
    load_figure_index,
    render_report_figures_from_run,
    write_figure_index,
)
from ehc_sn.reporting.inspection import OpenReport, open_report  # noqa: E402
from ehc_sn.reporting.loaders import (
    discover_eval_artifacts,
    load_eval_artifact_manifest,
    load_report_spec,
    select_eval_artifacts,
)
from ehc_sn.reporting.metrics.records import (
    load_metric_records,
    write_metric_records,
)
from ehc_sn.reporting.notebook import load_report_collection  # noqa: E402
from ehc_sn.reporting.pipeline import (
    OfflineEvalJobSpec,
    ReportPipelineSpec,
    build_report_run_from_checkpoint,
)
from ehc_sn.reporting.provenance import (
    CodeProvenance,
    ReportRunProvenance,
)
from ehc_sn.reporting.reader import ReportRun, load_report_run
from ehc_sn.reporting.render import load_render_manifest, render_report_run
from ehc_sn.reporting.schema import (
    CheckpointSpec,
    EvalArtifactReference,
    EvalArtifactSchema,
    FigureFormat,
    FigureIndex,
    FigureIndexEntry,
    FigureRenderSpec,
    MetricRecord,
    MetricSuiteSpec,
    RegimeKind,
    RegimeSelector,
    RenderedOutput,
    RenderManifest,
    ReportRenderSpec,
    ReportRunManifest,
    ReportRunSchema,
    ReportSpec,
    ReportSpecSchema,
)
from ehc_sn.reporting.writer import write_report_run_manifest

"""Post-hoc scientific measurements of model behavior and learned representations.

This package contains pure analysis primitives for computing scientific
quantities from trained model representations.  It must not import from
``ehp_sn.figures``, ``ehp_sn.metrics``, ``ehp_sn.reporting``,
``ehp_sn.diagnostics``, or ``ehp_sn.training``.
"""

from __future__ import annotations

from ehc_sn.analysis.registries import (
    AggregateRegistry,
    AnalysisRegistry,
    EvaluationRegistries,
    register_builtin_analysis_specs,
)
from ehc_sn.analysis.specs import AggregateSpec, AnalysisSpec, ConsumerPlacement


# Lazy imports for compiler (avoid circular import with evaluation/application).
def compile_figure_evaluation_plan(*args: object, **kwargs: object) -> object:
    from ehc_sn.analysis.compiler import compile_figure_evaluation_plan as _impl

    return _impl(*args, **kwargs)


def _get_compiled_figure_plan() -> type:
    from ehc_sn.analysis.compiler import CompiledFigurePlan

    return CompiledFigurePlan


def _get_figure_plan_error() -> type:
    from ehc_sn.analysis.compiler import FigurePlanError

    return FigurePlanError


def _get_compiled_figure() -> type:
    from ehc_sn.analysis.compiler import CompiledFigure

    return CompiledFigure


def _get_compilation_request() -> type:
    from ehc_sn.analysis.compiler import FigureCompilationRequest

    return FigureCompilationRequest


CompiledFigurePlan = _get_compiled_figure_plan()
FigurePlanError = _get_figure_plan_error()
CompiledFigure = _get_compiled_figure()
FigureCompilationRequest = _get_compilation_request()


# Lazy function wrappers.
def compile_figure_plan(*args: object, **kwargs: object) -> object:
    from ehc_sn.analysis.compiler import compile_figure_plan as _impl

    return _impl(*args, **kwargs)


def compile_inspection_plan(*args: object, **kwargs: object) -> object:
    from ehc_sn.analysis.compiler import compile_inspection_plan as _impl

    return _impl(*args, **kwargs)


def compile_report_plan(*args: object, **kwargs: object) -> object:
    from ehc_sn.analysis.compiler import compile_report_plan as _impl

    return _impl(*args, **kwargs)


# Lazy import for definitions (avoids circular import with figures/evaluation).
def get_definitions() -> object:
    from ehc_sn.analysis.definitions import get_definitions as _impl

    return _impl()


__all__ = [
    "AggregateRegistry",
    "AggregateSpec",
    "AnalysisRegistry",
    "AnalysisSpec",
    "CompiledFigure",
    "CompiledFigurePlan",
    "ConsumerPlacement",
    "EvaluationRegistries",
    "FigureCompilationRequest",
    "FigurePlanError",
    "compile_figure_evaluation_plan",
    "compile_figure_plan",
    "compile_inspection_plan",
    "compile_report_plan",
    "get_definitions",
    "register_builtin_analysis_specs",
]

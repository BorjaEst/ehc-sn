"""Core figure template infrastructure: base class and panel decorators."""

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.core.prediction_reasoning import PredictionReasoningTemplate
from ehc_sn.figures.core.task_overview import TaskOverviewTemplate

__all__ = [
    "BaseFigureTemplate",
    "PredictionReasoningTemplate",
    "TaskOverviewTemplate",
    "colorbar",
    "panel",
]

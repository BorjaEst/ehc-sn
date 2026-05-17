"""Core figure template infrastructure: base class and panel decorators."""

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel

__all__ = ["BaseFigureTemplate", "colorbar", "panel"]

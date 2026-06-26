"""Rendering primitives shared by figure templates.

Each task module provides standalone functions that map task-specific data
onto matplotlib axes, decoupled from any template or selector.

Extracted from ``task_overview_*`` templates so that both ``task_overview_*``
and ``prediction_reasoning_*`` families delegate to the same visual encoding.
"""

from ehc_sn.figures.renders.goaltrace import render_goaltrace_field
from ehc_sn.figures.renders.mazehard import render_maze_path_field
from ehc_sn.figures.renders.routebind import render_routebind_trajectory

__all__ = [
    "render_goaltrace_field",
    "render_maze_path_field",
    "render_routebind_trajectory",
]

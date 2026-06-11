"""MazeHard deliberation experiments.

Each module wires a (model family, version) into its Lightning regime.
"""

from ehc_sn.experiments.mazehard import hrm_v1, hrm_v2

__all__ = ["hrm_v1", "hrm_v2"]

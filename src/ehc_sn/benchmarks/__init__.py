"""Canonical benchmark MVP owner for ready tracks.

This package owns benchmark semantics and support-matrix gating. Script entry
points under ``scripts/benchmarks`` remain thin wrappers around this surface.
"""

from .model_comparison import (
    run_ready_track_model_comparison,
    run_ready_track_model_comparison_seeds,
)
from .runner import (
    READY_TRACKS,
    build_score_report,
    build_track_report,
    build_track_report_from_seed_scores,
    resolve_track_id,
    write_track_report,
)

__all__ = [
    "READY_TRACKS",
    "build_score_report",
    "build_track_report",
    "build_track_report_from_seed_scores",
    "run_ready_track_model_comparison",
    "run_ready_track_model_comparison_seeds",
    "resolve_track_id",
    "write_track_report",
]

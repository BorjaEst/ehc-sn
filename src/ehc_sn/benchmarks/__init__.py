"""Canonical benchmark MVP owner for ready tracks.

This package owns benchmark semantics and support-matrix gating. Script entry
points under ``scripts/benchmarks`` remain thin wrappers around this surface.
"""

from .runner import (
    READY_TRACKS,
    build_score_report,
    build_track_report,
    resolve_track_id,
    write_track_report,
)

__all__ = [
    "READY_TRACKS",
    "build_score_report",
    "build_track_report",
    "resolve_track_id",
    "write_track_report",
]

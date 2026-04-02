"""Timing helpers for benchmark execution."""

from __future__ import annotations

from time import perf_counter


def elapsed_seconds(start_time: float) -> float:
    """Return the elapsed wall-clock seconds since ``start_time``."""
    return perf_counter() - start_time


__all__ = ["elapsed_seconds"]

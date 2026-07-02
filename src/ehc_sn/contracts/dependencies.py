"""Declarative dependency vocabulary shared by evaluation and traces subsystems.

Both ``traces.observer`` and ``evaluation.contracts`` depend on this module;
it must not import from either.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DependencyKind(StrEnum):
    MODEL_VIEW = "model_view"
    RECORD_FIELD = "record_field"
    RUN_METADATA = "run_metadata"


@dataclass(frozen=True, order=True)
class Dependency:
    """A typed dependency on a model view, record field, or run metadata key."""

    kind: DependencyKind
    name: str


def model_view(name: str) -> Dependency:
    """Convenience constructor for a model-view dependency."""
    return Dependency(DependencyKind.MODEL_VIEW, name)


def record_field(name: str) -> Dependency:
    """Convenience constructor for a record-field dependency."""
    return Dependency(DependencyKind.RECORD_FIELD, name)


def run_metadata(name: str) -> Dependency:
    """Convenience constructor for a run-metadata dependency."""
    return Dependency(DependencyKind.RUN_METADATA, name)

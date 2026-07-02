"""Model trace capabilities — static declarations of what fields each model
architecture can emit during evaluation.

Capabilities are derived from the same implementation contracts that define
trace emission.  They are populated by hand, never queried from a live model.

Usage::

    from ehc_sn.traces.capabilities import TRACE_CAPABILITIES, resolve_capture_spec

    caps = TRACE_CAPABILITIES.get(("tem", "tem-v1"))
    resolved = resolve_capture_spec(profile, "tem", "tem-v1")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ehc_sn.traces.specs import CaptureProfileSpec


# =============================================================================
@dataclass(frozen=True)
class TraceCapabilities:
    """Static declaration of which trace fields a model family can produce.

    Attributes
    ----------
    paradigm:
        Execution paradigm (``"tem"``, ``"act"``, ``"rl"``, ``"ehp"``).
    model_family:
        Model-family identifier (e.g. ``"tem-v1"``, ``"hrm-v1"``).
    fields:
        Set of trace field paths the model can emit.
    """

    paradigm: str
    model_family: str
    fields: frozenset[str] = field(default_factory=frozenset)


# =============================================================================
@dataclass(frozen=True)
class ResolvedCaptureSpec:
    """Result of intersecting a capture profile with model capabilities.

    ``fields`` returns the union of required and available-optional fields —
    the set that can be safely used by figures.
    """

    required_fields: frozenset[str] = field(default_factory=frozenset)
    available_optional_fields: frozenset[str] = field(default_factory=frozenset)
    unavailable_optional_fields: frozenset[str] = field(
        default_factory=frozenset
    )

    @property
    def fields(self) -> frozenset[str]:
        """All fields this model/profile combination can produce."""
        return self.required_fields | self.available_optional_fields


# =============================================================================
# Registry: (paradigm, model_family) -> TraceCapabilities
#
# Populated from actual evaluation trace output.  Empty fields means the
# model family has no trace-capability declaration yet — validation skips
# resolution for that entry.
# =============================================================================

TRACE_CAPABILITIES: dict[tuple[str, str], TraceCapabilities] = {
    # --- TEM v1 ---
    # Resolved fields from arena-tem-v1 evaluation:
    #   act/halted, act/steps, pred/observation_id/path,
    #   pred/observation_id/post, pred/observation_id/recall,
    #   protocol/is_revisit, world_step/observation,
    #   world_step/observation_id
    ("tem", "tem-v1"): TraceCapabilities(
        paradigm="tem",
        model_family="tem-v1",
        fields=frozenset(
            {
                "act/halted",
                "act/steps",
                "pred/observation_id/path",
                "pred/observation_id/post",
                "pred/observation_id/recall",
                "protocol/is_revisit",
                "world_step/observation",
                "world_step/observation_id",
            }
        ),
    ),
    # --- TEM v2 ---
    # Currently same as v1.  Update when TEM v2 evaluation trace is
    # inspected and shows additional fields.
    ("tem", "tem-v2"): TraceCapabilities(
        paradigm="tem",
        model_family="tem-v2",
        fields=frozenset(
            {
                "act/halted",
                "act/steps",
                "pred/observation_id/path",
                "pred/observation_id/post",
                "pred/observation_id/recall",
                "protocol/is_revisit",
                "world_step/observation",
                "world_step/observation_id",
            }
        ),
    ),
    # --- HRM v1 ---
    # TODO: populate from actual mazehard-hrm-v1 trace output.
    ("act", "hrm-v1"): TraceCapabilities(
        paradigm="act",
        model_family="hrm-v1",
        fields=frozenset(),
    ),
    # --- HRM v2 ---
    # TODO: populate from actual mazehard-hrm-v2 trace output.
    ("act", "hrm-v2"): TraceCapabilities(
        paradigm="act",
        model_family="hrm-v2",
        fields=frozenset(),
    ),
}


class CaptureCompatibilityError(ValueError):
    """Raised when a capture profile requires fields the model cannot produce."""


def resolve_capture_spec(
    profile: "CaptureProfileSpec",
    paradigm: str,
    model_family: str,
) -> ResolvedCaptureSpec:
    """Intersect a capture profile with model capabilities.

    Args:
        profile: The capture profile specification.
        paradigm: Execution paradigm (``"tem"``, ``"act"``, etc.).
        model_family: Model-family identifier (e.g. ``"tem-v1"``).

    Returns:
        ResolvedCaptureSpec with required/available/unavailable fields.

    Raises:
        CaptureCompatibilityError: If any required profile field is not
            in the model's capabilities.
    """
    caps = TRACE_CAPABILITIES.get((paradigm, model_family))
    if caps is None or not caps.fields:
        # No capability declaration — cannot validate.
        # Return empty resolution; caller may skip.
        return ResolvedCaptureSpec()

    profile_fields = profile.paradigm_fields.get(paradigm)
    if profile_fields is None:
        return ResolvedCaptureSpec()

    required = frozenset(profile_fields.required)
    optional = frozenset(profile_fields.optional)

    # Required fields must be in capabilities.
    missing_required = required - caps.fields
    if missing_required:
        raise CaptureCompatibilityError(
            f"Capture profile {profile.name!r} requires fields that "
            f"model {paradigm}/{model_family} cannot produce: "
            f"{', '.join(sorted(missing_required))}."
        )

    available_opt = optional & caps.fields
    unavailable_opt = optional - caps.fields

    return ResolvedCaptureSpec(
        required_fields=required,
        available_optional_fields=available_opt,
        unavailable_optional_fields=unavailable_opt,
    )


__all__ = [
    "CaptureCompatibilityError",
    "ResolvedCaptureSpec",
    "TRACE_CAPABILITIES",
    "TraceCapabilities",
    "resolve_capture_spec",
]

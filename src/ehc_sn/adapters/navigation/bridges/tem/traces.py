"""Navigation-TEM trace fields (phase 1).

These fields bind Navigation task semantics to TEM-specific output surfaces.
They live here — in the shared Navigation+TEM bridge namespace — because they
read from the normalised TEM controller output (``ctx.outputs.obs_logits``) and
from the navigation step payload (``ctx.carry.data``), both of which carry
task-model coupling that belongs at the adapter boundary, not in the root
Navigation adapter package.

Usage
-----
    from ehc_sn.adapters.navigation.bridges.tem.traces import (
        NAVIGATION_TEM_TRACE_FIELDS,
        select_navigation_tem_trace_fields,
    )
    # Default spec — all five phase-1 keys included:
    self.trace_specs = build_trace_spec("tem", extra_fields=NAVIGATION_TEM_TRACE_FIELDS)

    # Filtered eval spec — only the requested Navigation keys appended:
    nav_extra = select_navigation_tem_trace_fields(self._eval_trace_keys)
    trace_specs = build_trace_spec("tem", include_keys=self._eval_trace_keys, extra_fields=nav_extra)
"""

from __future__ import annotations

from typing import Iterable, Protocol

from torch import Tensor

from ehc_sn.tasks.navigation.evaluation import coerce_observation_ids
from ehc_sn.traces import TraceField, TraceValue

# =================================================================================================
# Minimal typed context for Navigation+TEM trace getters
# =================================================================================================


class _NavTEMCarryData(Protocol):
    """Minimal carry-data surface read by Navigation+TEM trace getters."""

    def __getitem__(self, key: str) -> Tensor: ...
    def get(self, key: str, default: Tensor | None = None) -> Tensor | None: ...


class _NavTEMCarry(Protocol):
    """Minimal controller carry state accessed by Navigation+TEM trace getters."""

    data: _NavTEMCarryData


class _NavTEMOutputs(Protocol):
    """Minimal TEM step-output surface consumed by Navigation+TEM trace getters.

    ``obs_logits`` is the ordered triple ``(inference, retrieved, ancestral)``,
    each of shape ``(B, obs_dim)``.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]


class _NavigationTEMTraceContext(Protocol):
    """Minimal trace-step context expected by Navigation+TEM phase-1 trace getters."""

    carry: _NavTEMCarry
    outputs: _NavTEMOutputs


# =================================================================================================
# Getter functions
# =================================================================================================


def _get_world_observation_id(ctx: _NavigationTEMTraceContext) -> TraceValue:
    """Current-step observation id aligned with this step's TEM outputs.

    Reads ``ctx.carry.data["observation_id"]`` and normalises to shape ``(B,)``
    via :func:`~ehc_sn.tasks.navigation.evaluation.coerce_observation_ids`.
    """
    obs_id: Tensor = ctx.carry.data["observation_id"]
    return coerce_observation_ids(obs_id).detach()


def _get_is_revisit(ctx: _NavigationTEMTraceContext) -> TraceValue:
    """Boolean revisit flag for each batch slot, shape ``(B,)``.

    Returns ``None`` when the step payload does not carry revisit annotations
    (e.g. first step before any visit bookkeeping is applied).
    """
    is_revisit: Tensor | None = ctx.carry.data.get("is_revisit")
    if is_revisit is None:
        return None
    return is_revisit.view(-1).bool().detach()


def _get_pred_obs_id_inference(ctx: _NavigationTEMTraceContext) -> TraceValue:
    """Predicted observation id from the inference pathway, shape ``(B,)``."""
    logits: Tensor = ctx.outputs.obs_logits[0]  # (B, obs_dim)
    return logits.detach().argmax(dim=-1)


def _get_pred_obs_id_retrieved(ctx: _NavigationTEMTraceContext) -> TraceValue:
    """Predicted observation id from the retrieved pathway, shape ``(B,)``."""
    logits: Tensor = ctx.outputs.obs_logits[1]  # (B, obs_dim)
    return logits.detach().argmax(dim=-1)


def _get_pred_obs_id_ancestral(ctx: _NavigationTEMTraceContext) -> TraceValue:
    """Predicted observation id from the ancestral pathway, shape ``(B,)``."""
    logits: Tensor = ctx.outputs.obs_logits[2]  # (B, obs_dim)
    return logits.detach().argmax(dim=-1)


# =================================================================================================
# Named field objects
# =================================================================================================

NAVIGATION_TEM_TRACE_WORLD_OBS_ID = TraceField(
    name="world_step/observation_id",
    get=_get_world_observation_id,
)

NAVIGATION_TEM_TRACE_IS_REVISIT = TraceField(
    name="protocol/is_revisit",
    get=_get_is_revisit,
)

NAVIGATION_TEM_TRACE_PRED_INFERENCE = TraceField(
    name="pred/observation_id/inference",
    get=_get_pred_obs_id_inference,
)

NAVIGATION_TEM_TRACE_PRED_RETRIEVED = TraceField(
    name="pred/observation_id/retrieved",
    get=_get_pred_obs_id_retrieved,
)

NAVIGATION_TEM_TRACE_PRED_ANCESTRAL = TraceField(
    name="pred/observation_id/ancestral",
    get=_get_pred_obs_id_ancestral,
)

NAVIGATION_TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    NAVIGATION_TEM_TRACE_WORLD_OBS_ID,
    NAVIGATION_TEM_TRACE_IS_REVISIT,
    NAVIGATION_TEM_TRACE_PRED_INFERENCE,
    NAVIGATION_TEM_TRACE_PRED_RETRIEVED,
    NAVIGATION_TEM_TRACE_PRED_ANCESTRAL,
)
"""All phase-1 Navigation TEM trace fields in canonical order."""


# =================================================================================================
def select_navigation_tem_trace_fields(
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the Navigation TEM trace fields matching a requested key set.

    Used in the filtered evaluation-spec rebuild path so that only the
    Navigation extra fields that overlap with *include_keys* are appended.
    ``build_trace_spec`` does not filter ``extra_fields`` by ``include_keys``,
    so callers must pre-filter via this helper before passing them in.

    Args:
        include_keys: Requested trace key names, or ``None`` to return all
            phase-1 Navigation fields.

    Returns:
        Filtered sub-tuple of :data:`NAVIGATION_TEM_TRACE_FIELDS`.
    """
    if include_keys is None:
        return NAVIGATION_TEM_TRACE_FIELDS
    requested = set(include_keys)
    return tuple(f for f in NAVIGATION_TEM_TRACE_FIELDS if f.name in requested)


# =================================================================================================
__all__ = [
    "NAVIGATION_TEM_TRACE_FIELDS",
    "NAVIGATION_TEM_TRACE_WORLD_OBS_ID",
    "NAVIGATION_TEM_TRACE_IS_REVISIT",
    "NAVIGATION_TEM_TRACE_PRED_INFERENCE",
    "NAVIGATION_TEM_TRACE_PRED_RETRIEVED",
    "NAVIGATION_TEM_TRACE_PRED_ANCESTRAL",
    "select_navigation_tem_trace_fields",
]

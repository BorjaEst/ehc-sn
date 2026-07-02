"""Trace field vocabulary and per-paradigm TraceSpec constructors.

Mirrors :mod:`ehp_sn.metrics.signals` (scalar diagnostic names) and
:mod:`ehp_sn.metrics.routes` (metric routing tables) for the temporal rollout
data path.  All :class:`~ehp_sn.traces.TraceField` definitions live
here so model files contain *no* trace wiring.

Usage
-----
    from ehc_sn.traces import build_trace_spec
    self.trace_spec = build_trace_spec("act")   # or "rl" / "tem" / "ehp"

Naming convention
-----------------
Trace keys follow the same namespace hierarchy as diagnostic signals:

    act/*      — ACT halting/stepping signals
    pred/*     — model output predictions
    value/*    — value / Q-function outputs
    reward/*   — environment reward signals
    policy/*   — policy / action outputs
    dopamine/* — reward prediction error signals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import torch
from torch import Tensor

from ehc_sn.contracts.dependencies import model_view
from ehc_sn.traces.observer import (
    StepContext,
    TraceField,
    TraceSpec,
    TraceValue,
)
from ehc_sn.types import MemoryEntry, MultiScaleView, ScaleMetadata


# =============================================================================
class ReplayableEnvironments:
    """Non-pytree wrapper for replayable environment metadata."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


# =============================================================================
# Common — usable in any executed rollout paradigm
# =============================================================================


def _get_halted(ctx: StepContext) -> TraceValue:
    return ctx.record.snapshot.halted


def _get_steps(ctx: StepContext) -> TraceValue:
    return ctx.record.snapshot.steps


def _get_input_ids_meta(ctx: StepContext) -> TraceValue:
    """Static token-id batch captured as metadata when figures request it."""
    value = ctx.record.snapshot.data.get("input_ids")
    return None if value is None else value


def _get_labels_meta(ctx: StepContext) -> TraceValue:
    """Static label batch captured as metadata when figures request it."""
    value = ctx.record.snapshot.data.get("labels")
    return None if value is None else value


TRACE_HALTED = TraceField(
    name="act/halted",
    get=_get_halted,
)
TRACE_STEPS = TraceField(
    name="act/steps",
    get=_get_steps,
)
TRACE_INPUT_IDS_META = TraceField(
    name="input_ids",
    get=_get_input_ids_meta,
    storage="meta",
)
TRACE_LABELS_META = TraceField(
    name="labels",
    get=_get_labels_meta,
    storage="meta",
)

COMMON_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_INPUT_IDS_META,
    TRACE_LABELS_META,
)


# =============================================================================
# ACT-specific — produced by ACTSupervisedScorer / ACTController
# =============================================================================


def _get_action_logits_act(ctx: StepContext) -> TraceValue:
    """Action logits over halt/continue from the raw ACT controller step."""
    logits_q: Tensor = ctx.record.outputs.action_logits  # (B, n_actions)
    return logits_q


TRACE_ACTION_LOGITS_ACT = TraceField(
    name="value/action_logits",
    get=_get_action_logits_act,
)

ACT_TRACE_FIELDS: tuple[TraceField, ...] = (TRACE_ACTION_LOGITS_ACT,)


# =============================================================================
# RL-specific — produced by HybridRLObjective.compute_step() via validation scorer
# =============================================================================


def _get_q_logits_rl(ctx: StepContext) -> TraceValue:
    """Value-control Q-values over actions from the RL controller."""
    return ctx.record.outputs.q_values


def _get_state_value_rl(ctx: StepContext) -> TraceValue:
    """Critic state-value estimates V(s) from the value head."""
    return ctx.record.outputs.state_value


def _get_reward_env(ctx: StepContext) -> TraceValue:
    """Scalar environment reward for each batch slot."""
    return ctx.record.outputs.reward.squeeze(-1)


def _get_action(ctx: StepContext) -> TraceValue:
    """Selected action index for each batch slot."""
    return ctx.record.outputs.sampled_action


def _get_rpe(ctx: StepContext) -> TraceValue:
    """Reward prediction error: reward − V(s)."""
    reward: Tensor = ctx.record.outputs.reward.squeeze(-1)
    value: Tensor = ctx.record.outputs.state_value.squeeze(-1)
    return reward - value


def _get_world_observation_tem(ctx: StepContext) -> TraceValue:
    """Current-step observation encoding aligned with this step's TEM outputs."""
    v = ctx.record.snapshot.data.get("observation_id")
    return None if v is None else v


def _get_world_location_ids_tem(ctx: StepContext) -> TraceValue:
    """Current-step location ids aligned with this step's TEM outputs."""
    v = ctx.record.snapshot.data.get("location_id")
    return None if v is None else v.squeeze(-1)


def _get_diagnostic_lec_cells_tem(ctx: StepContext) -> TraceValue:
    """Replayable LEC activations by frequency for diagnostic figures."""
    return ctx.views.get("lec.cells")


def _get_diagnostic_lec_filtered_tem(ctx: StepContext) -> TraceValue:
    """Replayable LEC filtered by frequency for diagnostic figures."""
    return ctx.views.get("lec.filtered")


def _get_diagnostic_lec_sensory_code_tem(ctx: StepContext) -> TraceValue:
    """Replayable raw sensory code entering LEC inference by frequency."""
    return ctx.views.get("lec.sensory_code")


def _get_diagnostic_mec_location_mean_tem(ctx: StepContext) -> TraceValue:
    """Replayable MEC location codes by frequency for diagnostic figures."""
    return ctx.views.get("mec.cells")


def _get_diagnostic_hpc_location_mean_tem(ctx: StepContext) -> TraceValue:
    """Replayable HPC grounded-location codes by frequency for diagnostic figures."""
    return ctx.views.get("hpc.cells")


def _get_diagnostic_hpc_memory_summary_tem(ctx: StepContext) -> TraceValue:
    """Replayable per-step HPC memory summary for diagnostic figures.

    Returns lightweight scalars (active slots, write count, occupancy)
    without materialising dense memory matrices.
    """
    return ctx.views.get("hpc.memory.summary")


def _get_diagnostic_hpc_memory_final_tem(ctx: StepContext) -> TraceValue:
    """Replayable final-step HPC memory state for diagnostic figures.

    Only captured when ``hpc.memory.final`` is explicitly requested;
    uses ``storage="meta"`` to avoid per-step dense matrix materialisation.
    """
    memory = ctx.views.get("hpc.memory.final")
    if memory is None:
        return None
    return {
        "g_cued": memory["g_cued"],
        "x_cued": memory["x_cued"],
    }


def _memory_entry_for_trace(memory: MemoryEntry) -> Tensor:
    """Return the canonical dense memory operator for trace storage."""
    return memory.to_dense()


def _get_lec_alpha_sigmoid_tem(ctx: StepContext) -> TraceValue:
    """Static sigmoid-transformed LEC filter alpha values captured in carry data."""
    value = ctx.record.snapshot.data.get("lec_alpha_sigmoid")
    if value is None:
        static_data = getattr(ctx.record.snapshot, "static_data", None)
        value = (
            None
            if static_data is None
            else static_data.get("lec_alpha_sigmoid")
        )
    return None if value is None else value


def _get_lec_w_f_sigmoid_tem(ctx: StepContext) -> TraceValue:
    """Static sigmoid-transformed LEC frequency weights captured in carry data."""
    value = ctx.record.snapshot.data.get("lec_w_f_sigmoid")
    if value is None:
        static_data = getattr(ctx.record.snapshot, "static_data", None)
        value = (
            None if static_data is None else static_data.get("lec_w_f_sigmoid")
        )
    return None if value is None else value


TRACE_Q_LOGITS_RL = TraceField(
    name="value/q_values",
    get=_get_q_logits_rl,
)
TRACE_STATE_VALUE_RL = TraceField(
    name="value/state_value",
    get=_get_state_value_rl,
)
TRACE_REWARD_ENV = TraceField(
    name="reward/env",
    get=_get_reward_env,
)
TRACE_ACTION = TraceField(
    name="policy/action",
    get=_get_action,
)
TRACE_RPE = TraceField(
    name="dopamine/rpe",
    get=_get_rpe,
)
TRACE_WORLD_OBSERVATION_TEM = TraceField(
    name="world_step/observation",
    get=_get_world_observation_tem,
)
TRACE_WORLD_LOCATION_IDS_TEM = TraceField(
    name="world_step/location_ids",
    get=_get_world_location_ids_tem,
)
TRACE_DIAGNOSTIC_LEC_CELLS_TEM = TraceField(
    name="diagnostic/lec/cells",
    get=_get_diagnostic_lec_cells_tem,
    dependencies=frozenset({model_view("lec.cells")}),
)
TRACE_DIAGNOSTIC_LEC_FILTERED_TEM = TraceField(
    name="diagnostic/lec/filtered",
    get=_get_diagnostic_lec_filtered_tem,
    dependencies=frozenset({model_view("lec.filtered")}),
)
TRACE_DIAGNOSTIC_LEC_SENSORY_CODE_TEM = TraceField(
    name="diagnostic/lec/sensory_code",
    get=_get_diagnostic_lec_sensory_code_tem,
    dependencies=frozenset({model_view("lec.sensory_code")}),
)
TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM = TraceField(
    name="diagnostic/mec/location_mean",
    get=_get_diagnostic_mec_location_mean_tem,
    dependencies=frozenset({model_view("mec.cells")}),
)
TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM = TraceField(
    name="diagnostic/hpc/location_mean",
    get=_get_diagnostic_hpc_location_mean_tem,
    dependencies=frozenset({model_view("hpc.cells")}),
)
TRACE_DIAGNOSTIC_HPC_MEMORY_SUMMARY_TEM = TraceField(
    name="diagnostic/hpc/memory_summary",
    get=_get_diagnostic_hpc_memory_summary_tem,
    dependencies=frozenset({"hpc.memory.summary"}),
)
TRACE_DIAGNOSTIC_HPC_MEMORY_FINAL_TEM = TraceField(
    name="diagnostic/hpc/memory_final",
    get=_get_diagnostic_hpc_memory_final_tem,
    dependencies=frozenset({"hpc.memory.final"}),
    storage="meta",
)
TRACE_LEC_FILTER_ALPHA_SIGMOID_TEM = TraceField(
    name="lec/filter/alpha_sigmoid",
    get=_get_lec_alpha_sigmoid_tem,
)
TRACE_LEC_W_F_SIGMOID_TEM = TraceField(
    name="lec/w_f_sigmoid",
    get=_get_lec_w_f_sigmoid_tem,
)

RL_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_Q_LOGITS_RL,
    TRACE_STATE_VALUE_RL,
    TRACE_REWARD_ENV,
    TRACE_ACTION,
    TRACE_RPE,
)


# =============================================================================
# HRM/PFC hidden-state — additive diagnostic fields for HRM-family models
# =============================================================================


def _get_hrm_z_H(ctx: StepContext) -> TraceValue:
    """High-level HRM/PFC working-memory state.

    Expected shape ``(B, S+1, D)`` where *S+1* includes the controller
    slot.  Returns ``None`` when the view is unavailable.
    """
    z_H = ctx.views.get("pfc.z_H")
    if z_H is None:
        return None
    return z_H


def _get_hrm_z_L(ctx: StepContext) -> TraceValue:
    """Low-level HRM/PFC working-memory state.

    Expected shape ``(B, S+1, D)``.  Returns ``None`` when the view
    is unavailable.
    """
    z_L = ctx.views.get("pfc.z_L")
    if z_L is None:
        return None
    return z_L


TRACE_HRM_Z_H = TraceField(
    name="pfc/z_H",
    get=_get_hrm_z_H,
    storage="dense",
    dependencies=frozenset({"pfc.z_H"}),
)

TRACE_HRM_Z_L = TraceField(
    name="pfc/z_L",
    get=_get_hrm_z_L,
    storage="dense",
    dependencies=frozenset({"pfc.z_L"}),
)

HRM_HIDDEN_STATE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HRM_Z_H,
    TRACE_HRM_Z_L,
)

# WM task metadata — ground-truth labels for cue-dependent recall / working
# memory tasks.  These are placeholders; the full set of WM-specific fields
# will be populated when the cue-recall evaluation pipeline is implemented.
# For now, this tuple is deliberately empty to avoid NameError from the
# HRM_REASONING_TRACE_FIELDS reference below.
WM_TASK_METADATA_FIELDS: tuple[TraceField, ...] = ()

# Shared HRM reasoning fields — available to any HRM-family paradigm
# (act and rl are controller variants; the hidden-state and task-metadata
# surface is the same across ACT-style and RL-style HRM models.)
HRM_REASONING_TRACE_FIELDS: tuple[TraceField, ...] = (
    HRM_HIDDEN_STATE_FIELDS + WM_TASK_METADATA_FIELDS
)


# =============================================================================
# Goaltrace-specific — stored as metadata, consumed by figure selectors
# =============================================================================


def _get_goaltrace_firing_field(ctx: StepContext) -> TraceValue:
    """Predicted firing field from goaltrace task metadata."""
    return ctx.record.snapshot.data.get("firing_field")


def _get_goaltrace_target_field(ctx: StepContext) -> TraceValue:
    """Target firing field from goaltrace task metadata."""
    return ctx.record.snapshot.data.get("target_field")


def _get_goaltrace_node_mask(ctx: StepContext) -> TraceValue:
    """Node validity mask from goaltrace task metadata."""
    return ctx.record.snapshot.data.get("node_mask")


def _get_goaltrace_observation_id(ctx: StepContext) -> TraceValue:
    """Observation IDs from goaltrace task metadata."""
    return ctx.record.snapshot.data.get("observation_id")


GOALTRACE_TRACE_FIRING_FIELD = TraceField(
    name="goaltrace/firing_field",
    get=_get_goaltrace_firing_field,
    storage="meta",
)
GOALTRACE_TRACE_TARGET_FIELD = TraceField(
    name="goaltrace/target_field",
    get=_get_goaltrace_target_field,
    storage="meta",
)
GOALTRACE_TRACE_NODE_MASK = TraceField(
    name="goaltrace/node_mask",
    get=_get_goaltrace_node_mask,
    storage="meta",
)
GOALTRACE_TRACE_OBSERVATION_ID = TraceField(
    name="goaltrace/observation_id",
    get=_get_goaltrace_observation_id,
    storage="meta",
)

GOALTRACE_TRACE_FIELDS: tuple[TraceField, ...] = (
    GOALTRACE_TRACE_FIRING_FIELD,
    GOALTRACE_TRACE_TARGET_FIELD,
    GOALTRACE_TRACE_NODE_MASK,
    GOALTRACE_TRACE_OBSERVATION_ID,
)


# =============================================================================
# TEM-specific — TEM uses only the rollout-safe baseline fields for now.
# =============================================================================


TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_WORLD_OBSERVATION_TEM,
    TRACE_WORLD_LOCATION_IDS_TEM,
    TRACE_DIAGNOSTIC_LEC_CELLS_TEM,
    TRACE_DIAGNOSTIC_LEC_FILTERED_TEM,
    TRACE_DIAGNOSTIC_LEC_SENSORY_CODE_TEM,
    TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_MEMORY_SUMMARY_TEM,
    TRACE_DIAGNOSTIC_HPC_MEMORY_FINAL_TEM,
    TRACE_LEC_FILTER_ALPHA_SIGMOID_TEM,
    TRACE_LEC_W_F_SIGMOID_TEM,
)


# =============================================================================
def _select_trace_fields(  # --------------------------------------------------
    fields: tuple[TraceField, ...],
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the selected trace fields for a requested public key set."""
    if include_keys is None:
        return fields
    requested = {key for key in include_keys}
    return tuple(field for field in fields if field.name in requested)


# =============================================================================
def build_trace_spec(  # ------------------------------------------------------
    paradigm: Literal["act", "rl", "tem", "ehp"],
    *,
    include_keys: Iterable[str] | None = None,
    extra_fields: Iterable[TraceField] | None = None,
) -> TraceSpec:
    """Build a :class:`~ehp_sn.traces.TraceSpec` for a training paradigm.

    Returns common fields plus paradigm-specific fields.  Pass the returned
    spec to :class:`~ehp_sn.traces.TraceObserver` instead of
    constructing :class:`~ehp_sn.traces.TraceField` lists in model files.

    Args:
        paradigm: ``"act"`` for ACT-based models (hrm_v1) or
            ``"rl"`` for RL-based models (hrm_v2), or
            ``"tem"`` for TEM-based models (tem_v1), or
            ``"ehp"`` for EHP-based models (ehc_v1).

    Returns:
        A :class:`~ehp_sn.traces.TraceSpec` instance.

    Raises:
        ValueError: If *paradigm* is not ``"act"``, ``"rl"``, ``"tem"``, or ``"ehp"``.
    """
    if paradigm == "act":
        fields = _select_trace_fields(
            COMMON_TRACE_FIELDS
            + ACT_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS,
            include_keys,
        )
    elif paradigm == "rl":
        fields = _select_trace_fields(
            COMMON_TRACE_FIELDS
            + RL_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS,
            include_keys,
        )
    elif paradigm == "tem":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    elif paradigm == "ehp":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    else:
        raise ValueError(
            f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', 'tem', or 'ehp'."
        )
    if extra_fields is not None:
        fields = fields + tuple(extra_fields)
    return TraceSpec(fields=list(fields))


# =============================================================================
# Capture profiles — named sets of trace fields declared by analysis consumers
# ---------------------------------------------------------------------------
# Each profile maps paradigm → (required_keys, default_optional_keys).
# Required keys cannot be excluded.  Unsupported profile–paradigm combinations
# raise ValueError at experiment construction time.
# =============================================================================


@dataclass(frozen=True)
class CaptureProfileSpec:
    """A named capture profile with paradigm-specific field requirements.

    This is the consumer-facing contract: the fields a given analysis
    type (e.g. prediction-reasoning, full-diagnostic) needs to function.
    """

    name: str
    version: int
    description: str
    paradigm_fields: dict[str, CaptureParadigmBinding]


@dataclass(frozen=True)
class CaptureParadigmBinding:
    """Required and optional trace fields for one paradigm under a profile."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


# ── Helper: compose two bindings ─────────────────────────────────────────────


def _compose_bindings(
    base: CaptureParadigmBinding,
    additions: CaptureParadigmBinding,
) -> CaptureParadigmBinding:
    """Return a new binding with *additions* appended to *base*."""
    return CaptureParadigmBinding(
        required=base.required + additions.required,
        optional=base.optional + additions.optional,
    )


# ── Profile: prediction ──────────────────────────────────────────────────────
# Task outputs plus compact controller execution fields.

_PREDICTION_BINDINGS: dict[str, CaptureParadigmBinding] = {
    "act": CaptureParadigmBinding(
        required=("act/halted", "act/steps"),
        optional=(
            "value/action_logits",
            "input_ids",
            "pred/prediction_overlay",
            "goaltrace/firing_field",
            "routebind/trajectory_field",
        ),
    ),
    "rl": CaptureParadigmBinding(
        required=("act/halted", "act/steps"),
        optional=(
            "value/q_values",
            "value/state_value",
            "input_ids",
            "pred/prediction_overlay",
            "goaltrace/firing_field",
            "routebind/trajectory_field",
        ),
    ),
    "tem": CaptureParadigmBinding(
        required=(
            "pred/observation_id/post",
            "pred/observation_id/recall",
            "pred/observation_id/path",
        ),
        optional=(),
    ),
    "ehp": CaptureParadigmBinding(
        required=(
            "pred/observation_id/post",
            "pred/observation_id/recall",
            "pred/observation_id/path",
        ),
        optional=(),
    ),
}

# ── Profile: diagnostic ──────────────────────────────────────────────────────
# Everything in ``prediction`` plus internal representations, memory state,
# and supporting world-state fields.  Built by composition so the two profiles
# cannot diverge.

_DIAGNOSTIC_ADDITIONS: dict[str, CaptureParadigmBinding] = {
    "act": CaptureParadigmBinding(
        required=("pfc/z_H", "pfc/z_L"),
        optional=(),
    ),
    "rl": CaptureParadigmBinding(
        required=("pfc/z_H", "pfc/z_L"),
        optional=(),
    ),
    "tem": CaptureParadigmBinding(
        required=(
            "world_step/observation",
            "world_step/location_ids",
            "diagnostic/lec/cells",
            "diagnostic/lec/filtered",
            "diagnostic/mec/location_mean",
            "diagnostic/hpc/location_mean",
            "diagnostic/hpc/memory",
        ),
        optional=(),
    ),
    "ehp": CaptureParadigmBinding(
        required=(
            "world_step/observation",
            "world_step/location_ids",
            "diagnostic/lec/cells",
            "diagnostic/lec/filtered",
            "diagnostic/mec/location_mean",
            "diagnostic/hpc/location_mean",
            "diagnostic/hpc/memory",
        ),
        optional=(),
    ),
}

_DIAGNOSTIC_BINDINGS: dict[str, CaptureParadigmBinding] = {
    paradigm: _compose_bindings(
        _PREDICTION_BINDINGS[paradigm], _DIAGNOSTIC_ADDITIONS[paradigm]
    )
    for paradigm in ("act", "rl", "tem", "ehp")
}

# ── Profile: metrics_only ────────────────────────────────────────────────────
# No trace fields; only scalar metrics are collected.

_METRICS_ONLY_BINDINGS: dict[str, CaptureParadigmBinding] = {
    paradigm: CaptureParadigmBinding(required=())
    for paradigm in ("act", "rl", "tem", "ehp")
}

# ── Registry ─────────────────────────────────────────────────────────────────

TRACE_PROFILES: dict[str, CaptureProfileSpec] = {
    "metrics_only": CaptureProfileSpec(
        name="metrics_only",
        version=1,
        description="Scalar metrics and aggregate artifacts; no bounded trace capture.",
        paradigm_fields=_METRICS_ONLY_BINDINGS,
    ),
    "prediction": CaptureProfileSpec(
        name="prediction",
        version=1,
        description="Task outputs plus compact controller execution fields.",
        paradigm_fields=_PREDICTION_BINDINGS,
    ),
    "diagnostic": CaptureProfileSpec(
        name="diagnostic",
        version=1,
        description="Prediction fields plus internal representations, memory state, "
        "and supporting world-state fields.",
        paradigm_fields=_DIAGNOSTIC_BINDINGS,
    ),
}


def _paradigm_all_field_names(paradigm: str) -> set[str]:
    """Return the set of all known trace-field names for *paradigm*.

    Includes both paradigm-specific ``TraceField`` names and vocabulary
    field names (``traces.vocabulary``).  Vocabulary names are shared
    across tasks and are provided by ``trace_task_fields()`` in task
    packages.
    """
    if paradigm == "act":
        fields = (
            COMMON_TRACE_FIELDS
            + ACT_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS
        )
    elif paradigm == "rl":
        fields = (
            COMMON_TRACE_FIELDS
            + RL_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS
        )
    elif paradigm in ("tem", "ehp"):
        fields = TEM_TRACE_FIELDS
    else:
        raise ValueError(
            f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', 'tem', or 'ehp'."
        )

    known = {f.name for f in fields}
    from ehc_sn.traces.vocabulary import list_vocabulary_names as _list_voc

    known.update(_list_voc())
    return known


def resolve_capture_profile(
    paradigm: Literal["act", "rl", "tem", "ehp"],
    profile: str = "metrics_only",
    *,
    profile_version: int = 1,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    task_bindings: dict[str, CaptureParadigmBinding] | None = None,
    extra_fields: Iterable[TraceField] | None = None,
) -> TraceSpec:
    """Resolve a named capture profile to a concrete :class:`TraceSpec`.

    Parameters
    ----------
    paradigm:
        Trace paradigm for the experiment being evaluated.
    profile:
        Named capture profile registered in :data:`TRACE_PROFILES`.
    profile_version:
        Version of the profile contract.  Currently only version 1 exists; a
        mismatch raises ``ValueError``.
    include:
        Extra trace-field names to include beyond the profile.  Each name must
        correspond to a known trace field for *paradigm*.
    exclude:
        Profile field names to omit.  May only exclude *optional* profile
        fields — required profile fields raise ``ValueError``.
    task_bindings:
        Per-task field bindings appended to the profile's paradigm fields.
        Keys are task names (e.g. ``"goaltrace"``).  Each binding's required
        and optional fields are merged into the resolution.  Unknown field
        names in task bindings raise ``ValueError``.
    extra_fields:
        Additional ``TraceField`` objects to include in the resolved spec.
        Unlike *include* (which selects from known paradigm fields),
        *extra_fields* injects new fields with arbitrary getters (e.g.
        task-specific fields from ``trace_task_fields()``).  These are
        appended after all profile and include fields.

    Returns
    -------
    TraceSpec
        A concrete trace specification ready for use by the evaluation runner.

    Raises
    ------
    ValueError
        If *profile* is unknown, if *profile_version* does not match, if
        *paradigm* is unsupported for the profile, if *exclude* intersects
        required profile fields, if *include* references unknown field names,
        or if a task binding references unknown field names.
    """
    profile_spec = TRACE_PROFILES.get(profile)
    if profile_spec is None:
        raise ValueError(
            f"Unknown capture profile {profile!r}. "
            f"Available: {sorted(TRACE_PROFILES)}."
        )
    if profile_spec.version != profile_version:
        raise ValueError(
            f"Capture profile {profile!r} version mismatch: "
            f"requested v{profile_version}, available v{profile_spec.version}."
        )

    binding = profile_spec.paradigm_fields.get(paradigm)
    if binding is None:
        raise ValueError(
            f"Capture profile {profile!r} does not support paradigm "
            f"{paradigm!r}. Supported: {sorted(profile_spec.paradigm_fields)}."
        )

    # Collect required and optional from profile + task bindings.
    all_known = _paradigm_all_field_names(paradigm)
    required_set: set[str] = set(binding.required)
    optional_set: set[str] = set(binding.optional)

    if task_bindings:
        for task_name, task_binding in task_bindings.items():
            for f in task_binding.required:
                # Task bindings may reference task-specific meta-keys not in
                # the core paradigm field set (e.g. goaltrace/firing_field).
                # Include validation is relaxed; build_trace_spec will filter
                # to known fields.
                required_set.add(f)
            for f in task_binding.optional:
                optional_set.add(f)

    include_set = set(include)
    exclude_set = set(exclude)

    # Validate include keys are known.
    unknown_include = include_set - all_known
    if unknown_include:
        raise ValueError(
            f"Capture include keys unknown for paradigm {paradigm!r}: "
            f"{sorted(unknown_include)}. Known: {sorted(all_known)}."
        )

    # Validate exclude does not intersect required.
    forbidden_exclude = exclude_set & required_set
    if forbidden_exclude:
        raise ValueError(
            f"Cannot exclude required capture fields for profile "
            f"{profile!r}: {sorted(forbidden_exclude)}."
        )

    # Deterministic ordering: required (profile order), then optional (profile
    # order), then include (user order).  Uses dict.fromkeys for order-preserving
    # dedup.
    final_names: list[str] = []
    seen: set[str] = set()

    # 1. Required profile fields in declaration order.
    for f in binding.required:
        if f not in seen:
            final_names.append(f)
            seen.add(f)
    # 1b. Required task-binding fields (appended after profile required).
    if task_bindings:
        for tb in task_bindings.values():
            for f in tb.required:
                if f not in seen:
                    final_names.append(f)
                    seen.add(f)

    # 2. Optional profile fields in declaration order, minus exclude.
    for f in binding.optional:
        if f not in seen and f not in exclude_set:
            final_names.append(f)
            seen.add(f)
    # 2b. Optional task-binding fields.
    if task_bindings:
        for tb in task_bindings.values():
            for f in tb.optional:
                if f not in seen and f not in exclude_set:
                    final_names.append(f)
                    seen.add(f)

    # 3. Explicit includes in user order.
    for f in include:
        if f not in seen:
            final_names.append(f)
            seen.add(f)

    return build_trace_spec(
        paradigm,
        include_keys=final_names,
        extra_fields=extra_fields,
    )


# =============================================================================
__all__ = [
    "ReplayableEnvironments",
    "COMMON_TRACE_FIELDS",
    "ACT_TRACE_FIELDS",
    "RL_TRACE_FIELDS",
    "HRM_HIDDEN_STATE_FIELDS",
    "HRM_REASONING_TRACE_FIELDS",
    "TEM_TRACE_FIELDS",
    "GOALTRACE_TRACE_FIELDS",
    "TRACE_PROFILES",
    "CaptureProfileSpec",
    "CaptureParadigmBinding",
    "build_trace_spec",
    "resolve_capture_profile",
]

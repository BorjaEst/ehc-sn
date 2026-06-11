"""Cue-recall block-encoding diagnostic probe.

Produces a WM-contract-conformant eval artifact from the trained HRM v1
checkpoint using **block encoding**: three 300-slot blocks each carrying
one item prototype, with the cued block scaled at cue step.

This is a **research probe**, not a production task or adapter.  The
block encoding is an experimental schema-token representation chosen
because it produced the strongest between-episode separation in the
gating experiment (``scripts/probes/hrm_cue_recall_schema_probe.py``).

Public API
----------
    produce_cue_recall_block_artifact(model) -> TraceTree
        Run one episode and return the trace.

    persist_block_artifact(model, output_dir, *, episode_label) -> Path
        Produce and persist a WM-contract-conformant artifact bundle.

Usage::

    from ehc_sn.diagnostics.cue_recall_probe import persist_block_artifact
    from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
    import torch

    config = ModelSettingsV1.from_config("config/models/hrm-v1-base.toml")
    model = HRModelV1(config)
    sd = torch.load("checkpoints/hrm-v1/eval-weights-only.pt",
                    map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=False)
    model.eval()

    artifact_dir = persist_block_artifact(model, Path("/tmp/artifact"))
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ehc_sn.eval.artifacts import persist_regime_artifact_bundle
from ehc_sn.eval.contracts import EvaluationCaseResult, EvaluationRegimeResult
from ehc_sn.models.hrm.hrm_v1 import HRMInputV1, HRModelV1
from ehc_sn.traces import TraceTree
from ehc_sn.traces.wm_contract import (
    WM_PHASE_CUE,
    WM_PHASE_DELAY,
    WM_PHASE_INPUT,
    WM_PHASE_OUTPUT,
)

# ---------------------------------------------------------------------------
# Episode constants (Section 13.4 step layout)
#   step | phase   | item | position | cue | target | sample_id
#   ------+---------+------+----------+-----+--------+-----------
#   0    | input   | 10   | 0        | -1  | -1     | 1
#   1    | input   | 20   | 1        | -1  | -1     | 1
#   2    | input   | 30   | 2        | -1  | -1     | 1
#   3    | delay   | -1   | -1       | -1  | -1     | 1
#   4    | cue     | -1   | -1       | 20  | -1     | 1
#   5    | output  | -1   | -1       | -1  | 30     | 1
# ---------------------------------------------------------------------------

NUM_STEPS: int = 6
BATCH_SIZE: int = 1

PHASE_SEQUENCE: list[int] = [
    WM_PHASE_INPUT,
    WM_PHASE_INPUT,
    WM_PHASE_INPUT,
    WM_PHASE_DELAY,
    WM_PHASE_CUE,
    WM_PHASE_OUTPUT,
]

ITEMS: list[str] = ["A", "B", "C"]
CUE_POSITION: int = 1  # 0-based position to recall
TARGET: str = "B"

EXPECTED_ITEMS: list[int] = [10, 20, 30, -1, -1, -1]
EXPECTED_POSITIONS: list[int] = [0, 1, 2, -1, -1, -1]
EXPECTED_CUES: list[int] = [-1, -1, -1, -1, 20, -1]
EXPECTED_TARGETS: list[int] = [-1, -1, -1, -1, -1, 30]
EXPECTED_SAMPLE_ID: int = 1

# ---------------------------------------------------------------------------
# Prototypes
# ---------------------------------------------------------------------------


def _make_prototypes(model: HRModelV1, seed: int = 42) -> dict[str, Tensor]:
    """Generate fixed random prototype vectors for items A, B, C.

    Prototypes live in the model's hidden dimension and are scaled by
    0.1 to stay in a reasonable activation range.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    D = model.config.pfc.cortex.embedding_dim
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return {
        name: torch.randn(D, generator=g, device=device, dtype=dtype) * 0.1
        for name in ["A", "B", "C"]
    }


# =============================================================================
# Block encoding
# =============================================================================


def _build_block_schema_tokens(
    step_idx: int,
    model: HRModelV1,
    prototypes: dict[str, Tensor],
) -> Tensor:
    """Build schema tokens for one step using block encoding.

    Each position owns a 300-slot block:
        position 0 -> slots   0:300
        position 1 -> slots 300:600
        position 2 -> slots 600:900

    Within each block, the first 10 slots carry the item prototype.
    At the cue step (step >= 4), the cued-position block is scaled by 2.

    Returns:
        Tensor of shape ``(1, S, D)``.
    """
    S = model.config.num_schema_slots
    D = model.config.pfc.cortex.embedding_dim
    BLOCK_SIZE = S // 3
    ACTIVE_PER_BLOCK = 10

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tokens = torch.zeros(BATCH_SIZE, S, D, device=device, dtype=dtype)

    for i in range(3):
        start = i * BLOCK_SIZE
        end = start + ACTIVE_PER_BLOCK
        item_name = ITEMS[i]
        scale = 2.0 if (step_idx >= 4 and i == CUE_POSITION) else 1.0
        tokens[0, start:end] = prototypes[item_name] * scale

    return tokens


# =============================================================================
# Public API
# =============================================================================


def produce_cue_recall_block_artifact(
    model: HRModelV1,
) -> TraceTree:
    """Run one block-encoded cue-recall episode and return a TraceTree.

    The returned trace contains ``pfc/z_H``, ``pfc/z_L``, and all six
    ``task/*`` dense labels for a 6-step episode with phase sequence
    [0,0,0,1,2,3].

    Args:
        model: Loaded and weight-hydrated HRModelV1 in eval mode.

    Returns:
        TraceTree with 8 dense leaves, 6 time steps, batch size 1.
    """
    prototypes = _make_prototypes(model)
    state = None
    trace = TraceTree()

    for step_idx in range(NUM_STEPS):
        schema_tokens = _build_block_schema_tokens(step_idx, model, prototypes)
        payload = HRMInputV1(schema_tokens=schema_tokens)
        _, state = model.step(payload, state=state)

        z_H = state.pfc.scratch.memory.z_H.detach().cpu()
        z_L = state.pfc.scratch.memory.z_L.detach().cpu()

        trace.append(
            {
                "pfc/z_H": z_H,
                "pfc/z_L": z_L,
                "task/phase": torch.tensor([PHASE_SEQUENCE[step_idx]]),
                "task/item": torch.tensor([EXPECTED_ITEMS[step_idx]]),
                "task/position": torch.tensor([EXPECTED_POSITIONS[step_idx]]),
                "task/cue": torch.tensor([EXPECTED_CUES[step_idx]]),
                "task/target": torch.tensor([EXPECTED_TARGETS[step_idx]]),
                "task/sample_id": torch.tensor([EXPECTED_SAMPLE_ID]),
            }
        )

    trace.finalize()
    return trace


def persist_block_artifact(
    model: HRModelV1,
    output_dir: Path,
    *,
    episode_label: str = "cue_recall_trial_001",
) -> Path:
    """Produce and persist a block-encoded cue-recall WM artifact.

    Runs one episode, builds the TraceTree, wraps in
    ``EvaluationCaseResult`` / ``EvaluationRegimeResult``, persists
    via ``persist_regime_artifact_bundle``, and returns the resolved
    artifact directory path.

    Args:
        model: Loaded and weight-hydrated HRModelV1 in eval mode.
        output_dir: Target directory for the artifact bundle.
        episode_label: Case identifier written into the manifest.

    Returns:
        Resolved absolute path to the written artifact directory.
    """
    trace = produce_cue_recall_block_artifact(model)

    case_result = EvaluationCaseResult(
        case_id=episode_label,
        trace=trace,
        source_context={
            "encoding": "block_encoding",
            "items": ITEMS,
            "cue_position": CUE_POSITION,
            "target": TARGET,
        },
        evaluated=None,
    )
    regime_result = EvaluationRegimeResult(
        regime_id="cue_recall_block_probe",
        summary={"n_cases": 1, "n_steps": NUM_STEPS},
        case_results=(case_result,),
    )

    persist_regime_artifact_bundle(
        run_dir=output_dir,
        task="cue_recall",
        regime_kind="diagnostic",
        regime_id="cue_recall_block_probe",
        trigger_kind="diagnostic_probe",
        epoch=0,
        step=0,
        regime_result=regime_result,
    )
    return output_dir.resolve()

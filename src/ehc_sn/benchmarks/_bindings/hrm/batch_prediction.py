"""HRM batch-prediction benchmark binding module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ehc_sn.adapters.maze_hard.bridges.hrm.hrm_v1 import MazeHardHRMV1BridgeAdapter
from ehc_sn.adapters.maze_hard.bridges.hrm.hrm_v2 import MazeHardHRMV2BridgeAdapter
from ehc_sn.adapters.maze_hard.objectives import MazeHardACTTaskBinding
from ehc_sn.benchmarks._bindings.hrm.load import load_hrm_v1_bridge_adapter, load_hrm_v2_model
from ehc_sn.benchmarks._capabilities.batch_prediction import BatchPrediction, BatchPredicts
from ehc_sn.controllers.act import ACTController, ACTControllerConfig
from ehc_sn.models.hrm.hrm_v1 import HRModelV1
from ehc_sn.models.hrm.hrm_v2 import HRModelV2
from ehc_sn.tasks.maze_hard.runtime import coerce_maze_hard_batch


# =============================================================================
class HRMV1BatchPredictionAdapter(BatchPredicts):
    """Benchmark-time HRM v1 adapter for MazeHard batch prediction."""

    def __init__(  # ----------------------------------------------------------
        self,
        bridge_adapter: MazeHardHRMV1BridgeAdapter,
        *,
        compute_budget: int,
        done_action: int = 0,
    ) -> None:
        """Initialize the HRM v1 batch-prediction adapter with a bridge adapter."""
        self._bridge_adapter = bridge_adapter.eval()
        self._model = self._bridge_adapter.model
        controller_config = ACTControllerConfig(
            exploration_prob=0.0,
            max_steps=compute_budget,
            done_action=done_action,
        )
        self._controller = ACTController(
            self._bridge_adapter,
            controller_config,
        )

    @property
    def model(self) -> HRModelV1:
        """Return the wrapped pure HRM model."""
        return self._model

    def predict_batch(  # -----------------------------------------------------
        self,
        batch: Mapping[str, Any],
    ) -> BatchPrediction:
        """Run HRM deliberation until halt or the configured compute budget."""
        return _predict_with_controller(self._model, self._controller, batch)


# =============================================================================
class HRMV2BatchPredictionAdapter(BatchPredicts):
    """Benchmark-time HRM v2 adapter for MazeHard batch prediction."""

    def __init__(  # ----------------------------------------------------------
        self,
        model: HRModelV2,
        *,
        compute_budget: int,
        done_action: int = 0,
    ) -> None:
        """Initialize the HRM v2 batch-prediction adapter with a bridge adapter."""
        self._model = model.eval()
        controller_config = ACTControllerConfig(
            exploration_prob=0.0,
            max_steps=compute_budget,
            done_action=done_action,
        )
        self._controller = ACTController(
            MazeHardHRMV2BridgeAdapter(self._model),
            controller_config,
        )

    @property
    def model(self) -> HRModelV2:
        """Return the wrapped pure HRM model."""
        return self._model

    def predict_batch(  # -----------------------------------------------------
        self,
        batch: Mapping[str, Any],
    ) -> BatchPrediction:
        """Run HRM v2 deliberation until halt or the configured compute budget."""
        return _predict_with_controller(self._model, self._controller, batch)


# =============================================================================
def build_hrm_v1_batch_prediction(  # -----------------------------------------
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    compute_budget: int,
    device: str = "cpu",
    done_action: int = 0,
) -> HRMV1BatchPredictionAdapter:
    """Build the benchmark-time HRM v1 batch-prediction adapter."""
    bridge_adapter = load_hrm_v1_bridge_adapter(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return HRMV1BatchPredictionAdapter(
        bridge_adapter,
        compute_budget=compute_budget,
        done_action=done_action,
    )


# =============================================================================
def build_hrm_v2_batch_prediction(  # -----------------------------------------
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    compute_budget: int,
    device: str = "cpu",
    done_action: int = 0,
) -> HRMV2BatchPredictionAdapter:
    """Build the benchmark-time HRM v2 batch-prediction adapter."""
    model = load_hrm_v2_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return HRMV2BatchPredictionAdapter(
        model,
        compute_budget=compute_budget,
        done_action=done_action,
    )


# =============================================================================
def _predict_with_controller(  # ----------------------------------------------
    model: HRModelV1 | HRModelV2,
    controller: ACTController,
    batch: Mapping[str, Any],
) -> BatchPrediction:
    """Run one benchmark batch through the shared ACT-driven prediction path."""
    canonical_batch = batch if {"input_ids", "labels"}.issubset(batch.keys()) else coerce_maze_hard_batch(batch)
    device = next(model.parameters()).device
    input_ids = canonical_batch["input_ids"].to(device=device, dtype=torch.int64)
    labels = canonical_batch["labels"].to(device=device, dtype=torch.int64)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    step_batch = {"input_ids": input_ids, "labels": labels}

    state = controller.initial_state(step_batch)
    outputs = None
    with torch.no_grad():
        while True:
            state, outputs = controller.step(state, step_batch, explore=False)
            if bool(state.halted.all().item()):
                break

    if outputs is None:
        raise RuntimeError("ACTController produced no outputs during benchmark prediction.")

    task_binding = MazeHardACTTaskBinding()
    logits = task_binding.extract_logits(step_batch, state, outputs)  # (B, S, vocab)
    targets = task_binding.extract_targets(step_batch, state, outputs).labels  # (B, S)

    return BatchPrediction(
        predictions=logits.argmax(dim=-1).squeeze(0).detach().cpu(),
        targets=targets.squeeze(0).detach().cpu(),
        logits=logits.squeeze(0).detach().cpu(),
        steps=int(state.steps.max().item()),
        halted=bool(state.halted.all().item()),
    )


# =============================================================================
__all__ = [
    "HRMV1BatchPredictionAdapter",
    "HRMV2BatchPredictionAdapter",
    "build_hrm_v1_batch_prediction",
    "build_hrm_v2_batch_prediction",
]

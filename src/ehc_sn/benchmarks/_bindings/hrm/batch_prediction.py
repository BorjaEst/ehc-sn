"""HRM batch-prediction benchmark binding module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ehc_sn.benchmarks._bindings.hrm.load import load_hrm_v1_model, load_hrm_v2_model
from ehc_sn.benchmarks._bindings.hrm.preprocess import supervised_maze_tokenize
from ehc_sn.benchmarks._capabilities import BatchPrediction, BatchPredicts
from ehc_sn.controllers.act import ACTController, ACTControllerConfig
from ehc_sn.models.hrm.hrm_v1 import HRModelV1
from ehc_sn.models.hrm.hrm_v2 import HRModelV2


class HRMV1BatchPredictionAdapter(BatchPredicts):
    """Benchmark-time HRM v1 adapter for MazeHard batch prediction."""

    def __init__(self, model: HRModelV1, *, compute_budget: int, done_action: int = 0) -> None:
        self._model = model.eval()
        self._controller = ACTController(
            model,
            ACTControllerConfig(exploration_prob=0.0, max_steps=compute_budget, done_action=done_action),
        )

    @property
    def model(self) -> HRModelV1:
        """Return the wrapped pure HRM model."""
        return self._model

    def predict_batch(self, batch: Mapping[str, Any]) -> BatchPrediction:
        """Run HRM deliberation until halt or the configured compute budget."""
        return _predict_with_controller(self._model, self._controller, batch)


class HRMV2BatchPredictionAdapter(BatchPredicts):
    """Benchmark-time HRM v2 adapter for MazeHard batch prediction."""

    def __init__(self, model: HRModelV2, *, compute_budget: int, done_action: int = 0) -> None:
        self._model = model.eval()
        self._controller = ACTController(
            model,
            ACTControllerConfig(exploration_prob=0.0, max_steps=compute_budget, done_action=done_action),
        )

    @property
    def model(self) -> HRModelV2:
        """Return the wrapped pure HRM model."""
        return self._model

    def predict_batch(self, batch: Mapping[str, Any]) -> BatchPrediction:
        """Run HRM v2 deliberation until halt or the configured compute budget."""
        return _predict_with_controller(self._model, self._controller, batch)


def build_hrm_v1_batch_prediction(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    compute_budget: int,
    device: str = "cpu",
    done_action: int = 0,
) -> HRMV1BatchPredictionAdapter:
    """Build the benchmark-time HRM v1 batch-prediction adapter."""
    model = load_hrm_v1_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return HRMV1BatchPredictionAdapter(
        model,
        compute_budget=compute_budget,
        done_action=done_action,
    )


def build_hrm_v2_batch_prediction(
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


def _predict_with_controller(
    model: HRModelV1 | HRModelV2,
    controller: ACTController,
    batch: Mapping[str, Any],
) -> BatchPrediction:
    """Run one benchmark batch through the shared ACT-driven prediction path."""
    tokenized = supervised_maze_tokenize(batch)
    device = next(model.parameters()).device
    step_batch = {
        "inputs": tokenized["inputs"].unsqueeze(0).to(device=device, dtype=torch.int64),
        "labels": tokenized["labels"].unsqueeze(0).to(device=device, dtype=torch.int64),
    }

    state = controller.initial_state(step_batch)
    outputs = None
    with torch.no_grad():
        while True:
            state, outputs = controller.step(state, step_batch, allow_halt=True, explore=False, td_target=False)  # fmt: skip
            if bool(state.halted.all().item()):
                break

    if outputs is None:
        raise RuntimeError("ACTController produced no outputs during benchmark prediction.")

    return BatchPrediction(
        predictions=outputs.lm_logits.argmax(dim=-1).squeeze(0).detach().cpu(),
        targets=step_batch["labels"].squeeze(0).detach().cpu(),
        logits=outputs.lm_logits.squeeze(0).detach().cpu(),
        steps=int(state.steps.max().item()),
        halted=bool(state.halted.all().item()),
    )


__all__ = [
    "HRMV1BatchPredictionAdapter",
    "HRMV2BatchPredictionAdapter",
    "build_hrm_v1_batch_prediction",
    "build_hrm_v2_batch_prediction",
]

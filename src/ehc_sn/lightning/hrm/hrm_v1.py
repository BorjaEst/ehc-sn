"""HRM v1 Lightning module (ACT-supervised).

This module defines a PyTorch Lightning :class:`~lightning.LightningModule` wrapper
around the HRM v1 architecture (:class:`HRModelV1`) with Adaptive Computation Time
(ACT) control and loss computation.

Key behaviors:
    - **Manual optimization**: sets ``automatic_optimization = False`` and performs
        explicit backward/optimizer/scheduler steps for legacy parity.
    - **Stateful training carry**: forwards a carry object across mini-batches to
        support continuation / halting semantics.
    - **Partial reset batching**: halted examples are replaced with fresh rows
        using a FIFO buffer and
        :class:`~ehc_sn.training.partial_reset.PartialResetBatchAssembler`.

The batch structure used throughout this file is a plain ``dict[str, Tensor]``
with keys ``"input_ids"`` and ``"labels"``.
"""

from pathlib import Path
from typing import Any, Optional

import lightning as L
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from pydantic import BaseModel, Field, model_validator
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn.adapters.mazehard.hrm import (
    MazeHardHRMAdapterSettings,
    MazeHardHRMV1ACTTaskBinding,
    MazeHardHRMV1BridgeAdapter,
    build_mazehard_hrm_trace_meta,
)
from ehc_sn.adapters.mazehard.hrm.traces import MAZE_HARD_HRM_ACT_TRACE_FIELDS
from ehc_sn.controllers.deliberation.act import ACTController, ACTControllerConfig
from ehc_sn.lightning._rollout import evaluate_rollout, observe_rollout_chunk, update_metric_collection_from_evaluated_chunk
from ehc_sn.lightning.eval.contracts import EvaluationBatchArtifacts, EvaluationTraceRequest
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.metrics import build_train_metrics, build_val_metrics
from ehc_sn.metrics.routes import ACT_EPISODE_ROUTES, ACT_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.objectives.act import ACTLossConfig, ACTLossHead
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource, SingleStepRunner
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.types import Batch


# =================================================================================================
class ModelConfig_HRM_V1(BaseModel, extra="forbid"):
    """Configuration for the HRM v1 Lightning module.

    This config is intentionally "spec-first": it wires together the HRM core model,
    the ACT controller (adaptive computation time / halting logic), the rollout objective, and
    the optimizer/scheduler settings used during training.

    Notes:
        - `extra="forbid"` ensures unknown keys fail fast when parsing configs.
        - `global_batch_size` is used for scaling losses/metrics in a distributed setup.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the HRM v1 architecture.",
    )
    adapter: MazeHardHRMAdapterSettings = Field(
        default_factory=MazeHardHRMAdapterSettings,
        description="Settings for the MazeHard bridge adapter that binds the HRM core to task inputs/outputs.",
    )

    controller: ACTControllerConfig = Field(
        ...,
        description=(
            "Configuration for the ACT controller, which manages halting and partial resets during "
            "training. "
            "The keys in `controller` are passed to the ACTController constructor."
        ),
    )
    objective: ACTLossConfig = Field(
        ...,
        description="Objective config for the ACT rollout scorer.",
    )
    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description=(
            "Main optimizer config for model parameters (e.g. Adam). " "The keys in `optim_main` are passed to the optimizer constructor."
        ),
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=(
            "Learning rate scheduler config. If not set, no learning rate scheduling is applied. "
            "The keys in `scheduler` are passed to the scheduler constructor."
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="HRM runtime-owned validation safety settings.",
    )
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. " "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )  # TODO: consider moving to BufferSettings or similar


# =================================================================================================
class TrainingModel(L.LightningModule):
    """LightningModule wrapper for HRM v1 training.

    This module composes:
        - `HRModel`: the core architecture
        - `ACTController`: halting/partial-reset logic
        - A step module that computes loss and aggregates metrics

    Training uses manual optimization (`automatic_optimization = False`) to preserve
    legacy behavior (one backward pass, then explicit optimizer/scheduler steps).
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_HRM_V1,
    ) -> None:  # fmt: skip
        """Initialize the HRM v1 Lightning module.

        Args:
            config: Parsed `ModelConfig_HRM_V1` with architecture/controller/loss/optim settings.

        Notes:
            - Initializes a FIFO buffer and `PartialResetBatchAssembler` used to implement
              partial-reset semantics during training.
            - `_train_carry` is initialized lazily from the first batch via `step_module`.
        """
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = HRModelV1(model_settings)
        self.bridge_adapter = MazeHardHRMV1BridgeAdapter(self.model, config.adapter)
        self.controller = ACTController(self.bridge_adapter, config.controller)
        self.objective = ACTLossHead(config.objective, task_binding=MazeHardHRMV1ACTTaskBinding())
        self._config = config
        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()

        # Manual optimization: one backward, explicit opt/scheduler steps (legacy parity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(ACT_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(ACT_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("act", extra_fields=MAZE_HARD_HRM_ACT_TRACE_FIELDS)

        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,  # or local batch size if you prefer
            keys=("input_ids", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("input_ids", "labels"),
        )

    @property
    def config(self) -> ModelConfig_HRM_V1:
        """Return the parsed configuration used by this module."""
        return self._config

    def configure_optimizers(  # ------------------------------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[SequentialLR]]:  # fmt: skip
        """Build optimizers and LR schedulers.

        Returns:
            A tuple `(optimizers, schedulers)` in the format Lightning expects.
        """
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the main model parameters
        sup_params = [p for p in self.bridge_adapter.parameters() if p.requires_grad]
        opt_sup = AdamATan2(sup_params, self._config.optimizer)
        sch_sup = CosineAnnealingLRWithWarmup(opt_sup, total_steps, self.config.scheduler)

        return [opt_sup], [sch_sup]

    def on_train_epoch_start(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset per-epoch training state.

        Clears the training carry and the FIFO buffer so that partial-reset state does not leak
        across epochs.
        """
        self._train_carry = None
        self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # ------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset validation metrics at the start of each epoch."""
        self.val_metrics.reset()

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> dict[str, object]:  # fmt: skip
        """Run one training step with manual optimization.

        The training logic uses partial reset to replace halted slots with fresh examples.

        Notes:
            - Horizon is effectively 1: we run exactly one rollout step per mini-batch.
            - Loss is normalized by the local batch size; DDP averages gradients across ranks.
        """
        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.controller.initial_state(batch)

        evaluation = evaluate_rollout(
            runner=self._train_runner,
            source=PartialResetSource(incoming=batch, assembler=self._train_batch_assembler, carry0=self._train_carry),
            controller=self.controller,
            carry=self._train_carry,
            objective=self.objective,
            runner_options={"allow_halt": True, "explore": True},
            objective_options={"controller": self.controller, "td_target": True},
        )
        self._train_carry = evaluation.chunk.final_carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch["input_ids"].shape[0])
        loss = normalize_loss_for_backward(evaluation.evaluated.loss, local_bs=local_bs)

        optimizers = self.optimizers()
        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.step()  # type: ignore

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Update metrics with unnormalized loss and log to TensorBoard.
        update_metric_collection_from_evaluated_chunk(self.train_metrics, evaluation.evaluated, ACT_STEP_ROUTES)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss.detach(), "signals": evaluation.evaluated.last_step.outputs.signals}

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> dict[str, object]:  # fmt: skip
        """Run a deterministic fixed-budget ACT rollout for validation.

        Validation uses a repeated source, so learned early halting would
        immediately refresh completed rows onto the same sample. ``allow_halt``
        is therefore disabled here and the controller runs to its configured
        budget without exploration.
        """
        carry0 = self.controller.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=self.controller,
            carry=carry0,
            objective=self.objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": False, "explore": False},
            objective_options={"controller": self.controller, "td_target": False},
        )
        trace = observe_rollout_chunk(evaluation.chunk, self.trace_specs, trace_meta=build_mazehard_hrm_trace_meta(batch))
        update_metric_collection_from_evaluated_chunk(self.val_metrics, evaluation.evaluated, ACT_EPISODE_ROUTES)
        return {"trace": trace}

    # -- Evaluation regime surface ---------------------------------------------------------------

    def build_evaluation_metrics(  # -------------------------------------------------------------
        self, namespace: str,
    ) -> MetricCollection:  # fmt: skip
        """Return a fresh HRM-family metric collection with the given namespace prefix.

        Args:
            namespace: Metric namespace prefix, e.g. ``"diag/my_probe/"``.

        Returns:
            A fresh :class:`~torchmetrics.MetricCollection` keyed by ACT episode routes.
        """
        return build_val_metrics(ACT_EPISODE_ROUTES).clone(prefix=namespace)

    def execute_evaluation_batch(  # -------------------------------------------------------------
        self,
        batch: Batch,
        trace_request: Optional[EvaluationTraceRequest],
    ) -> EvaluationBatchArtifacts:  # fmt: skip
        """Execute one HRM v1 evaluation batch and return scored artifacts.

        Runs a deterministic ACT rollout identical to ``validation_step`` but does **not**
        update ``self.val_metrics``.

        Args:
            batch: A task batch in MazeHard format.
            trace_request: Trace key request, or ``None`` for no trace.

        Returns:
            :class:`~ehc_sn.lightning.eval.contracts.EvaluationBatchArtifacts`.
        """
        carry0 = self.controller.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=self.controller,
            carry=carry0,
            objective=self.objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": False, "explore": False},
            objective_options={"controller": self.controller, "td_target": False},
        )

        trace = None
        if trace_request is not None and trace_request.enabled:
            trace = observe_rollout_chunk(evaluation.chunk, self.trace_specs, trace_meta=build_mazehard_hrm_trace_meta(batch))

        def _apply(collection: MetricCollection) -> None:
            update_metric_collection_from_evaluated_chunk(collection, evaluation.evaluated, ACT_EPISODE_ROUTES)

        return EvaluationBatchArtifacts(
            regime_id="_inline",
            metric_namespace="",
            evaluated=evaluation.evaluated,
            apply_to_metrics=_apply,
            trace=trace,
        )


# =============================================================================
__all__ = ["ModelConfig_HRM_V1", "TrainingModel"]

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
        :class:`~ehc_sn.rollouts.partial_reset.PartialResetBatchAssembler`.
    - **Target network** (optional): EMA-lagged target backbone for TD bootstrap
        stabilization. Controlled by the ``[target_network]`` config section.
    - **Warmup phase** (optional): disables learned halting for the first
        ``supervised_only_warmup_steps`` steps so the PFC representations become
        informative before the control head learns.

The batch structure used throughout this file is a plain ``dict[str, Tensor]``
with keys ``"input_ids"`` and ``"labels"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from pydantic import AliasChoices, BaseModel, Field
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn.adapters.mazehard.hrm import (
    MazeHardHRMAdapterSettings,
    MazeHardHRMV1ACTTaskBinding,
    MazeHardHRMV1BridgeAdapter,
    build_mazehard_hrm_trace_meta,
)
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    ACTControllerConfig,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.diagnostics import DiagnosticTraceSpec
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.reducers import HiddenNormHistogram, compute_nonempty
from ehc_sn.metrics.rollout import update_metric_collection_from_evaluated_chunk
from ehc_sn.metrics.routes.act import ACT_EPISODE_ROUTES, ACT_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.objectives.act import ACTObjective, ACTObjectiveConfig
from ehc_sn.rollouts.buffers import FifoBuffer
from ehc_sn.rollouts.partial_reset import PartialResetBatchAssembler
from ehc_sn.rollouts.runtime import RecurrentRunner, SingleStepRunner
from ehc_sn.rollouts.sources import PartialResetSource
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.rollout import score_captured_rollout
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
)
from ehc_sn.training.stabilization import (
    TargetAdapterModule,
    TargetNetworkConfig,
)
from ehc_sn.types import Batch


# =============================================================================
class HRMV1ModelConfig(BaseModel, extra="forbid"):
    """Configuration for the HRM v1 Lightning module.

    This config is intentionally "spec-first": it wires together the HRM core
    model, the ACT controller (adaptive computation time / halting logic), the
    rollout objective, and the optimizer/scheduler settings used during training.

    Notes:
        - `extra="forbid"` ensures unknown keys fail fast when parsing configs.
        - `global_batch_size` is used for scaling losses/metrics in a distributed
          setup.
    """

    model_config_path: Path = Field(
        ...,
        description=(
            "Path to the model configuration TOML file that specifies the HRM "
            "v1 architecture."
        ),
    )
    adapter: MazeHardHRMAdapterSettings = Field(
        default_factory=MazeHardHRMAdapterSettings,
        description="Settings for the MazeHard bridge adapter that binds the "
        "HRM core to task inputs/outputs.",
    )

    controller: ACTControllerConfig = Field(
        ...,
        description=(
            "Configuration for the ACT controller, which manages halting and "
            "partial resets during training. The keys in `controller` are "
            "passed to the ACTController constructor."
        ),
    )
    objective: ACTObjectiveConfig = Field(
        ...,
        description="Objective config for the ACT rollout scorer.",
    )
    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description=(
            "Main optimizer config for model parameters (e.g. Adam). "
            "The keys in `optim_main` are passed to the optimizer constructor."
        ),
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=(
            "Learning rate scheduler config. If not set, no learning rate "
            "scheduling is applied. The keys in `scheduler` are passed to the "
            "scheduler constructor."
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="HRM runtime-owned validation safety settings.",
    )
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. The per-device batch size "
            "is computed as `global_batch_size // world_size`."
        ),
    )  # TODO: consider moving to BufferSettings or similar
    target_network: TargetNetworkConfig = Field(
        default_factory=TargetNetworkConfig,
        description="Optional EMA-lagged target network config for "
        "q_continue bootstrap stabilization.",
    )
    supervised_only_warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of optimizer steps during which learned halting "
        "is disabled (allow_halt=False). Pattern-matched from "
        "HRM-v2 warmup phase.",
    )


# =============================================================================
class HRMV1TrainingModel(L.LightningModule):
    """LightningModule wrapper for HRM v1 training.

    This module composes:
        - `HRModel`: the core architecture
        - `ACTController`: halting/partial-reset logic
        - A step module that computes loss and aggregates metrics

    Training uses manual optimization (`automatic_optimization = False`) to preserve
    legacy behavior (one backward pass, then explicit optimizer/scheduler steps).
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: HRMV1ModelConfig,
    ) -> None:
        """Initialize the HRM v1 Lightning module.

        Args:
            config: Parsed `HRMV1ModelConfig` with architecture/controller/loss/optim settings.

        Notes:
            - Initializes a FIFO buffer and `PartialResetBatchAssembler` used to implement
              partial-reset semantics during training.
            - `_train_carry` is initialized lazily from the first batch via `step_module`.
        """
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = HRModelV1(model_settings)
        self.adapter = MazeHardHRMV1BridgeAdapter(self.model, config.adapter)
        self.controller = ACTController(self.adapter, config.controller)
        self.objective = ACTObjective(config.objective, task_binding=MazeHardHRMV1ACTTaskBinding())  # fmt: skip
        self._config = config
        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()

        # Manual optimization: one backward, explicit opt/scheduler steps.
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        self.train_metrics = build_train_metrics(ACT_STEP_ROUTES).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(ACT_EPISODE_ROUTES).clone(
            prefix="val/"
        )
        self._trace_paradigm: str = "act"
        self.diagnostic_trace_spec: DiagnosticTraceSpec = DiagnosticTraceSpec(
            enabled=False,
            max_batches=2,
            keys=(),
        )
        self._diagnostic_traces: list[Any] = []

        # Bounded diagnostic reducers (hidden-state norm histogram).
        self._val_hidden_norms = HiddenNormHistogram(
            bin_edges=torch.linspace(0.0, 50.0, 51), max_batches=10
        )
        self._val_reducer_collection = MetricCollection(
            {"hidden_norms": self._val_hidden_norms},
            prefix="val_diag/",
        )

        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,
            keys=("input_ids", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("input_ids", "labels"),
        )

        # Optional target network for TD bootstrap stabilization.
        self._target_adapter: TargetAdapterModule | None = None
        if config.target_network.enabled:
            self._target_adapter = TargetAdapterModule(self.adapter)

    @property
    def config(self) -> HRMV1ModelConfig:
        """Return the parsed configuration used by this module."""
        return self._config

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """Build optimizers and LR schedulers.

        Returns:
            A tuple `(optimizers, schedulers)` in the format Lightning expects.
        """
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Optimizer for the main model parameters
        sup_params = [p for p in self.adapter.parameters() if p.requires_grad]
        opt_sup = AdamATan2(sup_params, self._config.optimizer)
        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_sup, total_steps, self.config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/main",
            }
        ]

        return [opt_sup], schedulers

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        """Reset per-epoch training state.

        Clears the training carry and the FIFO buffer so that partial-reset
        state does not leak across epochs.
        """
        self._train_carry = None
        self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        """Reset validation metrics and diagnostic reducers at the start of each epoch."""
        self.val_metrics.reset()
        self._val_reducer_collection.reset()
        self._diagnostic_traces.clear()

    @property
    def diagnostic_traces(self) -> tuple[Any, ...]:
        """Return bounded diagnostic traces captured during the just-completed
        validation epoch.  Empty tuple when trace capture was disabled or no
        batches were processed.
        """
        return tuple(self._diagnostic_traces)

    def reset_diagnostic_traces(  # -------------------------------------------
        self,
    ) -> None:
        """Clear the internal diagnostic trace buffer."""
        self._diagnostic_traces.clear()

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        """Run one training step with manual optimization.

        The training logic uses partial reset to replace halted slots with fresh
        examples.

        Notes:
            - Horizon is effectively 1: we run exactly one rollout step per
              mini-batch.
            - Loss is normalized by the local batch size; DDP averages gradients
              across ranks.
            - ``supervised_only_warmup_steps > 0`` disables learned halting
              during the warmup phase so the PFC representations become
              informative before the control head learns.
            - ``target_network.enabled`` activates an EMA-lagged target backbone
              for q_continue TD bootstrap targets.
        """
        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.controller.initial_state(batch)

        # Warmup gate: do not allow learned halting during the first N steps.
        is_warmup = self.global_step < self.config.supervised_only_warmup_steps

        source = PartialResetSource(
            incoming=batch,
            assembler=self._train_batch_assembler,
            carry0=self._train_carry,
        )

        # Target backbone is passed to the objective so _compute_td_target can
        # step it on the *same executed inputs* (record.carry.data) that the
        # online backbone consumed — not on the raw incoming batch.
        target_backbone: TargetAdapterModule | None = (
            self._target_adapter if self._target_adapter is not None else None
        )

        evaluation = score_captured_rollout(
            runner=self._train_runner,
            source=source,
            controller=self.controller,
            carry=self._train_carry,
            objective=self.objective,
            runner_options={"allow_halt": not is_warmup, "explore": True},
            objective_options={
                "controller": self.controller,
                "td_target": True,
                "use_token_weights": self.config.objective.use_token_weights,
                "target_backbone": target_backbone,
            },
        )
        update_metric_collection_from_evaluated_chunk(
            self.train_metrics, evaluation.evaluated, ACT_STEP_ROUTES
        )
        self._train_carry = evaluation.chunk.final_carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch["input_ids"].shape[0])
        loss = normalize_loss_for_backward(
            evaluation.evaluated.loss, local_bs=local_bs
        )

        optimizers = self.optimizers()
        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
            opt.step()  # type: ignore

        # EMA update of target backbone after optimizer step.
        if self._target_adapter is not None:
            self._target_adapter.ema_update(
                self.adapter, self.config.target_network.tau
            )

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        # Update metrics with unnormalized loss and log to TensorBoard.
        self.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )  # fmt: skip

        return {
            "loss": loss.detach(),
            "signals": evaluation.evaluated.last_step.outputs.signals,
        }

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        """Run a deterministic fixed-budget ACT rollout for validation.

        Validation uses a repeated source, so learned early halting would
        immediately refresh completed rows onto the same sample. ``allow_halt``
        is therefore disabled here and the controller runs to its configured
        budget without exploration.
        """
        trace_request = None
        ds = self.diagnostic_trace_spec
        if ds.enabled and batch_idx < ds.max_batches and ds.keys:
            trace_request = EvaluationTraceRequest(
                trace_spec=build_trace_spec("act", include_keys=set(ds.keys)),
                trace_meta=dict(build_mazehard_hrm_trace_meta(batch)),
            )
        result = self.execute_evaluation_batch(
            EvaluationCaseBatch(
                batch=batch,
                case_id=f"val-{batch_idx:04d}",
            ),
            trace_request=trace_request,
        )
        update_metric_collection_from_evaluated_chunk(
            self.val_metrics, result.evaluated, ACT_EPISODE_ROUTES
        )

        # Feed bounded diagnostic reducers.
        inp = self.adapter.prepare_inputs(batch)
        model_out, _ = self.model(inp, state=None)
        self._val_hidden_norms.update(
            model_out.theta_summary.norm(dim=-1).detach()
        )

        # Store bounded diagnostic traces for callback consumption.
        if (
            result.trace is not None
            and len(self._diagnostic_traces) < ds.max_batches
        ):
            self._diagnostic_traces.append(result.trace)

        return {"trace": result.trace}

    def on_validation_epoch_end(  # -------------------------------------------
        self,
    ) -> None:
        """Compute, log, and reset bounded diagnostic reducers."""
        compute_nonempty(self._val_reducer_collection)
        self._val_reducer_collection.reset()

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay case through the ACT eval path."""
        carry0 = self.controller.initial_state(case.batch)
        return execute_replay_evaluation_batch(
            case=case,
            runner=self._eval_runner,
            controller=self.controller,
            carry=carry0,
            objective=self.objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": False, "explore": False},
            objective_options={
                "controller": self.controller,
                "td_target": False,
            },
            trace_request=trace_request,
        )


# =============================================================================
__all__ = ["HRMV1ModelConfig", "HRMV1TrainingModel"]

"""EHC v1 MazeHard Lightning module (phase-2 controller pretrain).

Phase-2 trains only the cortical/controller subsystem of EHC v1 on the
MazeHard deliberation task.  The spatial pathway (LEC, MEC, HPC, and all
four inter-region projections) is frozen by setting requires_grad=False at
construction time.

Optimizer layout:
    opt_ctrl   — PFC backbone (excl. pfc.estimator) + adapter encoder + decoder.
    opt_heads  — pfc.estimator (vmPFC Q-head) + STR (V(s) critic).

Frozen (not in any optimizer, requires_grad=False):
    model.lec, model.mec, model.hpc,
    model.lec_to_hpc, model.mec_to_hpc,
    model.pfc_to_hpc, model.hpc_to_pfc.

Training style: horizon-1 SingleStepRunner (not arena-style streaming TBPTT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import lightning as L
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn import utils
from ehc_sn.adapters.mazehard.ehc import (
    MazeHardEHCAdapterSettings,
    MazeHardEHCV1BridgeAdapter,
    MazeHardEHCV1HybridTaskBinding,
)
from ehc_sn.controllers.deliberation.actor_critic import DeliberationACController, DeliberationACControllerConfig
from ehc_sn.lightning._rollout import evaluate_rollout, observe_rollout_chunk, update_metric_collection_from_evaluated_chunk
from ehc_sn.lightning.eval.contracts import EvaluationBatchArtifacts, EvaluationTraceRequest
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1, ModelSettingsV1
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLLossHead
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource, SingleStepRunner
from ehc_sn.tasks.mazehard.capabilities.deliberation import MazeHardDeliberationCapability, MazeHardDeliberationConfig
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.training.actor_critic import TD0ActorCriticBatchBuilder, ZeroBootstrapActorCriticValidationScorer
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.types import Batch


# =================================================================================================
class ModelConfig_EHC_V1_MazeHard(BaseModel, extra="forbid"):
    """Configuration for the EHC v1 MazeHard phase-2 Lightning module.

    Notes:
        - ``extra="forbid"`` ensures unknown keys fail fast.
        - ``mazehard_phase`` enforces that only the phase-2 regime is accepted.
    """

    # ~~ Model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file (use ehc-v1-mazehard.toml for phase 2).",
    )
    adapter: MazeHardEHCAdapterSettings = Field(
        default_factory=MazeHardEHCAdapterSettings,
        description="Settings for the MazeHard+EHC bridge adapter.",
    )
    deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="Deliberation capability config (halt_action, episode_horizon).",
    )
    controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="Deliberation actor-critic controller configuration.",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL objective configuration.",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer_ctrl: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for controller params: PFC backbone (excl. estimator) + encoder + decoder.",
    )
    optimizer_heads: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for head params: pfc.estimator (vmPFC) + STR (V(s) critic).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config applied to both optimizers.",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description=(
            "Steps during which only opt_ctrl trains (allow_halt=False). "
            "Prevents 'halt immediately' collapse before PFC representations are informative."
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Validation runner safety settings.",
    )

    # ~~ Phase gate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mazehard_phase: Literal["mazehard_controller_pretrain"] = Field(
        default="mazehard_controller_pretrain",
        description="MazeHard training phase.  Only 'mazehard_controller_pretrain' is accepted.",
    )

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices.",
    )


# =================================================================================================
class TrainingModel(L.LightningModule):
    """LightningModule for EHC v1 phase-2 MazeHard controller pretraining.

    Manages:
        - Freezing the spatial pathway (LEC, MEC, HPC, projections) via
          requires_grad=False at construction time.
        - Two-optimizer training: opt_ctrl (controller) and opt_heads (value heads).
        - Horizon-1 deliberation via SingleStepRunner + DeliberationACController.
        - Partial-reset batching via FifoBuffer + PartialResetBatchAssembler.
        - Warmup: during the first supervised_only_warmup_steps, only opt_ctrl
          is stepped and halting is disabled.
    """

    _FROZEN_MODULES = ("lec", "mec", "hpc")
    _FROZEN_PROJECTIONS = ("lec_to_hpc", "mec_to_hpc", "pfc_to_hpc", "hpc_to_pfc")

    def __init__(
        self,
        config: ModelConfig_EHC_V1_MazeHard,
    ) -> None:
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = EHCModelV1(model_settings)
        self.bridge_adapter = MazeHardEHCV1BridgeAdapter(self.model, config.adapter)

        # Freeze the spatial pathway immediately so requires_grad=False is set
        # before any optimizer is built.  This ensures .grad remains None on
        # frozen parameters even without explicit detach inside the model forward.
        self._freeze_spatial_pathway()

        self.controller: DeliberationACController | None = None
        self.objective: HybridRLLossHead | None = None
        self.learner: TD0ActorCriticBatchBuilder | None = None
        self.val_scorer: ZeroBootstrapActorCriticValidationScorer | None = None
        self._config = config
        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()

        self.automatic_optimization = False
        self._train_carry = None

        self.train_metrics = build_train_metrics(RL_STEP_ROUTES).clone(prefix="train/")
        self.val_metrics = build_val_metrics(RL_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("rl")

        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,
            keys=("input_ids", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("input_ids", "labels"),
        )

    def _freeze_spatial_pathway(self) -> None:
        """Set requires_grad=False on all frozen spatial parameters."""
        for name in self._FROZEN_MODULES:
            for p in getattr(self.model, name).parameters():
                p.requires_grad_(False)
        for name in self._FROZEN_PROJECTIONS:
            for p in getattr(self.model, name).parameters():
                p.requires_grad_(False)

    @property
    def config(self) -> ModelConfig_EHC_V1_MazeHard:
        """Return the parsed configuration used by this LightningModule."""
        return self._config

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        """Initialize the deliberation controller, objective, and learner."""
        task_config = self.config.deliberation
        finalizer = MazeHardDeliberationCapability(task_config, MazeHardRewardProjector())
        self.controller = DeliberationACController(self.bridge_adapter, self.config.controller, finalizer)
        self.objective = HybridRLLossHead(self.config.objective)
        task_binding = MazeHardEHCV1HybridTaskBinding()
        self.learner = TD0ActorCriticBatchBuilder(
            self.bridge_adapter, None, gamma=self.config.objective.gamma, task_binding=task_binding
        )
        self.val_scorer = ZeroBootstrapActorCriticValidationScorer(self.objective, task_binding)

    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[SequentialLR]]:
        """Build two optimizers and their schedulers.

        Returns:
            ``([opt_ctrl, opt_heads], [sch_ctrl, sch_heads])`` where:
                0) opt_ctrl  — PFC backbone (excl. estimator) + encoder + decoder.
                1) opt_heads — pfc.estimator (vmPFC) + STR (V(s) critic).
        """
        total_steps = int(self.trainer.estimated_stepping_batches)

        _frozen_ids = set()
        for name in self._FROZEN_MODULES:
            _frozen_ids |= {id(p) for p in getattr(self.model, name).parameters()}
        for name in self._FROZEN_PROJECTIONS:
            _frozen_ids |= {id(p) for p in getattr(self.model, name).parameters()}

        _heads_ids = (
            {id(p) for p in self.model.pfc.estimator.parameters()}
            | {id(p) for p in self.model.str.parameters()}
        )

        ctrl_params = [
            p for p in self.bridge_adapter.parameters()
            if p.requires_grad and id(p) not in _frozen_ids and id(p) not in _heads_ids
        ]
        heads_params = [
            p for p in self.bridge_adapter.parameters()
            if p.requires_grad and id(p) in _heads_ids
        ]

        opt_ctrl = AdamATan2(ctrl_params, self.config.optimizer_ctrl)
        opt_heads = AdamATan2(heads_params, self.config.optimizer_heads)

        sch_ctrl = CosineAnnealingLRWithWarmup(opt_ctrl, total_steps, self.config.scheduler)
        sch_heads = CosineAnnealingLRWithWarmup(opt_heads, total_steps, self.config.scheduler)

        return [opt_ctrl, opt_heads], [sch_ctrl, sch_heads]

    def on_train_epoch_start(self) -> None:
        """Reset training carry, buffer, and metrics."""
        self._train_carry = None
        self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics."""
        self.val_metrics.reset()

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Run one horizon-1 training step with manual dual-optimizer update."""
        if self.controller is None or self.objective is None or self.learner is None:
            raise RuntimeError("EHC v1 MazeHard runtime is not initialized. Call setup() before training.")

        if self._train_carry is None:
            self._train_carry = self.controller.initial_state(batch)

        is_warmup = self.global_step < self.config.supervised_only_warmup_steps

        execution = self._train_runner.run(
            source=PartialResetSource(
                incoming=batch,
                assembler=self._train_batch_assembler,
                carry0=self._train_carry,
            ),
            controller=self.controller,
            carry=self._train_carry,
            options={"allow_halt": not is_warmup, "explore": True},
        )
        self._train_carry = execution.final_carry.detach()

        record = execution.last_record
        if record.snapshot.steps is None:
            raise RuntimeError("EHC v1 MazeHard training runner produced a record without step counters.")

        ac_batch = self.learner.build_deliberation_ac_batch(record.outputs, record.snapshot)
        step_output = self.objective.compute_step(ac_batch, is_warmup=is_warmup)

        local_bs = int(batch["input_ids"].shape[0])
        loss = normalize_loss_for_backward(step_output.loss, local_bs=local_bs)

        optimizer_list = self.optimizers()
        optimizer_list = list(optimizer_list) if isinstance(optimizer_list, (list, tuple)) else [optimizer_list]
        scheduler_list = self.lr_schedulers()
        scheduler_list = list(scheduler_list) if isinstance(scheduler_list, (list, tuple)) else [scheduler_list]

        for opt in optimizer_list:
            opt.zero_grad(set_to_none=True)  # type: ignore[arg-type]

        self.manual_backward(loss)

        # During warmup only opt_ctrl (index 0) trains; opt_heads is deferred.
        active_indices = [0] if is_warmup else list(range(len(optimizer_list)))
        for idx in active_indices:
            opt = optimizer_list[idx]
            if utils.has_any_grad(opt):
                opt.step()  # type: ignore[misc]
                scheduler_list[idx].step()  # type: ignore[misc]

        update_metrics_from_step(self.train_metrics, step_output.metrics, RL_STEP_ROUTES)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss.detach(), "signals": step_output.signals}

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Run a full rollout to collect validation metrics."""
        if self.controller is None or self.val_scorer is None:
            raise RuntimeError("EHC v1 MazeHard runtime is not initialized. Call setup() before validation.")

        carry0 = self.controller.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=self.controller,
            carry=carry0,
            objective=self.val_scorer,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"explore": False, "allow_halt": False},
        )
        trace = observe_rollout_chunk(evaluation.chunk, self.trace_specs, trace_meta={})
        update_metric_collection_from_evaluated_chunk(self.val_metrics, evaluation.evaluated, RL_EPISODE_ROUTES)
        return {"trace": trace}

    def build_evaluation_metrics(
        self,
        namespace: str,
    ) -> MetricCollection:
        """Return a fresh EHC MazeHard metric collection with the given namespace prefix."""
        return build_val_metrics(RL_EPISODE_ROUTES).clone(prefix=namespace)

    def execute_evaluation_batch(
        self,
        batch: Batch,
        trace_request: Optional[EvaluationTraceRequest],
        source_context: object | None = None,
    ) -> EvaluationBatchArtifacts:
        """Execute one EHC MazeHard evaluation batch and return scored artifacts."""
        if self.controller is None or self.val_scorer is None:
            raise RuntimeError("EHC v1 MazeHard runtime is not initialized. Call setup() before evaluation.")

        carry0 = self.controller.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._eval_runner,
            source=RepeatSource(batch),
            controller=self.controller,
            carry=carry0,
            objective=self.val_scorer,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"explore": False, "allow_halt": False},
        )

        trace = None
        if trace_request is not None and trace_request.enabled:
            trace = observe_rollout_chunk(evaluation.chunk, self.trace_specs, trace_meta={})

        def _apply(collection: MetricCollection) -> None:
            update_metric_collection_from_evaluated_chunk(collection, evaluation.evaluated, RL_EPISODE_ROUTES)

        return EvaluationBatchArtifacts(
            regime_id="_inline",
            metric_namespace="",
            evaluated=evaluation.evaluated,
            apply_to_metrics=_apply,
            trace=trace,
            source_context=source_context,
            trace_supplements_applied=(),
        )


# =============================================================================
__all__ = ["ModelConfig_EHC_V1_MazeHard", "TrainingModel"]

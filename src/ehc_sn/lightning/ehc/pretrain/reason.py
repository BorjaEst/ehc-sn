"""EHC v1 controller pretrain regime: MazeHard deliberation, hybrid RL objective."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lightning as L
from pydantic import BaseModel, Field
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn import utils
from ehc_sn.adapters.mazehard.ehc import (
    MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS,
    MazeHardEHCAdapterSettings,
    MazeHardEHCV1HybridTaskBinding,
)
from ehc_sn.adapters.mazehard.ehc.traces import build_mazehard_ehc_trace_meta
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACController,
    DeliberationACControllerConfig,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.ehc.core._base import (
    freeze_params,
    resolve_controller_heads_ids,
    resolve_spatial_core_ids,
)
from ehc_sn.lightning.ehc.core.runtime import RuntimeConfig
from ehc_sn.metrics.adapter import update_metrics_from_step
from ehc_sn.metrics.builders import build_val_metrics
from ehc_sn.metrics.rollout import update_metric_collection_from_evaluated_chunk
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLObjective
from ehc_sn.rollouts.buffers import FifoBuffer
from ehc_sn.rollouts.partial_reset import PartialResetBatchAssembler
from ehc_sn.rollouts.runtime import RecurrentRunner, SingleStepRunner
from ehc_sn.rollouts.sources import PartialResetSource
from ehc_sn.tasks.mazehard.capabilities.deliberation import (
    MazeHardDeliberationCapability,
    MazeHardDeliberationConfig,
)
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.actor_critic import (
    TD0ActorCriticBatchBuilder,
    ZeroBootstrapActorCriticValidationScorer,
)
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
    SequentialLR,
)
from ehc_sn.types import Batch


# =============================================================================
class EHCReasonPretrainConfig(BaseModel, extra="forbid"):
    """Config for controller pretrain (MazeHard deliberation, hybrid RL objective)."""

    mode: Literal["reason_pretrain"] = "reason_pretrain"

    model_config_path: Path = Field(
        ...,
        description="",
    )
    adapter: MazeHardEHCAdapterSettings = Field(
        default_factory=MazeHardEHCAdapterSettings,
        description="",
    )
    deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="",
    )
    controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="",
    )
    optimizer_ctrl: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="",
    )
    optimizer_heads: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description="",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="",
    )


# =============================================================================
class EHCReasonPretrainRegime:
    """MazeHard deliberation controller pretrain regime.

    Trains controller_body, controller_heads, controller_bridge.
    Freezes spatial_core (lec, mec, hpc, lec_to_hpc, mec_to_hpc).
    pfc_to_hpc and hpc_to_pfc are trainable (owned by controller_bridge).
    """

    def __init__(  # ----------------------------------------------------------
        self,
        lm: L.LightningModule,
        model: EHCModelV1,
        config: EHCReasonPretrainConfig,
    ) -> None:
        """Initialize regime with model and config. Build controller, objective, learner, scorer."""
        self._lm = lm
        self._config = config

        # Freeze spatial core; bridge projections are now trainable in reason_pretrain.
        freeze_params(model, "lec", "mec", "hpc", "lec_to_hpc", "mec_to_hpc")

        self._controller: DeliberationACController | None = None
        self._objective: HybridRLObjective | None = None
        self._learner: TD0ActorCriticBatchBuilder | None = None
        self._val_scorer: ZeroBootstrapActorCriticValidationScorer | None = None

        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()
        self._train_carry = None

        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None
        self._eval_trace_keys: set[str] | None = None

    def _ensure_train_batch_assembler(
        self, batch: Batch
    ) -> PartialResetBatchAssembler:
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler
        local_bs = int(batch["input_ids"].shape[0])
        capacity = 4 * local_bs
        self._train_buffer = FifoBuffer(
            capacity, ("input_ids", "labels"), pin_memory=True
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer, keys=("input_ids", "labels")
        )
        return self._train_batch_assembler

    # -- Regime hooks ---------------------------------------------------------

    def setup(  # -------------------------------------------------------------
        self,
        stage: str | None,
    ) -> None:
        """Initialize controller, objective, learner, scorer."""
        cfg = self._config
        lm = self._lm
        finalizer = MazeHardDeliberationCapability(
            cfg.deliberation, MazeHardRewardProjector()
        )
        self._controller = DeliberationACController(
            lm.bridge_adapter, cfg.controller, finalizer
        )
        self._objective = HybridRLObjective(cfg.objective)
        task_binding = MazeHardEHCV1HybridTaskBinding()
        self._learner = TD0ActorCriticBatchBuilder(
            lm.bridge_adapter,
            None,
            gamma=cfg.objective.gamma,
            task_binding=task_binding,
        )
        self._val_scorer = ZeroBootstrapActorCriticValidationScorer(
            self._objective, task_binding
        )

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """Configure separate optimizers and schedulers for controller body + bridge, and controller heads."""
        total_steps = int(self._lm.trainer.estimated_stepping_batches)
        model = self._lm.model

        spatial_ids = resolve_spatial_core_ids(model)
        heads_ids = resolve_controller_heads_ids(model)

        # controller_body + controller_bridge go to opt_ctrl.
        # controller_heads go to opt_heads.
        # spatial_core is frozen (requires_grad=False), not included.
        ctrl_params = [
            p
            for p in self._lm.bridge_adapter.parameters()
            if p.requires_grad
            and id(p) not in spatial_ids
            and id(p) not in heads_ids
        ]
        heads_params = [
            p
            for p in self._lm.bridge_adapter.parameters()
            if p.requires_grad and id(p) in heads_ids
        ]

        opt_ctrl = AdamATan2(ctrl_params, self._config.optimizer_ctrl)
        opt_heads = AdamATan2(heads_params, self._config.optimizer_heads)
        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_ctrl, total_steps, self._config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/ctrl",
            },
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_heads, total_steps, self._config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/heads",
            },
        ]
        return [opt_ctrl, opt_heads], schedulers

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        """Reset train carry and buffer at the start of each epoch."""
        self._train_carry = None
        if self._train_buffer is not None:
            self._train_buffer.clear()
        self._lm.train_metrics.reset()

    def on_validation_epoch_start(
        self,
    ) -> None:
        """Reset val metrics at the start of each validation epoch."""
        self._lm.val_metrics.reset()

    def set_eval_trace_keys(self, keys: set[str]) -> None:
        """Set semantic trace keys for replay-evaluation capture."""
        self._eval_trace_keys = set(keys)
        extra_fields = tuple(
            field
            for field in MAZE_HARD_EHC_ACTOR_CRITIC_TRACE_FIELDS
            if field.name in self._eval_trace_keys
        )
        self._lm.trace_spec = build_trace_spec(
            "rl",
            include_keys=self._eval_trace_keys,
            extra_fields=extra_fields,
        )
        self._lm.trace_specs = self._lm.trace_spec

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Run a training step: execute one step of the controller, compute loss with the objective, and optimize."""
        lm = self._lm
        if (
            self._controller is None
            or self._objective is None
            or self._learner is None
        ):
            raise RuntimeError(
                "Controller pretrain runtime not initialized — call setup() first."
            )

        if self._train_carry is None:
            self._train_carry = self._controller.initial_state(batch)

        is_warmup = lm.global_step < self._config.supervised_only_warmup_steps
        assembler = self._ensure_train_batch_assembler(batch)
        execution = self._train_runner.run(
            source=PartialResetSource(
                incoming=batch,
                assembler=assembler,
                carry0=self._train_carry,
            ),
            controller=self._controller,
            carry=self._train_carry,
            options={"allow_halt": not is_warmup, "explore": True},
        )
        self._train_carry = execution.final_carry.detach()

        record = execution.last_record
        if record.snapshot.steps is None:
            raise RuntimeError(
                "Training runner produced a record without step counters."
            )

        ac_batch = self._learner.build_deliberation_ac_batch(
            record.outputs, record.snapshot
        )
        step_output = self._objective.compute_step(
            ac_batch, is_warmup=is_warmup
        )
        local_bs = int(batch["input_ids"].shape[0])
        loss = normalize_loss_for_backward(step_output.loss, local_bs=local_bs)

        optimizer_list = lm.optimizers()
        optimizer_list = (
            list(optimizer_list)
            if isinstance(optimizer_list, (list, tuple))
            else [optimizer_list]
        )
        scheduler_list = lm.lr_schedulers()
        scheduler_list = (
            list(scheduler_list)
            if isinstance(scheduler_list, (list, tuple))
            else [scheduler_list]
        )

        for opt in optimizer_list:
            opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        lm.manual_backward(loss)

        active = [0] if is_warmup else list(range(len(optimizer_list)))
        for idx in active:
            opt = optimizer_list[idx]
            if utils.has_any_grad(opt):
                opt.step()  # type: ignore[union-attr]
                scheduler_list[idx].step()  # type: ignore[union-attr]

        update_metrics_from_step(
            lm.train_metrics, step_output.metrics, RL_STEP_ROUTES
        )
        lm.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss.detach(), "signals": step_output.signals}

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Run a validation step: execute full rollout with the controller, compute episode metrics with the val_scorer."""
        lm = self._lm
        if self._controller is None or self._val_scorer is None:
            raise RuntimeError(
                "Controller pretrain runtime not initialized — call setup() first."
            )

        trace_request = EvaluationTraceRequest(
            trace_spec=lm.trace_spec,
            trace_meta=build_mazehard_ehc_trace_meta(batch),
        )
        evaluation = self.execute_evaluation_batch(
            EvaluationCaseBatch(
                batch=batch,
                case_id=f"val-{batch_idx:04d}",
            ),
            trace_request=trace_request,
        )
        update_metric_collection_from_evaluated_chunk(
            lm.val_metrics, evaluation.evaluated, RL_EPISODE_ROUTES
        )
        return {"trace": evaluation.trace}

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay case through the reason-pretrain eval path."""
        if self._controller is None or self._val_scorer is None:
            raise RuntimeError(
                "Controller pretrain runtime not initialized — call setup() first."
            )

        effective_trace_request = trace_request
        if trace_request is not None:
            trace_meta = dict(build_mazehard_ehc_trace_meta(case.batch))
            if trace_request.trace_meta is not None:
                trace_meta.update(trace_request.trace_meta)
            effective_trace_request = EvaluationTraceRequest(
                trace_spec=trace_request.trace_spec,
                trace_meta=trace_meta,
            )

        return execute_replay_evaluation_batch(
            case=case,
            runner=self._eval_runner,
            controller=self._controller,
            carry=self._controller.initial_state(case.batch),
            objective=self._val_scorer,
            max_rollout_steps=self._config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self._config.runtime.validation.hard_max_rollout_steps,
            runner_options={"explore": False, "allow_halt": False},
            trace_request=effective_trace_request,
        )


# =============================================================================
__all__ = ["EHCReasonPretrainConfig", "EHCReasonPretrainRegime"]

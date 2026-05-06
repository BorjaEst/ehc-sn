"""EHC v1 phase-3 initial alignment Lightning module.

Phase-3 initial alignment trains a single shared ``EHCModelV1`` on both the
arena predictive task and the MazeHard deliberation task in strict batch
alternation.

Freezing contract
-----------------
Trainable (single optimizer):
    model.pfc_to_hpc

Frozen (requires_grad=False):
    model.lec, model.mec, model.hpc
    model.lec_to_hpc, model.mec_to_hpc
    model.hpc_to_pfc
    model.pfc (incl. model.pfc.estimator), model.str
    arena_adapter._decoder
    mazehard_adapter._encoder, mazehard_adapter._decoder

Checkpoint-merge handoff
------------------------
The model is initialized from two separate checkpoints:
    arena checkpoint   → spatial substrate + arena decoder
    mazehard checkpoint → controller weights + mazehard adapter

``pfc_to_hpc`` and ``hpc_to_pfc`` are intentionally NOT loaded from either
checkpoint — they start at fresh initialization because aligning these
projections is the explicit objective of this phase.

Training style
--------------
Strict alternation: arena batch → arena loss → pfc_to_hpc update;
                    mazehard batch → mazehard loss → pfc_to_hpc update; repeat.

Arena stop-gradient seam is preserved exactly as in phase 1 (see
``EHCModelV1.forward`` line 444).  The seam prevents arena loss from reaching
PFC backbone parameters, which also means the arena loss cannot update
``pfc_to_hpc`` through the PFC branch — it updates ``pfc_to_hpc`` through the
HPC spatial pathway only.  Both paths are correct for initial alignment.

hpc_to_pfc is frozen in this first phase-3 slice.  The optional later thaw of
hpc_to_pfc is not implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import lightning as L
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn import utils
from ehc_sn.adapters.arena.ehc import ArenaEHCAdapterSettings, ArenaEHCTaskBinding, ArenaEHCV1BridgeAdapter
from ehc_sn.adapters.arena.ehc.traces import ARENA_EHC_TRACE_FIELDS
from ehc_sn.adapters.mazehard.ehc import (
    MazeHardEHCAdapterSettings,
    MazeHardEHCV1BridgeAdapter,
    MazeHardEHCV1HybridTaskBinding,
)
from ehc_sn.controllers.deliberation.actor_critic import DeliberationACController, DeliberationACControllerConfig
from ehc_sn.controllers.replay.trajectory import ReplayTrajectoryController, ReplayTrajectoryControllerConfig
from ehc_sn.lightning._rollout import (
    evaluate_rollout,
    evaluate_rollout_streaming,
    observe_rollout_chunk,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.lightning.ehc.core.checkpoint_merge import MergeReport, merge_ehc_alignment_checkpoints
from ehc_sn.lightning.eval.contracts import EvaluationBatchArtifacts, EvaluationTraceRequest
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.metrics import build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import EHC_EPISODE_ROUTES, EHC_STEP_ROUTES, RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1, ModelSettingsV1
from ehc_sn.objectives.ehc import EHCObjective, EHCObjectiveConfig
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig, HybridRLLossHead
from ehc_sn.rollouts import PartialResetSource, RecurrentRunner, RepeatSource, SingleStepRunner
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.mazehard.capabilities.deliberation import MazeHardDeliberationCapability, MazeHardDeliberationConfig
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.training.actor_critic import TD0ActorCriticBatchBuilder, ZeroBootstrapActorCriticValidationScorer
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.types import Batch

logger = logging.getLogger(__name__)


# =============================================================================
class ModelConfig_EHC_V1_Alignment(BaseModel, extra="forbid"):
    """Configuration for the EHC v1 phase-3 initial alignment Lightning module."""

    # ~~ Shared model architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_config_path: Path = Field(
        ...,
        description=(
            "Path to the shared EHC v1 model configuration TOML file. "
            "Use ehc-v1-mazehard.toml (pfc.seq_length=900) so both adapters are compatible."
        ),
    )

    # ~~ Checkpoint handoff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    arena_ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to the phase-1 arena training checkpoint (.ckpt). None = skip merge.",
    )
    mazehard_ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to the phase-2 mazehard training checkpoint (.ckpt). None = skip merge.",
    )

    # ~~ Arena task ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    arena_adapter: ArenaEHCAdapterSettings = Field(
        ...,
        description="Arena bridge adapter settings (observation_dim, action_count).",
    )
    arena_controller: ReplayTrajectoryControllerConfig = Field(
        default_factory=ReplayTrajectoryControllerConfig,
        description="Arena replay trajectory controller config.",
    )
    arena_objective: EHCObjectiveConfig = Field(
        default_factory=EHCObjectiveConfig,
        description="EHC predictive objective config for the arena task.",
    )

    # ~~ MazeHard task ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mazehard_adapter: MazeHardEHCAdapterSettings = Field(
        default_factory=MazeHardEHCAdapterSettings,
        description="MazeHard bridge adapter settings.",
    )
    mazehard_deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="Deliberation capability config (halt_action, episode_horizon).",
    )
    mazehard_controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="Deliberation actor-critic controller config.",
    )
    mazehard_objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL objective config for the MazeHard task.",
    )

    # ~~ Optimizer & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Single optimizer covering model.pfc_to_hpc parameters only.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Validation runner safety settings.",
    )

    # ~~ Phase gate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    alignment_phase: Literal["initial_alignment"] = Field(
        default="initial_alignment",
        description="Phase gate. Only 'initial_alignment' is accepted in this surface.",
    )

    # ~~ Extra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_batch_size: int = Field(
        ...,
        description="Global batch size (used to size the MazeHard partial-reset buffer).",
    )


# =============================================================================
class TrainingModel(L.LightningModule):
    """LightningModule for EHC v1 phase-3 initial alignment.

    Manages:
        - One shared ``EHCModelV1`` with two task adapters.
        - Freezing all parameters except ``model.pfc_to_hpc``.
        - Optional checkpoint merge from arena and mazehard sources.
        - Strict batch alternation (arena → mazehard → arena → …) via the
          ``AlternatingLoader`` from ``ehc_sn.data.alternating``.
        - Single-optimizer updates on ``pfc_to_hpc`` from both task losses.

    hpc_to_pfc is frozen in this initial phase-3 slice.
    """

    # All modules and projections that are NOT trainable in this phase.
    _FROZEN_MODULES = ("lec", "mec", "hpc", "pfc", "str")
    _FROZEN_PROJECTIONS = ("lec_to_hpc", "mec_to_hpc", "hpc_to_pfc")
    # pfc_to_hpc is the ONLY trainable parameter block.
    _TRAINABLE_PROJECTION = "pfc_to_hpc"

    def __init__(self, config: ModelConfig_EHC_V1_Alignment) -> None:
        super().__init__()
        model_settings = ModelSettingsV1.from_config(config.model_config_path)
        self.model = EHCModelV1(model_settings)

        self.arena_adapter = ArenaEHCV1BridgeAdapter(self.model, config.arena_adapter)
        self.mazehard_adapter = MazeHardEHCV1BridgeAdapter(self.model, config.mazehard_adapter)

        # Freeze all parameters except model.pfc_to_hpc.
        self._freeze_all_except_pfc_to_hpc()

        # Runtimes — initialized in setup().
        self._arena_ctrl: ReplayTrajectoryController | None = None
        self._arena_objective_head: EHCObjective | None = None
        self._mh_ctrl: DeliberationACController | None = None
        self._mh_objective_head: HybridRLLossHead | None = None
        self._mh_learner: TD0ActorCriticBatchBuilder | None = None
        self._mh_val_scorer: ZeroBootstrapActorCriticValidationScorer | None = None

        self._config = config
        self._arena_carry = None  # persisted across arena steps; reset each epoch
        self._arena_runner = RecurrentRunner()
        self._mh_train_runner = SingleStepRunner()
        self._mh_eval_runner = RecurrentRunner()

        self.automatic_optimization = False
        self._mh_carry = None

        self.val_metrics = build_val_metrics(RL_EPISODE_ROUTES).clone(prefix="val/")
        self.trace_specs = build_trace_spec("rl")

        self._mh_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,
            keys=("input_ids", "labels"),
            pin_memory=True,
        )
        self._mh_assembler = PartialResetBatchAssembler(
            buffer=self._mh_buffer,
            keys=("input_ids", "labels"),
        )

    @property
    def config(self) -> ModelConfig_EHC_V1_Alignment:
        """Return the parsed configuration for this LightningModule."""
        return self._config

    # ---- Freezing -------------------------------------------------------------------------------

    def _freeze_all_except_pfc_to_hpc(self) -> None:
        """Set requires_grad=False on every parameter except model.pfc_to_hpc.

        After this call only ``model.pfc_to_hpc`` parameters have
        ``requires_grad=True``.  Both adapters' encoder/decoder modules are
        also frozen since they do not contain pfc_to_hpc parameters.
        """
        pfc_to_hpc_ids = {id(p) for p in self.model.pfc_to_hpc.parameters()}

        # Freeze via the shared model first (covers both adapters since both
        # store a reference to the same EHCModelV1 object).
        for name in self._FROZEN_MODULES:
            for p in getattr(self.model, name).parameters():
                p.requires_grad_(False)
        for name in self._FROZEN_PROJECTIONS:
            for p in getattr(self.model, name).parameters():
                p.requires_grad_(False)

        # Freeze adapter-specific components (encoder/decoder); they do not
        # contain pfc_to_hpc params so the id check is redundant but kept for
        # safety.
        for p in self.arena_adapter._decoder.parameters():  # type: ignore[attr-defined]
            if id(p) not in pfc_to_hpc_ids:
                p.requires_grad_(False)
        for p in self.mazehard_adapter._encoder.parameters():  # type: ignore[attr-defined]
            if id(p) not in pfc_to_hpc_ids:
                p.requires_grad_(False)
        for p in self.mazehard_adapter._decoder.parameters():  # type: ignore[attr-defined]
            if id(p) not in pfc_to_hpc_ids:
                p.requires_grad_(False)

    # ---- Lifecycle ------------------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize runtimes and perform the checkpoint merge."""
        # Arena runtime: reuses phase-1 controller and objective.
        self._arena_ctrl = ReplayTrajectoryController(
            backbone=self.arena_adapter,
            config=self.config.arena_controller,
            runtime=ArenaReplayCapability(),
        )
        self._arena_objective_head = EHCObjective(
            self.config.arena_objective,
            task_binding=ArenaEHCTaskBinding(),
        )

        # MazeHard runtime: reuses phase-2 controller and objective.
        finalizer = MazeHardDeliberationCapability(
            self.config.mazehard_deliberation,
            MazeHardRewardProjector(),
        )
        self._mh_ctrl = DeliberationACController(
            self.mazehard_adapter,
            self.config.mazehard_controller,
            finalizer,
        )
        self._mh_objective_head = HybridRLLossHead(self.config.mazehard_objective)
        mh_task_binding = MazeHardEHCV1HybridTaskBinding()
        self._mh_learner = TD0ActorCriticBatchBuilder(
            self.mazehard_adapter,
            None,
            gamma=self.config.mazehard_objective.gamma,
            task_binding=mh_task_binding,
        )
        self._mh_val_scorer = ZeroBootstrapActorCriticValidationScorer(
            self._mh_objective_head,
            mh_task_binding,
        )

        # Checkpoint merge (optional — skip if either path is None).
        if self.config.arena_ckpt_path is not None and self.config.mazehard_ckpt_path is not None:
            report = merge_ehc_alignment_checkpoints(
                self.config.arena_ckpt_path,
                self.config.mazehard_ckpt_path,
                model=self.model,
                arena_decoder=self.arena_adapter._decoder,  # type: ignore[attr-defined]
                mazehard_encoder=self.mazehard_adapter._encoder,  # type: ignore[attr-defined]
                mazehard_decoder=self.mazehard_adapter._decoder,  # type: ignore[attr-defined]
            )
            logger.info(
                "alignment: checkpoint merge complete. " "spatial keys=%d, ctrl keys=%d, fresh_init_keys=%d",
                len(report.spatial_keys_from_arena),
                len(report.controller_keys_from_mazehard),
                len(report.fresh_init_keys),
            )
        else:
            logger.info(
                "alignment: one or both checkpoint paths are None — " "model initialized from scratch (expected in tests / debug runs)."
            )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[SequentialLR]]:
        """Build a single optimizer covering only ``model.pfc_to_hpc`` parameters.

        Returns:
            ``([opt], [sch])`` with a single optimizer and scheduler.
        """
        total_steps = int(self.trainer.estimated_stepping_batches)
        trainable = [p for p in self.model.pfc_to_hpc.parameters() if p.requires_grad]
        opt = AdamATan2(trainable, self.config.optimizer)
        sch = CosineAnnealingLRWithWarmup(opt, total_steps, self.config.scheduler)
        return [opt], [sch]

    def on_train_epoch_start(self) -> None:
        """Reset mazehard carry, arena carry, and buffer at the start of each epoch."""
        self._arena_carry = None
        self._mh_carry = None
        self._mh_buffer.clear()

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics."""
        self.val_metrics.reset()

    # ---- Task-specific step helpers --------------------------------------------------------------

    def _step_arena(self, batch: Batch) -> Tensor:
        """Run one arena step and return the arena predictive loss.

        Carries the arena PFC state across steps within an epoch so that the
        ``prev_pfc_summary`` entering ``pfc_to_hpc`` is the actual PFC output
        from the previous arena observation — not the reset-zero summary that
        would result from calling ``initial_state`` every step.

        The arena stop-gradient seam in ``EHCModelV1.forward`` (line 444) is
        preserved: the PFC summary is detached before entering ``pfc_to_hpc``,
        so arena loss cannot update PFC backbone parameters.
        """
        if self._arena_ctrl is None or self._arena_objective_head is None:
            raise RuntimeError("Arena runtime not initialized. Call setup() first.")

        carry = self._arena_carry if self._arena_carry is not None else self._arena_ctrl.initial_state(batch)
        evaluation = evaluate_rollout_streaming(
            runner=self._arena_runner,
            source=RepeatSource(batch),
            controller=self._arena_ctrl,
            carry=carry,
            objective=self._arena_objective_head,
            max_rollout_steps=1,
        )
        self._arena_carry = evaluation.execution.final_carry.detach()
        return evaluation.loss

    def _step_mazehard(self, batch: Batch) -> Tensor:
        """Run one MazeHard horizon-1 RL step and return the mazehard loss.

        Mirrors the phase-2 training_step with the warmup removed (all params
        except pfc_to_hpc are already frozen so the warmup distinction is moot).
        """
        if self._mh_ctrl is None or self._mh_objective_head is None or self._mh_learner is None:
            raise RuntimeError("MazeHard runtime not initialized. Call setup() first.")

        if self._mh_carry is None:
            self._mh_carry = self._mh_ctrl.initial_state(batch)

        execution = self._mh_train_runner.run(
            source=PartialResetSource(
                incoming=batch,
                assembler=self._mh_assembler,
                carry0=self._mh_carry,
            ),
            controller=self._mh_ctrl,
            carry=self._mh_carry,
            options={"allow_halt": True, "explore": True},
        )
        self._mh_carry = execution.final_carry.detach()

        record = execution.last_record
        if record.snapshot.steps is None:
            raise RuntimeError("MazeHard training runner produced a record without step counters.")

        ac_batch = self._mh_learner.build_deliberation_ac_batch(record.outputs, record.snapshot)
        step_output = self._mh_objective_head.compute_step(ac_batch, is_warmup=False)

        local_bs = int(batch["input_ids"].shape[0])
        return normalize_loss_for_backward(step_output.loss, local_bs=local_bs)

    # ---- Training -------------------------------------------------------------------------------

    def training_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Route the batch to the arena or mazehard step based on source tag.

        Batch alternation is enforced by the :class:`~ehc_sn.data.alternating.AlternatingLoader`.
        The ``batch["__source__"]`` key carries the routing tag.  Both steps
        update only ``model.pfc_to_hpc`` since all other parameters are frozen.
        """
        source = batch.pop("__source__")  # type: ignore[attr-defined]
        # Remove the redundant arena tag if present (AlternatingLoader adds "arena" key)
        batch.pop("arena", None)  # type: ignore[attr-defined]

        opt = self.optimizers()
        sch = self.lr_schedulers()

        if isinstance(opt, list):
            opt = opt[0]
        if isinstance(sch, list):
            sch = sch[0]

        opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]

        if source == "arena":
            loss = self._step_arena(batch)
        else:
            loss = self._step_mazehard(batch)

        if loss.grad_fn is not None:
            self.manual_backward(loss)

        if utils.has_any_grad(opt):
            opt.step()  # type: ignore[union-attr]
            sch.step()  # type: ignore[union-attr]

        self.log(f"train/{source}/loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss.detach(), "source": source}

    # ---- Validation -----------------------------------------------------------------------------

    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Run a full MazeHard rollout to collect validation metrics."""
        if self._mh_ctrl is None or self._mh_val_scorer is None:
            raise RuntimeError("MazeHard runtime not initialized. Call setup() first.")

        carry0 = self._mh_ctrl.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._mh_eval_runner,
            source=RepeatSource(batch),
            controller=self._mh_ctrl,
            carry=carry0,
            objective=self._mh_val_scorer,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"explore": False, "allow_halt": False},
        )
        trace = observe_rollout_chunk(evaluation.chunk, self.trace_specs, trace_meta={})
        update_metric_collection_from_evaluated_chunk(self.val_metrics, evaluation.evaluated, RL_EPISODE_ROUTES)
        return {"trace": trace}

    def build_evaluation_metrics(self, namespace: str) -> MetricCollection:
        """Return a fresh metric collection with the given namespace prefix."""
        return build_val_metrics(RL_EPISODE_ROUTES).clone(prefix=namespace)

    def execute_evaluation_batch(
        self,
        batch: Batch,
        trace_request: Optional[EvaluationTraceRequest],
        source_context: object | None = None,
    ) -> EvaluationBatchArtifacts:
        """Execute one MazeHard evaluation batch and return scored artifacts.

        Mirrors :class:`~ehc_sn.lightning.ehc.ehc_v1_mazehard.TrainingModel`
        behaviour.  Does not write into ``self.val_metrics``.
        """
        if self._mh_ctrl is None or self._mh_val_scorer is None:
            raise RuntimeError("MazeHard runtime not initialized. Call setup() before evaluation.")

        carry0 = self._mh_ctrl.initial_state(batch)
        evaluation = evaluate_rollout(
            runner=self._mh_eval_runner,
            source=RepeatSource(batch),
            controller=self._mh_ctrl,
            carry=carry0,
            objective=self._mh_val_scorer,
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

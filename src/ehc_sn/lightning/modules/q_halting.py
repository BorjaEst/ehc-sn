"""Generic LightningModule for actor-critic training regimes.

Supports HRM v2 (and future variants) via parameterized component classes.
Key difference from ACT-supervised: three-optimizer training, deliberation
value-control pipeline, warmup gating for RL losses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import lightning as L
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn import utils
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.data.episode_sources import ShuffledEpisodeSource
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.diagnostics import DiagnosticTraceSpec
from ehc_sn.metrics.adapter import update_metrics_from_step
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.reducers import HiddenNormHistogram, compute_nonempty
from ehc_sn.metrics.rollout import update_metric_collection_from_evaluated_chunk
from ehc_sn.metrics.step_metrics import StepMetrics
from ehc_sn.objectives.composites.hybrid_rl import HybridRLLossConfig
from ehc_sn.rollouts.runtime import RecurrentRunner, SingleStepRunner
from ehc_sn.rollouts.sources import DemandDrivenReplaySource, _move_batch_to
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.distributed import (
    SumOverBatch,
    normalize_loss_for_backward,
)
from ehc_sn.training.hrm import ValidationRuntimeConfig
from ehc_sn.training.hrm import load_weights_from_checkpoint as _loader
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.rollout import run_captured_rollout
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
)
from ehc_sn.types import Batch


# =============================================================================
class RuntimeConfigLike(Protocol):
    """Protocol for deliberation/execution policy configs consumed by
    :class:`QHaltingModule`.

    Both :class:`~ehc_sn.tasks.mazehard.runtime.MazeHardRuntimeConfig` and
    :class:`~ehc_sn.tasks.seqmaze.runtime.SeqMazeRuntimeConfig` satisfy this
    protocol.  The module does not import task-specific types.
    """

    halt_action: int
    episode_horizon: int
    validation: ValidationRuntimeConfig


# =============================================================================
def _payload_width(batch: Batch) -> int:
    """Return the leading (batch) dimension from an honest payload batch.

    The honest iterator yields a dict whose tensors share a leading dimension
    equal to the per-rank replay slot count.  Returns that dimension, or
    raises ``ValueError`` if the batch is empty.
    """
    for v in batch.values():
        if isinstance(v, Tensor):
            return int(v.shape[0])
    raise ValueError(
        "Honest payload batch is empty or contains no tensors. "
        "Cannot determine batch width."
    )


# =============================================================================
@dataclass(frozen=True, slots=True, eq=False)
class QHaltingBindings:
    """Immutable experiment-to-module bindings for actor-critic regimes."""

    # eq=False: fields include callables and classes; structural equality
    # is semantically meaningless; identity comparison is correct.

    model_cls: type[nn.Module]
    model_settings_cls: type[BaseModel]
    adapter_cls: type[nn.Module]
    adapter_settings_cls: type[BaseModel]
    controller_cls: type
    controller_config_cls: type[BaseModel]
    objective_cls: type[nn.Module]
    objective_config_cls: type[BaseModel]
    supervision_builder: Callable[[Any], object]
    """Task-owned supervision builder.

    Signature: ``(executed_batch) -> supervision``, where supervision is a
    task-owned dataclass with ``.labels`` and optionally ``.task_logits``.
    """
    learner_cls: type
    val_scorer_cls: type
    optimizer_cls: type[Optimizer]
    optimizer_config_cls: type[BaseModel]
    runtime_config_cls: type[BaseModel]
    reward_config_cls: type[BaseModel]
    runtime_cls: type
    reward_projector_cls: type
    trace_fields: tuple
    build_trace_meta_fn: Callable
    step_routes: tuple
    episode_routes: tuple
    hidden_state_fields: tuple
    token_weight_builder: Callable[[Any], Tensor] | None = None
    """Optional task-owned token weight builder.

    Signature: ``(record) -> tensor`` returning per-token weights.
    When ``None``, uniform weights are used.
    """


class QHaltingTrainingConfig(BaseModel, extra="forbid"):
    """Training-only configuration for an actor-critic experiment.

    Not required for evaluation — only used to construct optimizers
    and reward projection during training.
    """

    optimizer_supervised: AdamATan2Config = Field(
        ...,
        description="Optimizer config for supervised parameters.",
    )
    optimizer_rl: AdamATan2Config = Field(
        ...,
        description="Optimizer config for RL parameters.",
    )
    optimizer_qv: AdamATan2Config = Field(
        ...,
        description="Optimizer config for vmPFC parameters.",
    )
    reward: dict | BaseModel = Field(
        ...,
        description="Training-only reward configuration.  Concrete "
        "validation (MazeHardRewardConfig, SeqMazeRewardConfig) is "
        "performed by the training experiment builder.",
    )
    num_slots: int | None = Field(
        default=None,
        ge=1,
        description="Per-rank carry buffer width (concurrent trajectory slots). "
        "``None`` during evaluation — carry is allocated per batch from the "
        "DataLoader batch dimension.",
    )
    halt_disabled_steps: int = Field(
        default=5000,
        ge=0,
        description="Optimizer steps during which learned halting is disabled.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning-rate scheduler configuration.",
    )


class QHaltingComponentConfigs(BaseModel, extra="forbid"):
    """Concrete component configs for an actor-critic experiment.

    Validated and populated by the experiment builder, consumed by
    the regime module.  Fields hold opaque validated configs whose
    concrete Pydantic types are determined by the experiment builder.
    Contains only the configs needed to construct the computational
    graph — optimizers and training-only settings live in
    :class:`QHaltingTrainingConfig`.
    """

    adapter: BaseModel = Field(
        ...,
        description="Adapter settings (task-specific, validated by experiment builder).",
    )
    controller: DeliberationQHaltingControllerConfig = Field(
        default_factory=lambda: None,
        description="Deliberation AC controller configuration.",
    )
    objective: BaseModel = Field(
        ...,
        description="Objective configuration (task-specific, validated by "
        "experiment builder).",
    )


class QHaltingConfig(BaseModel, extra="forbid"):
    """Regime-owned settings for an actor-critic Lightning experiment.

    Contains only fields the regime module can validate without knowing
    the concrete experiment.  Component-specific configs live in
    :class:`QHaltingComponentConfigs`, validated by the
    experiment builder.

    ``num_slots`` is passed as a separate constructor argument,
    not a config field — the honest payload carries observed width.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file.",
    )
    halt_disabled_steps: int = Field(
        default=5000,
        ge=0,
        description="Optimizer steps during which learned halting is disabled.",
    )


class QHaltingModule(L.LightningModule):
    """Generic LightningModule for actor-critic deliberation training.

    Accepts an :class:`QHaltingConfig`, an
    :class:`QHaltingComponentConfigs`, and an
    :class:`QHaltingBindings` bundle.  Features three-optimizer
    training, partial-reset batching, and warmup gating.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: QHaltingConfig,
        component_configs: QHaltingComponentConfigs,
        bindings: QHaltingBindings,
        training_config: QHaltingTrainingConfig | None = None,
        *,
        execution: RuntimeConfigLike | None = None,
    ) -> None:
        super().__init__()
        self._bindings = bindings
        self._component_configs = component_configs
        self._training_config = training_config
        self._deliberation = execution
        self._num_slots: int | None = None

        model_settings = bindings.model_settings_cls.from_config(
            config.model_config_path
        )
        self.model = bindings.model_cls(model_settings)
        self.adapter = bindings.adapter_cls(
            self.model, component_configs.adapter
        )
        self.controller: Any = None
        self.objective: Any = None
        self.learner: Any = None
        self.val_scorer: Any = None
        self._config = config
        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()

        self.automatic_optimization = False
        self._train_carry = None
        self._train_source: DemandDrivenReplaySource | None = None
        self._episode_source: ShuffledEpisodeSource | None = None
        self._episode_source_for_lazy: ShuffledEpisodeSource | None = None
        self._pending_source_state: dict[str, Any] | None = None

        self.train_metrics = build_train_metrics(bindings.step_routes).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(bindings.episode_routes).clone(
            prefix="val/"
        )
        self._trace_paradigm: str = "rl"
        self.diagnostic_trace_spec: DiagnosticTraceSpec = DiagnosticTraceSpec(
            enabled=False, max_batches=2, keys=()
        )
        self._diagnostic_traces: list[Any] = []
        self._extra_trace_fields: tuple = bindings.trace_fields

        self._val_hidden_norms = HiddenNormHistogram(
            bin_edges=torch.linspace(0.0, 50.0, 51), max_batches=10
        )
        self._val_reducer_collection = MetricCollection(
            {"hidden_norms": self._val_hidden_norms},
            prefix="val_diag/",
        )

    def load_weights_from_checkpoint(self, path, groups):
        return _loader(self.model, path, groups)

    @property
    def config(self) -> QHaltingConfig:
        return self._config

    def validate_run_plan(  # -------------------------------------------------
        self,
    ) -> None:
        """Assert that all preconditions are satisfied before the first optimizer step.

        Runs once at the start of ``training_step``.  Catches misconfiguration
        that would otherwise produce silent incorrect behaviour or delayed crashes.
        """
        if self._deliberation is None:
            raise RuntimeError(
                "validate_run_plan: deliberation config is None. "
                "It must be set before training begins."
            )
        if self._training_config is None:
            raise RuntimeError(
                "validate_run_plan: training_config is None. "
                "Training requires a full QHaltingTrainingConfig."
            )
        rp = self._num_slots
        if rp is not None and rp <= 0:
            raise ValueError(
                f"validate_run_plan: num_slots must be > 0, " f"got {rp}."
            )
        world_size = max(getattr(self.trainer, "world_size", 1), 1)
        if rp is not None and rp % world_size != 0:
            raise ValueError(
                f"validate_run_plan: num_slots ({rp}) "
                f"not divisible by world_size ({world_size})."
            )
        if self._train_source is None:
            raise RuntimeError(
                "validate_run_plan: training source is None. "
                "Call setup('fit') before training."
            )
        if self._train_carry is None:
            raise RuntimeError(
                "validate_run_plan: training carry is None. "
                "Call setup('fit') before training."
            )
        if self.controller is None:
            raise RuntimeError(
                "validate_run_plan: controller is None. "
                "Call setup('fit') before training."
            )
        if self.objective is None:
            raise RuntimeError(
                "validate_run_plan: objective is None. "
                "Call setup('fit') before training."
            )
        if self.learner is None:
            raise RuntimeError(
                "validate_run_plan: learner is None. "
                "Call setup('fit') before training."
            )
        if self.trainer is None:
            raise RuntimeError(
                "validate_run_plan: trainer is None. "
                "Module must be attached to a Trainer before training."
            )
        for i, opt in enumerate(self.optimizers() or []):
            if not any(
                p.requires_grad
                for group in opt.param_groups
                for p in group["params"]
            ):
                raise RuntimeError(
                    f"validate_run_plan: optimizer {i} has no parameters "
                    "with requires_grad=True."
                )

    def setup(  # -------------------------------------------------------------
        self,
        stage: Optional[str] = None,
    ) -> None:
        """Initialize controller, objective, learner, val scorer, and training source.

        During evaluation (training_config is None), reward config defaults
        to a vanilla instance — the runtime always receives a real projector.

        In the ``fit`` stage, also creates the episode source, initial carry,
        and demand-driven replay source for honest-iterator training.
        """
        reward_config = (
            self._training_config.reward
            if self._training_config is not None
            else self._bindings.reward_config_cls()
        )
        if self._deliberation is None:
            raise RuntimeError("deliberation config must be set before setup()")
        runtime = self._bindings.runtime_cls(
            self._deliberation,
            self._bindings.reward_projector_cls(reward_config),
        )
        self.controller = self._bindings.controller_cls(
            self.adapter,
            self._component_configs.controller,
            runtime,
        )
        self.objective = self._bindings.objective_cls(
            self._component_configs.objective
        )
        self.learner = self._bindings.learner_cls(
            self.adapter,
            None,
            gamma=self._component_configs.objective.gamma,
            supervision_builder=self._bindings.supervision_builder,
            token_weight_builder=self._bindings.token_weight_builder,
        )
        self.val_scorer = self._bindings.val_scorer_cls(
            self.objective,
            supervision_builder=self._bindings.supervision_builder,
            token_weight_builder=self._bindings.token_weight_builder,
        )

        # Training source creation moved to on_fit_start() — device-dependent.
        if stage == "fit" and self._training_config is not None:
            tc = self._training_config
            if tc.num_slots is None:
                raise RuntimeError(
                    "num_slots is required for training. "
                    "Set it via QHaltingTrainingConfig.num_slots."
                )
            self._num_slots = tc.num_slots

    def on_fit_start(  # ------------------------------------------------------
        self,
    ) -> None:
        """Create training source and carry on the correct accelerator device."""
        if self._training_config is not None and self._num_slots is not None:
            carry_width = self._num_slots // max(
                getattr(self.trainer, "world_size", 1), 1
            )
            init_batch = self._ensure_episode_source().take(carry_width)
            init_batch = _move_batch_to(init_batch, self.device)
            self._train_carry = self.controller.initial_state(init_batch)
            self._train_carry = _move_batch_to(self._train_carry, self.device)
            self._train_source = DemandDrivenReplaySource(
                episode_source=self._ensure_episode_source(),
                carry0=self._train_carry,
                device=self.device,
            )

    def _assert_setup(self) -> None:
        if self.controller is None:
            raise RuntimeError("setup() not called.")
        if self.objective is None:
            raise RuntimeError("setup() not called.")
        if self.learner is None:
            raise RuntimeError("setup() not called.")
        if self.val_scorer is None:
            raise RuntimeError("setup() not called.")

    def _assert_setup(self) -> None:
        if self.controller is None:
            raise RuntimeError("setup() not called.")
        if self.objective is None:
            raise RuntimeError("setup() not called.")
        if self.learner is None:
            raise RuntimeError("setup() not called.")
        if self.val_scorer is None:
            raise RuntimeError("setup() not called.")

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]] | list[Optimizer]:
        if self._training_config is None:
            return []
        total_steps = int(self.trainer.estimated_stepping_batches)
        c = self._bindings
        tc = self._training_config

        vmPFC_ids = {id(p) for p in self.model.pfc.estimator.parameters()}
        str_ids = {id(p) for p in self.model.str.parameters()}
        excluded_ids = vmPFC_ids | str_ids
        sup_params = [
            p for p in self.adapter.parameters() if id(p) not in excluded_ids
        ]

        opt_sup = c.optimizer_cls(sup_params, tc.optimizer_supervised)
        opt_rl = c.optimizer_cls(
            list(self.model.str.parameters()),
            tc.optimizer_rl,
        )
        opt_qv = c.optimizer_cls(
            list(self.model.pfc.estimator.parameters()),
            tc.optimizer_qv,
        )

        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_sup, total_steps, tc.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/supervised",
            },
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_rl, total_steps, tc.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/rl",
            },
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_qv, total_steps, tc.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/qv",
            },
        ]
        return [opt_sup, opt_rl, opt_qv], schedulers

    def on_train_epoch_start(self) -> None:
        # Carry persists across all DataLoader boundaries per legacy HRM
        # contract — do NOT reset. Buffer is managed by FIFO eviction.
        pass

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()
        self._val_reducer_collection.reset()
        self._diagnostic_traces.clear()

    @property
    def diagnostic_traces(self) -> tuple[Any, ...]:
        return tuple(self._diagnostic_traces)

    def reset_diagnostic_traces(self) -> None:
        self._diagnostic_traces.clear()

    def _ensure_episode_source(  # -------------------------------------------
        self,
    ) -> ShuffledEpisodeSource:
        """Return or create the demand-driven episode source."""
        if self._episode_source is not None:
            return self._episode_source
        datamodule: Any = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError(
                "Demand-driven episode flow requires a DataModule to be "
                "attached to the trainer."
            )
        train_dataset = getattr(datamodule, "_train", None)
        if train_dataset is None:
            raise RuntimeError(
                "DataModule has no training dataset; call setup('fit') first."
            )
        world_size = max(getattr(self.trainer, "world_size", 1), 1)
        rank = getattr(self.trainer, "global_rank", 0)
        self._episode_source = ShuffledEpisodeSource(
            train_dataset,
            rank=rank,
            world_size=world_size,
            seed=(
                self._deliberation.validation.seed or 42
                if self._deliberation is not None
                else 42
            ),
        )

        if self._pending_source_state is not None:
            try:
                self._episode_source.load_state_dict(self._pending_source_state)
            except ValueError:
                import warnings

                saved_cycle = self._pending_source_state.get("cycle", 0)
                warnings.warn(
                    "Episode source fingerprint mismatch. Restarting "
                    f"coverage cycle {saved_cycle} from cursor 0.",
                    RuntimeWarning,
                )
                self._episode_source.restart_cycle(saved_cycle)
            self._pending_source_state = None

        return self._episode_source

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        self._assert_setup()
        if not getattr(self, "_run_plan_validated", False):
            self.validate_run_plan()
            self._run_plan_validated = True

        # Validate payload width matches requested geometry on first step.
        if not getattr(self, "_payload_validated", False):
            world_size = max(getattr(self.trainer, "world_size", 1), 1)
            observed = _payload_width(batch)
            requested = (self._num_slots or 0) // world_size
            if observed != requested:
                raise ValueError(
                    f"Honest payload width ({observed}) does not match "
                    f"requested num_slots // world_size "
                    f"({requested}). Check your data config."
                )
            self._payload_validated = True

        halt_disabled = self.global_step < self.config.halt_disabled_steps
        execution = run_captured_rollout(
            runner=self._train_runner,
            source=self._train_source,
            controller=self.controller,
            carry=self._train_carry,
            runner_options={
                "allow_halt": not halt_disabled,
                "explore": True,
                "halt_action": self._deliberation.halt_action,
                "max_halt_steps": self._deliberation.episode_horizon,
            },
        )
        self._train_carry = execution.final_carry.detach()
        self._train_source.update(carry=self._train_carry)

        record = execution.last_record
        if record.snapshot.steps is None:
            raise RuntimeError(
                "Training runner produced a record without step counters."
            )
        if self._train_carry is None:
            raise RuntimeError("Training carry missing after rollout.")
        next_obs = self._train_carry.data
        use_token_weights = hasattr(
            self.learner._task_binding, "extract_token_weights"
        )
        ac_batch = self.learner.build_deliberation_ac_batch(
            record.outputs,
            record.snapshot,
            next_obs=next_obs,
            carry=self._train_carry,
            use_token_weights=use_token_weights,
        )
        step_output = self.objective.compute_step(ac_batch, is_warmup=is_warmup)

        world_size_ = max(getattr(self.trainer, "world_size", 1), 1)
        rp = self._num_slots
        if rp is not None and rp % world_size_ != 0:
            raise ValueError(
                f"num_slots ({rp}) "
                f"not divisible by world_size ({world_size_})"
            )
        carry_width = (rp or 0) // world_size_
        loss = normalize_loss_for_backward(
            SumOverBatch(step_output.loss), local_bs=carry_width
        )

        # --- Auxiliary edge-prediction loss (multi-task training) ------------
        # Provides the dense (N² pairs per sample) signal the HRM needs to
        # learn disentanglement from composited slot representations.
        edge_logits = None
        try:
            edge_logits = record.outputs.task.edge_logits
        except (AttributeError, KeyError):
            pass
        if edge_logits is not None:
            # edge_logits: (B, N, N, 2), labels: (B, N, N)
            B, N, _, _ = edge_logits.shape
            # Get edge labels and mask from the step's batch data.
            step_batch = record.batch
            edge_labels = step_batch["edge_label"].to(
                dtype=torch.long, device=edge_logits.device
            )
            edge_mask = step_batch["edge_mask"].to(
                dtype=torch.bool, device=edge_logits.device
            )
            edge_loss = F.cross_entropy(
                edge_logits[edge_mask],
                edge_labels[edge_mask],
            )
            # Scale edge loss so it doesn't dominate the total or destabilize
            # the shared PFC backbone.  N² terms per sample vs ~N path tokens.
            loss = loss + (1.0 / (N * N)) * edge_loss
            edge_loss_weight = 1.0 / N  # normalize by graph size
            loss = loss + edge_loss_weight * edge_loss

        optimizer_list = self.optimizers()
        optimizer_list = (
            list(optimizer_list)
            if isinstance(optimizer_list, (list, tuple))
            else [optimizer_list]
        )
        scheduler_list = self.lr_schedulers()
        scheduler_list = (
            list(scheduler_list)
            if isinstance(scheduler_list, (list, tuple))
            else [scheduler_list]
        )

        for opt in optimizer_list:
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        active_indices = [0] if is_warmup else list(range(len(optimizer_list)))
        for idx in active_indices:
            opt = optimizer_list[idx]
            opt.step()
            scheduler_list[idx].step()

        update_metrics_from_step(
            self.train_metrics,
            step_output.metrics,
            self._bindings.step_routes,
        )
        self.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )  # fmt: skip

        return {
            "loss": loss.detach(),
            "signals": step_output.signals,
        }

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        self._assert_setup()

        trace_request = None
        ds = self.diagnostic_trace_spec
        if ds.enabled and batch_idx < ds.max_batches and ds.keys:
            trace_request = EvaluationTraceRequest(
                trace_spec=build_trace_spec(
                    "rl",
                    include_keys=set(ds.keys),
                    extra_fields=self._bindings.hidden_state_fields,
                ),
                trace_meta=dict(self._bindings.build_trace_meta_fn(batch)),
            )
        evaluation = self.execute_evaluation_batch(
            EvaluationCaseBatch(
                batch=batch,
                case_id=f"val-{batch_idx:04d}",
            ),
            trace_request=trace_request,
        )
        update_metric_collection_from_evaluated_chunk(
            collection=self.val_metrics,
            evaluated=evaluation.evaluated,
            routes=self._bindings.episode_routes,
        )

        inp = self.adapter.prepare_inputs(batch)
        model_out, _ = self.model(inp, state=None)
        self._val_hidden_norms.update(
            model_out.theta_summary.norm(dim=-1).detach().cpu()
        )

        if (
            evaluation.trace is not None
            and len(self._diagnostic_traces) < ds.max_batches
        ):
            self._diagnostic_traces.append(evaluation.trace)

        return {"trace": evaluation.trace}

    # ── Checkpoint hooks ────────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist episode source state into the Lightning checkpoint."""
        if self._episode_source is not None:
            checkpoint["episode_source"] = self._episode_source.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore episode source state from checkpoint (deferred).

        Stashes full state for deferred restore in ``_ensure_episode_source``.
        """
        source_state = checkpoint.get("episode_source")
        if source_state is not None:
            self._pending_source_state = source_state

    def on_validation_epoch_end(self) -> None:
        compute_nonempty(self._val_reducer_collection)
        self._val_reducer_collection.reset()

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        self._assert_setup()

        if trace_request is not None:
            trace_request = EvaluationTraceRequest(
                trace_spec=trace_request.trace_spec,
                trace_meta=dict(self._bindings.build_trace_meta_fn(case.batch)),
            )

        return execute_replay_evaluation_batch(
            case=case,
            runner=self._eval_runner,
            controller=self.controller,
            carry=self.controller.initial_state(case.batch),
            objective=self.val_scorer,
            max_rollout_steps=(
                self._deliberation.validation.max_rollout_steps
                if self._deliberation is not None
                else None
            ),
            hard_max_rollout_steps=(
                self._deliberation.validation.hard_max_rollout_steps
                if self._deliberation is not None
                else None
            ),
            runner_options={
                "explore": False,
                "allow_halt": False,
                "halt_action": self._deliberation.halt_action,
                "max_halt_steps": self._deliberation.episode_horizon,
            },
            trace_request=trace_request,
        )

    def aggregate_evaluation_case_metrics(  # ---------------------------------
        self,
        *,
        task: str,
        regime_id: str,
        regime_kind: str,
        case_results: Sequence[EvaluationCaseResult],
    ) -> dict[str, float | int]:
        _ = task, regime_id, regime_kind
        total_completed_count: int = 0
        total_eligible_count: int = 0
        total_accuracy_sum: float = 0.0
        total_exact_sum: float = 0.0
        total_token_correct: int = 0
        total_token_count: int = 0

        for case in case_results:
            for step in case.evaluated.steps:
                metrics = step.outputs.metrics
                if not isinstance(metrics, StepMetrics):
                    raise TypeError(
                        f"Expected StepMetrics, got {type(metrics).__name__}."
                    )
                ep = metrics.episode
                ep_tok = metrics.episode_tokens
                total_completed_count += int(ep.completed_count.item())
                total_eligible_count += int(ep.eligible_count.item())
                total_accuracy_sum += float(ep.accuracy_sum.item())
                total_exact_sum += float(ep.exact_sum.item())
                total_token_correct += int(ep_tok.token_correct_sum.item())
                total_token_count += int(ep_tok.token_count_sum.item())

        result: dict[str, float | int] = {
            "n_sequence_completed": total_completed_count,
            "n_sequence_eligible": total_eligible_count,
            "n_token_correct": total_token_correct,
            "n_token_total": total_token_count,
        }
        if total_completed_count > 0:
            result["sequence_accuracy"] = (
                total_accuracy_sum / total_completed_count
            )
            result["sequence_exact"] = total_exact_sum / total_completed_count
        if total_token_count > 0:
            result["token_accuracy"] = total_token_correct / total_token_count
        return result


__all__ = [
    "QHaltingBindings",
    "QHaltingComponentConfigs",
    "QHaltingConfig",
    "QHaltingModule",
]

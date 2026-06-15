"""Generic LightningModule for actor-critic training regimes.

Supports HRM v2 (and future variants) via parameterized component classes.
Key difference from ACT-supervised: three-optimizer training, deliberation
value-control pipeline, warmup gating for RL losses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import lightning as L
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn import utils
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACControllerConfig,
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
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig
from ehc_sn.rollouts.runtime import RecurrentRunner, SingleStepRunner
from ehc_sn.rollouts.sources import DemandDrivenReplaySource, _move_batch_to
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.rollout import run_captured_rollout
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
)
from ehc_sn.types import Batch


@dataclass(frozen=True, slots=True, eq=False)
class ActorCriticBindings:
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
    task_binding_cls: type
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


class ActorCriticTrainingConfig(BaseModel, extra="forbid"):
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
    hrm_runtime: HRMRuntimeConfig = Field(
        ...,
        description="HRM runtime configuration (validation safety limits).",
    )


class ActorCriticComponentConfigs(BaseModel, extra="forbid"):
    """Concrete component configs for an actor-critic experiment.

    Validated and populated by the experiment builder, consumed by
    the regime module.  Fields hold opaque validated configs whose
    concrete Pydantic types are determined by the experiment builder.
    Contains only the configs needed to construct the computational
    graph — optimizers and training-only settings live in
    :class:`ActorCriticTrainingConfig`.
    """

    adapter: BaseModel = Field(
        ...,
        description="Adapter settings (task-specific, validated by experiment builder).",
    )
    controller: DeliberationACControllerConfig = Field(
        default_factory=lambda: None,
        description="Deliberation AC controller configuration.",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL objective configuration.",
    )


class ActorCriticConfig(BaseModel, extra="forbid"):
    """Regime-owned settings for an actor-critic Lightning experiment.

    Contains only fields the regime module can validate without knowing
    the concrete experiment.  Component-specific configs live in
    :class:`ActorCriticComponentConfigs`, validated by the
    experiment builder.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description="Number of optimizer steps during which only the "
        "supervised optimizer trains.",
    )


class ActorCriticModule(L.LightningModule):
    """Generic LightningModule for actor-critic deliberation training.

    Accepts an :class:`ActorCriticConfig`, an
    :class:`ActorCriticComponentConfigs`, and an
    :class:`ActorCriticBindings` bundle.  Features three-optimizer
    training, partial-reset batching, and warmup gating.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ActorCriticConfig,
        component_configs: ActorCriticComponentConfigs,
        bindings: ActorCriticBindings,
        training_config: ActorCriticTrainingConfig | None = None,
    ) -> None:
        super().__init__()
        self._bindings = bindings
        self._component_configs = component_configs
        self._training_config = training_config

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

        # Deliberation (task runtime config) is required for both training
        # and evaluation.  It is set by the experiment builder after init
        # via setter, since the module is constructed before setup().
        self._deliberation: Any = None

    @property
    def config(self) -> ActorCriticConfig:
        return self._config

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize controller, objective, learner, and val scorer.

        During evaluation (training_config is None), reward config defaults
        to a vanilla instance — the runtime always receives a real projector.
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
        task_binding = self._bindings.task_binding_cls()
        self.learner = self._bindings.learner_cls(
            self.adapter,
            None,
            gamma=self._component_configs.objective.gamma,
            task_binding=task_binding,
        )
        self.val_scorer = self._bindings.val_scorer_cls(
            self.objective, task_binding
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
                    opt_sup, total_steps, self.config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/supervised",
            },
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_rl, total_steps, self.config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/rl",
            },
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_qv, total_steps, self.config.scheduler
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
                self._training_config.hrm_runtime.validation.seed or 42
                if self._training_config is not None
                else 42
            ),
        )
        return self._episode_source

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, Any]:
        self._assert_setup()

        if self._train_carry is None:
            local_bs = next(iter(batch.values())).shape[0]
            world_size_ = max(getattr(self.trainer, "world_size", 1), 1)
            init_batch = self._ensure_episode_source().take(
                local_bs * world_size_
            )
            init_batch = _move_batch_to(init_batch, self.device)
            self._train_carry = self.controller.initial_state(init_batch)

            # Create the persistent rollout source, lifetime = training run.
            self._train_source = DemandDrivenReplaySource(
                episode_source=self._ensure_episode_source(),
                carry0=self._train_carry,
                device=self.device,
            )

        source = self._train_source

        is_warmup = self.global_step < self.config.supervised_only_warmup_steps
        execution = run_captured_rollout(
            runner=self._train_runner,
            source=source,
            controller=self.controller,
            carry=self._train_carry,
            runner_options={
                "allow_halt": not is_warmup,
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
        ac_batch = self.learner.build_deliberation_ac_batch(
            record.outputs,
            record.snapshot,
            next_obs=next_obs,
            carry=self._train_carry,
            use_token_weights=True,
        )
        step_output = self.objective.compute_step(ac_batch, is_warmup=is_warmup)

        local_bs = max(
            next(iter(batch.values())).shape[0],
            1,
        )
        loss = normalize_loss_for_backward(step_output.loss, local_bs=local_bs)

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
            if utils.has_any_grad(opt):
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
                self._training_config.hrm_runtime.validation.max_rollout_steps
                if self._training_config is not None
                else None
            ),
            hard_max_rollout_steps=(
                self._training_config.hrm_runtime.validation.hard_max_rollout_steps
                if self._training_config is not None
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
    "ActorCriticBindings",
    "ActorCriticComponentConfigs",
    "ActorCriticConfig",
    "ActorCriticModule",
]

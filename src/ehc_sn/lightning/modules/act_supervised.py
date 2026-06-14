"""Generic LightningModule for ACT-supervised training regimes.

Supports HRM v1 (and future variants) via parameterized component classes.

Access pattern for config attributes:
    self.regime_config.*       — regime-owned settings (scheduler, batch scale, warmup, target-network)
    self.component_configs.*   — experiment-selected component configs (adapter, controller, objective, optimizer, runtime)
    self._component_configs.*  — same as above (internal; prefer the property)

    # grep-rule: no self.config.adapter|self.config.controller|self.config.objective|self.config.optimizer|self.config.runtime
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

from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.data.episode_sources import ShuffledEpisodeSource
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.diagnostics import DiagnosticTraceSpec
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.reducers import HiddenNormHistogram, compute_nonempty
from ehc_sn.metrics.rollout import update_metric_collection_from_evaluated_chunk
from ehc_sn.metrics.step_metrics import StepMetrics
from ehc_sn.objectives.act import ACTObjectiveConfig
from ehc_sn.rollouts.runtime import RecurrentRunner, SingleStepRunner
from ehc_sn.rollouts.sources import DemandDrivenReplaySource, _move_batch_to
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.distributed import normalize_loss_for_backward
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2Config
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


@dataclass(frozen=True, slots=True, eq=False)
class ACTSupervisedBindings:
    """Immutable experiment-to-module bindings for ACT-supervised regimes.

    Each field is a concrete class (not instance), so the module can
    instantiate them in its own lifecycle.
    """

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
    optimizer_cls: type[Optimizer]
    optimizer_config_cls: type[BaseModel]
    trace_fields: tuple
    build_trace_meta_fn: Callable
    step_routes: tuple
    episode_routes: tuple
    hidden_state_fields: tuple
    task_scorer_factory: Callable[[], Any] | None = None


class ACTSupervisedComponentConfigs(BaseModel, extra="forbid"):
    """Concrete component configs for an ACT-supervised experiment.

    Validated and populated by the experiment builder, consumed by
    the regime module.  Fields carry concrete Pydantic types.
    """

    adapter: BaseModel = Field(
        ...,
        description="Task-specific adapter settings. Validated by the experiment builder.",
    )
    controller: ACTControllerConfig = Field(
        ...,
        description="ACT deliberation controller configuration.",
    )
    objective: ACTObjectiveConfig = Field(
        ...,
        description="ACT objective configuration.",
    )
    optimizer: AdamATan2Config = Field(
        ...,
        description="Family-specific optimizer configuration.",
    )
    runtime: HRMRuntimeConfig = Field(
        ...,
        description="HRM/ACT runtime configuration (validation safety limits).",
    )


class ACTSupervisedConfig(BaseModel, extra="forbid"):
    """Regime-owned settings for an ACT-supervised Lightning experiment.

    Contains only fields the regime module can validate without knowing
    the concrete experiment.  Component-specific configs
    (adapter, controller, objective, optimizer, runtime) live in
    :class:`ACTSupervisedComponentConfigs`, validated by the
    experiment builder.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning-rate scheduler configuration.",
    )
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices.",
    )
    target_network: TargetNetworkConfig = Field(
        default_factory=TargetNetworkConfig,
        description="Optional EMA-lagged target network config.",
    )
    supervised_only_warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of optimizer steps during which learned halting "
        "is disabled.",
    )


class ACTSupervisedModule(L.LightningModule):
    """Generic LightningModule for ACT-supervised (halting-based) training.

    Accepts an :class:`ACTSupervisedConfig`, an
    :class:`ACTSupervisedComponentConfigs`, and an
    :class:`ACTSupervisedBindings` bundle that specifies concrete
    model, adapter, controller, objective, and optimizer classes.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ACTSupervisedConfig,
        component_configs: ACTSupervisedComponentConfigs,
        bindings: ACTSupervisedBindings,
    ) -> None:
        super().__init__()
        self._bindings = bindings
        self._component_configs = component_configs

        model_settings = bindings.model_settings_cls.from_config(
            config.model_config_path
        )
        self.model = bindings.model_cls(model_settings)
        self.adapter = bindings.adapter_cls(
            self.model, component_configs.adapter
        )
        self.controller = bindings.controller_cls(
            self.adapter, component_configs.controller
        )
        self.objective = bindings.objective_cls(
            component_configs.objective,
            task_binding=bindings.task_binding_cls(),
        )
        self._config = config
        self._train_runner = SingleStepRunner()
        self._eval_runner = RecurrentRunner()

        # Optional task-specific validation scorer.
        self._task_scorer = (
            bindings.task_scorer_factory()
            if bindings.task_scorer_factory is not None
            else None
        )

        # Manual optimization: one backward, explicit opt/scheduler steps.
        self.automatic_optimization = False
        self._train_carry = None
        self._train_source: DemandDrivenReplaySource | None = None
        self._episode_source: ShuffledEpisodeSource | None = None

        # Metrics are cloned for train/val to allow separate logging and state
        # management.
        self.train_metrics = build_train_metrics(bindings.step_routes).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(bindings.episode_routes).clone(
            prefix="val/"
        )
        self._trace_paradigm: str = "act"
        self._extra_trace_fields: tuple = bindings.trace_fields
        self.diagnostic_trace_spec: DiagnosticTraceSpec = DiagnosticTraceSpec(
            enabled=False, max_batches=2, keys=()
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

        # Optional target network for TD bootstrap stabilization.
        self._target_adapter: TargetAdapterModule | None = None
        if config.target_network.enabled:
            self._target_adapter = TargetAdapterModule(self.adapter)

    @property
    def regime_config(self) -> ACTSupervisedConfig:
        """Regime-owned settings (scheduler, batch scale, warmup, target-network policy)."""
        return self._config

    # Backward-compat alias — prefer regime_config for new code.
    @property
    def config(self) -> ACTSupervisedConfig:
        return self._config

    @property
    def component_configs(self) -> ACTSupervisedComponentConfigs:
        """Experiment-selected component configs (adapter, controller, objective, optimizer, runtime)."""
        return self._component_configs

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        total_steps = int(self.trainer.estimated_stepping_batches)
        sup_params = [p for p in self.adapter.parameters() if p.requires_grad]
        opt_sup = self._bindings.optimizer_cls(
            sup_params, self._component_configs.optimizer
        )
        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt_sup, total_steps, self.regime_config.scheduler
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
        # Carry persists across all DataLoader boundaries per legacy HRM
        # contract — do NOT reset. Buffer is managed by FIFO eviction.
        pass

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
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
            seed=self._component_configs.runtime.validation.seed or 42,
        )
        return self._episode_source

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        if self._train_carry is None:
            init_batch = self._ensure_episode_source().take(
                self.regime_config.global_batch_size
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

        is_warmup = (
            self.global_step < self.regime_config.supervised_only_warmup_steps
        )

        target_backbone: TargetAdapterModule | None = (
            self._target_adapter if self._target_adapter is not None else None
        )

        evaluation = score_captured_rollout(
            runner=self._train_runner,
            source=source,
            controller=self.controller,
            carry=self._train_carry,
            objective=self.objective,
            runner_options={
                "allow_halt": not is_warmup,
                "explore": True,
            },
            objective_options={
                "controller": self.controller,
                "td_target": True,
                "use_token_weights": self.component_configs.objective.use_token_weights,
                "target_backbone": target_backbone,
            },
        )
        update_metric_collection_from_evaluated_chunk(
            self.train_metrics,
            evaluation.evaluated,
            self._bindings.step_routes,
        )
        self._train_carry = evaluation.chunk.final_carry.detach()
        self._train_source.update(carry=self._train_carry)

        local_bs = max(
            self.regime_config.global_batch_size
            // max(getattr(self.trainer, "world_size", 1), 1),
            1,
        )
        loss = normalize_loss_for_backward(
            evaluation.evaluated.loss, local_bs=local_bs
        )

        optimizers = self.optimizers()
        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.step()

        if self._target_adapter is not None:
            self._target_adapter.ema_update(
                self.adapter, self.regime_config.target_network.tau
            )

        scheduler = self.lr_schedulers()
        for sch in (scheduler if isinstance(scheduler, list) else [scheduler]):
            sch.step()

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
        trace_request = None
        ds = self.diagnostic_trace_spec
        if ds.enabled and batch_idx < ds.max_batches and ds.keys:
            trace_request = EvaluationTraceRequest(
                trace_spec=build_trace_spec(
                    "act",
                    include_keys=set(ds.keys),
                    extra_fields=self._bindings.hidden_state_fields,
                ),
                trace_meta=dict(self._bindings.build_trace_meta_fn(batch)),
            )
        result = self.execute_evaluation_batch(
            EvaluationCaseBatch(
                batch=batch,
                case_id=f"val-{batch_idx:04d}",
            ),
            trace_request=trace_request,
        )
        update_metric_collection_from_evaluated_chunk(
            self.val_metrics,
            result.evaluated,
            self._bindings.episode_routes,
        )

        inp = self.adapter.prepare_inputs(batch)
        model_out, _ = self.model(inp, state=None)
        self._val_hidden_norms.update(
            model_out.theta_summary.norm(dim=-1).detach()
        )

        if (
            result.trace is not None
            and len(self._diagnostic_traces) < ds.max_batches
        ):
            self._diagnostic_traces.append(result.trace)

        if self._task_scorer is not None:
            self._task_scorer.update_from_evaluation(result)

        return {"trace": result.trace}

    def on_validation_epoch_end(  # -------------------------------------------
        self,
    ) -> None:
        compute_nonempty(self._val_reducer_collection)
        self._val_reducer_collection.reset()

        if self._task_scorer is not None:
            task_metrics = self._task_scorer.compute()
            self.log_dict(
                task_metrics, on_step=False, on_epoch=True, logger=True
            )
            self._task_scorer.reset()

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
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
            objective=self.objective,
            max_rollout_steps=(
                self._component_configs.runtime.validation.max_rollout_steps
                if hasattr(self._component_configs.runtime, "validation")
                and hasattr(
                    self._component_configs.runtime.validation,
                    "max_rollout_steps",
                )
                else None
            ),
            hard_max_rollout_steps=(
                self._component_configs.runtime.validation.hard_max_rollout_steps
                if hasattr(self._component_configs.runtime, "validation")
                and hasattr(
                    self._component_configs.runtime.validation,
                    "hard_max_rollout_steps",
                )
                else None
            ),
            runner_options={"allow_halt": False, "explore": False},
            objective_options={
                "controller": self.controller,
                "td_target": False,
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
    "ACTSupervisedBindings",
    "ACTSupervisedComponentConfigs",
    "ACTSupervisedConfig",
    "ACTSupervisedModule",
]

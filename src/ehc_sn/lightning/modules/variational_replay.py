"""Generic LightningModule for variational/replay training regimes.

Supports multiple model versions via parameterized component classes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import lightning as L
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn.adapters.tem import ArenaTEMAdapterSettings
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.lightning.diagnostics import DiagnosticTraceSpec
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_ANCESTRAL_ALL,
    TEM_ACC_OBS_ANCESTRAL_REVISIT,
    TEM_ACC_OBS_INFERENCE_ALL,
    TEM_ACC_OBS_INFERENCE_REVISIT,
    TEM_ACC_OBS_RETRIEVED_ALL,
    TEM_ACC_OBS_RETRIEVED_REVISIT,
)
from ehc_sn.metrics.reducers import (
    HiddenNormHistogram,
    OccupancyHistogram,
    compute_nonempty,
)
from ehc_sn.metrics.renderers import (
    log_reducer_figure,
    render_hidden_norm_histogram,
    render_occupancy_histogram,
)
from ehc_sn.metrics.rollout import (
    make_observed_step_metric_observer,
    update_metric_collection_from_evaluated_chunk,
)
from ehc_sn.metrics.step_metrics import StepMetrics
from ehc_sn.objectives.tem import TEMObjective, TEMObjectiveConfig
from ehc_sn.rollouts.buffers import FifoBuffer
from ehc_sn.rollouts.partial_reset import PartialResetBatchAssembler
from ehc_sn.rollouts.runtime import RecurrentRunner
from ehc_sn.rollouts.sources import PartialResetSource
from ehc_sn.tasks.arena.runtime import (
    ARENA_REPLAY_REQUIRED_KEYS,
    batch_size_from_arena_batch,
    infer_arena_replay_batch_keys,
)
from ehc_sn.tasks.arena.traces import (
    ArenaEvaluationSourceContext,
    apply_arena_trace_supplements,
    build_arena_trace_supplements,
)
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.optim import Adam, AdamConfig
from ehc_sn.training.rollout import score_rollout_streaming
from ehc_sn.training.schedules import (
    CosineAnnealingLRWithWarmup,
    SchedulerConfig,
)
from ehc_sn.training.tem import RuntimeConfig as TEMRuntimeConfig
from ehc_sn.training.tem import (
    TEMRuntimeState,
    resolve_tem_runtime,
)
from ehc_sn.types import Batch

# Community-standard map-style batch: plain dict returned by DataLoader.
TEM_STATIC_REQUIRED_KEYS = ("topology", "observations", "mask_valid")
TEM_STATIC_OPTIONAL_KEYS = ("regions", "start", "goals", "landmarks")
TEM_NONFINITE_DETAIL_LIMIT = 12

# =============================================================================
# Bindings — what the factory must provide
# =============================================================================


@dataclass(frozen=True, slots=True, eq=False)
class VariationalReplayBindings:
    """Immutable experiment-to-module bindings for variational/replay regimes.

    Stored as a simple namespace — each field is a concrete class, not an
    instance, so the module can instantiate them in its own lifecycle.
    """

    # eq=False: fields include callables and classes; structural equality
    # is semantically meaningless; identity comparison is correct.

    model_cls: type[nn.Module]
    model_settings_cls: type[BaseModel]
    adapter_cls: type[nn.Module]
    adapter_settings_cls: type[BaseModel]
    trace_fields: tuple
    build_trace_meta_fn: Callable
    replay_runtime_factory: Callable[[], object]
    task_binding_factory: Callable[[], object]


# =============================================================================
# Config
# =============================================================================


class VariationalReplayComponentConfigs(BaseModel, extra="forbid"):
    """Concrete component configs for a variational/replay experiment.

    Validated and populated by the experiment builder, consumed by
    the regime module.  Fields carry concrete Pydantic types — never
    bare ``BaseModel`` — so the config class does not need to know
    which experiment family produced them.
    """

    adapter: ArenaTEMAdapterSettings = Field(
        ...,
        description="Adapter settings for the task ↔ model bridge.",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Replay trajectory controller configuration.",
    )
    objective: TEMObjectiveConfig = Field(
        ...,
        description="Objective configuration (loss weights, etc.).",
    )


class VariationalReplayConfig(BaseModel, extra="forbid"):
    """Regime-owned settings for a variational replay Lightning experiment.

    Contains only fields the regime module can validate without knowing
    the concrete experiment.  Component-specific configs
    (adapter, controller, objective) live in
    :class:`VariationalReplayComponentConfigs`, validated by the
    experiment builder.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file.",
    )
    optimizer: AdamConfig = Field(
        default_factory=AdamConfig,
        description="Adam optimizer hyperparameters.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning-rate scheduler configuration.",
    )
    runtime: TEMRuntimeConfig = Field(
        default_factory=TEMRuntimeConfig,
        description="TEM runtime configuration (dynamics schedules, etc.).",
    )


# =============================================================================
# Lightning module
# =============================================================================


class VariationalReplayModule(L.LightningModule):
    """Generic LightningModule for variational/replay training.

    Accepts a :class:`VariationalReplayConfig`, a
    :class:`VariationalReplayComponentConfigs`, and a
    :class:`VariationalReplayBindings` bundle that specifies the concrete
    model, adapter, and related classes. All lifecycle methods
    (``training_step``, ``validation_step``, ``configure_optimizers``) are
    identical regardless of which model version is used.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: VariationalReplayConfig,
        component_configs: VariationalReplayComponentConfigs,
        bindings: VariationalReplayBindings,
        *,
        debug_env_var: str | None = None,
    ) -> None:
        super().__init__()
        self._bindings = bindings
        self._component_configs = component_configs
        self._debug_env_var = debug_env_var

        model_settings = bindings.model_settings_cls.from_config(
            config.model_config_path
        )
        self.model = bindings.model_cls(model_settings)
        self.adapter = bindings.adapter_cls(
            self.model, component_configs.adapter
        )

        self.train_controller: ReplayTrajectoryController | None = None
        self.train_objective: TEMObjective | None = None
        self.eval_controller: ReplayTrajectoryController | None = None
        self.eval_objective: TEMObjective | None = None
        self._config = config
        self._train_runner = RecurrentRunner()
        self._eval_runner = RecurrentRunner()

        # Manual optimization: explicit backward + opt step.
        self.automatic_optimization = False
        self._train_carry = None

        # Trace paradigm for offline trace-spec construction.
        self._trace_paradigm: str = "tem"
        self._extra_trace_fields: tuple = bindings.trace_fields

        # Metrics are cloned for train/val to allow separate logging and state
        # management.
        from ehc_sn.metrics.routes.tem import (
            TEM_EPISODE_ROUTES,
            TEM_PRIMARY_VAL_ROUTE_KEY,
            TEM_STEP_ROUTES,
        )

        self._step_routes = TEM_STEP_ROUTES
        self._episode_routes = TEM_EPISODE_ROUTES
        self._primary_val_metric_key = f"val/{TEM_PRIMARY_VAL_ROUTE_KEY}"

        self.train_metrics = build_train_metrics(self._step_routes).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(self._episode_routes).clone(
            prefix="val/"
        )

        self.diagnostic_trace_spec: DiagnosticTraceSpec = DiagnosticTraceSpec(
            enabled=False,
            max_batches=2,
            keys=(),
        )
        self._diagnostic_traces: list[Any] = []
        self._diag_params_cache: dict[str, object] | None = None

        # Bounded diagnostic reducers (observations + hidden-state norms).
        self._val_occupancy = OccupancyHistogram(
            n_locations=component_configs.adapter.observation_dim,
            max_batches=10,
        )
        self._val_hidden_norms = HiddenNormHistogram(
            bin_edges=torch.linspace(0.0, 50.0, 51), max_batches=10
        )
        self._val_reducer_collection = MetricCollection(
            {
                "occupancy": self._val_occupancy,
                "hidden_norms": self._val_hidden_norms,
            },
            prefix="val_diag/",
        )

        # Buffer + assembler implement partial-reset batching.
        self._train_buffer: FifoBuffer | None = None
        self._train_batch_assembler: PartialResetBatchAssembler | None = None

    @property
    def config(self) -> VariationalReplayConfig:
        """Return the parsed configuration."""
        return self._config

    def setup(  # -------------------------------------------------------------
        self,
        stage: Optional[str] = None,
    ) -> None:
        """Initialize phase-local train and evaluation runtimes."""
        if stage in (None, "fit"):
            self._ensure_train_runtime()
            self._ensure_eval_runtime()
        elif stage in ("validate", "test"):
            self._ensure_eval_runtime()

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """Build the optimizer and learning-rate scheduler."""
        total_steps = int(self.trainer.estimated_stepping_batches)

        sup_params = [p for p in self.adapter.parameters() if p.requires_grad]
        opt_sup = Adam(sup_params, self.config.optimizer)
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
        """Reset partial-reset carry and buffer at start of each epoch."""
        self._train_carry = None
        if self._train_buffer is not None:
            self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        """Reset evaluation metrics and diagnostic reducers."""
        self.val_metrics.reset()
        self._val_reducer_collection.reset()
        self._diagnostic_traces.clear()

    @property
    def diagnostic_traces(self) -> tuple[Any, ...]:
        """Return bounded diagnostic traces from the last validation epoch."""
        return tuple(self._diagnostic_traces)

    def reset_diagnostic_traces(  # -------------------------------------------
        self,
    ) -> None:
        """Clear the internal diagnostic trace buffer."""
        self._diagnostic_traces.clear()

    def _validation_seed(self, batch_idx: int) -> int:
        """Return the explicit evaluation seed for one validation batch."""
        seed = self.config.runtime.validation.seed
        if seed is None:
            raise ValueError(
                "TEM evaluation requires runtime.validation.seed to be set."
            )
        return int(seed) + int(batch_idx)

    def _train_chunk_steps(self) -> int:
        """Return the number of training chunk steps from config."""
        return self.config.runtime.sequence.tbptt_steps

    def _apply_runtime(self, step: int) -> TEMRuntimeState:
        """Resolve runtime state for the given optimizer step."""
        return resolve_tem_runtime(step, self.config.runtime)

    def _require_train_controller(self) -> ReplayTrajectoryController:
        if self.train_controller is None:
            raise RuntimeError("setup() not called before training.")
        return self.train_controller

    def _require_train_objective(self) -> TEMObjective:
        if self.train_objective is None:
            raise RuntimeError("setup() not called before training.")
        return self.train_objective

    def _require_eval_controller(self) -> ReplayTrajectoryController:
        if self.eval_controller is None:
            raise RuntimeError("setup() not called before evaluation.")
        return self.eval_controller

    def _require_eval_objective(self) -> TEMObjective:
        if self.eval_objective is None:
            raise RuntimeError("setup() not called before evaluation.")
        return self.eval_objective

    def _ensure_train_runtime(  # ---------------------------------------------
        self,
    ) -> None:
        """Initialize controller and objective for the training (fit) phase."""
        if (
            self.train_controller is not None
            and self.train_objective is not None
        ):
            return
        runtime = self._bindings.replay_runtime_factory()
        controller = ReplayTrajectoryController(
            self.adapter, self._component_configs.controller, runtime
        )
        task_binding = self._bindings.task_binding_factory()
        objective = TEMObjective(
            self._component_configs.objective,
            task_binding=task_binding,
        )
        buffer = FifoBuffer(
            capacity_rows=4 * 256,
            keys=ARENA_REPLAY_REQUIRED_KEYS,
            pin_memory=True,
        )
        self._train_buffer = buffer
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=buffer, keys=ARENA_REPLAY_REQUIRED_KEYS
        )
        self.train_controller = controller
        self.train_objective = objective

    def _ensure_eval_runtime(  # ----------------------------------------------
        self,
    ) -> None:
        """Initialize controller and objective for evaluation."""
        if self.eval_controller is not None and self.eval_objective is not None:
            return
        eval_runtime = self._bindings.replay_runtime_factory()
        self.eval_controller = ReplayTrajectoryController(
            self.adapter, self._component_configs.controller, eval_runtime
        )
        eval_task_binding = self._bindings.task_binding_factory()
        self.eval_objective = TEMObjective(
            self._component_configs.objective,
            task_binding=eval_task_binding,
        )

    def _ensure_train_batch_assembler(  # -------------------------------------
        self,
        batch: Batch,
    ) -> PartialResetBatchAssembler:
        if self._train_batch_assembler is not None:
            return self._train_batch_assembler
        buffer = FifoBuffer(
            capacity_rows=4 * 256,
            keys=infer_arena_replay_batch_keys(batch),
            pin_memory=True,
        )
        self._train_buffer = buffer
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=buffer, keys=infer_arena_replay_batch_keys(batch)
        )
        return self._train_batch_assembler

    def training_step(  # ------------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        """Run one chunked-TBPTT optimizer update through the recurrent runner."""
        runtime = self._apply_runtime(self.global_step)
        self.log(
            "train/runtime/eta",
            runtime.eta,
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip
        self.log(
            "train/runtime/hebbian_decay",
            runtime.hebbian_decay,
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip
        self.log(
            "train/runtime/p2g_use",
            runtime.p2g_use,
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip
        self.log(
            "train/runtime/p2g_uncertainty_offset",
            runtime.p2g_uncertainty_offset,
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip
        train_controller = self._require_train_controller()
        train_objective = self._require_train_objective()
        batch_assembler = self._ensure_train_batch_assembler(batch)

        if self._train_carry is None:
            self._train_carry = train_controller.initial_state(batch)

        source = PartialResetSource(
            incoming=batch,
            assembler=batch_assembler,
            carry0=self._train_carry,
        )
        objective_options = train_objective.runtime_loss_options(
            self.global_step, p2g_use=runtime.p2g_use
        )
        evaluation = score_rollout_streaming(
            runner=self._train_runner,
            source=source,
            controller=train_controller,
            carry=self._train_carry,
            objective=train_objective,
            max_rollout_steps=self._train_chunk_steps(),
            objective_options=objective_options,
            observed_step_observer=make_observed_step_metric_observer(
                self.train_metrics, self._step_routes
            ),
        )
        next_carry = evaluation.execution.final_carry.detach()
        self._train_carry = next_carry
        loss = evaluation.loss
        loss = loss / self._train_chunk_steps()

        optimizers = self.optimizers()
        optimizer_handles = (
            list(optimizers) if isinstance(optimizers, list) else [optimizers]
        )
        for opt in optimizer_handles:
            opt.zero_grad(set_to_none=True)  # type: ignore

        self.manual_backward(loss)

        for opt in optimizer_handles:
            opt.step()  # type: ignore

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        self._train_carry = evaluation.execution.final_carry.detach()
        final_carry = evaluation.execution.final_carry

        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/halted_fraction",
            final_carry.halted.float().mean(),
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip
        self.log(
            "train/fit_path/max_cursor",
            final_carry.cursor.max().float(),
            on_step=True,
            on_epoch=False,
            logger=True,
        )  # fmt: skip

        return {
            "loss": loss.detach(),
            "signals": evaluation.last_step.outputs.signals,
        }

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        """Run a full TEM rollout and optionally capture traces."""
        trace_request = None
        ds = self.diagnostic_trace_spec
        if ds.enabled and batch_idx < ds.max_batches and ds.keys:
            trace_request = EvaluationTraceRequest(
                trace_spec=build_trace_spec("tem", include_keys=set(ds.keys)),
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
            self.val_metrics, result.evaluated, self._episode_routes
        )

        obs_id = batch.get("observation_id")
        self._val_occupancy.update(
            obs_id.detach().cpu()
            if obs_id is not None
            else torch.empty(0, dtype=torch.long)
        )

        norms = None
        if result.evaluated.steps:
            model_state = result.evaluated.last_step.snapshot.model_state
            if model_state is not None:
                norms = _compute_hidden_state_norms(model_state)
        self._val_hidden_norms.update(
            norms if norms is not None else torch.empty(0)
        )

        if (
            result.trace is not None
            and len(self._diagnostic_traces) < ds.max_batches
        ):
            self._diagnostic_traces.append(result.trace)

        return {"trace": result.trace}

    def on_validation_epoch_end(  # -------------------------------------------
        self,
    ) -> None:
        """Compute, render, and reset bounded diagnostic reducers."""
        summaries = compute_nonempty(self._val_reducer_collection)

        if self.trainer is not None and self.trainer.is_global_zero:
            occ = summaries.get("val_diag/occupancy")
            if occ is not None:
                log_reducer_figure(
                    self.logger,
                    "val_diag/occupancy",
                    render_occupancy_histogram(occ),
                    global_step=self.global_step,
                )

            norm = summaries.get("val_diag/hidden_norms")
            if norm is not None:
                centers, density = norm
                log_reducer_figure(
                    self.logger,
                    "val_diag/hidden_norms",
                    render_hidden_norm_histogram(centers, density),
                    global_step=self.global_step,
                )

        self._val_reducer_collection.reset()

    def _diagnostic_params(self) -> dict[str, object]:
        """Return static diagnostic parameters injected into the rollout carry."""
        if self._diag_params_cache is None:
            model = self.model
            self._diag_params_cache = {
                "lec_alpha_sigmoid": torch.stack(
                    [
                        torch.sigmoid(a).detach().cpu()
                        for a in model.lec.filter.alpha
                    ]
                ),
                "lec_w_f_sigmoid": torch.stack(
                    [w.detach().cpu() for w in model.lec.w_f]
                ),
            }
        return self._diag_params_cache

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay case through the TEM eval path."""
        if trace_request is not None and trace_request.trace_meta is None:
            trace_request = EvaluationTraceRequest(
                trace_spec=trace_request.trace_spec,
                trace_meta=dict(self._bindings.build_trace_meta_fn(case.batch)),
            )

        runtime = self._apply_runtime(self.global_step)
        eval_controller = self._require_eval_controller()
        eval_objective = self._require_eval_objective()
        carry0 = eval_controller.initial_state(
            case.batch,
            static_data=self._diagnostic_params(),
        )

        objective_options = eval_objective.runtime_loss_options(
            self.global_step, p2g_use=runtime.p2g_use
        )
        result = execute_replay_evaluation_batch(
            case=case,
            runner=self._eval_runner,
            controller=eval_controller,
            carry=carry0,
            objective=eval_objective,
            max_rollout_steps=self.config.runtime.validation.max_rollout_steps,
            hard_max_rollout_steps=self.config.runtime.validation.hard_max_rollout_steps,
            runner_options={"allow_halt": True, "explore": False},
            objective_options=objective_options,
            trace_request=trace_request,
        )

        if result.trace is not None and isinstance(
            result.source_context, ArenaEvaluationSourceContext
        ):
            supplements = build_arena_trace_supplements(
                result.source_context, result.trace.length
            )
            apply_arena_trace_supplements(result.trace, supplements)

        return result

    def aggregate_evaluation_case_metrics(  # ---------------------------------
        self,
        *,
        task: str,
        regime_id: str,
        regime_kind: str,
        case_results: Sequence[EvaluationCaseResult],
    ) -> dict[str, float | int]:
        """Aggregate per-case observation accuracy into a regime summary."""
        _ = task, regime_id, regime_kind

        total_correct_all = 0.0
        total_count_all = 0.0
        total_correct_revisit = 0.0
        total_count_revisit = 0.0

        for case in case_results:
            for step in case.evaluated.steps:
                metrics = step.outputs.metrics

                ratio_all = metrics.extras.get(TEM_ACC_OBS_INFERENCE_ALL)
                if ratio_all is not None:
                    total_correct_all += float(ratio_all.numerator_sum.item())
                    total_count_all += float(ratio_all.denominator_sum.item())

                ratio_revisit = metrics.extras.get(
                    TEM_ACC_OBS_INFERENCE_REVISIT
                )
                if ratio_revisit is not None:
                    total_correct_revisit += float(
                        ratio_revisit.numerator_sum.item()
                    )
                    total_count_revisit += float(
                        ratio_revisit.denominator_sum.item()
                    )

        result: dict[str, float | int] = {}
        if total_count_all > 0:
            result["accuracy_all"] = total_correct_all / total_count_all
        if total_count_revisit > 0:
            result["accuracy_revisit"] = (
                total_correct_revisit / total_count_revisit
            )
        return result


# =============================================================================
# Free function helper (used in validation_step)
# =============================================================================


def _compute_hidden_state_norms(  # -------------------------------------------
    model_state: Any,
) -> Tensor:
    """Extract hidden-state L2 norms from a variable-depth model_state."""
    norms_list: list[Tensor] = []
    for f in fields(model_state) if is_dataclass(model_state) else ():
        val = getattr(model_state, f.name, None)
        if isinstance(val, Tensor):
            norms_list.append(val.norm(dim=-1))
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, Tensor):
                    norms_list.append(v.norm(dim=-1))
    if norms_list:
        return torch.stack(norms_list, dim=-1).norm(dim=-1)
    return torch.empty(0)


# =============================================================================
__all__ = [
    "VariationalReplayBindings",
    "VariationalReplayComponentConfigs",
    "VariationalReplayConfig",
    "VariationalReplayModule",
]

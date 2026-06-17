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
import torch.nn.functional as F
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
from ehc_sn.tasks.seqmaze.contracts import SEQMAZE_IGNORE_LABEL_ID
from ehc_sn.traces import build_trace_spec
from ehc_sn.training.distributed import (
    SumOverBatch,
    normalize_loss_for_backward,
)
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.hrm import load_weights_from_checkpoint as _loader
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


# =============================================================================
def _payload_width(batch: Batch) -> int:
    """Return the leading (batch) dimension from an honest payload batch."""
    for v in batch.values():
        if isinstance(v, Tensor):
            return int(v.shape[0])
    raise ValueError(
        "Honest payload batch is empty or contains no tensors. "
        "Cannot determine batch width."
    )


# =============================================================================
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
    param_group_fn: Callable[[nn.Module, nn.Module], list[dict]] | None = None
    """Optional callable ``(model, adapter) -> list[dict]``.

    Returns parameter groups for the optimizer.  Each dict must have
    ``"params"`` (list of ``nn.Parameter``) and may have optional overrides
    such as ``"lr"`` or ``"weight_decay"``.  When ``None``, all adapter
    parameters are placed in a single group.
    """


class ACTSupervisedTrainingConfig(BaseModel, extra="forbid"):
    """Training-only configuration for an ACT-supervised experiment.

    Not required for evaluation — only used to construct optimizers
    and instantiate the training loop.
    """

    optimizer: AdamATan2Config = Field(
        ...,
        description="Base optimizer configuration.  Per-group overrides "
        "(e.g. lr, weight_decay) can be supplied by the experiment "
        "builder via ``ACTSupervisedBindings.param_group_fn``.",
    )
    num_slots: int | None = Field(
        default=None,
        ge=1,
        description="Per-rank carry buffer width (concurrent trajectory slots). "
        "``None`` during evaluation — carry is allocated per batch from the "
        "DataLoader batch dimension.",
    )
    gradient_clip_val: float | None = Field(
        default=None,
        ge=0,
        description="Maximum global gradient norm for clipping.  ``None`` means "
        "no gradient clipping.",
    )


class ACTSupervisedComponentConfigs(BaseModel, extra="forbid"):
    """Concrete component configs for an ACT-supervised experiment.

    Validated and populated by the experiment builder, consumed by
    the regime module.  Fields carry concrete Pydantic types.
    Contains only the configs needed to construct the computational
    graph — optimizers and training-only settings live in
    :class:`ACTSupervisedTrainingConfig`.
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
    simple_supervised: bool = Field(
        default=False,
        description="Bypass the ACT rollout and use a single forward pass with "
        "pure cross-entropy loss.  Diagnostic flag to isolate whether the "
        "rollout mechanics (multi-step, Q-losses, state management) prevent "
        "convergence.",
    )


class ACTSupervisedModule(L.LightningModule):
    """Generic LightningModule for ACT-supervised (halting-based) training.

    Accepts an :class:`ACTSupervisedConfig`, an
    :class:`ACTSupervisedComponentConfigs`, an
    :class:`ACTSupervisedBindings` bundle that specifies concrete
    model, adapter, controller, objective, and optimizer classes,
    and an optional :class:`ACTSupervisedTrainingConfig` (``None``
    during evaluation).
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ACTSupervisedConfig,
        component_configs: ACTSupervisedComponentConfigs,
        bindings: ACTSupervisedBindings,
        training_config: ACTSupervisedTrainingConfig | None = None,
        *,
        execution: HRMRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self._bindings = bindings
        self._component_configs = component_configs
        self._num_slots: int | None = None
        self._gradient_clip_val: float | None = None

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
        self._pending_source_state: dict[str, Any] | None = None

        # Metrics are cloned for train/val to allow separate logging and state
        # management.
        self.train_metrics = build_train_metrics(bindings.step_routes).clone(
            prefix="train/"
        )
        self.val_metrics = build_val_metrics(bindings.episode_routes).clone(
            prefix="val/"
        )
        # Training config is set once at construction time — never mutated
        # post-construction.
        self._training_config = training_config
        self._runtime: HRMRuntimeConfig | None = execution

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

    def load_weights_from_checkpoint(self, path, groups):
        return _loader(self.model, path, groups)

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
        """Experiment-selected component configs (adapter, controller, objective)."""
        return self._component_configs

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]] | list[Optimizer]:
        if self._training_config is None:
            return []
        total_steps = int(self.trainer.estimated_stepping_batches)
        tc = self._training_config

        # Build parameter groups from experiment-provided callable,
        # falling back to a single group over all adapter parameters.
        param_group_fn = self._bindings.param_group_fn
        if param_group_fn is not None:
            raw_groups = param_group_fn(self.model, self.adapter)
        else:
            raw_groups = [{"params": list(self.adapter.parameters())}]

        # Merge base config with per-group overrides.
        base_cfg = tc.optimizer
        param_groups: list[dict[str, Any]] = []
        for g in raw_groups:
            merged: dict[str, Any] = {"params": g["params"]}
            for key in ("lr", "weight_decay", "betas"):
                merged[key] = g.get(key, getattr(base_cfg, key))
            param_groups.append(merged)

        opt = self._bindings.optimizer_cls(param_groups, base_cfg)
        schedulers: list[dict[str, Any]] = [
            {
                "scheduler": CosineAnnealingLRWithWarmup(
                    opt, total_steps, self.regime_config.scheduler
                ),
                "interval": "step",
                "frequency": 1,
                "name": "optim/main",
            }
        ]
        return [opt], schedulers

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the training source and carry for source-driven training.

        Called by Lightning after the datamodule's ``setup()``. Creates the
        episode source, initial carry, and demand-driven replay source so
        ``training_step`` receives real episode batches from the DataLoader.
        """
        if stage == "fit" and self._training_config is not None:
            tc = self._training_config
            if tc.num_slots is None:
                raise RuntimeError(
                    "num_slots is required for training. "
                    "Set it via ACTSupervisedTrainingConfig.num_slots."
                )
            self._num_slots = tc.num_slots
            self._gradient_clip_val = tc.gradient_clip_val
            local_bs = tc.num_slots // max(
                getattr(self.trainer, "world_size", 1), 1
            )
            init_batch = self._ensure_episode_source().take(local_bs)
            init_batch = _move_batch_to(init_batch, self.device)
            self._train_carry = self.controller.initial_state(init_batch)
            # Move carry to target device so the replay source's template,
            # carry, and index tensors are all on the same device.
            self._train_carry = _move_batch_to(self._train_carry, self.device)

            self._train_source = DemandDrivenReplaySource(
                episode_source=self._ensure_episode_source(),
                carry0=self._train_carry,
                device=self.device,
            )

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
        seed = (
            self._runtime.validation.seed or 42
            if self._runtime is not None
            else 42
        )
        self._episode_source = ShuffledEpisodeSource(
            train_dataset,
            rank=rank,
            world_size=world_size,
            seed=seed,
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

    # ── Checkpoint hooks ────────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist episode source state into the Lightning checkpoint."""
        if self._episode_source is not None:
            checkpoint["episode_source"] = self._episode_source.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore episode source state from checkpoint (deferred)."""
        source_state = checkpoint.get("episode_source")
        if source_state is not None:
            self._pending_source_state = source_state

    def _training_step_simple(  # ------------------------------------------------
        self,
        batch: Batch,
    ) -> dict[str, object]:
        """Single forward pass with pure cross-entropy, no rollout.

        Bypasses the ACT controller's multi-step mechanics and Q-losses.
        Trains only the encoder and decoder (backbone receives no gradient
        from this path).
        """
        B = _payload_width(batch)
        state = self.adapter.init_state(B)
        out, _ = self.adapter(batch, state)
        logits = out.task.path_logits  # (B, T, V)

        # Infer vocabulary size from logits
        V = logits.shape[-1]

        # Extract labels and mask non-supervised positions
        labels = batch["target_path"].to(dtype=torch.long)
        mask = batch["path_mask"].to(dtype=torch.bool)
        labels = torch.where(
            mask,
            labels,
            torch.full_like(labels, SEQMAZE_IGNORE_LABEL_ID),
        )

        loss = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
        )

        # --- Auxiliary edge-prediction loss (multi-task training) ------------
        # Provides the dense (N² pairs per sample) signal the HRM needs to
        # learn disentanglement from composited slot representations.
        edge_logits = getattr(out.task, "edge_logits", None)
        if edge_logits is not None:
            B, N, _, _ = edge_logits.shape
            edge_labels = batch["edge_label"].to(
                dtype=torch.long, device=edge_logits.device
            )
            edge_mask = batch["edge_mask"].to(
                dtype=torch.bool, device=edge_logits.device
            )
            edge_loss = F.cross_entropy(
                edge_logits[edge_mask],
                edge_labels[edge_mask],
            )
            # Scale so edge loss (N² pairs) doesn't dominate path loss (T tokens).
            # Both at random: path ≈ T * log(V), edge ≈ N² * log(2).
            loss = loss + (1.0 / N) * edge_loss

        optimizers = self.optimizers()
        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        if self._gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self._gradient_clip_val
            )

        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.step()

        self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True)
        self.log(
            "train/loss/token", loss, on_step=True, on_epoch=False, logger=True
        )

        # Compute token accuracy for logging
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == batch["target_path"]) & mask
            acc = correct.sum().float() / mask.sum().float()
            self.log(
                "train/acc", acc, on_step=True, on_epoch=False, logger=True
            )

        if self.global_step % 10 == 0:
            pre_clip_norm = 0.0
            for p in self.adapter.parameters():
                if p.grad is not None:
                    pre_clip_norm += p.grad.norm().item() ** 2
            pre_clip_norm = pre_clip_norm**0.5
            self.log(
                "train/grad_norm",
                pre_clip_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

        return {"loss": loss}

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict[str, object]:
        # --- Simple supervised bypass: single forward pass, pure CE ---------
        if self.regime_config.simple_supervised:
            return self._training_step_simple(batch)

        # --- Normal ACT rollout training ------------------------------------
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

        is_warmup = (
            self.global_step < self.regime_config.supervised_only_warmup_steps
        )

        target_backbone: TargetAdapterModule | None = (
            self._target_adapter if self._target_adapter is not None else None
        )

        evaluation = score_captured_rollout(
            runner=self._train_runner,
            source=self._train_source,
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

        carry_width = max(
            (self._num_slots or 0)
            // max(getattr(self.trainer, "world_size", 1), 1),
            1,
        )
        loss = normalize_loss_for_backward(
            SumOverBatch(evaluation.evaluated.loss), local_bs=carry_width
        )

        optimizers = self.optimizers()
        for opt in (
            optimizers if isinstance(optimizers, list) else [optimizers]
        ):
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        # Gradient clipping: prevents PFC backbone updates from destroying
        # encoder representations (ratio ~3600 vs ~1500 update-to-param).
        # Log the pre-clip norm for diagnostic, then clamp.
        if self._gradient_clip_val is not None:
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self._gradient_clip_val
            )
            if self.global_step % 10 == 0:
                self.log(
                    "train/grad_norm_clipped",
                    pre_clip_norm,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )
        elif self.global_step % 10 == 0:
            # Pre-clip total norm for diagnostic when clipping is off
            pre_clip_norm = 0.0
            for p in self.adapter.parameters():
                if p.grad is not None:
                    pre_clip_norm += p.grad.norm().item() ** 2
            pre_clip_norm = pre_clip_norm**0.5
            self.log(
                "train/grad_norm_clipped",
                pre_clip_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

        # Log per-component gradient norms for diagnostic
        if self.global_step % 10 == 0:
            # Total gradient norm (backward-compat)
            total_norm = 0.0
            for p in self.adapter.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm**0.5
            self.log(
                "train/grad_norm",
                total_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

            # Per-component: model backbone (PFC) vs adapter (encoder+decoder)
            model_param_ids = {id(p) for p in self.model.parameters()}
            for group_name, group_params in [
                ("pfc", self.model.parameters()),
                (
                    "adapter",
                    [
                        p
                        for p in self.adapter.parameters()
                        if id(p) not in model_param_ids
                    ],
                ),
            ]:
                grad_norm_sq = 0.0
                param_norm_sq = 0.0
                has_grad = False
                for p in group_params:
                    param_norm_sq += p.norm().item() ** 2
                    if p.grad is not None:
                        grad_norm_sq += p.grad.norm().item() ** 2
                        has_grad = True
                grad_norm = grad_norm_sq**0.5
                param_norm = param_norm_sq**0.5
                if has_grad:
                    self.log(
                        f"train/grad/{group_name}_norm",
                        grad_norm,
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                    )
                self.log(
                    f"train/param/{group_name}_norm",
                    param_norm,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )
                if has_grad and param_norm > 0:
                    self.log(
                        f"train/ratio/{group_name}_update_to_param",
                        grad_norm / (param_norm + 1e-8),
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                    )

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
            self._task_scorer.update_from_evaluation(result, batch)

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
                self._runtime.validation.max_rollout_steps
                if self._runtime is not None
                and hasattr(
                    self._runtime.validation,
                    "max_rollout_steps",
                )
                else None
            ),
            hard_max_rollout_steps=(
                self._runtime.validation.hard_max_rollout_steps
                if self._runtime is not None
                and hasattr(
                    self._runtime.validation,
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

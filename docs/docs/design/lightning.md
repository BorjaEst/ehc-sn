---
title: Lightning Adapter Design
description: Narrow execution adapter for framework-independent training runtimes
---

# Lightning Adapter Design (`ehp_sn.lightning`)

> A narrow execution adapter around framework-independent training runtimes.

The Lightning package is the **framework integration layer**, not the owner of
models, objectives, metrics, data semantics, rollout logic, or experiment
definitions. Its job is to translate the framework-independent EHP system into
the lifecycle expected by `lightning.pytorch.Trainer`:

```
EHP domain objects
    model, objective, task runtime, metrics, data module, configuration
        │
        ▼
ehp_sn.lightning
    LightningModule adapters
    LightningDataModule adapters
    callbacks, Trainer construction, checkpoint integration
        │
        ▼
lightning.pytorch.Trainer
```

---

## 1. Core principle: adapt, do not define

```
ehp_sn.lightning does not define how TEM, HRM, or EHP training works.

It defines how an already-defined training runtime is executed by
lightning.pytorch.Trainer.
```

A poor design implements the scientific system _inside_ `LightningModule`. A
professional design keeps the scientific system usable without Lightning and
makes `ehp_sn.lightning` replaceable.

### Appropriate responsibilities

- Adapt framework-independent runtimes to Lightning hooks.
- Convert project optimization specifications into Lightning's optimizer
  configuration format.
- Construct `Trainer`, callbacks, loggers, strategies, and checkpointing.
- Adapt project data providers to the `LightningDataModule` lifecycle where
  needed.
- Translate Lightning lifecycle events into project runtime lifecycle events.

### What it must not own

| Concern                                 | Owner                |
| --------------------------------------- | -------------------- |
| Neural network architectures            | `ehp_sn.models`      |
| Loss mathematics                        | `ehp_sn.objectives`  |
| Metric definitions and route selection  | `ehp_sn.metrics`     |
| Rollout algorithms                      | `ehp_sn.rollouts`    |
| Recurrent carry semantics               | `ehp_sn.rollouts`    |
| Dataset formats and source construction | `ehp_sn.data`        |
| Task contracts                          | `ehp_sn.tasks`       |
| Evaluation aggregation                  | `ehp_sn.evaluation`  |
| Trace schemas                           | `ehp_sn.traces`      |
| Experiment resolution                   | `ehp_sn.experiments` |
| Model construction policy               | `ehp_sn.experiments` |
| Figure generation                       | `ehp_sn.figures`     |

### Invariant: dependency direction

```
ehp_sn.lightning
    ↓
ehp_sn.training, ehp_sn.data, ehp_sn.metrics, ehp_sn.models,
ehp_sn.objectives, ehp_sn.evaluation, ehp_sn.traces

No domain package imports ehp_sn.lightning.
```

---

## 2. Package structure

```
src/ehp_sn/lightning/
├── __init__.py                     # Narrow public API — factories only
│
├── modules.py                      # TaskLightningModule, ManualOptimizationModule
├── datamodule.py                   # EHPDataModule (thin wrapper)
├── trainer.py                      # build_trainer(), build_callbacks()
├── optimization.py                 # build_lightning_optimization()
├── checkpointing.py                # Checkpoint schema metadata, naming policy
│
├── callbacks/
│   ├── __init__.py
│   ├── diagnostics.py              # Bounded scalar signal logging
│   ├── evaluation.py               # Scheduled evaluation invocation
│   ├── figures.py                  # Trace → figure rendering
│   └── progress.py                 # Step-keyed progress bar
│
└── logging.py                      # MLflow logger construction
```

Split `checkpointing.py` into a subpackage or extract `logging.py` into a
subpackage only when there are multiple independent implementations.
Premature directory subdivision increases navigation and public/private
boundary maintenance costs.

### Module size guideline

Module size is a diagnostic signal, not a conformance requirement. A
`LightningModule` that exceeds several hundred lines _usually_ signals that
domain logic has leaked into the framework adapter. But the real criteria
are number of responsibilities, domain-specific imports, independent
testability, lifecycle coupling, and state ownership — not line count.

---

## 3. Upstream contracts: the framework-independent runtime

Before defining the Lightning package, define the contracts it adapts.
These live in `ehp_sn.training`:

```
ehp_sn.training/
├── contracts.py                    # Typed step results
├── runtime.py                      # TrainingRuntime protocol
├── optimization.py                 # OptimizationSpec, OptimizerSpec, SchedulerSpec
├── trainer.py                      # Framework-independent execution spec (future)
├── runtimes/
│   ├── variational_replay.py       # VariationalReplayRuntime
│   ├── act_supervised.py           # ACTSupervisedRuntime
│   └── actor_critic.py            # ActorCriticRuntime
└── builders.py                     # build_training_system()
```

### 3.1 TrainingRuntime protocol — runtime owns all mutable state

There are two possible state-ownership models: (a) the runtime owns its
state internally, or (b) all state is external and passed in. Mixing both
produces lifecycle and checkpoint bugs. For EHP, the runtime **owns all
semantic runtime state internally** (carry, source cursor, schedule state,
controller runtime state, metric accumulators, stage isolation). The
Lightning module sees only the typed step result.

```python
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from torch import Tensor


class RuntimeStage(StrEnum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class TrainingRuntime(Protocol):
    """Framework-independent execution contract for one training regime.

    The runtime owns all mutable execution state: carry, source cursor,
    schedule state, controller runtime state, and metric accumulators.
    It exposes that state only through ``state_dict()`` /
    ``load_state_dict()`` for checkpointing.
    """

    @property
    def model(self) -> "nn.Module":
        """The plain ``nn.Module`` for inference, checkpoint, and evaluation.

        This is the canonical model access point.  Callers must not assume
        that the runtime exposes a generic ``forward()`` method; the model
        is the inference surface.
        """

    def parameter_groups(self) -> Mapping[str, Iterable["nn.Parameter"]]:
        """Return named parameter groups for optimizer construction.

        Each key is a stable name (``"all"``, ``"model"``, ``"controller"``,
        ``"critic"``) that the optimizer spec can target.  The runtime owns
        the mapping from name to parameters — the Lightning optimizer
        builder must not inspect arbitrary submodule paths.
        """

    # ── Process lifecycle ────────────────────────────────────────────────

    def initialize_process(self, context: "ProcessContext") -> None:
        """Process-local initialisation (data download, pre-processing).

        Called once per process before any device placement.
        """

    def start_stage(self, stage: RuntimeStage, context: "DeviceContext") -> None:
        """Device-dependent initialisation for a training stage.

        Called after Lightning has moved the model to the target device.
        The runtime constructs data sources, initialises carry, and
        allocates device-side buffers here.
        """

    def end_stage(self, stage: RuntimeStage) -> None:
        """Clean up stage-local resources when a training stage ends."""

    # ── Epoch lifecycle (optional — add only when needed) ────────────────

    def start_epoch(self, context: "StepContext") -> None:
        """Called at the start of each training or validation epoch."""

    def end_epoch(self, context: "StepContext") -> None:
        """Called at the end of each training or validation epoch.

        At validation epoch end the runtime may compute and expose
        accumulated metrics.
        """

    # ── Step execution ───────────────────────────────────────────────────

    def train_step(
        self,
        batch: object,
        context: "StepContext",
    ) -> "TrainingStepResult":
        """Execute one framework-independent training step."""

    def validation_step(
        self,
        batch: object,
        context: "StepContext",
    ) -> "EvaluationStepResult":
        """Execute one framework-independent validation step."""

    def test_step(
        self,
        batch: object,
        context: "StepContext",
    ) -> "EvaluationStepResult":
        """Execute one framework-independent test step."""

    # ── Metric access ────────────────────────────────────────────────────

    def compute_train_metrics(self) -> Mapping[str, float]:
        """Return computed (accumulated and reduced) training metrics."""

    def compute_validation_metrics(self) -> Mapping[str, float]:
        """Return computed validation metrics, including the primary
        monitoring key for checkpoint selection."""

    def reset_train_metrics(self) -> None:
        """Reset training metric accumulators after logging."""

    def reset_validation_metrics(self) -> None:
        """Reset validation metric accumulators before a new epoch."""

    @property
    def primary_validation_metric_key(self) -> str | None:
        """Canonical monitoring key for checkpoint selection."""

    # ── Checkpointing ────────────────────────────────────────────────────

    def state_dict(self) -> Mapping[str, Any]:
        """Return resumable runtime state (carry, source cursor, schedules)."""

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore resumable runtime state from a checkpoint."""
```

The runtime owns:

- how the model is called;
- how controllers are initialised;
- how rollouts are executed;
- how carry is propagated and detached;
- how objectives are evaluated;
- how task metrics are selected, updated, accumulated, and reduced;
- how runtime schedules affect the model;
- what diagnostic signals are produced.

### 3.2 Context types

```python
@dataclass(frozen=True, slots=True)
class ProcessContext:
    global_rank: int
    world_size: int


@dataclass(frozen=True, slots=True)
class DeviceContext:
    device: torch.device
    global_rank: int
    world_size: int


@dataclass(frozen=True, slots=True)
class StepContext:
    global_step: int
    batch_idx: int
    epoch: int = 0
```

### 3.3 Typed step results

The Lightning module must not receive arbitrary dictionaries from the
runtime. Results carry observations, not accumulated metrics.

Important: the `metrics` field in step results carries **per-step
observations** (e.g. `"correct": 3, "total": 8` for one batch), not
accumulated values. The runtime owns accumulation and exposes computed
values through `compute_train_metrics()` / `compute_validation_metrics()`.
Lightning must not independently average already-aggregated episode-level
ratios.

```python
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from torch import Tensor


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    loss: Tensor
    observations: Mapping[str, Tensor] = field(default_factory=dict)
    """Per-step scalar observations for logging (e.g. grad norm).

    These are NOT accumulated metrics.  The runtime owns accumulation."""
    batch_size: int
    signals: Mapping[str, Any] = field(default_factory=dict)
    """Diagnostic signals (carry fraction, halted fraction, etc.).

    Guaranteed detached; safe for callback consumption."""


@dataclass(frozen=True, slots=True)
class EvaluationStepResult:
    observations: Mapping[str, Tensor] = field(default_factory=dict)
    batch_size: int
    outputs: object | None = None
    signals: Mapping[str, Any] = field(default_factory=dict)
```

For regimes that need manual optimisation, the runtime returns a plan:

```python
@dataclass(frozen=True, slots=True)
class OptimizationAction:
    """One action in a multi-optimizer training step.

    All fields reference *named* entities that the Lightning module
    resolves; no Lightning optimizer objects cross the boundary.
    """
    optimizer_key: str
    loss_key: str
    zero_grad_before: bool = True
    backward: bool = True
    step_after: bool = True
    retain_graph: bool = False


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    actions: tuple[OptimizationAction, ...]
    losses: Mapping[str, Tensor]
    """Named losses produced by the step (e.g. ``"supervised"``, ``"rl"``)."""


@dataclass(frozen=True, slots=True)
class ManualTrainingStepResult(TrainingStepResult):
    optimization_plan: OptimizationPlan | None = None
```

### 3.4 OptimisationSpec

```python
@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    name: str
    optimizer_factory: str
    """Registered optimizer factory name (e.g. ``"adam.atan2"``).

    Resolved by the experiment builder against a registry, not an
    unrestricted string."""
    learning_rate: float
    weight_decay: float = 0.0
    parameter_group: str = "all"
    """Must match a key returned by ``runtime.parameter_groups()``."""
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SchedulerSpec:
    algorithm: str
    interval: Literal["step", "epoch"]
    frequency: int = 1
    monitor: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    optimizers: tuple[OptimizerSpec, ...]
    schedulers: tuple[SchedulerSpec, ...] = ()
```

---

## 4. `modules.py` — LightningModule adapters

### 4.1 TaskLightningModule (automatic optimisation)

Use for ordinary supervised learning, single-optimizer regimes, and
standard backward/step scheduling (TEM v1/v2, HRM v1 ACT).

```python
class TaskLightningModule(L.LightningModule):
    """Lightning adapter for a framework-independent training runtime.

    Thin: owns lifecycle translation, logging, and checkpoint bridging.
    The runtime owns training semantics and all mutable execution state.
    """

    def __init__(
        self,
        *,
        runtime: TrainingRuntime,
        optimization: OptimizationSpec,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.optimization = optimization

    @property
    def model(self) -> nn.Module:
        """Expose the plain model for inference and checkpoint access."""
        return self.runtime.model

    # ── Lifecycle ────────────────────────────────────────────────────────

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.runtime.start_stage(
                RuntimeStage.FIT,
                DeviceContext(
                    device=self.device,
                    global_rank=self.global_rank,
                    world_size=self.trainer.world_size,
                ),
            )
        elif stage in ("validate", "test"):
            stage_enum = (
                RuntimeStage.VALIDATE if stage == "validate"
                else RuntimeStage.TEST
            )
            self.runtime.start_stage(
                stage_enum,
                DeviceContext(
                    device=self.device,
                    global_rank=self.global_rank,
                    world_size=self.trainer.world_size,
                ),
            )

    def on_train_epoch_start(self) -> None:
        self.runtime.start_epoch(
            StepContext(
                global_step=self.global_step,
                batch_idx=0,
                epoch=self.current_epoch,
            )
        )

    def on_validation_epoch_start(self) -> None:
        self.runtime.reset_validation_metrics()
        self.runtime.start_epoch(
            StepContext(
                global_step=self.global_step,
                batch_idx=0,
                epoch=self.current_epoch,
            )
        )

    # ── Training ─────────────────────────────────────────────────────────

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        result = self.runtime.train_step(
            batch=batch,
            context=self._step_context(batch_idx),
        )

        self._log_observations("train", result.observations, result.batch_size)

        self.log(
            "train/loss",
            result.loss,
            batch_size=result.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return result.loss

    # ── Validation / Test ────────────────────────────────────────────────

    def validation_step(self, batch: object, batch_idx: int) -> None:
        result = self.runtime.validation_step(
            batch=batch,
            context=self._step_context(batch_idx),
        )
        self._log_observations("val", result.observations, result.batch_size)

    def test_step(self, batch: object, batch_idx: int) -> None:
        result = self.runtime.test_step(
            batch=batch,
            context=self._step_context(batch_idx),
        )
        self._log_observations("test", result.observations, result.batch_size)

    def on_validation_epoch_end(self) -> None:
        computed = self.runtime.compute_validation_metrics()
        self.log_dict(
            {f"val/{k}": v for k, v in computed.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        primary = self.runtime.primary_validation_metric_key
        if primary is not None and primary in computed:
            self.log(
                "val/primary",
                computed[primary],
                prog_bar=True,
                sync_dist=True,
            )

    # ── Optimisation ─────────────────────────────────────────────────────

    def configure_optimizers(self):
        return build_lightning_optimization(
            runtime=self.runtime,
            spec=self.optimization,
        )

    # ── Checkpointing ────────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ehp"] = {
            "schema_version": 1,
            "runtime": self.runtime.state_dict(),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ehp_state = checkpoint.get("ehp", {})
        self.runtime.load_state_dict(ehp_state.get("runtime", {}))
```

### 4.2 ManualOptimizationLightningModule

Use only where optimiser sequencing is part of the algorithm (HRM v2
actor-critic).

The runtime returns an `OptimizationPlan` — a sequence of named actions
that the Lightning module executes in order.

```python
class ManualOptimizationLightningModule(TaskLightningModule):
    """Lightning adapter for runtimes that require manual optimisation.

    The runtime returns an ``OptimizationPlan`` with named losses and
    a sequence of ``OptimizationAction`` descriptors.  The module owns
    only the execution: zero-grad, backward, step.
    """

    automatic_optimization = False

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        result = self.runtime.train_step(
            batch=batch,
            context=self._step_context(batch_idx),
        )

        # Execute the optimisation plan — the runtime owns sequencing
        # semantics; the module owns the Lightning primitives.
        plan = result.optimization_plan
        if plan is not None:
            optimizers = {
                key: opt
                for key, opt in zip(
                    self._optimizer_keys(), self.optimizers()
                )
            }
            for action in plan.actions:
                opt = optimizers[action.optimizer_key]
                loss = plan.losses[action.loss_key]

                if action.zero_grad_before:
                    opt.zero_grad(set_to_none=True)
                if action.backward:
                    self.manual_backward(
                        loss, retain_graph=action.retain_graph
                    )
                if action.step_after:
                    opt.step()

        self._log_observations(
            "train", result.observations, result.batch_size
        )
        self.log(
            "train/loss",
            result.loss,
            batch_size=result.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return result.loss.detach()

    def _optimizer_keys(self) -> tuple[str, ...]:
        """Return the ordered optimizer keys declared by the spec.

        Must match the order returned by ``configure_optimizers()`` and
        correspond to keys referenced by ``OptimizationAction.optimizer_key``.
        """
        return tuple(item.name for item in self.optimization.optimizers)
```

### 4.3 Deciding which adapter to use

| Regime       | Optimisers | Optimisation type | Adapter                             |
| ------------ | ---------- | ----------------- | ----------------------------------- |
| TEM v1/v2    | 1          | Standard          | `TaskLightningModule`               |
| HRM v1 (ACT) | 1          | Standard          | `TaskLightningModule`               |
| HRM v2 (AC)  | 3          | Per-optimiser     | `ManualOptimizationLightningModule` |

Do not create regime-specific subclasses of `LightningModule`. The two
adapters above — one for automatic, one for manual optimisation — cover
all three regimes.

---

## 5. `optimization.py` — translating specs to Lightning

```python
def build_lightning_optimization(
    *,
    runtime: TrainingRuntime,
    spec: OptimizationSpec,
) -> tuple[list[Optimizer], list[dict[str, Any]]]:
    """Translate a project OptimizationSpec into Lightning's return format.

    Parameter groups are resolved from ``runtime.parameter_groups()``.
    The Lightning layer does not inspect arbitrary submodule paths.
    """
    all_groups = runtime.parameter_groups()

    optimizers: list[Optimizer] = []
    for item in spec.optimizers:
        if item.parameter_group not in all_groups:
            raise KeyError(
                f"Optimizer '{item.name}' targets unknown "
                f"parameter group '{item.parameter_group}'. "
                f"Available: {sorted(all_groups)}"
            )
        params = list(all_groups[item.parameter_group])
        if not any(p.requires_grad for p in params):
            raise ValueError(
                f"Parameter group '{item.parameter_group}' has no "
                f"parameters with requires_grad=True."
            )
        opt = _build_optimizer(item, params)
        optimizers.append(opt)

    schedulers = [
        _build_scheduler(item, optimizers)
        for item in spec.schedulers
    ]

    return optimizers, schedulers
```

The Lightning layer must not decide which algorithm or hyperparameters to
use. That belongs to the resolved training definition. The
`optimizer_factory` field is resolved by the experiment builder against an
owner registry — it is not an unrestricted string.

### 5.1 Parameter group contract

The runtime owns the mapping from named groups to parameters:

```python
class VariationalReplayRuntime:
    def parameter_groups(self):
        return {
            "all": self.adapter.parameters(),
        }

class ActorCriticRuntime:
    def parameter_groups(self):
        vmPFC_ids = {id(p) for p in self.model.pfc.estimator.parameters()}
        str_ids = {id(p) for p in self.model.str.parameters()}
        excluded = vmPFC_ids | str_ids
        return {
            "supervised": [p for p in self.adapter.parameters()
                           if id(p) not in excluded],
            "rl": list(self.model.str.parameters()),
            "qv": list(self.model.pfc.estimator.parameters()),
        }
```

The Lightning optimizer builder must not inspect `runtime.model.pfc` or
arbitrary submodule paths.

---

## 6. `datamodule.py` — adapting data providers

### 6.1 DataProvider protocol

```python
class DataProvider(Protocol):
    """Framework-independent data access contract.

    The canonical data system lives in ``ehp_sn.data``.  This protocol
    allows the Lightning adapter to wrap any compliant provider without
    importing dataset internals.
    """

    def prepare(self) -> None: ...
    def setup(self, stage: str | None) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader | Sequence[DataLoader]: ...
    def test_dataloader(self) -> DataLoader | Sequence[DataLoader]: ...
    def predict_dataloader(self) -> DataLoader | Sequence[DataLoader]: ...
```

### 6.2 EHPDataModule

```python
class EHPDataModule(L.LightningDataModule):
    """Thin Lightning lifecycle wrapper around a project DataProvider."""

    def __init__(self, provider: DataProvider) -> None:
        super().__init__()
        self.provider = provider

    def prepare_data(self) -> None:
        self.provider.prepare()

    def setup(self, stage: str | None = None) -> None:
        self.provider.setup(stage)

    def train_dataloader(self):
        return self.provider.train_dataloader()

    def val_dataloader(self):
        return self.provider.val_dataloader()

    def test_dataloader(self):
        return self.provider.test_dataloader()

    def predict_dataloader(self):
        return self.provider.predict_dataloader()
```

### 6.3 Prohibited access pattern

The `LightningModule` must never do this:

```python
dataset = self.trainer.datamodule._train   # ❌ reaches into private field
```

Instead, the runtime receives a **source factory** during construction:

```python
runtime = VariationalReplayRuntime(
    source_factory=data_system.training_source_factory,
    ...
)
```

Then during `setup()`:

```python
def setup(self, context: RuntimeContext) -> None:
    self._source = self.source_factory.build(
        rank=context.global_rank,
        world_size=context.world_size,
    )
```

---

## 7. Episode source ownership

```
Responsibility            Owner
─────────────────────     ─────────────────────────────
EpisodeSource             ehp_sn.data
ShuffledEpisodeSource     ehp_sn.data
DemandDrivenReplaySource  ehp_sn.rollouts.sources
Source factories          ehp_sn.data or ehp_sn.experiments
Source consumption        ehp_sn.training.runtimes
Source checkpointing      ehp_sn.training.runtimes
Rank/world-size context   ehp_sn.lightning (via RuntimeContext)
```

The Lightning module is a conduit for device, rank, and world-size
information. It must not construct data sources itself.

---

## 8. Metric ownership (three levels)

For masked sequential metrics (e.g. valid-step-count denominators, episode
completion ratios), naïvely calling `self.log(..., on_epoch=True)` may
produce incorrect aggregation because Lightning averages per-batch means.
The runtime must own accumulation and reduction.

```
Layer              Owns                                            Example
─────              ────                                            ───────
ehp_sn.metrics     Metric definitions, keys,                       TEM_ACC_OBS_POST_ALL
                   numerator/denominator state,
                   route definitions, aggregation maths,
                   distributed-safe accumulation contracts
                   (torchmetrics.Metric subclasses or equivalent)

Training runtime   Metric route selection,                         TEM_STEP_ROUTES applied
                   output-to-route wiring,                          to TEM step outputs
                   step/episode metric update timing,
                   accumulation across batches,
                   compute() → reduced scalars,
                   primary metric designation

Lightning adapter  Prefix assignment ("train/", "val/"),           self.log_dict(...)
                   self.log / self.log_dict,                        on_step / on_epoch
                   sync_dist=True for distributed sync,
                   exposure of the primary key to callbacks
```

The Lightning module must not import metric keys like `TEM_ACC_OBS_POST_ALL`
or route tuples like `TEM_STEP_ROUTES`. Those belong to the runtime. The
Lightning module receives **computed** scalars from
`compute_validation_metrics()` — it never accumulates raw numerators.

---

## 9. Evaluation ownership

```
Layer                  Owns
─────                  ────
ehp_sn.evaluation      Evaluation specifications, protocols,
                       case definitions, metric aggregation,
                       result contracts, primary metric resolution,
                       result persistence

ehp_sn.lightning       Scheduling (when), invocation (that),
.callbacks.evaluation  scalar result forwarding to Lightning logging
```

The callback calls a framework-independent evaluator and logs only scalar
summaries:

```python
class EvaluationCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, ...):
        for regime in self._due_regimes(trainer):
            result = self.evaluator.evaluate(
                model=pl_module.runtime.model,
                cases=regime.cases,
            )
            pl_module.log_dict(result.scalar_metrics)
            self._persist_artifacts(result)
```

It must not aggregate TEM-specific numerator keys or know about
`TEM_ACC_OBS_POST_REVISIT`.

### 9.1 Execution constraints

Evaluation callbacks that fire during training must enforce:

- **Inference mode**: `torch.inference_mode()` or `torch.no_grad()` active
  during the entire evaluation span.
- **Model mode restored**: The model returns to `train()` mode after
  evaluation completes, regardless of exceptions.
- **Mixed precision**: Evaluation uses the same precision policy as
  validation, not training (no gradient scaling).
- **Rank isolation**: Whether evaluation runs on all ranks or only rank
  zero must be explicit. If metrics require collective reduction, all
  ranks must participate.
- **Carry protection**: The training carry must be saved before evaluation
  and restored afterwards. Evaluation must not mutate model memory that
  training depends on.
- **Dataloader isolation**: Evaluation uses its own dataloaders, not
  the training dataloader.
- **Checkpoint visibility**: An evaluation result logged in step N should
  be visible to a checkpoint callback that fires in the same hook cycle.

---

## 10. Callback structure

### 10.1 Responsibility map

| Callback                   | Hook(s)                   | Responsibility                        |
| -------------------------- | ------------------------- | ------------------------------------- |
| `StepProgressBar`          | `init_train_tqdm`,        | Step-keyed progress display           |
|                            | `on_train_batch_end`      |                                       |
| `DiagnosticsCallback`      | `on_*_batch_end`          | Filtered scalar signal logging        |
| `EvaluationCallback`       | `on_train_batch_end`,     | Scheduled evaluation regime execution |
|                            | `on_validation_epoch_end` |                                       |
| `FigureGenerationCallback` | `setup`,                  | Bounded trace → figure rendering      |
|                            | `on_validation_epoch_end` |                                       |
| `RuntimeLifecycleCallback` | `on_*_epoch_start`,       | Epoch/stage transitions → runtime     |
|                            | `on_*_epoch_end`          |                                       |
| `CheckpointCallback`       | (Lightning built-in)      | Custom naming, selection policy       |

### 10.2 Design rules

Good callbacks:

- have one clearly named responsibility;
- avoid changing core model mathematics invisibly;
- serialise their state when resumability requires it;
- are independently testable;
- work under distributed execution;
- avoid retaining batch tensors across steps;
- use rank-zero guards for non-distributed side effects.

Poor callbacks:

- locate arbitrary services through `trainer`;
- directly mutate domain model internals;
- perform major training algorithm steps;
- encode experiment-specific constants;
- silently change loss weights;
- collect every batch output in memory;
- write files from every distributed rank.

---

## 11. `trainer.py` — centralised Trainer construction

Direct `Trainer(...)` calls must be centralised in a single factory.

### 11.1 LightningTrainerSpec

The `LightningTrainerSpec` lives in `ehp_sn.lightning` because it directly
mirrors Lightning's constructor arguments. A future framework-independent
`ExecutionSpec` (expressing _intent_ such as device policy, distribution
strategy, numerical precision) would live in `ehp_sn.training` and be
_translated_ by this spec.

```python
@dataclass(frozen=True, slots=True)
class LightningTrainerSpec:
    """Lightning-specific Trainer configuration.

    This is intentionally a thin mirror of Lightning's constructor.
    A future ``ehp_sn.training.ExecutionSpec`` would express
    framework-independent execution policy and be translated here.
    """
    accelerator: str = "auto"
    devices: int | tuple[int, ...] | str = "auto"
    strategy: str = "auto"
    precision: str = "32-true"

    max_epochs: int | None = None
    max_steps: int = -1

    accumulate_grad_batches: int = 1
    gradient_clip_value: float | None = None
    gradient_clip_algorithm: str | None = None

    deterministic: bool | str | None = None
    benchmark: bool | None = None

    log_every_n_steps: int = 50
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1

    num_sanity_val_steps: int = 2
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
```

### 11.2 build_trainer

```python
def build_trainer(
    spec: LightningTrainerSpec,
    *,
    callbacks: Sequence[L.Callback] = (),
    loggers: Sequence[L.pytorch.loggers.Logger] = (),
) -> L.Trainer:
    """Construct a validated Lightning Trainer from project configuration.

    This is the only function that calls ``L.Trainer(...)`` in the
    repository.
    """
    validate_trainer_spec(spec)

    return L.Trainer(
        accelerator=spec.accelerator,
        devices=spec.devices,
        strategy=spec.strategy,
        precision=spec.precision,
        max_epochs=spec.max_epochs,
        max_steps=spec.max_steps,
        accumulate_grad_batches=spec.accumulate_grad_batches,
        gradient_clip_val=spec.gradient_clip_value,
        gradient_clip_algorithm=spec.gradient_clip_algorithm,
        deterministic=spec.deterministic,
        benchmark=spec.benchmark,
        log_every_n_steps=spec.log_every_n_steps,
        val_check_interval=spec.val_check_interval,
        check_val_every_n_epoch=spec.check_val_every_n_epoch,
        num_sanity_val_steps=spec.num_sanity_val_steps,
        enable_progress_bar=spec.enable_progress_bar,
        enable_model_summary=spec.enable_model_summary,
        callbacks=list(callbacks),
        logger=list(loggers) or False,
    )
```

### 11.3 build_callbacks and build_loggers

```python
def build_callbacks(
    system: TrainingSystem,
) -> tuple[L.Callback, ...]:
    """Build Lightning callbacks from resolved project policies."""

def build_lightning_loggers(
    spec: LoggingSpec,
) -> tuple[Logger, ...]:
    """Build Lightning logger adapters."""
```

---

## 12. Checkpoint design

### 12.1 What goes in a Lightning checkpoint

Lightning checkpoints may include model, optimizer, scheduler, scaler, loop,
callback, and hyperparameter state. EHP should add only state that Lightning
cannot understand. The runtime owns all resumable state; there is no
separate `training_state`.

```
checkpoint["ehp"] = {
    "schema_version": 1,
    "runtime":           runtime.state_dict(),         # carry, source cursor, schedule
    "experiment": {
        "definition_id":   definition.id,
        "config_digest":   definition.config_digest,
        "model_ref":       definition.model_ref,
        "task_ref":        definition.task_ref,
    },
}
```

### 12.2 What must not be serialised

- Arbitrary Pydantic model trees
- Factory functions
- Class objects
- Registries
- Complete dataset objects
- Dataloaders
- Large trace buffers

### 12.3 Checkpoint separation

Consider separate concepts for different purposes:

```
training checkpoint
    complete resumable state (model + optimiser + runtime + source)

model bundle
    model weights + model construction metadata

evaluation reference
    immutable pointer to a selected model bundle or checkpoint
```

Do not use Lightning's `.ckpt` file as the sole artifact and experiment
contract for the whole repository.

---

## 13. Builder design — the construction path

### 13.1 TrainingSystem

```python
@dataclass(frozen=True, slots=True)
class TrainingSystem:
    runtime: TrainingRuntime
    optimization: OptimizationSpec
    data: DataProvider
    trainer: LightningTrainerSpec
    callbacks: CallbackSpec
    logging: LoggingSpec
    checkpointing: CheckpointPolicy
```

### 13.2 Recommended construction flow

```
definition = resolve_training_definition(config)

system = build_training_system(definition)

module = build_lightning_module(system)
datamodule = build_lightning_datamodule(system.data)
callbacks = build_callbacks(system)
loggers = build_loggers(system.logging)

trainer = build_trainer(
    system.trainer,
    callbacks=callbacks,
    loggers=loggers,
)

trainer.fit(module, datamodule=datamodule)
```

### 13.3 Factories in `__init__.py`

```python
from .datamodule import EHPDataModule
from .modules import ManualOptimizationLightningModule, TaskLightningModule
from .trainer import build_callbacks, build_lightning_loggers, build_lightning_module, build_trainer

__all__ = [
    "EHPDataModule",
    "ManualOptimizationLightningModule",
    "TaskLightningModule",
    "build_callbacks",
    "build_lightning_loggers",
    "build_lightning_module",
    "build_trainer",
]
```

Do not expose:

- individual callback implementation classes;
- private metric conversion helpers;
- checkpoint dictionary keys;
- runtime-state conversion utilities;
- internal Lightning compatibility helpers;
- regime-specific Lightning module subclasses.

---

## 14. Automatic vs manual optimisation

| Criteria                                   | Automatic | Manual |
| ------------------------------------------ | :-------: | :----: |
| One optimizer                              |    ✅     |   —    |
| One backward pass per step                 |    ✅     |   —    |
| Standard step/epoch scheduler stepping     |    ✅     |   —    |
| All trainable components update together   |    ✅     |   —    |
| Actor and critic use separate optimisers   |     —     |   ✅   |
| Controller and memory update at diff rates |     —     |   ✅   |
| Separate losses with controlled gradient   |     —     |   ✅   |
| Optimiser steps occur conditionally        |     —     |   ✅   |
| TBPTT requires explicit boundaries         |     —     |   ✅   |
| Alternating optimisation                   |     —     |   ✅   |

Do not switch to manual optimisation merely because the training logic is
complex. Use it only when optimiser execution itself is nonstandard.

---

## 15. Lightning as an optional dependency

The key invariant:

> Removing `ehp_sn.lightning` may remove the Lightning execution backend,
> but it must not remove the definitions of EHP models, tasks, objectives,
> data, metrics, evaluation, or artifacts.

A concrete test: a `TrainingRuntime` should be able to execute a correct
training step without constructing a `Trainer`:

```python
runtime = VariationalReplayRuntime(model, adapter, controller, objective)
runtime.start_stage(RuntimeStage.FIT, device_context)

result = runtime.train_step(batch, step_context)

assert result.loss.ndim == 0
assert torch.isfinite(result.loss)
assert result.batch_size == expected

# Metrics are accumulated internally and can be computed:
computed = runtime.compute_train_metrics()
assert "accuracy" in computed
```

This is the professional test. If it passes, the architecture is correct.

---

## 16. Definition of done

The architecture satisfies this design when all these statements are true:

1. A training runtime can execute a unit step without constructing a `Trainer`.
2. No Lightning module imports task-specific metric keys.
3. No Lightning module constructs datasets or reaches into private
   `DataModule` fields.
4. No Lightning module contains rollout algorithms.
5. No Lightning module performs task-specific evaluation aggregation.
6. Models remain plain `nn.Module`.
7. The experiment builder owns concrete component selection.
8. The runtime owns regime-specific orchestration.
9. Lightning owns lifecycle translation and infrastructure.
10. Removing `ehp_sn.lightning` removes the Lightning backend, but not the
    definition or testability of training behavior.

The target conceptual boundary:

```
TrainingRuntime:
    "What does one correct EHP training step mean?"

LightningModule:
    "How does Lightning invoke, optimise, log, checkpoint, and distribute it?"
```

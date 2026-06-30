# Training Architecture

> Canonical design for `ehp_sn.training` — the package that owns **training
> execution policy** for EHP model systems.

`ehp_sn.training` provides framework-independent contracts and services for
executing optimization over EHP model systems. It owns training-step
composition, mutable training state, optimization policy, optimizer and
scheduler construction, gradient handling, and training lifecycle
coordination.

It does **not** own the scientific meaning of the model, rollout, task, or
objective.

---

## 1. Scope and ownership

### 1.1 Canonical responsibility

The package coordinates this transformation:

```
batch / rollout
    │
    ▼
training unit ──► named losses + metrics + next carry
    │
    ▼
backward
    │
    ▼
gradient policy
    │
    ▼
optimizer update
    │
    ▼
scheduler progression
    │
    ▼
training state / checkpoint / logging events
```

Formally, the training package owns **when and how** optimization occurs;
it does not own **what** the model computes, **how** experience is generated,
or **why** a loss is scientifically valid.

### 1.2 Concrete ownership

| Owns                                 | Examples                                                           |
| ------------------------------------ | ------------------------------------------------------------------ |
| Training-step composition contract   | `LossTerm`, `TrainStepOutput`, `TrainingUnit`                      |
| Aggregation of named loss terms      | Weighted sum, backward eligibility per term                        |
| Mutable training state               | `TrainingState` — batch step, optimizer step, samples seen         |
| Optimization policy                  | Gradient accumulation, clipping, nonfinite detection               |
| Optimizer and scheduler construction | `OptimizerSpec`, `ParameterGroupSpec`, `build_optimizer`           |
| Parameter ownership and validation   | Exactly-one-group invariant, frozen-param detection                |
| Recurrent carry truncation policy    | Autograd detach at TBPTT boundaries                                |
| Family-specific runtime schedules    | `tem.py` (TEM `resolve_tem_runtime`), `hrm.py` (validation config) |
| Distributed loss normalization       | `SumOverBatch`, `normalize_loss_for_backward`                      |
| Family-specific training computation | `training/units/` — `VariationalReplayTrainingUnit`, etc.          |

### 1.3 What training does NOT own

| Concern                                        | Owner                                         |
| ---------------------------------------------- | --------------------------------------------- |
| Neural network definitions                     | `ehp_sn.models`                               |
| Environment interaction and temporal execution | `ehp_sn.rollouts`                             |
| Task rules and targets                         | `ehp_sn.tasks`                                |
| Loss mathematics                               | `ehp_sn.objectives` / `ehp_sn.loss`           |
| Metric definitions and aggregation             | `ehp_sn.metrics`                              |
| Dataset and dataloader construction            | `ehp_sn.data`                                 |
| Logger implementations (TensorBoard, MLflow)   | `ehp_sn.logging` / `ehp_sn.lightning.loggers` |
| Offline evaluation recipes                     | `ehp_sn.evaluation`                           |
| Trace schemas and capture                      | `ehp_sn.traces`                               |
| Figures and visualizations                     | `ehp_sn.figures`                              |
| Lightning lifecycle implementation             | `ehp_sn.lightning`                            |
| Experiment-specific model assembly             | `ehp_sn.experiments`                          |

The critical distinction is:

> **objectives** define _what_ is optimized; **training** defines _how_ and
> _when_ optimization occurs.

### 1.4 Forbidden dependencies

```
models ────────X──► training
objectives ────X──► training
rollouts ──────X──► training
tasks ─────────X──► training
metrics ───────X──► training
training ──────X──► lightning
training ──────X──► experiments
training ──────X──► eval
training ──────X──► figures
training ──────X──► traces
```

`training` may import from `models` (through narrow contracts only), from
`objectives` (through `TrainStepOutput`), and from `rollouts` (through
carry types). It must never import from `lightning` — Lightning integration
lives in `ehp_sn.lightning`, which imports `ehp_sn.training`, not the
reverse.

---

## 2. Package structure

```
src/ehp_sn/training/
├── __init__.py               # Public API
├── contracts.py              # LossTerm, TrainStepOutput, TrainingUnit
├── state.py                  # TrainingState
├── config.py                 # TrainingConfig, GradientConfig, PrecisionConfig
├── carry.py                  # TruncationPolicy, TBPTTConfig
├── distributed.py            # SumOverBatch, normalize_loss_for_backward
├── optim.py                  # OptimizerSpec, SchedulerSpec, build_optimizer, build_scheduler
├── gradients.py              # grad_l2_norm, gradient policy functions
├── tem.py                    # TEM runtime schedules (see relocation plan below)
├── hrm.py                    # HRM runtime config (see relocation plan below)
└── units/                    # Reusable TrainingUnit implementations
    ├── __init__.py
    ├── variational_replay.py
    └── act_supervised.py
```

**Deliberately excluded from `training/`:**

| Current location                | Preferred location                                   | Reason                                                              |
| ------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| `lightning/modules/*.py`        | `ehp_sn.lightning.modules`                           | Lightning adapters; `lightning` imports `training`, not the reverse |
| `lightning/callbacks/*.py`      | `ehp_sn.lightning.callbacks`                         | Lightning lifecycle observers                                       |
| `artifact_publisher.py`         | `ehp_sn.model_artifacts` or orchestration layer      | Artifact packaging, not training execution policy                   |
| `step_loop.py`                  | `ehp_sn.diagnostics` or `ehp_sn.traces`              | Diagnostic iteration, not optimizer control                         |
| `rollout.py` (scoring wrappers) | `ehp_sn.rollouts`                                    | Rollout orchestration, not optimizer policy                         |
| Callback/event types            | Defer until a non-Lightning execution backend exists | Currently Lightning-only; no second consumer                        |

### 2.1 Submodule roles

| File                | Role                                                                      |
| ------------------- | ------------------------------------------------------------------------- |
| `contracts.py`      | Semantic center: `LossTerm`, `TrainStepOutput`, `TrainingUnit`            |
| `state.py`          | Mutable runtime counters and metadata                                     |
| `config.py`         | Validated, framework-neutral optimization policy                          |
| `carry.py`          | Autograd truncation policy at TBPTT boundaries                            |
| `distributed.py`    | DDP loss normalisation, rank-safe utilities                               |
| `optim.py`          | Optimizer/scheduler spec, construction, parameter-group validation        |
| `gradients.py`      | `grad_l2_norm`, gradient validation, clipping, accumulation normalisation |
| `tem.py` / `hrm.py` | Family-specific runtime resolution (see relocation classification below)  |
| `units/`            | Concrete `TrainingUnit` implementations for each training paradigm        |

### 2.2 Relocation plan for `tem.py` and `hrm.py`

These files currently mix training-owned and non-training concerns. During
Phase 4–5 of the migration, split them by content ownership:

| Content                                           | Current in         | Correct owner                                | Reason                                                 |
| ------------------------------------------------- | ------------------ | -------------------------------------------- | ------------------------------------------------------ |
| `resolve_tem_runtime(step, config)`               | `tem.py`           | `ehp_sn.training`                            | Independent variable is `optimizer_step`               |
| `MemoryRuntimeConfig`, `UncertaintyRuntimeConfig` | `tem.py`           | `ehp_sn.training` or `ehp_sn.controllers`    | Schedule parameterisation tied to training progression |
| `SequenceRuntimeConfig` (`tbptt_steps`)           | `tem.py`           | `ehp_sn.rollouts` or `ehp_sn.training.carry` | Chunk-length policy, not optimizer concern             |
| `ValidationRuntimeConfig` (both families)         | `tem.py`, `hrm.py` | `ehp_sn.experiments` or `ehp_sn.evaluation`  | Validation recipe, not execution policy                |
| `load_weights_from_checkpoint(...)`               | `tem.py`, `hrm.py` | `ehp_sn.models.loading`                      | Model weight hydration, not training policy            |
| `VALID_INIT_GROUPS`                               | `tem.py`, `hrm.py` | `ehp_sn.models.loading`                      | Semantic-group definitions belong to model contracts   |
| HRM `RuntimeConfig` (validation-only)             | `hrm.py`           | `ehp_sn.experiments`                         | No schedule — pure validation safety limits            |

After relocation, `tem.py` and `hrm.py` may shrink to thin re-exports or
disappear entirely. The files remain in `training/` during Phase 1–3 to
avoid disrupting the existing import graph.

---

## 3. Core contracts

### 3.1 `LossTerm` and `TrainStepOutput`

`TrainStepOutput` is the return contract of every `TrainingUnit.train_step()`
call. It is the typed envelope that carries named loss terms, metrics, and
next-state information from the scientific layer to the optimization layer.

```python
@dataclass(frozen=True)
class LossTerm:
    value: Tensor              # local summed loss (compatible with SumOverBatch)
    normalizer: Tensor | int   # number of contributing units (samples, tokens, valid steps)
    weight: float = 1.0        # applied when computing total_loss()
    backward: bool = True      # participates in backward pass
    log: bool = True           # included in logging output


@dataclass
class TrainStepOutput(CarryT_co):
    losses: Mapping[str, LossTerm]
    metrics: Mapping[str, Tensor | float] = empty_dict()
    carry: CarryT_co | None = None

    def total_loss(self) -> Tensor: ...
```

**Normalization contract (canonical):**

Every `LossTerm.value` is a **local sum** — sum over the local batch of
unreduced loss contributions. This is the only supported representation
for backward-participating losses. It is directly compatible with
`SumOverBatch` for distributed reduction.

`LossTerm.normalizer` records the number of units that contributed to the
sum: samples, valid recurrence steps, unmasked tokens, or active replay
slots. The normalizer must be a **positive scalar** —
`int` or a 0-d detached `Tensor`. Vector normalizers are not
supported; per-example normalization should be applied by the objective
before forming the `LossTerm`.

Training computes the backward loss as a **per-term weighted
normalized sum**:

```
loss = sum(term.weight * term.value / term.normalizer
           for term in losses if term.backward)
```

This preserves each objective's scientific loss coefficient independently.
Formally:

```python
class LossTerm:
    ...
    def normalized(self) -> Tensor:
        """Return value / normalizer — per-unit mean loss."""
        return self.value / self.normalizer

class TrainStepOutput:
    ...
    def total_loss(self) -> Tensor:
        return sum(
            term.weight * term.normalized()
            for term in self.losses.values()
            if term.backward
        )
```

Under DDP, DDP averages gradients across ranks. For this to produce the
correct global per-unit mean, every rank must have equal contribution counts
— the invariant enforced by `validate_num_slots_divisibility`. Unequal
local normalizers require a separate global-denominator all-reduce and are
not currently supported.

Metrics are carried in `metrics` (detached `Tensor` or Python `float`),
not in `LossTerm`. The `normalizer` field on a logged `LossTerm` can be
used to aggregate per-step metrics into correctly weighted epoch-level
statistics.

**Rules:**

- Every `LossTerm` with `backward=True` must have a scalar `value`.
- At least one `LossTerm` must have `backward=True`.
- `normalizer` must be positive.
- Logged metrics are detached or converted to Python scalars.

### 3.2 `TrainingUnit`

`TrainingUnit` is the protocol that bridges scientific computation and
training infrastructure. It performs:

```
batch + carry + state
    │
    ▼
rollout (optional, depending on paradigm)
    │
    ▼
objective computation
    │
    ▼
TrainStepOutput
```

It does **not** call:

- `loss.backward()`
- `optimizer.step()`
- `scheduler.step()`
- `self.log(...)`

```python
class TrainingUnit(Protocol[BatchT, CarryT_co]):
    def train_step(
        self,
        batch: BatchT,
        *,
        carry: CarryT_co | None,
        state: TrainingState,
    ) -> TrainStepOutput[CarryT_co]:
        ...
```

**Validation is not part of `TrainingUnit`.** Training-time validation may
differ from training in carry semantics, memory behaviour (fixed vs.
streaming), deliberation depth, and trace production. Training owns _when_
validation runs (via `ValidationConfig`); how validation computation is
performed belongs to the existing evaluation contracts
(`ehp_sn.evaluation`, `ehp_sn.rollouts.scoring`) or to a separate
`ValidationUnit` protocol if a reusable domain abstraction is needed.

### 3.3 `TrainingState`

Mutable runtime counters that are scientifically or operationally relevant.
These are distinct from Lightning's lifecycle counters.

```python
@dataclass
class TrainingState:
    batch_step: int = 0
    optimizer_step: int = 0

    samples_seen: int = 0
    valid_steps_seen: int = 0
    tokens_seen: int = 0
    episodes_seen: int = 0

    should_stop: bool = False
    stop_reason: str | None = None
```

**Why multiple counters matter for EHP:**

| Counter            | Distinct from      | Why                                               |
| ------------------ | ------------------ | ------------------------------------------------- |
| `batch_step`       | `optimizer_step`   | Gradient accumulation: N batches per update       |
| `optimizer_step`   | `batch_step`       | TEM schedules depend on optimizer updates         |
| `samples_seen`     | both step counters | Loss normalisation, learning curves               |
| `valid_steps_seen` | `samples_seen`     | Recurrent rollouts: 1 sample may be N valid steps |
| `tokens_seen`      | `valid_steps_seen` | Token-based tasks (SeqMaze)                       |
| `episodes_seen`    | `samples_seen`     | Replay source admission, coverage tracking        |

Because Lightning remains the execution framework, `TrainingState` should not
duplicate every Lightning field. Lightning remains authoritative for epoch and
trainer lifecycle; `TrainingState` records domain-relevant counters.
`epoch` is deliberately excluded — the Lightning adapter synchronises it
from the framework context when needed for logging, never as an
independently mutated field.

### 3.4 `TrainingConfig`

Validated, framework-neutral training policy. Describes execution behaviour,
not model internals.

```python
@dataclass(frozen=True)
class GradientConfig:
    accumulation_steps: int = 1
    clip_norm: float | None = None
    clip_value: float | None = None
    error_if_nonfinite: bool = True

@dataclass(frozen=True)
class PrecisionConfig:
    mode: Literal["fp32", "fp16-mixed", "bf16-mixed"] = "fp32"

@dataclass(frozen=True)
class ValidationConfig:
    every_n_steps: int | None = None
    every_n_epochs: int | None = 1
    run_at_start: bool = False
    # The Lightning adapter translates these into val_check_interval,
    # check_val_every_n_epoch, and sanity_val_checking.  Training owns
    # *when* validation executes; Lightning owns *how*.

@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int | None = None
    max_optimizer_steps: int | None = None
    seed: int = 0
    gradients: GradientConfig = default_field(GradientConfig())
    precision: PrecisionConfig = default_field(PrecisionConfig())
    validation: ValidationConfig = default_field(ValidationConfig())
```

**What belongs here:**

- Maximum epochs or optimizer steps
- Precision
- Gradient accumulation
- Clipping
- Validation frequency
- Deterministic execution policy
- Training seed

**What does NOT belong here:**

- Model dimensions
- Dataset paths
- Replay slots
- Task horizon
- Controller halting limits
- TEM `eta`, Hebbian decay, `p2g`
- Environment dimensions
- Objective coefficients that define scientific loss semantics
- Checkpoint configuration (owned by Lightning integration layer)

TOML remains the serialised configuration format. `TrainingConfig` is the
validated in-memory representation. The existing TOML sections in
`config/training/` are interpreted into typed `TrainingConfig` objects.

### 3.5 `FitResult` (deferred)

A typed terminal output from a training run (`FitResult`) is a sensible
contract, but it requires a framework-independent `fit()` producer. While
the repository uses Lightning's `trainer.fit()`, `FitResult` lacks an
authoritative constructor.

Defer `FitResult` to the public API until either:

- a repository-level `run_training() -> FitResult` orchestration function
  exists; or
- a non-Lightning execution backend produces it.

In the interim, the Lightning adapter may return a plain dataclass or the
Lightning trainer result directly.

---

## 4. Carry policy

### 4.1 Motivation

For EHP, carry management is a concrete coordination problem between:

- Rollout segmentation
- Carry lifetime
- Graph detachment
- Loss normalisation
- Optimiser-update boundaries
- Episode boundaries
- Memory-bank persistence

### 4.2 Ownership

| Operation                            | Owner                                     |
| ------------------------------------ | ----------------------------------------- |
| Define carry structure               | `contracts` / model runtime               |
| Produce next carry                   | Rollout runner                            |
| Identify semantic episode boundaries | Task / rollout runtime                    |
| Apply episode reset masks            | Carry implementation (task/runtime-owned) |
| Decide when to detach carry graph    | Training policy (`TruncationPolicy`)      |
| Decide accumulation/update boundary  | Training policy (`GradientConfig`)        |
| Serialise persistent carry           | Checkpoint integration                    |
| Define loss per chunk                | Objective                                 |

### 4.3 Truncation policy contract

Training owns only the autograd-detachment decision at TBPTT boundaries.
Episode boundary semantics (reset masks, halt flags, cursor advancement)
belong to the task/runtime layer, not to training. Replay-source
synchronisation belongs to the adapter or rollout source, not to the
truncation policy.

```python
class TruncationPolicy(Protocol):
    def should_detach(self, state: TrainingState) -> bool:
        \"\"\"Return True at a TBPTT truncation boundary.

        Called once per chunk.  When True, ``detach()`` is invoked
        on the carry before the next chunk begins.
        \"\"\"
        ...

    def detach(self, carry: Any) -> Any:
        \"\"\"Detach autograd history from a carry.

        Breaks the computation graph while preserving carry *values*
        so the next chunk can reuse them as initial state.
        \"\"\"
        ...
```

### 4.4 TBPTT configuration

```python
@dataclass(frozen=True)
class TBPTTConfig:
    chunk_length: int
    detach_between_chunks: bool = True
    update_every_n_chunks: int = 1
    normalize_by: Literal["chunks", "valid_steps", "tokens"] = "valid_steps"
```

**Critical invariant:**

> Carry values may survive a chunk boundary; autograd graphs may **not**
> survive a configured truncation boundary.

This invariant should have direct tests.

---

## 5. Optimisation layer

### 5.1 Optimizer spec

```python
@dataclass(frozen=True)
class ParameterGroupSpec:
    name: str
    selector: str                 # module-name pattern or explicit parameter list
    lr_scale: float = 1.0
    weight_decay: float | None = None

@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    kind: Literal["adam", "adamw", "adamatan2"]
    learning_rate: float
    weight_decay: float = 0.0
    parameter_groups: tuple[ParameterGroupSpec, ...] = ()
    kwargs: Mapping[str, object] = default_field(dict)

@dataclass(frozen=True)
class SchedulerSpec:
    kind: str
    interval: Literal["optimizer_step", "epoch", "validation"]
    frequency: int = 1
    monitor: str | None = None
    kwargs: Mapping[str, object] = default_field(dict)
```

### 5.2 Parameter group validation

`build_optimizer(model, spec)` must detect:

| Condition                             | Action              |
| ------------------------------------- | ------------------- |
| Parameter assigned to multiple groups | Raise `ValueError`  |
| Trainable parameter in no group       | Raise `ValueError`  |
| Frozen parameter in a group           | Raise `ValueError`  |
| Empty group                           | Raise `ValueError`  |
| Selector matches nothing              | Raise `ValueError`  |
| Duplicate group names                 | Raise `ValueError`  |
| Unsupported optimizer kind            | Raise `ValueError`  |
| All conditions pass                   | Construct optimiser |

**Invariant:** Every trainable parameter belongs to exactly one intended
optimiser group. When `parameter_groups=()` (the default), one group
containing every trainable parameter is created automatically.

### 5.3 Gradient policy

Gradient operations are plain functions parameterised by `GradientConfig`.
A class hierarchy is unnecessary unless materially different algorithms
emerge (adaptive clipping, per-group policies, NaN recovery rollback).

```python
def prepare_backward_loss(
    loss: Tensor,
    *,
    accumulation_steps: int,
) -> Tensor:
    """Return loss / accumulation_steps for manual_backward."""
    ...

def validate_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    error_if_nonfinite: bool = True,
) -> None:
    """Detect nonfinite gradients.  Raise or warn per config."""
    ...

def clip_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    clip_norm: float | None = None,
    clip_value: float | None = None,
) -> None:
    """Apply norm or value clipping after AMP unscaling."""
    ...
```

**Accumulation invariant:** Dividing each microbatch loss by
`accumulation_steps` produces the correct large-batch gradient _only_
when every microbatch has equal contribution counts for each loss term.
For example, a batch of 10 tokens and a batch of 90 tokens do not
sum correctly to a 100-token batch under simple `loss / k` averaging.

For the current repository, this invariant is satisfied by fixed replay
slots and fixed-length tasks. Variable-normalizer accumulation is
unsupported initially. Add a runtime assertion or document which
training families satisfy the invariant.

### 5.4 Scheduler semantics

The training package must explicitly define:

- Whether a scheduler steps per batch, per optimiser update, per epoch,
  or per validation event
- Whether it depends on a monitored metric
- Whether it runs before or after optimiser stepping
- How it behaves during skipped updates (gradient accumulation, nonfinite
  gradient skip)
- How its state is restored

**Default for EHP:** step scheduler after successful optimiser update, not
after every batch. There should be exactly **one** owner of scheduler
progression.

### 5.5 Update sequence

The centralised update order is:

```
1. forward under precision context
2. compute loss
3. scale/backward if AMP
4. unscale gradients
5. validate gradients (nonfinite detection)
6. clip gradients
7. optimizer step
8. scaler update (if AMP)
9. zero gradients
10. scheduler step (if optimiser-step-based)
11. update training state counters
```

A callback must not perform an additional optimiser step or clip gradients
independently.

---

## 6. Training units

### 6.1 Purpose

A `TrainingUnit` assembles model, rollout runner, controller, and objective
into a composable step computation. It is the adapter between domain-specific
scientific logic and the generic optimisation engine.

### 6.2 Example: variational replay

```python
class VariationalReplayTrainingUnit:
    def __init__(
        self,
        *,
        model,
        rollout_runner,
        controller,
        objective,
        runtime_policy,
    ) -> None:
        self._model = model
        self._rollout_runner = rollout_runner
        self._controller = controller
        self._objective = objective
        self._runtime_policy = runtime_policy

    def train_step(
        self,
        batch,
        *,
        carry,
        state: TrainingState,
    ) -> TrainStepOutput:
        runtime = self._runtime_policy.resolve(
            optimizer_step=state.optimizer_step,
        )
        self._model.set_runtime(runtime)

        rollout = self._rollout_runner.run(
            model=self._model,
            controller=self._controller,
            batch=batch,
            carry=carry,
        )
        objective_output = self._objective(rollout)

        return TrainStepOutput(
            losses={
                name: LossTerm(
                    value=term.value,
                    normalizer=term.normalizer,
                    weight=term.weight,
                )
                for name, term in objective_output.losses.items()
            },
            metrics=objective_output.metrics,
            carry=rollout.final_carry,
        )
```

### 6.3 Placement rules

Only place a `TrainingUnit` in `training/units/` if it is reusable across
experiments. A unit specific to one experiment should stay under:

```
experiments/<task>/<family>/training.py
```

The dependency direction is:

```
experiment builder
    │
    ▼
constructs model + rollout + objective + training unit
    │
    ▼
wraps unit in Lightning adapter
```

The generic training package must not import every experiment family.

### 6.4 Expected units

| Unit                            | Consumes                                   | Produces                                  |
| ------------------------------- | ------------------------------------------ | ----------------------------------------- |
| `VariationalReplayTrainingUnit` | TEM model, recurrent runner, TEM objective | `TrainStepOutput` with ELBO terms         |
| `ACTSupervisedTrainingUnit`     | HRM model, ACT controller, ACT scorer      | `TrainStepOutput` with task+halt terms    |
| `ActorCriticTrainingUnit`       | Actor-critic model, Q-halting controller   | `TrainStepOutput` with value+policy terms |

---

## 7. Lightning integration

### 7.1 Architecture

Lightning remains the infrastructure backend. The architecture is:

```
EHP model / rollout / objective
            │
            ▼
      TrainingUnit
            │
            ▼
 TrainStepOutput + TrainingState
            │
            ▼
 thin LightningModule adapter
            │
            ▼
 Lightning Trainer
```

### 7.2 Adapter responsibility

A Lightning adapter owns:

- Implementing Lightning hooks
- Calling `manual_backward` or returning automatic loss
- Obtaining optimisers
- Applying gradient policy
- Calling optimiser and scheduler steps
- Translating metrics into `self.log`
- Synchronising Lightning state with `TrainingState`
- Serialising domain state in checkpoint hooks

A Lightning adapter does **not** own:

- Rollout semantics
- Task-specific logic
- Objective construction
- TEM schedule calculations
- Metric meaning
- Parameter selection rules
- Carry structure

### 7.3 Target `training_step`

The Lightning module should be a thin adapter. The example below models
gradient accumulation, skipped-update semantics, AMP unscaling, and
correct update-boundary indexing.

```python
class VariationalReplayLightningAdapter(L.LightningModule):
    def on_train_start(self):
        # Clear gradients before the first accumulation window.
        optimiser = self.optimizers()
        optimiser.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        output = self._unit.train_step(
            batch,
            carry=self._carry,
            state=self._training_state,
        )

        accumulation = self._training_config.gradients.accumulation_steps
        loss = output.total_loss()

        # Accumulation-aware backward: divide by accumulation_steps so
        # the summed gradient over the window matches a large-batch update.
        self.manual_backward(loss / accumulation)

        self._training_state.batch_step += 1
        update_due = self._training_state.batch_step % accumulation == 0

        if update_due:
            optimiser = self.optimizers()

            # AMP: unscale gradients before validation and clipping.
            # The adapter abstracts the Lightning-specific call.
            self._unscale_gradients_if_needed(optimiser)

            parameters = tuple(self._unit.trainable_parameters())
            validate_gradients(
                parameters,
                error_if_nonfinite=(
                    self._training_config.gradients.error_if_nonfinite
                ),
            )
            clip_gradients(
                parameters,
                clip_norm=self._training_config.gradients.clip_norm,
                clip_value=self._training_config.gradients.clip_value,
            )

            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            if self._scheduler_steps_per_update:
                self.lr_schedulers().step()

            self._training_state.optimizer_step += 1

        self._carry = output.carry
        if self._truncation_policy.should_detach(self._training_state):
            self._carry = self._truncation_policy.detach(self._carry)

        self._log_output(output)
        return loss.detach()

    def on_train_epoch_end(self) -> None:
        \"\"\"Discard incomplete accumulation-window gradients at epoch end.\"\"\"
        accumulation = self._training_config.gradients.accumulation_steps
        if self._training_state.batch_step % accumulation != 0:
            self.optimizers().zero_grad(set_to_none=True)
```

**Key semantics:**

- `on_train_start` clears gradients before the first accumulation window.
- Backward uses `loss / accumulation_steps` so the summed gradient over
  the accumulation window matches a single large-batch update.
- `batch_step` increments before the update check, so `batch_step % accumulation == 0`
  fires at the end of each complete window: batches `accumulation-1`,
  `2*accumulation-1`, etc.
- AMP unscaling uses the adapter's `_unscale_gradients_if_needed()`,
  which internally calls the Lightning precision plugin. The canonical
  pseudocode does not hard-code one framework method name.
- Gradient validation and clipping are plain functions parameterised by
  `GradientConfig`.
- Optimizer step, scheduler step, zero-grad, and `optimizer_step` increment
  occur only when the accumulation window completes.
- An incomplete final accumulation window at epoch end is **discarded** —
  `on_train_epoch_end` clears remaining gradients when the batch count
  is not divisible by `accumulation_steps`. For strict divisibility,
  configure epoch lengths to be multiples of `accumulation_steps`.
- Scheduler progression is gated on `_scheduler_steps_per_update` to
  distinguish per-optimizer-step from per-batch or per-epoch scheduling.

The precise sequence may differ under FSDP or custom precision plugins,
but the module should be an adapter rather than the scientific implementation.

---

## 8. Callbacks and policies (design principle, deferred implementation)

### 8.1 Distinction

| Type       | Role                     | Mutation allowed |
| ---------- | ------------------------ | ---------------- |
| `Callback` | Observe, log, report     | No               |
| `Policy`   | Alter training behaviour | Yes              |

This distinction is intellectually sound and should guide callback design.
However, the repository currently uses Lightning callbacks exclusively and
has no non-Lightning execution backend. Do **not** implement a
framework-independent event bus (`TrainingEvent`, `TrainingCallback`,
`TrainingContext`) until a concrete second consumer exists.

### 8.2 Current state

The existing callbacks in `ehp_sn.lightning.callbacks` are all
observational:

- `MetricsCallback` — logs metrics
- `StepProgressBar` — renders progress
- `EvaluationRegimesCallback` — triggers evaluation regimes
- `DiagnosticsCallback` — captures diagnostic traces
- `FigureGenerationCallback` — produces figures
- `LearningRateMonitor` — logs LR

None mutate training state. This is correct behaviour. Keep them in
`ehp_sn.lightning.callbacks`.

### 8.3 When to formalise

Introduce a typed callback/policy system when:

- a non-Lightning execution backend is added (e.g., Accelerate); or
- a mutation-capable callback appears (adaptive TBPTT, curriculum schedule,
  freeze/unfreeze schedule, NaN recovery).

Until then, rely on Lightning's callback system and document the
observational contract as a convention.

---

## 9. Distributed training

### 9.1 Scope

`training/distributed.py` remains focused on:

- Normalisation of summed losses under DDP
- Distributed scalar reduction
- Rank-safe utilities
- Correct batch-weighted metric aggregation

### 9.2 Loss normalisation contract

```python
@dataclass(frozen=True)
class SumOverBatch:
    \"\"\"Marker: the wrapped value is a sum over the local contributing units.

    \"Batch\" is historical naming; the contributing unit set may be
    samples, valid recurrent steps, unmasked tokens, or active replay
    slots.  The contract is that ``value`` is a *sum* (never a mean)
    and ``LossTerm.normalizer`` records the contribution count.
    \"\"\"
    value: Tensor

def normalize_loss_for_backward(
    loss: SumOverBatch,
    normalizer: int | float,
) -> Tensor:
    \"\"\"Return per-unit mean loss for ``manual_backward()``.

    ``loss.value`` must be a sum over the local contributing units.
    ``normalizer`` is the contribution count.  DDP averages gradients
    across ranks.  With equal local normalizers *n* across *W* ranks::

        DDP gradient
        = (1 / W) × Σ ∇(local_sum_r / n)
        = ∇(global_sum / (W × n))
        = ∇(global_sum / global_normalizer)
        = gradient of global per-unit mean

    This holds exactly when every rank has equal local normalizers.
    For replay-based training, this is guaranteed when each rank
    receives the same ``num_slots // world_size`` replay slots —
    the invariant enforced by ``validate_num_slots_divisibility``.

    Unequal local normalizers require a global-denominator all-reduce
    and a local backward scaling by::

        world_size × local_sum_r / global_normalizer

    (where ``global_normalizer`` is the all-reduced sum of per-rank
    normalizers).  This is not currently supported.
    \"\"\"
    if normalizer <= 0:
        raise ValueError(f\"normalizer must be positive, got {normalizer}.\")
    return loss.value / float(normalizer)
```

**Invariant:** The loss passed to `normalize_loss_for_backward` must be a
**sum** over the local contributing units, not a mean. Using a mean-reduced
loss produces incorrectly scaled gradients under DDP. Every rank must
receive an equal share of replay slots. Unequal-normalizer training is
unsupported until a dedicated global-denominator scaling function is
implemented.

### 9.3 What NOT to do

Do not turn `distributed.py` into a wrapper around Lightning strategies, DDP,
or FSDP. Lightning already owns strategy resolution; `training/distributed.py`
owns only the EHP-specific conventions (sum-reduced loss, slot divisibility).

---

## 10. Implementation sequence

### Phase 1: Contracts

Add `contracts.py` and `state.py` with `LossTerm`, `TrainStepOutput`,
`TrainingUnit`, and `TrainingState`.

No behavioral changes. These are pure dataclasses and protocols.

### Phase 2: Typed execution configuration

Add `config.py` with `TrainingConfig`, `GradientConfig`, `PrecisionConfig`,
and `ValidationConfig`. Interpret existing TOML sections
into typed objects.

Do not change the TOML schema initially.

### Phase 3: Optimizer builder

Expand `optim.py` (or create `optimization/specs.py`). Add `OptimizerSpec`,
`ParameterGroupSpec`, `SchedulerSpec`, and a validated `build_optimizer()`.

Move HRM parameter-group construction out of the Lightning module and into
the builder. Add validation tests for duplicates, empty groups, frozen
parameters, and unmatched trainable parameters.

### Phase 4: Training units

Extract one family first — preferably TEM, because it has the more complex
carry and TBPTT semantics.

Create `units/variational_replay.py`. Make the current Lightning module
delegate to it. The Lightning module moves to
`ehp_sn.lightning.modules/` as a thin adapter.

Then migrate ACT supervised training (`units/act_supervised.py`).

### Phase 5: Explicit state and checkpoint integration

Add `TrainingState` to the Lightning adapters. Use `optimizer_step` for
optimiser-dependent schedules. Serialise through
`on_save_checkpoint` / `on_load_checkpoint`.

Do not build a custom checkpoint manager until Lightning-independent or
distributed-sharded checkpoints become a real requirement.

---

## 11. Core invariants

### Step output

- Every backward-participating loss is scalar.
- At least one loss participates in backward.
- Logged metrics are detached or converted before storage.
- `LossTerm.normalizer` is positive and records the contribution count
  (samples, valid steps, or tokens).
- When no units contribute to a loss term (e.g., fully masked sequence),
  the objective should either omit the term or emit it with
  `backward=False` and `normalizer=1`. Forwarding a zero-normalizer
  term to backward is an error.
- Carry type is consistent between input and output.

### Optimisation

- Every trainable parameter belongs to exactly one intended optimiser group.
- Frozen parameters are excluded.
- Empty parameter groups are rejected.
- Backward is called by execution infrastructure, not by objectives.
- Gradient clipping occurs after AMP unscaling.
- Scheduler progression follows successful optimiser updates.
- Accumulation normalisation is centralised.

### State

- `batch_step` and `optimizer_step` are distinct.
- Runtime schedules (TEM `eta`, Hebbian decay) use `optimizer_step`.
- Resume restores domain state as well as weights.
- Carry is detached at TBPTT boundaries.
- Carry resets only at explicit semantic boundaries.
- Episode and optimiser steps are measured independently.

### Distributed

- Logging and checkpoint side effects are rank-safe.
- Metrics are reduced with correct weighting.
- Validation does not double-count samples.
- Data sharding is deterministic and complete.
- Checkpoints are portable across supported strategy configurations.

### Logging

- Training units return metrics but do not log.
- Lightning or logging callbacks perform emission.
- Metric names use stable namespaces:

```
train/loss/total
train/loss/reconstruction
train/loss/kl
train/optimization/gradient_norm
train/runtime/optimizer_step
train/runtime/valid_steps
```

---

## 12. Tests that matter

```
test_one_step_matches_manual_pytorch_update
test_gradient_accumulation_matches_large_batch_update
test_clip_occurs_after_amp_unscale
test_scheduler_steps_only_after_optimizer_update
test_parameter_groups_are_disjoint
test_unmatched_trainable_parameters_are_rejected
test_checkpoint_resume_matches_uninterrupted_run
test_rng_state_is_restored
test_tbptt_detaches_carry_at_boundary
test_episode_boundary_resets_carry
test_invalid_batch_does_not_advance_optimizer_step
test_rank_zero_side_effects_execute_once
test_validation_metrics_use_sample_weighting
```

For EHP specifically:

```
test_tem_memory_state_resume_equivalence
test_act_ponder_loss_participates_in_expected_backward_plan
test_actor_and_critic_optimizer_ownership
test_replay_slots_are_not_treated_as_dataloader_batch_size
test_rollout_step_and_optimizer_step_are_independent
test_tbptt_chunk_25_matches_configured_update_policy
test_tem_runtime_schedule_uses_optimizer_step_not_batch_step
test_weighted_multi_loss_normalization_preserves_per_term_coefficients
test_accumulation_window_boundary_fires_at_correct_batch_indices
test_incomplete_accumulation_window_is_discarded
```

---

## 13. Public API

```python
# ehp_sn/training/__init__.py

from .config import (
    GradientConfig,
    PrecisionConfig,
    TrainingConfig,
    ValidationConfig,
)
from .contracts import (
    LossTerm,
    TrainingUnit,
    TrainStepOutput,
)
from .state import TrainingState
from .optim import (
    OptimizerSpec,
    ParameterGroupSpec,
    SchedulerSpec,
    build_optimizer,
    build_scheduler,
)
from .distributed import normalize_loss_for_backward, SumOverBatch

__all__ = [
    "GradientConfig",
    "LossTerm",
    "OptimizerSpec",
    "ParameterGroupSpec",
    "PrecisionConfig",
    "SchedulerSpec",
    "SumOverBatch",
    "TrainingConfig",
    "TrainingState",
    "TrainingUnit",
    "TrainStepOutput",
    "ValidationConfig",
    "build_optimizer",
    "build_scheduler",
    "normalize_loss_for_backward",
]
```

**Do not export:**

- `CheckpointConfig` (Lightning-owned until non-Lightning checkpoints exist)
- `FitResult` (deferred — lacks a framework-independent producer)
- TEM-specific training units
- HRM-specific training units
- Lightning modules and adapters (those live in `ehp_sn.lightning`)
- Internal truncation policies
- Optimiser selector internals
- Event dispatchers and callback types (deferred)
- Distributed implementation helpers beyond `normalize_loss_for_backward`
- Family-specific runtime schedules (`tem.py`, `hrm.py`)

Explicit submodule access remains acceptable:

```python
from ehp_sn.training.units import VariationalReplayTrainingUnit
from ehp_sn.training.carry import TBPTTConfig, TruncationPolicy
```

---

## 14. User-facing composition

```python
model = build_model(model_config)
runner = build_rollout_runner(rollout_config)
objective = build_objective(objective_config)

unit = VariationalReplayTrainingUnit(
    model=model,
    rollout_runner=runner,
    controller=controller,
    objective=objective,
    runtime_policy=runtime_policy,
)

training_config = TrainingConfig(
    max_optimizer_steps=50_000,
    gradients=GradientConfig(
        accumulation_steps=1,
        clip_norm=1.0,
    ),
    precision=PrecisionConfig(mode="bf16-mixed"),
)

optimizer = build_optimizer(
    model,
    OptimizerSpec(
        name="main",
        kind="adamw",
        learning_rate=3e-4,
        weight_decay=1e-2,
    ),
)

module = VariationalReplayLightningAdapter(
    unit=unit,
    optimizer=optimizer,
    training_config=training_config,
)

trainer = build_lightning_trainer(
    training_config=training_config,
    callbacks=callbacks,
)

trainer.fit(module, datamodule=datamodule, ckpt_path=resume_checkpoint)
```

The builder performs composition. It does not contain training-loop mechanics.

---

## 15. Final target architecture

The final design should have four clearly separated layers:

```
Layer 1:  Scientific layer
          model + task + rollout + objective
          (ehp_sn.models, ehp_sn.tasks, ehp_sn.rollouts, ehp_sn.objectives)

Layer 2:  Training semantic layer
          TrainingUnit + TrainStepOutput + TrainingState
          (ehp_sn.training)

Layer 3:  Optimisation layer
          optimizer + scheduler + gradients + parameter ownership
          (ehp_sn.training)

Layer 4:  Framework layer
          LightningModule + Trainer + callbacks + checkpoint hooks
          (ehp_sn.lightning)
```

The critical boundary is between Layer 2 and Layer 3: `TrainingUnit`
produces a `TrainStepOutput`; the optimisation and framework layers
consume it to execute backward, stepping, scheduling, logging, and
checkpointing.

The package boundary can be summarised as:

> `ehp_sn.training` defines the contracts and policies required to convert
> scientific computation into reproducible parameter updates. It does not
> define the model, environment, objective mathematics, evaluation protocol,
> logging backend, or framework lifecycle.

For the current repository, the highest-value change is not a new trainer.
It is introducing `TrainingUnit`, `TrainStepOutput`, and `TrainingState`,
then reducing the Lightning modules to infrastructure adapters. Lightning
adapters live in `ehp_sn.lightning`, which imports `ehp_sn.training` —
never the reverse. The existing rollout and objective separation is
directionally correct; the missing boundary is between step computation
and step optimisation.

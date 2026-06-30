---
title: Objectives Design
description: Differentiable scoring layer between model/runtime outputs and optimization — typed contracts, task adapters, composite objectives
---

# Objectives Design Contract

> `ehp_sn.objectives` owns the **differentiable scoring layer between
> model/runtime outputs and optimization**. It transforms typed task,
> controller, and model outputs into the scalar used for backpropagation,
> named signed contributions, unreduced terms, and detached diagnostics.

This is broader than a "loss functions" package. A loss is one mathematical
term (cross-entropy, MSE, KL divergence). An **objective** combines several
such terms, applies masks and schedules, normalizes them correctly, and
returns structured diagnostics — it is the complete optimization criterion
for one training paradigm.

---

## 1. Scope and ownership

### 1.1 Package boundary

```
model / controller / runtime
        │
        ▼
  typed predictions and rollout records
        │
        ▼
  task evaluator          (task/ — task-specific adaptation)
        │
        ▼
  composite objective     (composites/ — paradigm-specific composition)
        │
        ▼
  ObjectiveResult
        │
        ├── loss ──────────► backward()
        ├── contributions ─► named signed weighted scalars
        ├── terms ─────────► unreduced tensors for aggregation
        └── diagnostics ───► detached observability
```

### 1.2 What objectives own

| Owns                                | Examples                                             |
| ----------------------------------- | ---------------------------------------------------- |
| Differentiable task scoring         | Token CE, field MSE, trajectory MSE                  |
| Differentiable controller scoring   | Halt BCE, Q-value MSE, state-value MSE               |
| Combination of objective components | ACT: task + halt + continue; TEM: obs + latent + reg |
| Coefficient application             | `c_task * task_sum`, `c_obs * obs_nll_sum`           |
| Mask-aware reductions               | Per-sequence mean, masked sum, masked mean           |
| Objective-specific validation       | Shape checks, value-range checks, empty-mask policy  |
| Objective result contracts          | `ObjectiveResult`, `ObjectiveContext`                |
| Task-to-objective adaptation        | `TaskStepEvaluator` protocol → `TaskStepEvaluation`  |

### 1.3 What objectives do NOT own

| Not owned                                | Owner                                        |
| ---------------------------------------- | -------------------------------------------- |
| Model forward execution                  | `ehp_sn.models`                              |
| Recurrent state evolution                | `ehp_sn.controllers`                         |
| Rollout traversal and step iteration     | `ehp_sn.rollouts.scoring`                    |
| Optimizer construction and stepping      | `ehp_sn.training` or `ehp_sn.lightning`      |
| `.backward()` calls                      | `ehp_sn.lightning`                           |
| Logging backends (MLflow, console)       | `ehp_sn.logging`, `ehp_sn.lightning.loggers` |
| Evaluation metric accumulation           | `ehp_sn.metrics`                             |
| Target generation from raw datasets      | `ehp_sn.data` or task packages               |
| Experiment selection and recipe dispatch | `scripts/` or `ehp_sn.evaluation`            |
| Checkpointing                            | `ehp_sn.lightning`                           |

### 1.4 Forbidden dependencies

```
loss ───────────────X──► objectives
objectives ─────────X──► lightning
objectives ─────────X──► logging backend
objectives ─────────X──► optimizer
objectives.task ────X──► objectives.composites
metrics ────────────X──► lightning module internals
```

The `loss/` package is a separate peer package, not a sub-package of
`objectives/`. `objectives` **depends on** `loss`; `loss` must never
depend on `objectives`. Dependency direction is always:

```
loss  ◄── objectives  ◄── rollouts / training / lightning
```

---

## 2. Architectural layers

The package uses a **four-level architecture**:

```
Level 1:  Mathematical loss primitives      ehp_sn.loss
Level 2:  Atomic semantic objectives        objectives.supervised, objectives.control
Level 3:  Task adaptation                   objectives.task
Level 4:  Paradigm-level optimization       objectives.composites
```

### 2.1 Level 1 — `ehp_sn.loss` (separate peer package)

Small, domain-independent mathematical functions. Pure tensor operations
with no knowledge of tasks, models, rollouts, or objectives.

| Module                   | Functions                                              | Use                               |
| ------------------------ | ------------------------------------------------------ | --------------------------------- |
| `loss/cross_entropy.py`  | `softmax_cross_entropy`, `stablemax_cross_entropy`     | Observation NLL, token prediction |
| `loss/consistency.py`    | `mse_consistency`, `nll_consistency`, `LatentRelation` | Latent-code consistency           |
| `loss/regularization.py` | `l1_penalty`, `l2_penalty`, `sum_regularization_terms` | Grid/place regularization         |
| `loss/divergences.py`    | `gaussian_kl_divergence`, `sum_gaussian_kl_divergence` | Variational KL terms              |
| `loss/reductions.py`     | `masked_mean`, `masked_sum`, `per_sequence_mean`       | Masked aggregation                |

These return unreduced `(B,)` or `(B, S)` tensors. They never call
`.item()`, `.detach()`, or log anything. See [Loss Design](loss.md) for
the full contract.

### 2.2 Level 2 — Atomic objectives (`supervised/`, `control/`)

Each atomic objective wraps one loss primitive with typed input validation,
masking, and reduction into an `ObjectiveResult`.

```
objects/supervised/token.py      TokenPredictionObjective
objects/supervised/field.py      FieldRegressionObjective
objects/control/halt.py          HaltClassificationObjective
objects/control/q_value.py       QValueObjective
objects/control/state_value.py   StateValueObjective
```

**Atomic objective test:**

> Can this be described by one coherent mathematical term?

If yes, it belongs at Level 2. If it combines task, halting, regularization,
or RL terms, it belongs at Level 4.

**Example — `TokenPredictionObjective`:**

```python
class TokenPredictionObjective(nn.Module):
    """Masked cross-entropy token prediction loss.

    One coherent term: per-element CE, masked by ignore_index,
    normalized per sequence, summed over the batch.
    """

    def forward(self, inputs: TokenObjectiveInput) -> ObjectiveResult:
        ...
```

### 2.3 Level 3 — Task adaptation (`task/`)

This is the **central architectural innovation** of the repository. Task
evaluators adapt heterogeneous task outputs and supervision into a common
task-level result, decoupling composite objectives from task modality.

```
MazeHard output + token labels
        │
        ▼
MazeHardTokenEvaluator          (knows about MazeHardTaskOutput)
        │
        ▼
TaskStepEvaluation ──────────►  ACTSupervisedScorer (task-agnostic)
```

```
Goaltrace field + field target
        │
        ▼
GoaltraceFieldEvaluator         (knows about GoaltraceTaskOutput)
        │
        ▼
TaskStepEvaluation ──────────►  ACTSupervisedScorer (task-agnostic)
```

**Protocol:**

```python
class TaskStepEvaluator(Protocol[TaskOutputT, SupervisionT]):
    """Protocol for task-owned step evaluators.

    Conforming types:
    - Receive one predicted task output and one supervision struct.
    - Return a task-agnostic evaluation containing only loss sums,
      counts, a completion target, and neutral sufficient statistics.

    The ACT scorer does not inspect task_output or supervision —
    only the returned TaskStepEvaluation.
    """

    def evaluate(
        self,
        *,
        task_output: TaskOutputT,
        supervision: SupervisionT,
    ) -> TaskStepEvaluation:
        ...
```

**Concrete evaluators:**

| Evaluator                               | Task      | Modality               | Loss Signal | Completion Signal                |
| --------------------------------------- | --------- | ---------------------- | ----------- | -------------------------------- |
| `GoaltraceFieldEvaluator`               | Goaltrace | Continuous field (B×N) | Masked MSE  | Field-quality sigmoid            |
| `MazeHardTokenEvaluator`                | MazeHard  | Token sequence (B×S×V) | Masked CE   | Exact-match (all tokens correct) |
| `RoutebindTrajectoryBootstrapEvaluator` | Routebind | Continuous field (B×S) | Masked MSE  | Field-quality sigmoid            |
| `SeqMazeTaskEvaluator`                  | SeqMaze   | Token sequence (B×T×V) | Masked CE   | EOS-canonicalized exact-match    |

Each evaluator:

1. Internally delegates to an atomic objective (`TokenPredictionObjective`
   or `FieldRegressionObjective`).
2. Computes a **completion target** (float in [0, 1]) — the model's
   readiness to halt.
3. Packs everything into a `TaskStepEvaluation` with sufficient
   statistics for metric aggregation.

### 2.4 Level 4 — Composite objectives (`composites/`)

Composite objectives define the actual training criterion for a paradigm.
They combine multiple atomic objectives and task evaluations into a single
differentiable scalar.

| Composite             | Regime             | Composition Formula                                       | Entry Point                                 |
| --------------------- | ------------------ | --------------------------------------------------------- | ------------------------------------------- |
| `ACTSupervisedScorer` | ACT (HRM v1)       | `c_task·task_sum + c_halt·halt_BCE + c_continue·cont_BCE` | `evaluate_step(record, *, inputs, context)` |
| `TEMObjective`        | TEM (v1, v2)       | `obs_nll + latent_consistency + reg`                      | `evaluate_step(record, *, inputs, context)` |
| `HybridRLObjective`   | Hybrid RL (HRM v2) | `token_CE + c_v·V_MSE + c_q·Q_MSE`                        | `compute_step(batch, *, context)`           |

**Key distinction:** ACT and TEM conform to the rollout-scored protocol
(called per `StepRecord`). Hybrid RL is a batch-computed objective (called
once per materialized TD(0) batch). This fundamental difference is why
there is no universal `Objective.forward()` base class.

---

## 3. Core contracts

### 3.1 `ObjectiveResult`

The single return type for all atomic objective forward calls and the
embedding type inside composite step results.

```python
@dataclass(frozen=True)
class ObjectiveResult:
    """Differentiable output from a single objective forward call.

    Distinguished fields:

    - ``loss``: scalar tensor for ``.backward()``.
    - ``contributions``: signed, coefficient-weighted scalar contributions
      whose **sum equals ``loss``**.
    - ``terms``: unreduced per-element tensors (may carry gradients).
    - ``diagnostics``: detached observations that do not participate in
      optimization.
    """

    loss: Tensor
    contributions: Mapping[str, Tensor]
    terms: Mapping[str, Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)
```

**Semantics:**

| Field           | Shape                      | Gradient?       | Invariant                                                       |
| --------------- | -------------------------- | --------------- | --------------------------------------------------------------- |
| `loss`          | scalar `()`                | yes             | `loss == sum(contributions.values())`                           |
| `contributions` | scalar per key             | yes             | Each value is signed, weighted, included in `loss`              |
| `terms`         | `(B,)` or `(B, S)` per key | yes (may carry) | Ephemeral — discard after reduction; do not retain across steps |
| `diagnostics`   | scalar or `(B,)` per key   | **no**          | Detached observability only                                     |

**Contributions must sum to loss:**

```python
torch.testing.assert_close(
    result.loss,
    torch.stack(tuple(result.contributions.values())).sum(),
)
```

**Example — ACT:**

```python
contributions = {
    "task": self._c_task * task.task_loss_sum,
    "halt": self._c_halt * halt_per_sample_sum,
    "continue": self._c_continue * continue_per_sample_sum,
}
```

**Example — entropy regularization:**

```python
contributions = {
    "policy": policy_loss,              # positive
    "value": c_value * value_loss,      # positive
    "entropy": -c_entropy * entropy,    # negative (entropy bonus)
}
```

This avoids ambiguous sign conventions. Every contribution is the **actual
signed, weighted value added to `loss`**.

**`terms` lifetime and graph-retention rules:**

`terms` are **ephemeral** values intended for aggregation within the
current backward pass. They must not be:

- retained across steps without explicit `.detach()`;
- accumulated into epoch-level rolling averages while carrying a graph;
- stored in checkpoint or state dictionaries;
- logged directly (use `diagnostics` for detached observability).

Callers that need step-level aggregation should extract
`RatioStat(numerator, denominator)` from terms and discard the
per-element tensors immediately after the forward pass.

**Atomic objective contract:** All atomic objectives return a full
`ObjectiveResult`, even when only one mathematical term is involved.
A single-entry `contributions` map (e.g., `{"token": loss}`) is
**intentional** — it preserves API uniformity so that every objective
consumer can depend on `result.contributions`, `result.terms`, and
`result.diagnostics` without branching on objective type.

**`detached_diagnostics()` helper:**

```python
def detached_diagnostics(self) -> dict[str, Tensor]:
    return {
        name: value.detach()
        for name, value in self.diagnostics.items()
    }
```

**Construction validation:** `ObjectiveResult.__post_init__` enforces
structural invariants at construction time:

```python
def __post_init__(self) -> None:
    if self.loss.ndim != 0:
        raise ValueError(f"loss must be scalar, got shape {self.loss.shape}")
    if not self.contributions:
        raise ValueError("contributions must be non-empty")
    for name, value in self.contributions.items():
        if value.ndim != 0:
            raise ValueError(
                f"contribution {name!r} must be scalar, got shape {value.shape}"
            )
```

**Additivity** (`loss == sum(contributions.values())`) is a **test and
debug invariant**, not enforced at construction time. Tensor equality is
not computed inside `__post_init__` for performance; it is verified by
the test suite (§12).

**Validation policy (compile-safe):** Shape and value-range checks in
`__post_init__` are **Python-side only** — they run at construction time
and are not preserved under `torch.compile`. For production deployment
where construction-time checks are bypassed, tensor invariants may
optionally be asserted via `torch._assert` inside the forward pass.
Debug-mode checks (shape assertions, value-range guards) may be gated
behind `if __debug__:` or a package-level validation flag.

### 3.2 `ObjectiveContext` and regime-specific scoring context

A minimal common context passed to composite objectives at each step.
Objectives must not change their mathematical formula based on stage
(e.g., using a different loss function in validation). Stage is provided
for schedule resolution and diagnostics only.

**Algorithmic conditions** such as warmup, exploration, or coefficient
annealing must be communicated via **resolved scalar fields** in the
regime-specific scoring context, not inferred from stage. For example,
whether value losses are zeroed is an algorithmic decision represented
by a resolved coefficient or an explicit `is_warmup: bool` in the
batch context — not by checking `stage == Stage.TRAIN`.

```python
from enum import StrEnum

class Stage(StrEnum):
    """Execution stage for objective evaluation.

    ``PREDICT`` indicates inference-only execution.  Objectives are
    normally not invoked during prediction — the training regime
    skips objective evaluation when ``stage == PREDICT`` and no
    gradient is requested.  If prediction-time diagnostics are
    required, use a separate diagnostic evaluation path rather than
    relaxing the ``ObjectiveResult`` contract.
    """
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"

@dataclass(frozen=True)
class ObjectiveContext:
    """Per-step context for objective evaluation.

    Attributes:
        global_step: Optimizer step counter (monotonic).
        stage: Execution stage.
    """

    global_step: int
    stage: Stage = Stage.TRAIN
```

Regime-specific resolved context extends or accompanies this:

```python
@dataclass(frozen=True)
class TEMScoringContext:
    """Per-step TEM algorithm parameters, already resolved from schedules."""
    objective: ObjectiveContext
    temperature: float
    p2g_use: float
    g_cell_reg: float
    p_cell_reg: float
```

```python
@dataclass(frozen=True)
class ACTScoringContext:
    """Per-step ACT context. Coefficients here allow future annealing
    without mutating the objective instance."""
    objective: ObjectiveContext
    halt_coefficient: float
    continue_coefficient: float
```

The objective **consumes** resolved values; it does not parse trainer state
or evaluate schedules internally. Schedule resolution belongs in the
training regime (e.g., `resolve_tem_runtime` in `training/tem.py`).

### 3.3 `TaskStepEvaluation`

The bridge between task-specific evaluators and the task-agnostic ACT scorer.
Each evaluator internally normalizes over its task elements (tokens, nodes,
etc.) so that every sample contributes one semantically comparable
per-sample loss value, regardless of modality.

```python
@dataclass(frozen=True)
class TaskStepEvaluation:
    """Task-agnostic evaluation result for one ACT step.

    Attributes:
        task_loss_per_sample: Per-sample task loss, shape ``(B,)``.
            Each evaluator normalizes internally over its task elements
            (tokens for MazeHard/SeqMaze, valid spatial nodes for
            Goaltrace/Routebind) so that every sample contributes a
            single semantically comparable scalar.  This makes task
            loss directly comparable to per-sample halt and continue
            losses under a common denominator.
        completion_target: Per-sample readiness to halt, shape ``(B,)``
            float in ``[0, 1]``. Higher = more ready.
        accuracy_stats: Optional token-level correctness statistics.
        statistics: Task-level sufficient statistics for metric
            aggregation. A metrics adapter converts these into
            ``StepMetrics``; the task evaluator does not import
            metric transport types.
    """

    task_loss_per_sample: Tensor
    completion_target: Tensor
    accuracy_stats: AccuracyStats | None = None
    statistics: Mapping[str, RatioStat] = field(default_factory=dict)

    @property
    def continuation_target(self) -> Tensor:
        """Per-sample continuation signal — derived from completion.

        For tasks where continuation is genuinely independent of
        completion, override this in a subclass or protocol
        implementation.  For all current EHP tasks, continuation
        is ``1.0 - completion_target``.
        """
        return 1.0 - self.completion_target
```

`continuation_target` is a **derived property**, not an independently stored
field. This prevents the invariant violation
`completion_target + continuation_target != 1`. If a future task requires a
genuinely independent continuation target, it should use a separate protocol
or an explicit `independent_continuation_target` field with documented
semantics.

**`task_loss_per_sample` contract:**

Each evaluator returns a per-sample loss, **not** a raw sum over all
elements in the batch. The evaluator is responsible for internal
normalization (e.g., per-sequence token CE mean, per-sample field MSE
mean over valid nodes). The ACT scorer then sums over active samples
and divides by `N_active`:

```
# ACT scorer — all terms are per-sample under a common denominator
task_active_sum = (task_loss_per_sample * active_mask).sum()
halt_active_sum = (halt_loss_per_sample * active_mask).sum()
continue_active_sum = (continue_loss_per_sample * active_mask).sum()

L_ACT = (
    c_task   * task_active_sum
  + c_halt   * halt_active_sum
  + c_continue * continue_active_sum
) / active_mask.sum().clamp_min(1)
```

This makes every coefficient directly comparable: `c_task = 1.0` and
`c_halt = 0.5` refer to the same per-sample scale.

**`RatioStat` and `AccuracyStats` ownership:** `RatioStat`
(a numerator/denominator pair) and `AccuracyStats` (token-level
correctness statistics) are neutral types. Their canonical home is
`ehp_sn.contracts.statistics` — a shared contracts layer that both
`objectives` and `metrics` may depend on but neither owns. They must
not be imported from a metrics transport module.

### 3.4 Step-specific result types (target architecture)

Composite objectives return richer step-level types. `ObjectiveResult`
is the **sole optimization authority**; paradigm-specific step wrappers
carry only the additional information that the training system must
aggregate beyond the optimized scalar.

The target design separates optimization from aggregation:

```python
from typing import Generic, TypeVar

OutputT = TypeVar("OutputT")

@dataclass(frozen=True)
class ACTSupervisedStep(Generic[OutputT]):
    """A single rollout/loss step produced by ACTSupervisedScorer.

    ``objective`` is the canonical optimization result. ``task`` and
    ``control`` carry paradigm-specific structured data for downstream
    metric and trace consumers. Metrics and signals are constructed
    externally by dedicated adapters.

    ``outputs`` carries the adapter-bridge output for this step.
    Its concrete type varies by adapter family; callers that need
    typed access should bind ``OutputT`` to the expected bridge output
    type.  Omit or set to ``None`` when the rollout record already
    owns the outputs.
    """

    objective: ObjectiveResult
    task: TaskStepEvaluation
    control: ACTControlEvaluation
    outputs: OutputT | None = None
```

```python
from typing import Generic, TypeVar

OutputT = TypeVar("OutputT")

@dataclass(frozen=True)
class TEMObjectiveStep(Generic[OutputT]):
    """A single rollout/loss step produced by TEMObjective."""

    objective: ObjectiveResult
    sufficient_statistics: TEMSufficientStatistics
    outputs: OutputT | None = None
```

```python
@dataclass(frozen=True)
class HybridRLObjectiveStep:
    """A single batch step produced by HybridRLObjective."""

    objective: ObjectiveResult
    policy_statistics: PolicyStatistics
```

**Transitional note:** The current codebase stores duplicate `losses`
bundles (`ACTStepLosses`, `TEMLosses`, `HybridRLLosses`), `StepMetrics`,
and `signals` dictionaries inside these step types. These are
**transitional duplication** that will be removed once metrics adapters
are introduced. New code should treat `objective.contributions` as the
authoritative representation of loss components.

`ObjectiveResult` answers "what is optimized." The wrapper answers "what
else must the training system aggregate from this step."

---

## 4. Reduction policy

### 4.1 Reduction utilities

**Mathematical primitives** live in `loss/reductions.py` — pure tensor
operations with no knowledge of objectives, tasks, or optimization
semantics:

```python
# In loss/reductions.py
def masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    """Scalar sum over mask-selected elements."""
    ...

def masked_mean(
    values: Tensor,
    mask: Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Mean over mask-selected elements, with configurable dimension."""
    ...

def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Divide with denominator clamping; returns zero when denominator is zero."""
    ...
```

**Semantic reduction policies** live in `objectives/reductions.py` — they
encode optimization-scale normalization decisions that affect gradient
magnitude and coefficient interpretation:

```python
# In objectives/reductions.py
def per_sequence_mean_then_batch_sum(
    values: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Per-sample mean over valid positions, then batch sum."""
    ...

def reduce_active_slots(
    per_sample: Tensor,
    active: Tensor,
) -> Tensor:
    """Sum over active samples only (zero contribution from inactive)."""
    ...

def normalize_per_eligible_sample(
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    """Masked mean per eligible entry — gradient magnitude independent of count."""
    ...
```

The split is:

- `loss.reductions` = tensor mathematics;
- `objectives.reductions` = optimization-scale semantics.

### 4.2 Normalization policies and declared scale semantics

Every objective declares one of two **scale semantics** that determine how
coefficient values are interpreted and whether the loss scales with batch
size:

| Scale policy   | Invariant                                   | Coefficient interpretation                                     |
| -------------- | ------------------------------------------- | -------------------------------------------------------------- |
| **Mean-scale** | `loss(duplicate(batch)) == loss(batch)`     | Coefficient controls relative weight independent of batch size |
| **Sum-scale**  | `loss(duplicate(batch)) == 2 × loss(batch)` | Effective weight grows linearly with batch; LR must compensate |

**Per-family normalization:**

| Objective         | Scale      | Mask source                           | Normalization                                                  | Empty-mask            |
| ----------------- | ---------- | ------------------------------------- | -------------------------------------------------------------- | --------------------- |
| TEM               | Mean       | `protocol_mask` (revisit eligibility) | Masked mean over eligible entries                              | `clamp_min(1)` → zero |
| ACT task          | Per-sample | Per-sample evaluator normalization    | Per-sample mean over task elements, then sum active / N_active | Zero contribution     |
| ACT halt          | Per-sample | Halt mask (optional)                  | `sum(BCE) / N_active` — per-sample, common denominator         | Zero when all masked  |
| ACT continue      | Per-sample | Continuation mask (optional)          | `sum(BCE) / N_active` — per-sample, common denominator         | Zero when all masked  |
| Hybrid RL / Token | Sum        | `ignore_index=-100`                   | Per-sequence mean, then batch sum                              | Zero contribution     |
| Field             | Sum        | `mask (B,N)` bool                     | Per-sample mean over valid nodes, then sum                     | Zero contribution     |
| Q-Value           | Sum        | None                                  | Raw MSE sum                                                    | N/A                   |
| State-Value       | Sum        | None                                  | Raw MSE sum                                                    | N/A                   |

**ACT denominator coherence:** The **target design** unifies all ACT
terms under a single common denominator so that coefficient interpretation
is independent of batch composition:

```
# Target ACT formula — all terms share one per-sample denominator
L_ACT = reduce_active_slots(
    c_task   * task_loss_per_sample
  + c_halt   * halt_loss_per_sample
  + c_continue * continue_loss_per_sample,
    active_mask
) / active_mask.sum().clamp_min(1)
```

The evaluator returns per-sample task losses (§3.3), making task loss
directly comparable to per-sample halt and continue losses. Every
coefficient refers to the same per-sample scale. The current
implementation's mixed-denominator policy (task with its own element
count, halt/continue with raw sums) is migration debt and must be
resolved before the design is considered normative.

### 4.3 Design invariants

**Mask invariance:** Adding invalid padded elements does not change the
objective value.

**Scale semantics:** Every objective declares one of two scale policies
(see §4.2). Replication tests must use the declared policy:

| Policy         | Invariant                                   | Test                                                  |
| -------------- | ------------------------------------------- | ----------------------------------------------------- |
| **Mean-scale** | `loss(duplicate(batch)) == loss(batch)`     | `assert_close(loss(batch), loss(batch_expanded))`     |
| **Sum-scale**  | `loss(duplicate(batch)) == 2 × loss(batch)` | `assert_close(2 * loss(batch), loss(batch_expanded))` |

There is no single package-wide replication invariant — each objective's
declared scale semantics determines the correct test.

**Empty-mask policy:** Every atomic objective has explicit behavior for
zero valid elements — returns zero gradient-contribution or raises a
domain-specific exception. No accidental `NaN`.

---

## 5. Protocols, not a universal base class

There is **no** generic `Objective.forward(predictions, targets, context)`
base class. The repository has two distinct execution modes that justify
separate protocols.

### Rollout-scored protocol

```python
class RolloutScorer(Protocol[InputT, StepT]):
    """Protocol for objectives scored per executed StepRecord."""

    def evaluate_step(
        self,
        record: StepRecord,
        *,
        inputs: InputT,
        context: ObjectiveContext,
    ) -> StepT:
        ...
```

Conformers: `ACTSupervisedScorer`, `TEMObjective`.

### Batch-computed protocol

```python
class BatchObjective(Protocol[BatchT, StepT]):
    """Protocol for objectives computed on a materialized batch."""

    def compute_step(
        self,
        batch: BatchT,
        *,
        context: ObjectiveContext,
    ) -> StepT:
        ...
```

Current conformer: `HybridRLObjective`. Introduce the protocol only when
a second batch-computed objective appears.

### Rule

> Introduce a protocol when multiple consumers need to depend on a shared
> behavior, not merely because several classes are called "objectives."

---

## 6. Configuration

Objective configuration lives close to the composite objective definition.
**The objective formula** (coefficients, loss function choice, normalization
mode) and **schedule specifications** (anneal horizons, warmup steps) are
distinct concerns:

```python
# ── Objective formula (owned by the composite objective) ──────────────────

@dataclass(frozen=True)
class ACTObjectiveConfig:
    """Coefficients that define the ACT optimization formula."""
    task_coefficient: float = 1.0
    halt_coefficient: float = 0.5
    continue_coefficient: float = 0.01

@dataclass(frozen=True)
class TEMObjectiveConfig:
    """Coefficients and loss-function choices for the TEM formula.

    Schedule horizons (``*_it`` fields) are **not** objective formula
    parameters — they belong in ``TEMScheduleConfig``.
    """
    observation_loss: LossType = "softmax_cross_entropy"
    c_obs: float = 1.0
    latent_loss: str = "mse_consistency"
    c_grid: float = 1.0
    c_place: float = 1.0
    grid_reg_norm: RegularizationNorm = "l2"
    c_grid_reg: float = 0.01
    place_reg_norm: RegularizationNorm = "l1"
    c_place_reg: float = 0.02

@dataclass(frozen=True)
class HybridRLObjectiveConfig:
    """Coefficients for the hybrid RL formula."""
    token_loss: LossType = "stablemax_cross_entropy"
    c_state_value: float = 0.5
    c_q_value: float = 0.5

# ── Schedule specifications (owned by the training regime) ────────────────

@dataclass(frozen=True)
class TEMScheduleConfig:
    """Anneal horizons for TEM temperature and regularization.

    These are schedule parameters, not objective formula parameters.
    The training regime evaluates them into ``TEMScoringContext``
    before calling the objective.
    """
    temp_it: int = 2000          # anneal horizon for temperature
    p2g_use_it: int = 0          # anneal horizon for place→sensory gate
    p2g_scale: float = 200.0
    g_reg_it: int = 40_000_000   # anneal horizon for grid reg
    p_reg_it: int = 4000         # anneal horizon for place reg
```

**Rule:** The composite configuration may reference schedule specifications,
but schedule evaluation belongs to the training regime. The objective
consumes already-resolved scalar values via its scoring context.

Runtime resolution transforms schedule-bearing config into current objective
context:

```python
resolved = resolve_tem_runtime(schedule_config, global_step=context.global_step)
objective.evaluate_step(record, inputs=inputs, context=resolved)
```

---

## 7. Metrics boundary

The clean target architecture separates metric construction from objective
computation:

```
objective
    │
    ▼
ObjectiveResult + step-specific result
    │
    ├──► metrics adapter (e.g., TaskMetricsAdapter.from_step) ──► StepMetrics
    │
    └──► diagnostics ──► logging / observability
```

**Target rule (normative):** Composite objectives must not import
`StepMetrics`, metric key constants, or metric transport types. Task
evaluators must not construct `StepMetrics` or reference metric routing
tables. Metric construction is the responsibility of dedicated adapters
that consume `ObjectiveResult` and paradigm-specific step data.

**Migration state (temporary):** The current codebase violates this
boundary: composite objectives directly construct `StepMetrics` objects,
and task evaluators carry `metrics` and `task_extras` fields that feed
into metric transport. The `TaskStepEvaluation` target contract (§3.3)
replaces these with `statistics: Mapping[str, RatioStat]`. This
violation is **acknowledged technical debt** to be resolved after result
and reduction semantics are standardized. It does not reflect the
normative target architecture.

**Principle:** Objectives own differentiable quantities. Metrics own metric
keys and aggregation. Logging owns external names and sinks.

---

## 8. Builder and registry policy

### Current practice (appropriate at scale)

Objectives are constructed inline in Lightning modules:

```python
self.act_scorer = ACTSupervisedScorer(
    c_task=scorer_config.task_loss_coefficient,
    c_halt=scorer_config.halt_loss_coefficient,
    c_continue=scorer_config.continue_loss_coefficient,
)
objective = TEMObjective(self._component_configs.objective)
```

### When to introduce a builder

A typed builder (`build_objective`) should be added when:

1. Configuration selects objective type dynamically.
2. Five or more composite objective families exist.
3. CLI or experiment specs need exhaustive objective construction.
4. Lightning modules duplicate construction logic.

Then use a **typed discriminated union**, not a string registry:

```python
ObjectiveConfig = ACTObjectiveConfig | TEMObjectiveConfig | HybridRLObjectiveConfig

def build_objective(config: ObjectiveConfig) -> nn.Module:
    match config:
        case ACTObjectiveConfig():
            return ACTSupervisedScorer(config)
        case TEMObjectiveConfig():
            return TEMObjective(config)
        case HybridRLObjectiveConfig():
            return HybridRLObjective(config)
```

Prefer this over a string registry until external extensibility is
genuinely required.

---

## 9. Trainer-side usage

```python
# 1. Forward pass through model and runtime
predictions = model(batch.inputs)

# 2. Evaluate via composite objective
step = scorer.evaluate_step(
    record,
    inputs=scoring_input,
    context=ObjectiveContext(
        global_step=trainer.global_step,
        stage=Stage.TRAIN,
    ),
)

# 3. Extract loss and contributions
loss = step.objective.loss
contributions = step.objective.contributions

# 4. Log diagnostics (detached)
for name, value in step.objective.detached_diagnostics().items():
    logger.log_metric(f"objective/{name}", value)

# 5. Construct metrics externally via adapter
step_metrics = act_step_metrics_from(step, detach=True)
```

Clean data flow:

```
batch
  │
  ├── inputs ───────────────► model/runtime ──► predictions
  │                                             │
  ├── supervision ──────────────────────────────┼──► task evaluator
  │                                             │       │
  ├── control logits ───────────────────────────┼──► composite objective
  │                                             │       │
  └── metric targets ───────────────────────────┼──► metrics
                                                        │
objective ──► ObjectiveResult(loss, contributions, terms, diagnostics)
```

---

## 10. Public API

### Root exports (narrow and architectural)

```python
from ehp_sn.objectives import (
    ObjectiveContext,
    ObjectiveResult,
    TaskStepEvaluation,
    TaskStepEvaluator,
)
```

```python
__all__ = [
    "ObjectiveContext",
    "ObjectiveResult",
    "TaskStepEvaluation",
    "TaskStepEvaluator",
]
```

### Submodule imports (paradigm implementations)

```python
from ehp_sn.objectives.composites.act import (
    ACTSupervisedScorer,
    ACTSupervisedScorerConfig,
    ACTSupervisedStep,
)
from ehp_sn.objectives.composites.tem import (
    TEMObjective,
    TEMObjectiveConfig,
    TEMObjectiveStep,
)
from ehp_sn.objectives.composites.hybrid_rl import (
    HybridRLObjective,
    HybridRLObjectiveConfig,
    HybridRLObjectiveStep,
)
from ehp_sn.objectives.task.goaltrace import GoaltraceFieldEvaluator
from ehp_sn.objectives.task.mazehard import MazeHardTokenEvaluator
from ehp_sn.objectives.task.routebind import RoutebindTrajectoryBootstrapEvaluator
from ehp_sn.objectives.task.seqmaze import SeqMazeTaskEvaluator
```

Do **not** root-export:

- Every low-level loss function (`masked_mean`, `softmax_cross_entropy`, etc.)
- Every private input dataclass (`TokenObjectiveInput`, `HaltObjectiveInput`)
- Internal aggregation records
- Metric key constants
- Rollout helpers

---

## 11. Implementations at a glance

### `ACTSupervisedScorer`

```
ACT total = reduce_active_slots(
    c_task   * task_loss_per_sample
  + c_halt   * halt_loss_per_sample
  + c_continue * continue_loss_per_sample,
    active_mask
) / active_mask.sum().clamp_min(1)

contributions = {
    "task":   c_task   * task_active_sum / N_active,
    "halt":   c_halt   * halt_active_sum / N_active,
    "continue": c_continue * continue_active_sum / N_active,
}
```

- Task-agnostic: receives pre-computed `TaskStepEvaluation` with
  per-sample task losses.
- Halt via `HaltClassificationObjective` (BCE with logits) — per-sample.
- Continue via `F.binary_cross_entropy_with_logits` — per-sample.
- All terms share one common denominator (`N_active`).
- Used in: Goaltrace, MazeHard, Routebind, SeqMaze (all via task evaluator bridge).

### `TEMObjective`

```
TEM total = obs_nll_sum + latent_consistency_sum + reg_sum

contributions:
    obs_nll ...... obs_post + obs_recall + obs_path
    latent ....... place_consistency + grid_kl
    reg .......... grid_l2 + place_l1
```

- Observation NLL via configured CE variant (`softmax` or `stablemax`).
- Latent consistency via `mse_consistency` (default) or `nll_consistency`.
- Grid regularization: L2. Place regularization: L1.
- Temperature, p2g_use, and regularization weights are scheduled via
  resolved `TEMScoringContext`.

### `HybridRLObjective`

```
Hybrid RL total = token_CE_sum + c_v × V_MSE_sum + c_q × Q_MSE_sum

contributions:
    token ....... token_CE (per-sequence mean, then sum)
    state_value . c_state_value * V_MSE
    q_value ..... c_q_value * Q_MSE
```

- Batch-computed, not rollout-scored.
- Value losses are zeroed during warmup (controlled by an explicit
  `is_warmup` flag or resolved coefficient in the batch context,
  never inferred from `ObjectiveContext.stage`).
- Used in: MazeHard HRM v2 training path.

---

## 12. Testing requirements

### Contract tests

```python
def test_objective_result_contract():
    result = ...
    assert result.loss.ndim == 0
    assert result.loss.is_floating_point()
    assert_close(result.loss, sum(result.contributions.values()))
```

### Additivity

```python
assert_close(result.loss, torch.stack(tuple(result.contributions.values())).sum())
```

### Mask invariance

```python
# Appending masked padding must not change loss
loss(original) == loss(padded_with_invalid_steps)
```

### Batch replication (scale-dependent)

Replication tests must use the objective's declared scale policy
(see §4.2):

```python
# Mean-scale objectives
assert_close(loss(batch), loss(batch_expanded))

# Sum-scale objectives
assert_close(2 * loss(batch), loss(batch_expanded))
```

There is no single package-wide replication invariant — each objective's
declared `BatchReduction` determines the correct test.

### Empty-mask behavior

Every atomic objective has explicit empty-mask policy:

- Return differentiable zero.
- Or raise `ValueError`.

No accidental `NaN`.

### Gradient ownership

```python
result.loss.backward()
assert predictions.logits.grad is not None
assert supervision.labels.grad is None
```

### Numerical reference tests

Hand-computed small tensors against known expected values for:

- Token CE (with `ignore_index`)
- Field MSE (with mask)
- Halt BCE
- Consistency losses
- KL divergence

### Serialization

```python
state = objective.state_dict()
restored.load_state_dict(state)
assert_close(restored(inputs).loss, objective(inputs).loss)
```

### Compile and autocast

```python
torch.compile(objective)
with torch.autocast(device_type="cuda"):
    result = objective(inputs)
```

---

## 13. Summary of design decisions

| Decision              | Choice                                                         | Rationale                                                                                                                          |
| --------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Base class?           | **No universal base**                                          | Two execution modes (rollout-scored, batch-computed) have genuinely different signatures                                           |
| Result type           | `ObjectiveResult` with `contributions`, `terms`, `diagnostics` | Enforces `loss == sum(contributions)`, separates gradient from diagnostics                                                         |
| Task bridge           | `TaskStepEvaluator` protocol + per-task evaluators             | Decouples ACT scorer from task modality                                                                                            |
| Primitives location   | `loss/` as separate peer package                               | Also consumed by `training/elbo.py` and analysis code                                                                              |
| Schedule resolution   | External (training regime), not in objective                   | Objective consumes resolved scalars; schedule logic is independently testable                                                      |
| Metrics boundary      | **Target — external adapters**                                 | Current direct `StepMetrics` construction is temporary migration debt; normative boundary: objectives own only diffable quantities |
| Builder               | **Not yet** — inline construction                              | Only 3 composites; factory adds indirection without solving a real problem                                                         |
| Registry              | **Not unless needed**                                          | Typed discriminated union when a builder is introduced                                                                             |
| Coefficient schedules | Via passed resolved context                                    | No `WeightSchedule` protocol yet; add when ACT needs annealing                                                                     |

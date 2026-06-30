# Controller Architecture

> Canonical design for `ehc_sn.controllers` — one-step, task-agnostic control
> transitions over recurrent model execution.

## 1. Architectural definition

Within EHP, a **controller** performs one logical control transition:

\[
(C\_{n+1}, O_n) = \operatorname{step}(C_n, B_n, \Omega)
\]

where:

| Symbol       | Meaning                                                               | Repository type                                       |
| ------------ | --------------------------------------------------------------------- | ----------------------------------------------------- |
| \(C_n\)      | Controller carry — recurrent model state + slot-local execution state | `RolloutState` or a subtype                           |
| \(B_n\)      | Current batch or admitted slot data                                   | Per-family input type (e.g. `dict[str, Tensor]`)      |
| \(\Omega\)   | Explicit execution context                                            | Per-family typed context (e.g. `DeliberationContext`) |
| \(O_n\)      | Typed controller-family output                                        | `QHaltingInteractionRecord`, `ReplayStepOutput`, etc. |
| \(C\_{n+1}\) | Next controller carry                                                 | Same type as \(C_n\)                                  |

The outer **runner** owns repeated invocation:

```
while not runner_stop_condition:
    carry, output = controller.step(carry, batch, context)
```

Therefore:

> The controller owns the semantics of **one control transition**.
> The runner owns iteration, collection, and global stopping.

### One-transition contract

```python
BatchT = TypeVar("BatchT", contravariant=True)
CarryT = TypeVar("CarryT")
ContextT = TypeVar("ContextT")
OutputT = TypeVar("OutputT", covariant=True)

@runtime_checkable
class StepController(Protocol[BatchT, CarryT, ContextT, OutputT]):
    """Protocol satisfied by all EHP controllers."""

    def initial_state(self, batch_sample: BatchT) -> CarryT:
        """Build the initial carry from a sample batch.

        Must be deterministic, explicit, and free from global device inference.
        The sample batch defines batch size, device, and dtype.
        """
        ...

    def step(
        self,
        state: CarryT,
        batch: BatchT,
        context: ContextT,
    ) -> tuple[CarryT, OutputT]:
        """Execute one controller transition.

        - Batch width is preserved.
        - Carry tensors remain on compatible devices.
        - Halted (non-active) slots retain their prior state.
        - Newly admitted slots are reset before backbone execution.
        - Output fields correspond to the same transition as the returned carry.
        - No optimization or persistence side effects occur.
        """
        ...
```

Execution context is typed per controller family rather than passed as opaque
keyword arguments:

```python
@dataclass(frozen=True)
class DeliberationContext:
    """Execution context for deliberation controllers."""
    allow_halt: bool = True
    explore: bool = True
    halt_action: int | None = None
    max_halt_steps: int | None = None

@dataclass(frozen=True)
class ReplayContext:
    """Execution context for replay controllers."""
    allow_halt: bool = True
```

This protocol is consumed by `RecurrentRunner` in `ehp_sn.rollouts.runtime`.
The runner uses a structural `StepController[Any, Any, Any, Any]` to remain
generic; typed callers provide the correct context.

---

## 2. Package scope

`ehp_sn.controllers` owns **control policies over recurrent model execution**.
A controller coordinates:

```
slot admission/reset  →  backbone transition  →  control decision  →  next carry
```

### 2.1 Controllers own

| Responsibility                | Examples                                                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Slot-local control state**  | Recurrent model carry, active/admitted status, step counters, replay cursor, trajectory identity, deliberation halt status                |
| **Backbone invocation**       | Reset selected recurrent states, prepare resident slot batch, invoke backbone once, interpret policy/critic/halt/task outputs             |
| **Control decisions**         | ACT continue/halt, Q-based continue/halt, environment action selection, replay cursor advancement, slot replacement signalling            |
| **Controller-family outputs** | Typed records consumed by objectives, training batch builders, traces, rollout records, evaluation                                        |
| **Algorithmic invariants**    | Masked updates for halted slots, monotonic step counters, replay cursor bounds, action mask enforcement, terminated/truncated propagation |

### 2.2 Controllers do not own

| Responsibility                         | Owner                                                              |
| -------------------------------------- | ------------------------------------------------------------------ |
| Outer execution loop                   | `ehp_sn.rollouts` (`RecurrentRunner`)                              |
| Losses and optimization                | `ehp_sn.objectives`, `ehp_sn.training`, `ehp_sn.lightning`         |
| Concrete task semantics                | Protocols in `ehp_sn.contracts`; implementations in `ehp_sn.tasks` |
| Dataset loading and data iteration     | Data modules, rollout sources                                      |
| Artifact persistence and visualization | `ehp_sn.traces`, `ehp_sn.eval`, analysis, figures                  |
| Experiment-level construction          | Experiment scripts, Lightning modules                              |

A controller may produce policy logits, values, halt probabilities, rewards, or
bootstrap state. It should **not** compute the final training loss.

---

## 3. Two time axes

EHP distinguishes two time axes in control:

| Axis  | Meaning                      | Convention                           |
| ----- | ---------------------------- | ------------------------------------ |
| \(t\) | Environment or physical time | One env transition per step          |
| \(k\) | Internal deliberation time   | Multiple internal steps per env step |

The runtime advances \(t\). A deliberation controller advances \(k\).

```
t:  0 ───── env_step ──── 1 ───── env_step ──── 2 ── ...

     │                     │
     ▼                     ▼
k:  0 → 1 → 2 → halt      0 → 1 → halt
    (deliberation)         (deliberation)
```

This separation is critical for ACT and prospective computation. A controller
may perform or describe internal deliberation, but it should not silently
become the owner of the entire environment rollout.

---

## 4. Backbone ownership: model/controller composition

The repository follows a **model/controller composition** architecture:

```
Controller
├── Backbone (model)
│   ├── recurrent representation
│   ├── task head
│   ├── policy head
│   └── critic head
├── Policy decision utility
└── Runtime/environment dependency
```

**The backbone owns trainable parameters** (representation, policy, value, task
heads). **The controller owns invocation, reset, decision interpretation, and
state progression.**

This is an intentional repository-specific design. It supports reusing the same
backbone under different control regimes:

```
same HRM backbone
    ├── ACTController              (halt/continue from Q-logits)
    └── DeliberationQHaltingController  (value-control with TaskRuntime)
```

The boundary is:

> The **backbone computes decision variables**. The **controller applies
> decision semantics**.

| Backbone computes    | Controller applies                     |
| -------------------- | -------------------------------------- |
| Halt/continue logits | Done masks via threshold + exploration |
| Policy logits        | Action sampling or greedy selection    |
| Value estimate       | Recording into interaction record      |

---

## 5. Package structure

```
src/ehp_sn/controllers/
├── __init__.py              # Public API — stable re-exports of controller classes
├── _base.py                 # BaseController — shared slot lifecycle infrastructure
├── base.py                  # StepController protocol (re-export from rollouts)
├── state.py                 # RolloutState, carries, snapshots
├── records.py               # QHaltingInteractionRecord, ReplayStepOutput, etc.
├── policies.py              # CategoricalPolicy, halting utilities
├── factory.py               # Advanced: configuration-driven construction utility
├── deliberation/
│   ├── __init__.py           # Re-exports ACTController, DeliberationQHaltingController
│   ├── act.py                # ACTController — adaptive computation time
│   └── q_halting.py          # DeliberationQHaltingController — runtime-backed value control
├── online/
│   ├── __init__.py           # Re-exports RLController
│   └── actor_critic.py       # RLController — online RL with EnvBase
└── replay/
    ├── __init__.py           # Re-exports ReplayTrajectoryController
    └── trajectory.py         # ReplayTrajectoryController — carry-owned stepwise replay
```

### 5.1 `base.py`

Defines the minimal structural controller interface and reusable transition
mechanics.

```python
BatchT = TypeVar("BatchT", contravariant=True)
CarryT = TypeVar("CarryT")
ContextT = TypeVar("ContextT")
OutputT = TypeVar("OutputT", covariant=True)

@runtime_checkable
class StepController(Protocol[BatchT, CarryT, ContextT, OutputT]):
    """One-transition controller protocol."""
    def initial_state(self, batch_sample: BatchT) -> CarryT: ...
    def step(self, state: CarryT, batch: BatchT, context: ContextT) -> tuple[CarryT, OutputT]: ...
```

Shared concrete base with invariant slot machinery:

```python
class BaseController(Generic[ModelStateT]):
    """Slot lifecycle and backbone-access infrastructure.  Implementation
    infrastructure — not part of the stable public API.

    Owns exactly:
    - slot data refresh (halted → admitted)
    - backbone state reset delegation
    - step counter advancement
    """
    def refresh_slot_data(...) -> dict[str, Tensor]: ...
    def reset_model_state(...) -> ModelStateT: ...
    def advance_steps(...) -> Tensor: ...
```

Good candidates for the base: masked state reset, slot data replacement,
step counter updates, carry validation, model-state delegation.

Bad candidates (belong in concrete families): ACT threshold logic, policy
sampling, replay cursor progression, TD bootstrap construction.

### 5.2 `state.py`

Owns carry types used during execution.

```python
@dataclass(kw_only=True)
class RolloutState(Generic[ModelStateT]):
    """Base per-slot carry across controller families.

    Fields:
        model_state: Backbone recurrent state.
        data: Per-slot input/label buffers, shape (B, ...).
        static_data: Optional rollout-scoped metadata.
        needs_admission: Bool (B,) — True means this slot requires
            fresh data on the next step (halted/admission boundary).
        rollout_steps: Int32 (B,) — steps since last admission.
    """
    model_state: ModelStateT
    data: dict[str, Tensor]
    static_data: dict[str, object] | None
    needs_admission: Tensor  # (B,) bool
    rollout_steps: Tensor    # (B,) int32 — steps since admission
```

Specialised subtypes add paradigm-specific fields:

```python
@dataclass(kw_only=True)
class ACTRolloutState(RolloutState[ModelStateT]):
    deliberation_halted: Tensor  # (B,) bool — k-time halt
    deliberation_steps: Tensor   # (B,) int64 — k-time steps

@dataclass(kw_only=True)
class ReplayRolloutState(RolloutState[ModelStateT]):
    cursor: Tensor             # (B,) int64 — current step within trajectory
    trajectory_length: Tensor  # (B,) int64 — effective length
    trajectory_id: Tensor      # (B,) int64 — stable identity, -1 = not admitted
    resident_payload: dict[str, Tensor]  # carry-owned trajectory arrays
    task_state: dict[str, Tensor]       # task-owned replay-local state
```

**State semantics convention:**

| Field                 | Meaning                                              | Scope                   |
| --------------------- | ---------------------------------------------------- | ----------------------- |
| `needs_admission`     | Slot requires fresh data (halted/admission boundary) | Universal               |
| `rollout_steps`       | Steps since last admission                           | Universal               |
| `deliberation_halted` | Internal (k-time) deliberation complete              | ACT                     |
| `deliberation_steps`  | Internal deliberation steps used                     | ACT                     |
| `terminated`          | Environment terminal state                           | Environment interaction |
| `truncated`           | Runtime limit reached                                | Environment interaction |

The three lifecycle axes — slot, deliberation, environment — should remain
conceptually distinct even when they coincide in practice.

### 5.3 `records.py`

Owns immutable outputs describing what one controller transition produced.

```python
@dataclass(frozen=True)
class QHaltingInteractionRecord(DetachMixin):
    """One-step actor-critic interaction record.

    Controller-owned.  Consumed by objectives, training batch builders,
    traces, and evaluation.

    ``done`` is a derived convenience property; the canonical signals
    are ``terminated`` and ``truncated``.
    """
    action: Tensor              # (B,) sampled action indices
    policy_logits: Tensor       # (B, A) actor-head logits
    log_prob: Tensor            # (B,) action log-probability
    entropy: Tensor             # (B,) policy entropy
    value: Tensor               # (B, 1) critic state value estimate
    reward: Tensor              # (B, 1) task-finalized reward
    terminated: Tensor          # (B,) bool — episode terminal
    truncated: Tensor           # (B,) bool — episode truncated
    q_values: Tensor | None = None  # (B, A) value scores (Q-halting families)
    task_output: object | None = None  # opaque task-side model output

    @property
    def done(self) -> Tensor:
        """Combined halt signal (terminal or truncated)."""
        return self.terminated | self.truncated

@dataclass(frozen=True)
class ACTControllerStepOutput(DetachMixin):
    """One-step ACT controller output."""
    backbone_output: ACTBackboneOutput
    done_action: int

@dataclass(frozen=True)
class ReplayStepOutput(DetachMixin):
    """One-step replay controller output."""
    backbone_output: Any
    cursor: Tensor  # (B,) int64
```

These are controller-owned even when training or objectives consume them.
Consumption by another layer does not transfer ownership. The test:

> Does this type describe the semantic output of a controller transition,
> or is it a neutral protocol that several layers independently implement?

If it describes a controller transition, it belongs under `controllers`.

### 5.4 `policies.py`

Stateless or lightly stateful decision policies used by controllers.

```python
class CategoricalPolicy:
    """Sample actions from logits with valid-action mask support."""
    def __call__(
        self,
        policy_input: PolicyInput,
        *,
        explore: bool = True,
    ) -> PolicyDecision: ...

@dataclass(frozen=True)
class CategoricalPolicyInput:
    """Minimal input for categorical action sampling over logits."""
    logits: Tensor                  # (B, A)
    valid_action_mask: Tensor | None = None  # (B, A) bool — None = all valid

@dataclass(frozen=True)
class PolicyDecision:
    """Action-selection result returned by a policy."""
    action: Tensor              # (B,) or (B, 1)
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
```

Policies are algorithmic controller components, not complete controllers.
Policy heads containing trainable parameters belong to the backbone.

### 5.5 Construction

Controllers are constructed explicitly — each family accepts only the
dependencies it genuinely requires:

```python
# ACT — self-contained, no external runtime
controller = ACTController(backbone=backbone, config=config)

# Deliberation Q-halting — depends on a TaskRuntime
controller = DeliberationQHaltingController(
    backbone=backbone,
    config=config,
    runtime=task_runtime,
)

# Online RL — depends on an environment and adapter
controller = RLController(
    backbone=backbone,
    env=env,
    config=config,
    runtime=adapter,
)

# Replay — depends on a trajectory runtime
controller = ReplayTrajectoryController(
    backbone=backbone,
    config=config,
    runtime=replay_runtime,
)
```

A `factory.py` module exists for configuration-driven construction from
resolved specifications. It is an **advanced/experimental** utility, not
part of the stable public API. Explicit constructors remain the preferred
pattern in experiment scripts.

### 5.6 Submodule `__init__.py` files

Each submodule re-exports its stable surface:

```python
# deliberation/__init__.py
from ehc_sn.controllers.deliberation.act import ACTController, ACTControllerConfig
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingController,
    DeliberationQHaltingControllerConfig,
)

__all__ = [
    "ACTController",
    "ACTControllerConfig",
    "DeliberationQHaltingController",
    "DeliberationQHaltingControllerConfig",
]
```

---

## 6. Controller families

### 6.1 ACT deliberation

```
File: deliberation/act.py
Backbone protocol: ACTRolloutBackbone
Task boundary: none (halt logic is self-contained)
Primary axis: k (internal deliberation time)
```

**Responsibilities:**

- Invoke the ACT-compatible backbone
- Compute halt/continue scores via Q-logit collapse
- Apply exploration-aware halt flipping
- Enforce maximum deliberation budget
- Freeze halted slots
- Return next carry and halt diagnostics

**Does not compute:** ponder loss, any objective.

**Invariants:**

- `deliberation_steps` is monotonic until reset
- `deliberation_halted` is sticky until admission/reset
- Halted samples do not continue updating
- Halt decisions are based only on current transition outputs and configured rules

### 6.2 Deliberation Q-halting

```
File: deliberation/q_halting.py
Backbone protocol: QHaltingRolloutBackbone
Task boundary: TaskRuntime[RuntimeStateT] (injected)
Primary axis: k (internal deliberation steps) → t (one env transition at halt)
```

**Responsibilities:**

- Invoke the Q-halting backbone
- Select continue/halt action via ACT collapse or policy
- Advance the abstract task runtime
- Collect reward and terminal signals
- Emit structured interaction record

**Does not own:** concrete task semantics — depends on `TaskRuntime` protocol.

**Invariants:**

- Continue/halt actions are derived from valid decision logits
- Task runtime advances once per active slot per step
- `terminated` and `truncated` remain distinguishable
- Reward and bootstrap state refer to the same transition

### 6.3 Online actor-critic (RL)

```
File: online/actor_critic.py
Class: RLController  (also known as ActorCriticController in documentation)
Alias: RLControllerConfig  (also known as ActorCriticControllerConfig)
Backbone protocol: QHaltingRolloutBackbone
Task boundary: EnvBase + TaskEnvironmentAdapter
Primary axis: t (environment time)
```

The class name `RLController` is retained for backward compatibility.
New documentation and code may refer to it as `ActorCriticController`.
Both names refer to the same implementation — a single online RL
controller using TorchRL `EnvBase`.

**Responsibilities:**

- Obtain controller inputs through a task-environment adapter
- Invoke the actor-critic backbone
- Sample or select an action via policy
- Step the TorchRL environment abstraction
- Update online carry (including `env_td`)
- Expose policy/value/reward/termination data

**Invariants:**

- Actions satisfy action-space or mask constraints
- Policy logits, sampled action, log-probability, and value correspond to the same state
- Deterministic evaluation is explicit
- Environment transition occurs once per active slot per step

### 6.4 Replay trajectory

```
File: replay/trajectory.py
Backbone protocol: RolloutBackbone (generic)
Task boundary: ReplayTrajectoryRuntime
Primary axis: replay cursor progression
```

**Responsibilities:**

- Admit a trajectory into an available slot
- Reset model and task state for admitted slots
- Invoke the backbone for the current replay position
- Advance cursor by exactly one step
- Detect trajectory completion (cursor ≥ trajectory_length)
- Preserve resident trajectory payload as sole continuity authority
- Return replay-specific records

**Invariants:**

- Cursor advances at most once per step
- Cursor never exceeds resident trajectory length
- Completed trajectories request replacement (halted=True)
- Resident payload is stable until replacement
- Active slots read exclusively from carry-owned `resident_payload`, never from the incoming source batch

---

## 7. Contract placement

The repository distinguishes three kinds of type across architectural boundaries:

| Kind                                        | Location                                                 | Examples                                                                                   |
| ------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Repository-level dependency protocols**   | `ehp_sn.contracts`                                       | `TaskRuntime`, `TaskEnvironmentAdapter`, `OnlineBootstrapCarry`, `ReplayTrajectoryRuntime` |
| **Controller-owned carry and output types** | `ehp_sn.controllers.state`, `ehp_sn.controllers.records` | `RolloutState`, `QHaltingInteractionRecord`, `ReplayStepOutput`                            |
| **Backbone-owned output protocols**         | Model/binding layer                                      | `QHaltingBackboneOutput`, `ACTRolloutBackbone`, `ActorCriticBackboneOutput`                |

### Dependency graph

```
ehp_sn.contracts                    ← neutral protocols only
    ↑
ehp_sn.controllers                  ← controller carry, records, policies
    ↑
ehp_sn.rollouts                     ← RecurrentRunner, StepRecord, sources
    ↑
ehp_sn.objectives / ehp_sn.training  ← loss construction, bootstrap targets
    ↑
ehp_sn.lightning / ehp_sn.eval       ← training loops, evaluation
```

**Rule:** `ehp_sn.contracts` must not import from `ehp_sn.controllers`.
Cross-layer protocols like `OnlineBootstrapCarry` belong at the `contracts`
level, not under `controllers/contracts/`.

### Current migration status

| Protocol                                      | Current location                        | Target location                            | Status     |
| --------------------------------------------- | --------------------------------------- | ------------------------------------------ | ---------- |
| `TaskRuntime`, `RuntimeReset`, `StepFeedback` | `contracts/task_runtime.py`             | `contracts/`                               | ✅ Done    |
| `TaskEnvironmentAdapter`                      | `contracts/task_environment.py`         | `contracts/` (transitional, may disappear) | ✅ Done    |
| `OnlineBootstrapCarry`                        | `controllers/contracts/actor_critic.py` | `contracts/`                               | 🔄 Migrate |
| `QHaltingInteractionRecord`                   | `controllers/contracts/actor_critic.py` | `controllers/records.py`                   | ✅ Owner   |
| `QHaltingBackboneOutput`                      | `controllers/contracts/actor_critic.py` | Model contract layer                       | 🔄 Migrate |

---

## 8. Public API

The package root exposes the stable controller classes and the structural
protocol that callers need for typing:

```python
from ehc_sn.controllers import (
    StepController,
)

from ehc_sn.controllers.deliberation import (
    ACTController,
    ACTControllerConfig,
    DeliberationQHaltingController,
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.controllers.online import (
    RLController,
    RLControllerConfig,
)
from ehc_sn.controllers.replay import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)

__all__ = [
    "ACTController",
    "ACTControllerConfig",
    "DeliberationQHaltingController",
    "DeliberationQHaltingControllerConfig",
    "ReplayTrajectoryController",
    "ReplayTrajectoryControllerConfig",
    "RLController",
    "RLControllerConfig",
    "StepController",
]
```

The package root should expose **capabilities** (controller classes, their
configs, and the protocol they satisfy). It should not expose implementation
infrastructure that callers rarely need to import directly.

Do not root-export:

| Not exported              | Import from                           | Reason                                        |
| ------------------------- | ------------------------------------- | --------------------------------------------- |
| `BaseController`          | `ehc_sn.controllers._base`            | Implementation infrastructure, not a contract |
| `RolloutState`            | `ehc_sn.controllers.state`            | Carry type; advanced callers only             |
| `build_controller`        | `ehc_sn.controllers.factory`          | Experimental; explicit constructors preferred |
| Every interaction record  | `ehc_sn.controllers.records`          | Consumed by objectives, not by callers        |
| Every concrete carry type | `ehc_sn.controllers.state`            | Advanced callers import from defining module  |
| Policy utility classes    | `ehc_sn.controllers.policies`         | Internal algorithmic components               |
| Internal masking helpers  | `ehc_sn.controllers.deliberation.act` | Implementation detail                         |
| Backbone output protocols | Model/binding layer                   | Not controller-owned                          |

---

## 9. Controller invariants

### Universal

- `step()` performs exactly one logical controller transition.
- Batch width is preserved.
- Carry tensors remain on compatible devices.
- Slots not selected for update retain their prior state.
- Newly admitted slots are reset before backbone execution.
- Output fields correspond to the same transition as the returned carry.
- No optimization or persistence side effects occur.

### ACT

- `deliberation_steps` is monotonic until reset.
- `deliberation_halted` is sticky until admission/reset.
- Halted samples do not continue updating.
- Halt decisions are based only on current transition outputs and configured rules.

### Deliberation Q-halting

- Continue/halt actions are derived from valid decision logits.
- Task runtime advances once per active slot.
- `terminated` and `truncated` remain distinguishable.
- Reward and bootstrap state refer to the same transition.

### Online actor-critic

- Actions satisfy action-space or mask constraints.
- Policy logits, sampled action, log-probability, and value correspond to the same state.
- Deterministic evaluation is explicit.
- Environment transition occurs once per active slot.

### Replay trajectory

- Cursor advances at most once per step.
- Cursor never exceeds resident trajectory length.
- Completed trajectories request replacement.
- Resident payload is stable until replacement.

---

## 10. Controller construction

Concrete constructors remain explicit — each family accepts only the
dependencies it genuinely requires:

```python
# ACT — self-contained, no external runtime
controller = ACTController(backbone=backbone, config=config)

# Deliberation Q-halting — depends on a TaskRuntime
controller = DeliberationQHaltingController(
    backbone=backbone,
    config=config,
    runtime=task_runtime,
)

# Online RL — depends on an environment and adapter
controller = RLController(
    backbone=backbone,
    env=env,
    config=config,
    runtime=adapter,
)

# Replay — depends on a trajectory runtime
controller = ReplayTrajectoryController(
    backbone=backbone,
    config=config,
    runtime=replay_runtime,
)
```

A `factory.py` module exists for configuration-driven construction from
resolved specifications. It is an advanced/experimental utility; explicit
constructors remain the preferred pattern in experiment scripts.

---

## 11. Summary

| Property               | Value                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------ |
| Package purpose        | One-step, task-agnostic control transitions over recurrent model execution           |
| Canonical operation    | `carry, output = controller.step(state, batch, context)`                             |
| Backbone relationship  | Controller invokes backbone; backbone owns parameters, controller owns orchestration |
| Task boundary          | Protocols in `ehp_sn.contracts`, not concrete task classes                           |
| State model            | `RolloutState[ModelStateT]` base with family-specific extensions                     |
| Outputs                | Typed per-family records (`QHaltingInteractionRecord`, `ReplayStepOutput`, etc.)     |
| Model/controller reuse | Same backbone, different controllers (ACT ↔ Q-halting)                               |
| Dependencies           | Explicit per constructor; factory available as advanced utility                      |
| Testing invariant      | One transition = one `step()` call; runner owns iteration                            |

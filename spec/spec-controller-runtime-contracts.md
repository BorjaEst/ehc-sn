# EHC-SN Controller Runtime Contracts Specification

> Non-required companion spec for controller-to-learner, runtime, and
> family-specific execution contracts above the model step surface.

## 1 Scope

This spec owns detailed controller, learner, and runtime execution contracts
that sit above the model step and adapter surfaces. Generic model and adapter
patterns remain in `spec/spec-model-interfaces.md`.

### 1.1 Runner Snapshot Contract

The canonical runner-owned per-step snapshot surface lives in
`ehc_sn.rollouts.runtime`:

- `CarrySnapshot`: frozen post-step projection stored on `StepRecord.snapshot`
  and `ObservedStep.snapshot`.
- `final_carry`: authoritative controller-owned continuity state returned at
  the end of execution.
- `StepRecord.executed_frame`: the exact tensors consumed by the model,
  objective, and diagnostics on this step. For replay controllers this is the
  carry-owned step slice, independent of the source batch.
- `StepRecord.sampled_input`: what the source proposed; the raw batch from the
  source before the controller processed it.
- `StepRecord.batch`: backward-compatible alias; always contains
  `executed_frame` content (not `sampled_input`).

Rules:

- Carry owns continuity. Snapshots are lean projections of that continuity for
  learners and observers; they are not a second source of replay truth.
- Snapshot fields may expose current-step data and lightweight slot-local
  continuity facts (`halted`, `steps`, model/runtime state, or small static
  metadata) when downstream consumers need them.
- Runner snapshots must not duplicate source-owned full replay rows or other
  full `(B, T, ...)` tensors. If a value is source-owned or time-major enough
  to bloat per-step records, it stays in the source batch or an out-of-band
  join path, not in carry. `resident_payload` is explicitly excluded from
  `CarrySnapshot` for this reason.
- Observability must not widen the carry contract by default. When traces,
  figures, or diagnostics need source-owned context, they should rejoin it from
  batch/source metadata or task-owned artifacts rather than smuggling it into
  every snapshot.
- Family-specific snapshot protocols are projections of this runner snapshot
  contract and must expose only the fields their consumer actually needs.
- Objectives and traces must read `executed_frame` (not `sampled_input` or the
  generic snapshot) when they need step-truth data. `record.batch` is a
  backward-compatible alias for `executed_frame`.

---

## 2 Neutral Actor-Critic Contracts

The actor-critic interaction record and TD(0) batch assembly are neutral
controller-to-learner contracts, not online-RL-specific types.

### 2.1 Canonical Ownership

The canonical definitions live in `ehc_sn.controllers.contracts.actor_critic`.

- `ActorCriticInteractionRecord`: per-step interaction snapshot emitted by any
  actor-critic controller.
- `ActorCriticRolloutBackbone`: backbone protocol with task payload, policy
  output, and required critic output.
- `ActorCriticBackboneOutput`: named backbone output; `critic` is mandatory,
  never `None`.
- `ActorCriticPolicyOutput`: policy head payload.
- `ActorCriticCriticOutput`: value head payload.
- `ActorCriticExecutionSnapshot`: minimal learner-side projection of the
  runner snapshot containing `steps` and `halted`.
- `OnlineBootstrapCarry`: that projection extended with `model_state` for TD(0)
  bootstrap.
- `OnlineBootstrapRuntime`: minimal runtime seam exposing
  `extract_next_step_obs(carry) -> Batch`.

All code must import from the canonical leaf modules directly.

Canonical family import paths:

- ACT deliberation: `ehc_sn.controllers.deliberation.act`
- AC deliberation: `ehc_sn.controllers.deliberation.actor_critic`
- Replay: `ehc_sn.controllers.replay.trajectory`
- Online RL: `ehc_sn.controllers.online.actor_critic`
- AC contracts: `ehc_sn.controllers.contracts.actor_critic`

### 2.2 TD(0) Assembly Split

`ehc_sn.training.actor_critic.TD0ActorCriticBatchBuilder` separates three
responsibilities:

- `assemble_batch(record, snapshot, bootstrap_value)`: pure batch math over
  `ActorCriticInteractionRecord`, `ActorCriticExecutionSnapshot`, and a
  pre-computed `bootstrap_value` tensor. No RL runtime types.
- `compute_bootstrap_value(carry)`: online-specific bootstrap acquisition using
  `OnlineBootstrapRuntime` and `OnlineBootstrapCarry`.
- `build_ac_batch(record, carry)`: orchestration helper that computes the
  bootstrap value and then assembles the batch.

### 2.3 Interaction Record Surface

The canonical reference shape is:

```python
@dataclass
class ActorCriticInteractionRecord(DetachMixin):
    observation_used_for_decision: dict[str, Tensor]
    policy_logits: Tensor
    sampled_action: Tensor
    reward: Tensor
    done: Tensor
    terminated: Tensor
    truncated: Tensor
    value_estimate: Tensor
    policy_decision: PolicyDecision
    task_output: object | None = None
```

Rules:

- Every actor-critic backbone must return a non-`None` critic output.
- Controllers must snapshot `observation_used_for_decision` from the exact
  batch passed to the backbone, not reconstruct it from carry.
- Learners receive `ActorCriticInteractionRecord` directly and compute TD
  targets and advantages before calling objectives.
- Objectives must not reach into controller-internal data structures through
  the record.
- `policy_logits` are action-selection logits, not Q-values.
- `value_estimate` is the critic state value, not a Q-logit.
- `reward` is the executed step reward after any task-runtime finalization.
- `sampled_action` and `policy_decision.log_prob` must stay consistent.

---

## 3 Online RL Training Contract — Learner-Owned Batches

Pure reward-first RL training (HRM v2 path) uses a learner-owned batch-assembly
model. The rollout-scoring objective path (`RLLossHead`) has been removed;
there are no runtime consumers.

### 3.1 RL Training Path

```text
source -> controller.step() -> ActorCriticInteractionRecord
-> Learner.build_ac_batch()       (online orchestrator: bootstrap + assemble)
  -> compute_bootstrap_value()   (online-specific: requires RL carry + runtime)
  -> assemble_batch()            (generic TD(0) assembler — no RL types)
-> HybridRLLossHead.compute_step(batch) -> loss, metrics, signals
```

### 3.2 Learner Contract

The learner owns:

- bootstrap value computation via `V(s_{t+1})` in a no-grad context;
- TD(0) return: `return = reward + gamma * bootstrap_value * (1 - done)`;
- advantage: `(return - value_estimate).detach()`;
- `HybridActorCriticBatch` assembly from `ActorCriticInteractionRecord`.

The learner does not own optimizer stepping or scheduler stepping.

### 3.3 Pure Batch Loss Contract

`HybridRLLossHead.compute_step(batch)` is a pure function over a fully
materialized `HybridActorCriticBatch`. It does not access controller internals,
compute bootstrap values, or reconstruct advantages internally.

### 3.4 Validation Path

Validation uses `ZeroBootstrapActorCriticValidationScorer` from
`ehc_sn.training.actor_critic`. It adapts each `ActorCriticInteractionRecord`
via an injected `HybridActorCriticTaskBinding` into a zero-bootstrap
`HybridActorCriticBatch` and delegates to `HybridRLLossHead.compute_step`.

Zero bootstrap is conservative and correct for evaluation: the focus is token
accuracy and episode metrics, not precise TD target estimation. RL loss numbers
during validation are approximate diagnostic values only.

The `HybridActorCriticTaskBinding` protocol owns task-specific extraction of
token logits and labels from `ActorCriticInteractionRecord`.

### 3.5 Online RL Task-Runtime Seam

Online env runtimes remain responsible for env-step shaping and next-
observation extraction only. Task-owned runtime helpers may finalize reward-
bearing transition metadata after the environment kernel has stepped.

```python
class RLTaskRuntime(Protocol):
    def build_reset_td(self, batch: Batch) -> TensorDictBase: ...

    def build_env_step_td(
        self,
        env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase: ...

    def finalize_env_transition(
        self,
        previous_env_td: TensorDictBase,
        next_env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase: ...

    def extract_next_step_obs(self, carry: OnlineBootstrapCarry) -> Batch: ...
```

Rules:

- `RLTaskRuntime` is the seam for live-environment actor-critic rollouts only.
  Deliberation-family controllers without an environment use
  `DeliberationStepFinalizer` instead.
- `build_env_step_td` prepares the concrete env input for the current step,
  including any partial-reset merging required by halted slots.
- `finalize_env_transition` may attach task-owned reward or diagnostic fields
  after `env.step()` returns.
- `extract_next_step_obs` returns the task-shaped bootstrap observation batch
  from the post-step carry when learner-side TD targets require a next-step
  value estimate.
- Environment kernels remain mechanical stepping layers. When reward semantics
  depend on task labels, scores, or decoded task outputs, they must live in the
  task runtime rather than in `envs/`.

---

## 4 Deliberation Actor-Critic Contract

### 4.1 Ownership Boundaries

- Semantic episode horizon is task-owned via `DeliberationStepFinalizer` as
  `episode_horizon`.
- Learned-halt termination is task-owned via `DeliberationStepFinalizer` as
  `terminated`.
- Semantic truncation is task-owned via `DeliberationStepFinalizer` as
  `truncated`.
- Validation-time execution safety caps remain runtime-owned via
  `RuntimeConfig.max_rollout_steps` or `RuntimeConfig.hard_max_rollout_steps`.

### 4.2 DeliberationStepFinalizer Contract

`DeliberationStepFinalizer` owns all reward, `terminated`, and `truncated`
signals for deliberation tasks.

```python
@dataclass(frozen=True)
class DeliberationStepResult:
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    next_runtime_state: object | None = None

class DeliberationStepFinalizer(Protocol):
    def finalize_step(
        self,
        data: Batch,
        task_output: object,
        action: Tensor,
        steps: Tensor,
        runtime_state: object | None,
    ) -> DeliberationStepResult: ...
```

Rules:

- `terminated` is set when the model emits the task-defined halt action.
- `truncated` is set when `steps >= task_config.episode_horizon`.
- The controller must not reconstruct or override these signals via its own
  budget field.
- `DeliberationStepFinalizer` implementations live in task-owned or
  adapter-owned layers, not in `controllers/`.
- The finalizer receives only `data`, `task_output`, `action`, `steps`, and
  `runtime_state`; it must not access controller internals.
- `next_runtime_state` is a lightweight carry threaded per-step through
  `DeliberationACRolloutState.runtime_state`. Use `None` for stateless
  finalizers.
- `DeliberationStepFinalizer` is orthogonal to `RLTaskRuntime`; the two must
  not be mixed in the same controller.

### 4.3 `allow_halt` Semantics

`allow_halt=False` means suppress learned-halt termination. It does not mean
ignore task-owned semantic truncation.

```text
allow_halt=True:   terminated = finalizer.terminated | truncated = finalizer.truncated
allow_halt=False:  terminated = zeros                | truncated = finalizer.truncated
```

`done = terminated | truncated` in both cases.

### 4.4 Controller Does Not Own Semantic Budget

`DeliberationACControllerConfig` must not expose a `max_steps` or equivalent
public field for semantic budget enforcement. Any semantic truncation must be
owned by the task via the finalizer.

`TD0ActorCriticBatchBuilder` handles the training path and is injected with the
same `HybridActorCriticTaskBinding`. These remain generic capability helpers in
`training/`, not HRM-version-specific classes.

---

## 5 TEM Family Step-Output Contract

`TEMStepOutput` is a TEM-family contract, not a neutral replay-variational
protocol. Its field names (`logits_inference`, `logits_retrieved`,
`logits_ancestral`, `GRID_TRANSITION_RELATION`,
`PLACE_TRANSITION_RELATION`) are intentionally TEM-shaped.

`TEMStepOutput` is the core protocol consumed by `TEMObjective`. It covers only
the fields required for ELBO-style loss computation: `latent_relations`,
`reg_terms`, `logits_inference`, `logits_retrieved`, and
`logits_ancestral`.

`theta_cls` is not part of the core contract. It is an optional PFC/CLS
diagnostic surface defined in the separate `TEMThetaClsCapable` protocol.
`TEMObjective.compute_signals` detects the capability via
`getattr(outputs, "theta_cls", None)` and emits `THETA_CLS_NORM` only when a
non-`None` tensor is present.

Rules:

- Bridge adapters that do not expose `theta_cls` need not define the attribute
  at all.
- Bridge adapters must not add a fake `theta_cls = None` property purely to
  satisfy the core contract.
- Bridge adapters that do expose `theta_cls` satisfy both `TEMStepOutput` and
  `TEMThetaClsCapable` structurally; no explicit base class is required.
- `THETA_CLS_NORM` in `metrics/signals.py` is unchanged; only the detection
  mechanism changed.

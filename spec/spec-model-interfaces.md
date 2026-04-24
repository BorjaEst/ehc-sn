# EHC-SN Model Interfaces Specification

> Non-required companion spec for detailed state, step, and adapter interface
> patterns.

## 1 Scope

This spec documents detailed interface patterns for pure models and canonical
adapters. Core ownership rules remain in `spec/spec-architecture.md`.

---

## 2 State Management

Each recurrent model should define an explicit dataclass for its recurrent
state.

Rules:

- State is passed into and returned from the public model step surface.
- No hidden recurrent state is stored in module attributes between calls.
- State dataclasses should provide `detach()` when TBPTT-style truncation is
  required.
- Composed models may nest sub-states.

---

## 3 Model Step Surface

The preferred model-native public step surface is:

```python
def step(self, payload: ModelInput, state: ModelState) -> tuple[ModelOutput, ModelState]:
    ...
```

Rules:

- `ModelInput` is model-native, not a raw task batch.
- `ModelOutput` is architecture-native, not task-decoded output.
- `ModelOutput` may be purely predictive and need not expose a direct action-
  selection head.
- Goal-directed action selection may live above the model in controllers,
  policies, adapters, or other attached execution layers.
- Rollout controllers and objectives live above the model step surface and
  are not part of the model-native public API.
- `init_state()` and `reset_state()` are part of the stable public surface when
  the model is recurrent.
- Backbone `__call__` and bridge `forward` surfaces return `(output, next_state)`
  — output first, state second. This is the canonical backbone seam ordering.

---

## 4 Adapter Surface

The preferred adapter surface is explicit about every task-to-model
transformation.

```python
class ModelTaskAdapter(Protocol):
    def prepare_inputs(self, task_obs: object) -> object: ...
    def step(self, task_obs: object, state: object) -> tuple[object, object]: ...
    def compute_loss(self, model_outputs: object, task_targets: object) -> object: ...
    def postprocess(self, model_outputs: object) -> object: ...
```

Not every adapter needs every method, but all task-to-model transformations
must be explicit in the public surface.

Bridge adapters must treat `postprocess` as the stable output-decoding verb;
`prepare_outputs` is not a canonical public name. The stable public bridge
surface is the task-family barrel plus adapter members `config`, `model`,
`init_state()`, `reset_state()`, `prepare_inputs()`, `postprocess()`, and
`forward()`. Encoder/decoder submodules, builder helpers, and other bridge
composition internals are private unless explicitly re-exported from the family
barrel. Objectives and controllers should depend on capability protocols for
the fields they consume rather than on concrete family bridge-output dataclasses
when the concrete type is not semantically required.

---

## 5 Replay Controller Surface

Replay controllers operate on source-provided replay rows and current-step
state, not on full trajectories stored in carry.

Preferred replay-carry pattern:

```python
@dataclass
class ReplayCarry:
    model_state: object
    cursor: Tensor
    trajectory_length: Tensor
    halted: Tensor
    data: dict[str, Tensor]
```

Rules:

- Replay sources own full replay rows and any full batch-major replay tensors.
- Replay carry owns only recurrent state, cursor-local stop facts, and the
  current-step task payload.
- Replay carry must not retain full `(B, T, ...)` trajectory tensors or other
  full replay batches that would be cloned into rollout records on every step.
- Replay runtime batches are batch-major: `B` is the leading axis and `T` is
  the second axis.
- Current-step replay payloads are derived from the source batch plus the
  current cursor. Step records should snapshot only current-step data.

---

## 6 Online RL Training Contract — Learner-Owned Batches

Pure reward-first RL training (HRM v2 path) uses a learner-owned batch
assembly model. The rollout-scoring objective path (`RLLossHead`) has been
removed; there are no runtime consumers.

### 6.1 RL Training Path

```
source -> controller.step() -> ActorCriticInteractionRecord
-> Learner.build_ac_batch()       (online orchestrator: bootstrap + assemble)
  -> compute_bootstrap_value()   (online-specific: requires RL carry + runtime)
  -> assemble_batch()            (generic TD(0) assembler — no RL types)
-> HybridRLLossHead.compute_step(batch) -> loss, metrics, signals
```

### 6.2 Learner Contract

The learner owns:

- bootstrap value computation via `V(s_{t+1})` (backbone, no-grad context)
- TD(0) return: `return = reward + gamma * bootstrap_value * (1 - done)`
- advantage: `(return - value_estimate).detach()`
- `HybridActorCriticBatch` assembly from `ActorCriticInteractionRecord`

The learner does **not** own optimizer stepping or scheduler stepping.

### 6.3 Pure Batch Loss Contract

`HybridRLLossHead.compute_step(batch)` is a pure function over a fully
materialized `HybridActorCriticBatch`. It does not access controller internals,
compute bootstrap values, or reconstruct advantages internally.

### 6.4 Validation Path

Validation uses `ZeroBootstrapActorCriticValidationScorer` from
`ehc_sn.training.actor_critic`. It adapts each `ActorCriticInteractionRecord`
(via an injected `HybridActorCriticTaskBinding`) into a zero-bootstrap
`HybridActorCriticBatch` and delegates to `HybridRLLossHead.compute_step`.

Zero bootstrap is conservative and correct for evaluation — the focus is token
accuracy and episode metrics, not precise TD target estimation. All RL loss
numbers during validation are approximate diagnostic values only.

The `HybridActorCriticTaskBinding` protocol owns task-specific extraction
(token logits, labels) from `ActorCriticInteractionRecord`. The concrete
implementation for MazeHard+HRM is `MazeHardHRMV2HybridTaskBinding` in the
adapter layer.

---

## 7 Deliberation Actor-Critic Contract

### 7.1 Ownership Boundaries

| Concept                           | Owner                              | Field name                                     |
| --------------------------------- | ---------------------------------- | ---------------------------------------------- |
| Semantic episode horizon          | Task (`DeliberationStepFinalizer`) | `episode_horizon`                              |
| Learned-halt termination          | Task (`DeliberationStepFinalizer`) | `terminated`                                   |
| Semantic truncation               | Task (`DeliberationStepFinalizer`) | `truncated`                                    |
| Execution safety cap (validation) | Runtime (`RuntimeConfig`)          | `max_rollout_steps` / `hard_max_rollout_steps` |

### 7.2 DeliberationStepFinalizer Contract

`DeliberationStepFinalizer` owns **all** of reward, `terminated`, and
`truncated` for deliberation tasks. Specifically:

- `terminated`: set when the model emits the task-defined halt action (learned
  halt signal).
- `truncated`: set when `steps >= task_config.episode_horizon` (task-owned
  semantic horizon).

The controller must not reconstruct or override these signals via its own
budget field.

### 7.3 DeliberationACController `allow_halt` Semantics

`allow_halt=False` means **"suppress learned-halt termination"** — i.e., zero
out the finalizer's `terminated` signal. It does **not** mean "ignore
task-owned semantic truncation". The `truncated` signal from the finalizer
always passes through, regardless of `allow_halt`.

```
allow_halt=True:   terminated = finalizer.terminated | truncated = finalizer.truncated
allow_halt=False:  terminated = zeros              | truncated = finalizer.truncated
```

`done = terminated | truncated` in both cases.

### 7.4 Controller Does Not Own Semantic Budget

`DeliberationACControllerConfig` must **not** expose a `max_steps` or
equivalent public field for semantic budget enforcement. Any semantic truncation
must be owned by the task via the finalizer.

`TD0ActorCriticBatchBuilder` handles the training path, also injected with
the same `HybridActorCriticTaskBinding`. These are generic capability helpers
in `training/` — not HRM-version-specific classes.

---

## 8 Actor-Critic Interaction Record and Learner Contract

The actor-critic interaction record and TD(0) batch assembly are
**neutral controller-to-learner contracts**, not online-RL-specific types.

### 8.1 Neutral Ownership

The canonical definitions live in `ehc_sn.controllers.contracts.actor_critic`:

| Type                           | Role                                                                 |
| ------------------------------ | -------------------------------------------------------------------- |
| `ActorCriticInteractionRecord` | Per-step interaction snapshot emitted by any actor-critic controller |
| `ActorCriticRolloutBackbone`   | Backbone protocol (task + policy + **required** critic output)       |
| `ActorCriticBackboneOutput`    | Named backbone output — `critic` is mandatory, never `None`          |
| `ActorCriticPolicyOutput`      | Policy head payload                                                  |
| `ActorCriticCriticOutput`      | Value head payload                                                   |
| `ActorCriticExecutionSnapshot` | Minimal learner-side snapshot: `steps` + `halted`                    |
| `OnlineBootstrapCarry`         | Extends snapshot with `model_state` for TD(0) bootstrap              |
| `OnlineBootstrapRuntime`       | Minimal runtime seam: `extract_next_step_obs(carry) -> Batch`        |

All code must import from the canonical leaf modules directly.

Canonical family import paths:

| Family           | Canonical path                                 |
| ---------------- | ---------------------------------------------- |
| ACT deliberation | `ehc_sn.controllers.deliberation.act`          |
| AC deliberation  | `ehc_sn.controllers.deliberation.actor_critic` |
| Replay           | `ehc_sn.controllers.replay.trajectory`         |
| Online RL        | `ehc_sn.controllers.online.actor_critic`       |
| AC contracts     | `ehc_sn.controllers.contracts.actor_critic`    |

### 8.2 Generic TD(0) Assembler vs Online Bootstrap Acquisition

These two responsibilities are explicitly separated in
`ehc_sn.training.actor_critic.TD0ActorCriticBatchBuilder`:

- **Generic assembler** — `assemble_batch(record, snapshot, bootstrap_value)`:  
  Pure batch math. Depends only on `ActorCriticInteractionRecord`,
  `ActorCriticExecutionSnapshot`, and a pre-computed `bootstrap_value` tensor.
  No RL runtime types.

- **Online bootstrap acquisition** — `compute_bootstrap_value(carry)`:  
  Online-specific. Requires an `OnlineBootstrapRuntime` (set at construction)
  and an `OnlineBootstrapCarry` (post-step controller state). Both Protocols
  are canonically defined in `ehc_sn.controllers.contracts.actor_critic`.

- **Online orchestrator** — `build_ac_batch(record, carry)`:  
  Calls `compute_bootstrap_value` then `assemble_batch`. Entry point used by
  the Lightning training loop for online RL.

### 8.3 Online RL Task-Runtime Seam (unchanged)

The `ActorCriticInteractionRecord` contract (field set, field semantics, rules)
is reproduced below for reference. The canonical definition is in
`ehc_sn.controllers.contracts.actor_critic`.

```python
@dataclass
class ActorCriticInteractionRecord(DetachMixin):
    observation_used_for_decision: dict[str, Tensor]  # exact batch passed to backbone
    policy_logits: Tensor                              # (B, A) policy logits from actor head
    sampled_action: Tensor                             # (B,) sampled action indices
    reward: Tensor                                     # (B, 1) task-finalized reward for the step
    done: Tensor                                       # (B,) combined termination flag
    terminated: Tensor                                 # (B,) episode terminated
    truncated: Tensor                                  # (B,) episode truncated
    value_estimate: Tensor                             # (B, 1) critic state value
    policy_decision: PolicyDecision                    # rollout-time log_prob, entropy
    task_output: object | None = None                  # optional task-side model output
```

### 8.4 Record Field Rules

- **Backbone contract**: every actor-critic backbone **must** return a non-None
  `critic` output. `ActorCriticBackboneOutput.critic` is `ActorCriticCriticOutput`,
  not optional. Backbones without a value head violate the contract.
- The controller must snapshot `observation_used_for_decision` from the exact
  `data` dict used in the backbone forward pass, not reconstructed from carry.
- Learners receive `ActorCriticInteractionRecord` directly and are responsible
  for computing TD targets and advantage estimates before calling objectives.
- Objectives must not access controller-internal data structures through the
  record.
- `policy_logits` are the actor logits used for action selection, not Q-values.
- `value_estimate` is the critic's state value, not a Q-logit.
- `reward` is the executed step reward after any task-runtime finalization.
- `sampled_action` and `policy_decision.log_prob` are consistent: `log_prob`
  is always computed under the policy distribution for `sampled_action`.

### 8.5 Online RL Task-Runtime Seam

Online env runtimes remain responsible for env-step shaping and
next-observation extraction only. Task-owned runtime helpers may finalize
reward-bearing transition metadata after the environment kernel has stepped.

> **Scope boundary**: `RLTaskRuntime` is the seam for _live-environment_ actor-critic
> rollouts only. Deliberation-family controllers without an environment use
> `DeliberationStepFinalizer` (§ 8.6) instead. The two seams must not be mixed.

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

### 8.6 Deliberation Step-Finalizer Seam

Deliberation actor-critic controllers without an environment delegate reward
and termination production to an injected `DeliberationStepFinalizer`.

```python
@dataclass(frozen=True)
class DeliberationStepResult:
    reward: Tensor           # (B, 1) — task reward for this step
    terminated: Tensor       # (B,)   — per-slot termination flag
    truncated: Tensor        # (B,)   — per-slot truncation flag
    next_runtime_state: object | None = None  # lightweight carry

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

- `DeliberationStepFinalizer` implementations live in task-owned or
  adapter-owned layers, not in `controllers/`.
- The finalizer must not access controller-internal data structures; it receives
  only `data`, `task_output`, `action`, `steps`, and `runtime_state`.
- `next_runtime_state` is a lightweight carry threaded per-step through
  `DeliberationACRolloutState.runtime_state`. Use `None` for stateless
  finalizers.
- When `DeliberationACController` receives `allow_halt=False`, the finalizer's
  `terminated` signal (learned-halt) is suppressed (zeroed). The finalizer's
  `truncated` signal (task-owned semantic horizon) always passes through
  unchanged, regardless of `allow_halt`. See §7.3 for the canonical
  `allow_halt` semantics table.
- `DeliberationACControllerConfig` owns no semantic budget field. There is no
  `max_steps` on the controller config. All truncation is task-owned via the
  finalizer's `truncated` output.
- `DeliberationStepFinalizer` is orthogonal to `RLTaskRuntime`; the two must
  not be mixed in the same controller.

### 8.7 TEM Step-Output Contract

> **Scope note.** `TEMStepOutputs` is a **TEM-family contract**, not a neutral
> replay-variational protocol. The field names (`logits_inference`,
> `logits_retrieved`, `logits_ancestral`, `GRID_TRANSITION_RELATION`,
> `PLACE_TRANSITION_RELATION`) are intentionally TEM-shaped. A generic
> replay-variational protocol must wait until a second concrete consumer (beyond
> TEM) exists. EHC is design pressure only — it is not sufficient evidence to
> extract a neutral protocol now.

`TEMStepOutputs` is the **core** protocol consumed by `TEMLossHead`. It covers
only the fields required for ELBO-style loss computation: `latent_relations`,
`reg_terms`, `logits_inference`, `logits_retrieved`, and `logits_ancestral`.

`theta_cls` is **not** part of the core contract. It is an optional
PFC/CLS diagnostic surface defined in the separate `TEMThetaClsCapable`
protocol. `TEMLossHead.compute_signals` detects the capability via
`getattr(outputs, "theta_cls", None)` and emits `THETA_CLS_NORM` only when a
non-`None` tensor is present.

Rules:

- Bridge adapters that do not expose `theta_cls` need not define the attribute
  at all; they must not add a fake `theta_cls = None` property purely to
  satisfy the core contract.
- Bridge adapters that do expose `theta_cls` satisfy both `TEMStepOutputs` and
  `TEMThetaClsCapable` structurally; no explicit base class is required.
- `THETA_CLS_NORM` in `metrics/signals.py` is unchanged; only the detection
  mechanism (protocol vs. `getattr`) changed.

---

## 7 Module Reuse Protocol

Module configs use stable field names across all models that share that module.

Rules:

- Shared module fields use the same Pydantic type in every model that embeds
  the module.
- Model-specific fields may vary but must not collide with shared module field
  names.
- Pre-trained module weights should be loadable from a parent checkpoint by
  parameter-name prefix when standalone and composite models share the same
  module contract.

Example optional checkpoint fields:

```python
hpc_checkpoint: Optional[Path] = None
pfc_checkpoint: Optional[Path] = None
```

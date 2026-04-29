# EHC-SN Model Interfaces Specification

> Non-required companion spec for generic state, step, adapter, replay-carry,
> and module-reuse interface patterns.

## 1 Scope

This spec documents generic interface patterns for pure models, canonical
adapters, and replay controllers. Core ownership rules remain in
`spec/spec-architecture.md`. Detailed controller-to-learner and family-specific
runtime contracts live in `spec/spec-controller-runtime-contracts.md`.

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

Detailed controller/runtime execution contracts are intentionally split into
`spec/spec-controller-runtime-contracts.md` so this companion stays focused on
model-facing interfaces.

---

## 6 Module Reuse Protocol

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

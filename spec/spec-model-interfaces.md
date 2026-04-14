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
def step(self, payload: ModelInput, state: ModelState) -> tuple[ModelState, ModelOutput]:
    ...
```

Rules:

- `ModelInput` is model-native, not a raw task batch.
- `ModelOutput` is architecture-native, not task-decoded output.
- `init_state()` and `reset_state()` are part of the stable public surface when
  the model is recurrent.

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

---

## 5 Module Reuse Protocol

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

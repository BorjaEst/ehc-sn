# Model Design Contract

> Complete parameterized architectures and their model-local contracts.

An **architecture** defines a parameterized neural computation with a stable
transition contract:

```
output, next_state = model(inputs, state)
```

`ehp_sn.models` owns the neural architectures, their configuration, their
recurrent state, and their typed input/output contracts. It does not own
loss computation, training loops, rollout execution, task adaptation, or
infrastructure coupling.

---

## 1. Package responsibility

### 1.1 Owns

| Responsibility                             | Examples                                                             |
| ------------------------------------------ | -------------------------------------------------------------------- |
| Complete model architectures               | `TEMModelV1`, `TEMModelV2`, `HRMModelV1`, `HRMModelV2`, `EHPModelV1` |
| Architecture configuration                 | `TEMSettingsV1`, `HRMSettingsV2`, `EHPSettingsV1`                    |
| Model inputs, outputs, recurrent state     | `TEMInputV1`, `TEMOutputV1`, `TEMStateV1`                            |
| Composition of reusable neural modules     | Wiring `LECModel` → projection → `HPCAttractor` → output             |
| Canonical parameter initialisation         | `reset_parameters()`                                                 |
| Architecture-level runtime dynamics        | `set_runtime()`, `finalize_memory()` (TEM)                           |
| Model construction from validated settings | `TEMSettingsV1.model_validate(...)` → `TEMModelV1(...)`              |
| Stable model identifiers                   | Family-qualified class names                                         |
| Model-local observability                  | `trace_views()`                                                      |
| State-dict compatibility                   | Version-aware parameter naming, `load_state_dict`                    |

### 1.2 Does not own

| Never in `models`                     | Belongs in                                |
| ------------------------------------- | ----------------------------------------- |
| Datasets, batching, data loading      | `data/`                                   |
| Task-specific input adaptation        | `adapters/`                               |
| Losses, objectives, training criteria | `objectives/`, `loss/`                    |
| Optimizer or scheduler creation       | `training/`                               |
| Lightning modules                     | `lightning/`                              |
| Rollout execution                     | `rollouts/`                               |
| ACT deliberation loop                 | `controllers/`                            |
| Environment interaction               | `tasks/`                                  |
| Metric accumulation                   | `metrics/`                                |
| Checkpoint selection, MLflow lookup   | `model_artifacts/`, `evaluation/`         |
| Experiment configuration              | `config/` (training configs), experiments |
| Reporting or plotting                 | `reporting/`, `figures/`                  |

### 1.3 Architectural boundary

```
tasks/data
    ↓
adapters              ← task→model translation (outside models)
    ↓
models                ← architecture, state, configuration (this package)
    ↓
model outputs         ← typed predictions, latent codes, next state

objectives            ← consume model outputs
controllers           ← govern repeated model execution
rollouts              ← temporal execution + state lifecycle
training              ← coordinates models, objectives, optimizers
evaluation            ← coordinates models, adapters, metrics, artifacts
```

The model returns neural predictions and model state. The controller decides
whether to continue. The runner decides when to reset or detach. The
objective computes loss. The metric system evaluates predictions.

---

## 2. Distinguish `models` from `modules`

The repository separates reusable building blocks from complete architectures:

```
ehp_sn.modules
    reusable parameterized components
    reusable state types
    local mathematical operations
    no knowledge of complete model families

ehp_sn.models
    complete executable architectures
    composition and information flow
    family-specific state and output contracts
    architecture configuration
```

Concrete separation:

| Package              | Examples                                   | Role                       |
| -------------------- | ------------------------------------------ | -------------------------- |
| `modules.lec`        | `LECModel`, `LECState`, `LECSettings`      | Sensory feature processing |
| `modules.mec`        | `MECModel`, `MECState`, `MECSettings`      | Grid-cell path integration |
| `modules.hpc`        | `HPCAttractor`, `HPCAttention`, `HPCState` | Hippocampal memory         |
| `modules.pfc`        | `PFCModel`, `PFCState`, `PFCSettings`      | Prefrontal reasoning       |
| `modules.str`        | `STRModelLinear`, `STRState`               | Striatal actor-critic      |
| `modules.projection` | `ProjectionBundle`, `ProjectionSettings`   | Cross-region projections   |
| `models.tem`         | `TEMModelV1`, `TEMModelV2`                 | Complete TEM architectures |
| `models.hrm`         | `HRMModelV1`, `HRMModelV2`                 | Complete HRM architectures |
| `models.ehp`         | `EHPModelV1`                               | Complete EHP architectures |

A model family may use multiple modules, but a module must **not** import or
know about the enclosing model.

---

## 3. Professional foundation

The professional foundation is `torch.nn.Module`, which provides:

- recursive parameter registration;
- recursive buffer registration;
- device and dtype movement;
- training/evaluation mode;
- hooks;
- hierarchical composition;
- `state_dict()` serialization;
- compatibility with compilation, distributed training, mixed precision.

Every concrete model inherits directly from `nn.Module`:

```
class TEMModelV1(nn.Module): ...
class TEMModelV2(nn.Module): ...
class HRMModelV1(nn.Module):  ...
class HRMModelV2(nn.Module):  ...
class EHPModelV1(nn.Module):  ...
```

There is no shared `BaseModel` base class. Project-specific interoperability
is expressed through typed input/output objects, configuration dataclasses,
and narrow protocols.

---

## 4. Package structure

```
src/ehp_sn/models/
├── __init__.py                # Stable public API: model classes + factory
├── factory.py                 # build_model(settings) — discriminated match + ModelSettings union
├── protocols.py               # TraceableModel, StatefulArchitecture
│
├── tem/
│   ├── __init__.py            # Family API: inputs, outputs, state, model
│   ├── tem_v1.py              # TEMModelV1 + its local contracts
│   ├── tem_v2.py              # TEMModelV2 + its local contracts
│   └── _shared.py             # Shared TEM definitions (GridCodes, PlaceCodes, etc.)
│
├── hrm/
│   ├── __init__.py            # Family API
│   ├── hrm_v1.py              # HRMModelV1 + its local contracts
│   ├── hrm_v2.py              # HRMModelV2 + its local contracts
│   └── _shared.py             # Shared HRM definitions
│
└── ehp/
    ├── __init__.py            # Family API
    ├── ehp_v1.py              # EHPModelV1 + its local contracts
    ├── ehp_v2.py              # EHPModelV2 + its local contracts
    └── _shared.py             # Shared EHP projection settings, slot names
```

### 4.1 Root files

| File           | Purpose                                                                                         |
| -------------- | ----------------------------------------------------------------------------------------------- |
| `__init__.py`  | Stable public API — model classes, `build_model`                                                |
| `factory.py`   | `build_model(settings: ModelSettings)` — discriminated construction; `ModelSettings` union type |
| `protocols.py` | Cross-family capability protocols (`TraceableModel`, `StatefulArchitecture`)                    |

### 4.2 Family submodule structure

Each version file contains the architecture and its local contracts in one
coherent module. Keep contracts and implementation together while they form
one cohesive unit. Split when sections have distinct consumers, evolve
independently, or make navigation and review materially harder.

```
ehp_sn/models/tem/tem_v1.py:
    TEMSettingsV1               # Pydantic BaseModel, extra="forbid"
    TEMInputV1                  # @dataclass frozen
    TEMOutputV1                 # @dataclass frozen
    TEMStateV1                  # @dataclass frozen (detach returns new state)
    TEMModelV1(nn.Module)       # complete architecture
```

### 4.3 Naming rules

Class names must be **family-qualified** to avoid cross-family collisions:

✅ Correct:

```
TEMSettingsV1, TEMInputV1, TEMOutputV1, TEMStateV1, TEMModelV1
HRMSettingsV1, HRMInputV1, HRMOutputV1, HRMStateV1, HRMModelV1
HRMSettingsV2, HRMInputV2, HRMOutputV2, HRMStateV2, HRMModelV2
```

❌ Avoid (collision between TEM and HRM):

```
ModelSettingsV1   # ambiguous — which family?
```

---

## 5. Concrete model contract

Every recurrent or memory-bearing model supports five operations.

### 5.1 Construction

```python
model = TEMModelV1(settings)
```

Construction creates parameters and persistent buffers. It does **not** load
datasets, checkpoints, experiment state, or trainer state.

### 5.2 State initialisation

```python
state = model.init_state(
    batch_size=batch_size,
    device=device,
    dtype=dtype,
)
```

This produces model-owned neural state. Initialisation must not depend on a
`DataModule`, environment, or trainer.

### 5.3 Forward transition

```python
output, next_state = model(inputs, state)
```

The canonical contract across all model families. Explicit functional
recurrence — no hidden mutable state.

### 5.4 State reset

```python
state = model.reset_state(state, reset_mask)
```

Selectively reset rows (new episodes) while preserving active rows. The
runner decides _when_ reset is required; the model implements _how_.

When to use a model method versus a state method:

- Use `model.reset_state(state, mask)` when reset depends on model settings,
  parameters, or model-created defaults (e.g. default memory state, fresh
  frequency buffers tied to architecture config).
- Use `state.reset(mask)` when reset is purely structural tensor masking and
  every sub-state implements its own recursive reset.

For EHP composition, recursive state methods compose cleanly:

```python
@dataclass(frozen=True, slots=True)
class EHPStateV1:
    tem: TEMStateV2
    hrm: HRMStateV1

    def reset(self, mask: Tensor) -> "EHPStateV1":
        return EHPStateV1(
            tem=self.tem.reset(mask),
            hrm=self.hrm.reset(mask),
        )
```

### 5.5 Architecture-level finalisation

Some architectures need a boundary operation — clamping Hebbian weights,
compactifying memory, updating persistent caches. The model owns the
operation; the runner owns when it is called.

Not every model requires finalisation. TEM v1 needs `finalize_memory()`
to clamp Hebbian weights at TBPTT chunk boundaries. TEM v2 (factor memory)
requires none. HRM requires none. Do not add a universal no-op
`finalize_state()` on every model; use an optional capability protocol
or let adapters call architecture-specific methods when present:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MemoryFinalizer(Protocol):
    def finalize_memory(self, state: object) -> object:
        ...
```

---

## 6. Inputs, outputs, and state

### 6.1 Family-specific, not universal

Each model family has its own concrete types. No universal output with
optional fields:

❌ Bad:

```python
@dataclass
class ModelOutput:
    grid_codes: GridCodes | None = None
    place_codes: PlaceCodes | None = None
    action_logits: Tensor | None = None
    q_values: Tensor | None = None
    state_value: Tensor | None = None
    halt_logits: Tensor | None = None
```

✅ Correct:

```python
@dataclass(frozen=True, slots=True)
class TEMOutputV1:
    grid_codes: GridCodes
    place_codes: PlaceCodes
    pred_codes: PredCodes

@dataclass(frozen=True, slots=True)
class HRMOutputV1:
    theta_summary: Tensor
    schema_slots: Tensor
    schema_readout: Tensor
    action_logits: Tensor

@dataclass(frozen=True, slots=True)
class HRMOutputV2:
    theta_summary: Tensor
    schema_slots: Tensor
    q_values: Tensor
    state_value: Tensor
```

### 6.2 Output field ordering

Even though dataclasses are normally accessed by name, preserve a consistent
semantic order across versions within a family.

For TEM across v1 and v2:

```
grid_codes, place_codes, pred_codes
```

or:

```
pred_codes, grid_codes, place_codes
```

Choose one ordering and use it consistently.

### 6.3 State ownership boundaries

Separate model state from controller and rollout state:

| Category             | Owned by      | Contents                                              |
| -------------------- | ------------- | ----------------------------------------------------- |
| **Model state**      | `models`      | LEC/MEC/HPC/PFC latent state, differentiable memory   |
| **Controller state** | `controllers` | Halted mask, deliberation step count, ACT probability |
| **Runtime carry**    | `rollouts`    | Composition of model_state + controller_state         |

Do not put `halted` and `steps` into `TEMState` or `HRMState` merely because
every controller currently uses them.

A prospective EHP state:

```python
@dataclass(frozen=True, slots=True)
class EHPStateV1:
    lec: LECState
    mec: MECState
    hpc: HPCState
    pfc: PFCState

    def detach(self) -> "EHPStateV1":
        return EHPStateV1(
            lec=self.lec.detach(),
            mec=self.mec.detach(),
            hpc=self.hpc.detach(),
            pfc=self.pfc.detach(),
        )
```

or compositionally:

```python
@dataclass(frozen=True, slots=True)
class EHPStateV1:
    tem: TEMStateV2
    hrm: HRMStateV1

    def detach(self) -> "EHPStateV1":
        return EHPStateV1(
            tem=self.tem.detach(),
            hrm=self.hrm.detach(),
        )
```

The choice depends on whether EHP literally composes existing TEM and HRM
implementations or only reuses their lower-level modules.

---

## 7. Configuration design

### 7.1 Architecture configuration only

Model settings contain architecture and neural-dynamics parameters only.

✅ Good fields:

```
embedding dimensions
latent dimensions
number of layers
attention heads
frequency configuration
memory width or capacity
projection topology
fixed versus trainable connections
activation functions
dropout
normalization
architectural iteration count
```

❌ Do not include:

```
learning rate
optimizer
weight decay
batch size
number of workers
checkpoint path
MLflow run ID
evaluation recipe
metric names
logging interval
trainer precision
```

### 7.2 Runtime neural parameters

Parameters such as `eta`, `hebbian_decay`, and `p2g_uncertainty_offset` are a
boundary case. The model should own their _current values_ because they affect
neural dynamics. Training should own the _schedule_ that selects those values.

```python
# models owns value semantics
model.set_dynamics(eta=eta, hebbian_decay=decay, p2g_uncertainty_offset=offset)

# training owns scheduling policy — it computes eta from step count
```

### 7.3 Configuration versioning

Model architecture configuration is a durable artifact. Version it explicitly
using Pydantic v2 conventions:

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict

class TEMSettingsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    transition_action_count: int
    f_initial: list[float]
    ...
```

A loader can migrate old dictionaries:

```python
from collections.abc import Mapping

def load_tem_settings(raw: Mapping[str, object]) -> TEMSettingsV1:
    version = int(raw.get("schema_version", 1))
    match version:
        case 1:
            return TEMSettingsV1.model_validate(raw)
        case _:
            raise ValueError(f"Unsupported TEM settings version: {version}")
```

Distinguish three independent version axes that are easily conflated:

| Axis                             | Meaning                                               | Example bump                                  |
| -------------------------------- | ----------------------------------------------------- | --------------------------------------------- |
| **Architecture version**         | Parameter topology, state structure, memory mechanism | Hebbian → factor (TEM v1 → v2)                |
| **Checkpoint schema version**    | State-dict key naming, serialisation layout           | Renamed module path, added buffer             |
| **Configuration schema version** | Settings field names, defaults, validation rules      | New mandatory field, removed deprecated field |

These can change independently. A checkpoint-schema migration does not imply
an architecture change.

---

## 8. Model factory

### 8.1 Discriminated settings union

The settings type itself identifies the model family and version. No separate
spec object is needed:

```python
ModelSettings = (
    TEMSettingsV1
    | TEMSettingsV2
    | HRMSettingsV1
    | HRMSettingsV2
    | EHPSettingsV1
    | EHPSettingsV2
)
```

### 8.2 Factory (not registry)

Use an explicit `match`-based factory, not a mutable global registry:

```python
from torch import nn

def build_model(settings: ModelSettings) -> nn.Module:
    match settings:
        case TEMSettingsV1(): return TEMModelV1(settings)
        case TEMSettingsV2(): return TEMModelV2(settings)
        case HRMSettingsV1(): return HRMModelV1(settings)
        case HRMSettingsV2(): return HRMModelV2(settings)
        case EHPSettingsV1(): return EHPModelV1(settings)
        case EHPSettingsV2(): return EHPModelV2(settings)
        case _:
            raise TypeError(
                f"Unsupported model settings type: {type(settings).__name__}"
            )
```

The discriminating type is already known after configuration parsing. Passing
family/version strings would be unnecessary indirection and risks disagreement
between the string identifier and the settings type.

**When to add a registry:**

- external packages must add models;
- experiments dynamically discover installed model families;
- model implementations are loaded through Python entry points;
- the repository grows to dozens of independently maintained families.

Until then, an explicit factory is easier to type, inspect, test, and
refactor.

- external packages must add models;
- experiments dynamically discover installed model families;
- model implementations are loaded through Python entry points;
- the repository grows to dozens of independently maintained families.

Until then, an explicit factory is easier to type, inspect, test, and
refactor.

---

## 9. Public API

### 9.1 Root API

```python
# ehp_sn/models/__init__.py

from .factory import build_model
from .ehp import EHPModelV1
from .hrm import HRMModelV1, HRMModelV2
from .tem import TEMModelV1, TEMModelV2

__all__ = [
    "EHPModelV1",
    "HRMModelV1",
    "HRMModelV2",
    "TEMModelV1",
    "TEMModelV2",
    "build_model",
]
```

### 9.2 Family API

```python
# ehp_sn/models/tem/__init__.py

from .tem_v1 import (
    TEMInputV1,
    TEMModelV1,
    TEMOutputV1,
    TEMSettingsV1,
    TEMStateV1,
)
from .tem_v2 import (
    TEMInputV2,
    TEMModelV2,
    TEMOutputV2,
    TEMSettingsV2,
    TEMStateV2,
)

__all__ = [
    "TEMInputV1", "TEMInputV2",
    "TEMModelV1", "TEMModelV2",
    "TEMOutputV1", "TEMOutputV2",
    "TEMSettingsV1", "TEMSettingsV2",
    "TEMStateV1", "TEMStateV2",
]
```

### 9.3 Import conventions

Consumers use stable paths:

```python
from ehp_sn.models import TEMModelV1, build_model
from ehp_sn.models.tem import TEMSettingsV1, TEMStateV1
```

Do **not** import from internal module paths outside model tests:

```python
# ❌ Avoid outside model tests
from ehp_sn.models.tem.tem_v1 import ...
```

### 9.4 What not to export

Do not expose:

```
MODEL_REGISTRY
_INTERNAL_BUILDERS
_DefaultProjection
_TEMImplementationV3
checkpoint resolver functions
training wrappers
LightningModules
objective classes
task adapters
```

---

## 10. Observability API

`trace_views()` returns model-owned semantic tensors and does not perform
logging or artifact writing. This is the appropriate observability surface.

### 10.1 Protocol

```python
from typing import Protocol, TypeVar
from torch import Tensor

StateT = TypeVar("StateT", contravariant=True)
OutputT = TypeVar("OutputT", contravariant=True)

class TraceableModel(Protocol[StateT, OutputT]):
    @property
    def available_trace_views(self) -> frozenset[str]:
        ...

    def trace_views(
        self,
        requested: frozenset[str],
        *,
        state: StateT,
        output: OutputT | None = None,
    ) -> dict[str, Tensor]:
        ...
```

Temporal semantics must be explicit. Document whether traced values come from:

- the state **before** the transition;
- the state **after** the transition;
- the output of the transition;
- module-local cached activations.

For recurrent models this distinction is critical — a trace key like
`hpc.cells` could refer to pre-step, post-step, or mid-step activations
depending on when `trace_views()` is called.

### 10.2 What the model should not know

The model must not import or reference:

```
Zarr
MLflow
trace sampling frequency
artifact paths
figure names
evaluation recipes
```

It only maps semantic view names to tensors.

### 10.3 Key namespace conventions

Use dot-separated namespaced keys:

```
lec.cells
lec.filtered
lec.sensory_code
mec.cells
mec.location_mean
hpc.cells
hpc.memory.summary
pfc.z_H
pfc.z_L
str.q_values
str.state_value
```

The trace profile decides which are captured.

---

## 11. Shape and axis conventions

Define a project-wide vocabulary used consistently across all docstrings:

| Symbol | Meaning                                        |
| ------ | ---------------------------------------------- |
| `B`    | Batch or replay-slot axis                      |
| `T`    | External/environment sequence time             |
| `K`    | Internal deliberation step                     |
| `N`    | Node, token, slot, or memory-entry count       |
| `D`    | Latent feature dimension                       |
| `A`    | Action dimension                               |
| `F`    | Frequency count                                |
| `H`    | Attention heads                                |
| `S`    | Sequence length (sum of `T + K` when combined) |

For EHP, preserve the distinction between physical time and internal
deliberation:

```
t ∈ [0, T)    physical or environment time
k ∈ [0, K)    internal reasoning or deliberation time
```

Every input, output, and state docstring documents tensor shapes using these
symbols:

```python
def forward(
    self,
    observation: Tensor,      # (B, T, O) sensory observations
    previous_action: Tensor,  # (B, T) action indices
    state: TEMStateV1,
) -> TEMOutputV1:
    """Run one TEM step."""
```

---

## 12. Versioning policy

Version concrete architecture contracts, not merely filenames.

A version change is warranted when one of these changes incompatibly:

- parameter topology;
- state structure;
- input contract;
- output contract;
- state-dict key structure;
- interpretation of a major architectural setting;
- memory mechanism;
- execution semantics that affect learned behaviour.

A new experiment configuration alone does **not** require a new model version.

| Change                                | Version bump?                        |
| ------------------------------------- | ------------------------------------ |
| Hebbian memory → factor memory        | ✅ Yes (TEM v1 → v2)                 |
| New learning rate schedule            | ❌ No                                |
| Additional frequency module           | ✅ Yes (new state structure)         |
| New evaluation dataset                | ❌ No                                |
| Cross-entropy → variational objective | ❌ No (objective belongs in `loss/`) |
| Different projection topology         | ✅ Yes (new parameters)              |

---

## 13. Stable state-dict naming and versioning

Parameter names become part of the practical compatibility surface. A
refactor from `self.encoder` to `self.lec` changes state-dict keys from
`encoder.weight` to `lec.weight` and breaks old checkpoints.

### 13.1 Policy

- accept checkpoint incompatibility and increment the **checkpoint schema
  version** (which is independent of the architecture version);
- provide an explicit migration function for known renames within the same
  architecture version;
- use `strict=True` by default; `strict=False` only for deliberate transfer
  learning, not as the default fix for architecture drift.

### 13.2 Migration within a single architecture version

When a module is renamed but the parameter topology is unchanged, provide a
key-remapping migration. This is a checkpoint-schema change, not an
architecture change:

```python
def migrate_tem_v1_checkpoint_schema_1_to_2(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Rename keys from TEM v1 checkpoint schema 1 to schema 2.

    The architecture is unchanged; only the serialised key namespace
    has been reorganised.
    """
    renamed = {}
    for key, tensor in state_dict.items():
        key = key.replace("hpc.encoder.", "hpc.lec.")  # example
        renamed[key] = tensor
    return renamed
```

### 13.3 Cross-architecture-version migration

When the architecture itself changes (e.g. Hebbian → factor memory),
state-dict migration is generally impossible or scientifically invalid.
The new architecture has different parameter topology, semantics, and
initialisation requirements. Do not provide a migration; declare the
checkpoint as incompatible and train from scratch or initialise from a
subset of compatible weights.

### 13.4 Independent version axes

Distinguish the three version axes defined in §7.3:

| Axis                         | Change example                    | Migration?                        |
| ---------------------------- | --------------------------------- | --------------------------------- |
| Architecture version         | Hebbian → factor memory           | ❌ New architecture; no migration |
| Checkpoint schema version    | Renamed `hpc.encoder` → `hpc.lec` | ✅ Key remapping possible         |
| Configuration schema version | New mandatory field in settings   | ✅ Default or loader migration    |

---

## 14. Initialisation

Initialisation belongs near the architecture because it depends on module
semantics.

### 14.1 Canonical initialisation

```python
class TEMModelV1(nn.Module):
    def reset_parameters(self) -> None:
        """Canonical architecture initialisation."""
        self.projections.reset_parameters()
```

### 14.2 Experimental initialisation

```python
def initialize_tem_experimental(
    model: TEMModelV1,
    scheme: InitScheme,
) -> None:
    """Experiment-level reinitialisation policy."""
```

The first is appropriate for the model's canonical initialisation. The second
is appropriate for optional experiment-level policies.

---

## 15. Dependency rules

### 15.1 Allowed imports

```
models → modules
models → low-level types (ehc_sn.types)
models → model-local utilities (ehc_sn.utils.detach)
```

### 15.2 Disallowed imports

```
models → objectives
models → controllers
models → rollouts
models → training
models → evaluation
models → figures
models → logging infrastructure
models → MLflow
models → Lightning
models → adapters (in the model direction — adapters import models)
```

### 15.3 Architecture test

The repository should enforce an architecture test that verifies no model
module imports from `objectives`, `controllers`, `rollouts`, `training`,
`evaluation`, `figures`, `logging`, `lightning`, or `adapters`.

---

## 16. Anti-patterns

### God model

```python
model.training_step(...)      # ❌
model.validation_step(...)    # ❌
model.compute_metrics(...)    # ❌
model.save_report(...)        # ❌
model.load_dataset(...)       # ❌
```

### Hidden mutable recurrence

```python
self.carry = ...   # ❌ inside forward()
```

### Loose dictionaries as output

```python
return {"x": ..., "foo": ..., "misc": ...}   # ❌ undocumented keys
```

### Universal output with all optional fields

```python
@dataclass
class ModelOutput:
    tem_g: Tensor | None = None
    tem_p: Tensor | None = None
    act_halted: Tensor | None = None          # ❌
```

### Stringly typed construction

```python
build_model(config: dict[str, Any])   # ❌ no schema
```

### Global registry side effects

```python
@register_model("tem_v1")             # ❌ import-order dependence
class TEMModelV1: ...
```

### Model-task coupling

```python
class MazeHardHRMModel(...)           # ❌
class ArenaTEMModel(...)              # ❌
```

Single architecture should serve multiple tasks through adapters.

### Architecture-loss coupling

```python
model.forward(batch, labels) -> loss   # ❌
```

### Infrastructure imports

```python
import mlflow      # ❌ inside models/
import lightning   # ❌
import wandb       # ❌
```

---

## 17. Testing requirements

The design defines many architectural invariants. These must be enforced by
automated tests that establish compliance independently of any particular
consumer.

### 17.1 Required architecture tests

| Test                  | Requirement                                                            | Verification                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Construction**      | Every valid settings type builds the intended model class              | `build_model(TEMSettingsV1(...))` returns `TEMModelV1`                                                                                                      |
| **Invalid config**    | Unknown or extra settings fields fail before parameter allocation      | `TEMSettingsV1.model_validate({"extra": 1})` raises `ValidationError`                                                                                       |
| **Forward contract**  | `output, next_state = model(inputs, state)` returns correct types      | `isinstance(output, TEMOutputV1)`, `isinstance(next_state, TEMStateV1)`                                                                                     |
| **Shape contract**    | All documented tensor shapes are verified on a sample batch            | Concrete shape assertions per docstring                                                                                                                     |
| **State isolation**   | Forward does not mutate input state unless explicitly documented       | `state_before = copy(state); model(inputs, state); assert state == state_before`                                                                            |
| **Selective reset**   | Masked rows reset to initial values; other rows are bitwise unchanged  | Compare reset rows vs `init_state(1)`, active rows vs pre-reset                                                                                             |
| **Serialisation**     | Strict `state_dict` round trip preserves all parameters                | `model.load_state_dict(copy.deepcopy(model.state_dict()))` succeeds with `strict=True`                                                                      |
| **Device movement**   | State and output remain on expected device after forward               | `output.grid_codes.post[0].device == model_device`                                                                                                          |
| **Gradient flow**     | Trainable pathways receive gradients; fixed pathways do not            | Backward pass on a scalar of output; check `.grad is None` for frozen params                                                                                |
| **Dependency rule**   | Forbidden imports fail architecture tests                              | Script that scans `models/` for imports of `objectives`, `controllers`, `rollouts`, `training`, `evaluation`, `figures`, `logging`, `lightning`, `adapters` |
| **Public API**        | Required symbols import from documented namespaces                     | `from ehp_sn.models import TEMModelV1, build_model` succeeds                                                                                                |
| **Trace contract**    | Declared trace keys are available and produce correctly shaped tensors | `model.trace_views(model.available_trace_views, state)` returns dict with documented keys                                                                   |
| **Batch equivalence** | Batched and per-sample execution agree for stateless components        | Compare `model(x_batch)[0]` vs `torch.stack([model(x_i)[0] for x_i in x_unbatched])` within tolerance                                                       |

### 17.2 Test placement

- Model-specific tests (e.g. `test_tem_v1_forward_shapes`) live in `tests/models/tem/test_tem_v1.py`.
- Cross-family architecture tests (e.g. `test_no_training_imports`) live in `tests/models/test_architecture.py`.
- Public API tests live in `tests/models/test_public_api.py`.

### 17.3 Test expectations

- Tests must run without Lightning, MLflow, or any training infrastructure.
- Tests must be deterministic under a fixed seed.
- Model construction tests must complete in under one second on CPU.
- Forward-contract tests may use a batch size of 2 to keep memory low.

---

## 18. Recommended final design summary

```
ehp_sn.models
│
├── stable public construction API
│   └── build_model(settings) → nn.Module
│
├── tem/
│   ├── TEMModelV1, TEMSettingsV1, TEMInputV1, TEMOutputV1, TEMStateV1
│   ├── TEMModelV2, TEMSettingsV2, TEMInputV2, TEMOutputV2, TEMStateV2
│   └── _shared (GridCodes, PlaceCodes, PredCodes, TEMProjectionSettings)
│
├── hrm/
│   ├── HRMModelV1, HRMSettingsV1, HRMInputV1, HRMOutputV1, HRMStateV1
│   ├── HRMModelV2, HRMSettingsV2, HRMInputV2, HRMOutputV2, HRMStateV2
│   └── _shared
│
└── ehp/
    ├── EHPModelV1, EHPSettingsV1, EHPInputV1, EHPOutputV1, EHPStateV1
    ├── EHPModelV2, EHPSettingsV2, EHPInputV2, EHPOutputV2, EHPStateV2
    └── _shared (EHCProjectionSettings, slot constants)
```

### Canonical usage

```python
from ehp_sn.models import build_model
from ehp_sn.models.ehp import EHPSettingsV1, EHPInputV1

# Construction
settings = EHPSettingsV1.model_validate(config)
model = build_model(settings)

# Initialisation
state = model.init_state(
    batch_size=batch_size,
    device=device,
    dtype=dtype,
)

# Forward
output, next_state = model(inputs, state)

# The controller decides whether to continue.
# The runner decides when to reset or detach.
# The objective computes loss.
# The metric system evaluates predictions.
# The artifact system records outputs.
```

The model returns neural predictions and model state. Everything else lives
in adjacent packages with clear, unidirectional dependencies.

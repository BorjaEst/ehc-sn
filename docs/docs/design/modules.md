# Module Design Contract

> Reusable `torch.nn.Module` components and their local state/output contracts.

`ehp_sn.modules` provides the **shared neural vocabulary** of the repository:
parameterized, composable, independently testable components from which
complete architectures are built. It does *not* contain complete model-family
subsystems, task orchestration, controller loops, objectives, training logic,
or experiment configuration.

---

## 1. Architectural position

```
ehp_sn.contracts / types.py
        ^
ehp_sn.functional
        ^
ehp_sn.modules                    ← this package
        ^
ehp_sn.models
        ^
ehp_sn.controllers / objectives / training / evaluation
```

### 1.1 Dependencies

`ehp_sn.modules` may import:

- Python standard library;
- PyTorch (`torch`, `torch.nn`, `torch.nn.functional`);
- `einops` for explicit tensor rearrangement;
- `ehp_sn.contracts` or lightweight `ehp_sn.types`;
- `ehp_sn.functional` (stateless tensor equations);
- narrowly justified numerical utilities.

It must **not** import `ehp_sn.models`, `ehp_sn.controllers`, `ehp_sn.objectives`,
`ehp_sn.metrics`, `ehp_sn.lightning`, `ehp_sn.evaluation`, `ehp_sn.figures`,
`ehp_sn.experiments`, or `ehp_sn.logging`.

The repository currently respects this rule: `modules/` imports only from
`ehc_sn.types`, `ehc_sn.utils`, and `ehc_sn.activations`.

### 1.2 Ownership

| `modules` owns | `modules` does not own |
|---|---|
| Attention, MLP, transformer blocks | Complete TEM/HRM/EHP architectures |
| Embeddings, projections | Subsystem orchestration (sensory-read -> transition cycle) |
| Recurrent cells (one transition) | Grid-cell lifecycle (`generative` / `inference`) |
| Memory read/write operators | PFC reasoning schedule (`step()` loop) |
| Spatial primitives (path integration, filters, norms) | ACT deliberation, halting policy, step budgets |
| Prediction heads (categorical, value, field) | Loss objectives, metric accumulation |
| Workspace slot-addressing | Training loops, evaluation recipes |
| Observation encoders/decoders | Experiment configuration |

---

## 2. Distinguish `modules` from `models`

> `nn.Module` is an implementation mechanism.
> `ehp_sn.modules` is an architectural ownership boundary.

A component belongs in `modules` when it is a **reusable building block** ---
one bounded transformation, no knowledge of which model family uses it, and
independently testable.

A component belongs in `models` when it is a **subsystem orchestrator** ---
coordinating multiple transformations with lifecycle, phase semantics, or
family-specific information flow.

### 2.1 Acceptance test

Two questions decide placement:

> **Q1.** Can this component be instantiated and tested without knowing which
> model family, task, dataset, controller, or training recipe uses it?

> **Q2.** Does the component execute *one* neural transformation, or does it
> coordinate a sequence of transformations with lifecycle and phase semantics?

**Q1 yes + Q2 "one transformation"** -> `modules`.
**Q1 no or Q2 "coordination"** -> `models` or `controllers`.

### 2.2 Concrete classification

The following table applies these questions to every component currently in
the repository's `modules/` package.

| Current location | Component | Classification | Recommendation |
|---|---|---|---|
| `modules/attention.py` | `Attention` | Reusable primitive | **Keep** |
| `modules/mlp.py` | `SwiGLU`, `MLP` | Reusable primitive | **Keep** |
| `modules/transformer.py` | `TransformerBlock`, `TransformerStack`, `TransformerSequenceSummary`, `TokenSummarizer` | Reusable primitive | **Keep** |
| `modules/autoencoder.py` | `Autoencoder`, `TwoHotEncoder` | Reusable primitive | **Keep** |
| `modules/projection.py` | `ProjectionModule`, `ProjectionBundle`, `ProjectionSettings` | Reusable primitive | **Keep** |
| `modules/hpc/location.py` | `PlaceInference` | Reusable primitive | **Keep** (-> `modules/memory/`) |
| `modules/hpc/query.py` | `AttractorRead`, `FactorRead` | Reusable primitive | **Keep** (-> `modules/memory/`) |
| `modules/hpc/update.py` | `HebbianWrite`, `EpisodicWrite` | Reusable primitive | **Keep** (-> `modules/memory/`) |
| `modules/hpc/_base.py` | `HPCBase` | Subsystem orchestration | **Move to** `models.tem` |
| `modules/hpc/modules.py` | `HPCAttractor`, `HPCAttention` | Complete memory subsystems | **Move to** `models.tem` |
| `modules/hpc/query_policy.py` | `ReadCues`, `CueRead`, `TargetRead` | TEM-specific query semantics | **Move to** `models.tem` |
| `modules/mec/__init__.py` | `MECModel` | Subsystem orchestrator | **Move to** `models.tem` |
| `modules/mec/path.py` | `PathIntegrator` | Reusable primitive | **Keep** (-> `modules/spatial/`) |
| `modules/lec/__init__.py` | `LECModel` | Subsystem orchestrator | **Move to** `models.tem` |
| `modules/lec/filter.py` | `FrequencyFilter` | Reusable primitive | **Keep** (-> `modules/spatial/`) |
| `modules/lec/norm.py` | `FeatureNorm` | Reusable primitive | **Keep** (-> `modules/spatial/`) |
| `modules/pfc/__init__.py` | `PFCModel` | Subsystem orchestrator | **Move to** `models.hrm` |
| `modules/pfc/reasoning.py` | `HighLvRModule`, `LowLvRModule` | Recurrent cells | **Keep** (-> `modules/recurrent/`) |
| `modules/pfc/values.py` | `QValueEstimator` | Reusable primitive | **Keep** (-> `modules/heads/`) |
| `modules/pfc/workspace.py` | `Workspace`, `WorkspaceSchema`, `WorkspaceLayout` | Reusable primitive | **Keep** (-> `modules/workspace/`) |
| `modules/str/__init__.py` | `STRModelLinear` | Actor-critic subsystem | **Move to** `models.hrm` |
| `modules/str/__init__.py` | `compute_rpe()` | Pure function | **Move to** `ehp_sn.functional` |
| `modules/bg/` | (empty stubs) | Pre-emptive placeholder | **Remove** |

Components not listed (e.g. `MECLayout`, `P2GMemory`, `OVCCorrection`,
`DenseHebbianStoreBackend`, `HebbianLayout`) are tightly coupled to TEM's
two-phase memory cycle and should move to `models.tem` with their owning
subsystem.

---

## 3. Module contract

### 3.1 Core requirements

Every public module in `ehp_sn.modules`:

- subclasses `torch.nn.Module`;
- owns one bounded neural transformation or local state transition;
- accepts explicit tensor and local-state inputs;
- documents input shape, output shape, mask semantics, and state semantics;
- is independently testable;
- does not depend on a particular task, experiment, or training recipe.

### 3.2 Return types

A module returns **one of**:

- A bare `Tensor` when the result is a single unambiguous value.
- A frozen `dataclass` or `NamedTuple` when multiple semantically distinct
  values are produced.

A bare tuple is **never** acceptable for a multi-value return.

```python
# Single result --- bare Tensor is fine
class FeatureNorm(nn.Module):
    def forward(self, x: list[Tensor]) -> list[Tensor]:
        ...

# Multiple results --- use NamedTuple or frozen dataclass
@dataclass(frozen=True)
class MemoryReadOutput:
    value: Tensor
    weights: Tensor
    logits: Tensor | None = None

class FactorRead(nn.Module):
    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        valid: Tensor | None = None,
    ) -> MemoryReadOutput:
        ...
```

A return type must **never** change shape or type depending on a boolean flag
or branch. If the caller configures what is returned, use a consistent
structured output with an `Optional` field.

### 3.3 Forward API conventions

**Explicit tensors, keyword-only control parameters:**

```python
def forward(
    self,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    valid: Tensor | None = None,
    temperature: float = 1.0,
) -> MemoryReadOutput:
    ...
```

**No model-level verbs.** Generic module methods should be `forward`. If the
module owns a learned initial state (e.g., a reset-vector buffer), it may
expose `initial_state(batch_size) -> State`. Methods such as `generative`,
`inference`, `deliberate`, `rollout`, or `evaluate` encode model-phase
semantics and belong in `models` or `controllers`.

**Episode-boundary reset belongs to the model, not the module.** When
runtime state is explicit (passed in/out of `forward`, never stored on
`self`), reset is an external operation --- replace the state object --- and
should not be a module method. Modules that expose `reset_state(state,
reset_flag) -> State` are absorbing a model-level lifecycle decision.

### 3.4 Constructor conventions

Accept explicit primitive arguments:

```python
class QValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        ...
```

Complex components may accept a component-local immutable config dataclass.
Do **not** pass a root experiment configuration. The current Pydantic
component-scoped pattern (`AttentionConfig`, `MLPConfig`, `PathSettings`) is
appropriate.

### 3.5 State ownership

| Category | Ownership | Example |
|---|---|---|
| **Persistent learned** | `nn.Parameter` on the module | Embedding weights, projection matrices |
| **Persistent non-learned** | `register_buffer()` | Fixed Fourier frequencies, static masks |
| **Runtime episode state** | Explicit dataclass, passed in/out of `forward` | `MemoryState`, `TwoTimescaleState` |

Do **not** retain episode tensors on the module as attributes --- that obscures
lifecycle, harms reentrancy, and can retain autograd graphs.

### 3.6 Initialization

Document the initialization strategy. PyTorch defaults are acceptable when
they are appropriate. Where a module deviates from PyTorch defaults (e.g.,
truncated normal, logit-space initialization, zero-init for specific gates),
document the deviation and its rationale. Do not mandate custom initialization
for every module.

### 3.7 Numerical stability

Public modules should define:

- **Mask handling** --- documented behaviour for masked positions and
  all-masked rows. Returning `NaN` because softmax received all negative
  infinity is not a harmless implementation detail.
- **Mixed-precision behaviour** --- tested under `torch.amp.autocast`.
- **Gradient behaviour** --- no unexpected discontinuities.
- **Empty sequence behaviour** --- documented and handled.

---

## 4. Distinguish `modules` from `functional`

The distinction is about **interface**, not parameterization:

| | `ehp_sn.functional` | `ehp_sn.modules` |
|---|---|---|
| Interface | `def fn(tensor, ...) -> tensor` | `class Mod(nn.Module)` with `forward` |
| Composition | Called directly in code | Plugged into `nn.Sequential`, `torch.compile`, model tree |
| State | No persistent state by contract | May own `nn.Parameter`, `register_buffer`, or neither |
| Use case | Atomic tensor equations, math utilities | Reusable architectural building blocks |

A module with **zero parameters** (e.g. `FeatureNorm`) is still correctly an
`nn.Module` --- it is a composable building block that participates in device
placement, serialization, and module-tree traversal.

| Component | Correct home | Reason |
|---|---|---|
| `compute_rpe(r, V_s, V_s', gamma, done) -> Tensor` | `functional` | Direct tensor equation |
| `hebbian_outer_product(p_inf, p_gen) -> Tensor` | `functional` | Direct tensor equation |
| `cosine_read_weights(query, keys, mask, tau) -> Tensor` | `functional` | Direct tensor equation |
| `masked_softmax(logits, mask) -> Tensor` | `functional` | Direct tensor equation |
| `FeatureNorm(nn.Module)` | `modules.spatial` | Composable architectural building block |
| `AttractorRead(nn.Module)` | `modules.memory` | Composable architectural building block |
| `HebbianWrite(nn.Module)` | `modules.memory` | Composable architectural building block |

The repository currently does not have an `ehp_sn.functional` package. Pure
tensor equations live inline in module files or in `ehc_sn.utils`. Creating
`functional/` is the recommended first migration step.

---

## 5. Public API

### 5.1 Root exports

The root `ehp_sn.modules` API promotes **high-frequency stable building
blocks** --- components that appear in model construction code across multiple
model families. The set is intentionally small:

```python
from ehp_sn.modules import (
    Attention,
    MLP,
    ProjectionModule,
    SwiGLU,
    TransformerBlock,
    TransformerStack,
    TwoTimescaleCell,
    Workspace,
)
```

Specialized operators remain available through subpackages:

```python
from ehp_sn.modules.memory import AttractorRead, HebbianWrite, FactorRead
from ehp_sn.modules.recurrent import TwoTimescaleState, TwoTimescaleOutput
from ehp_sn.modules.spatial import PathIntegrator, FrequencyFilter
from ehp_sn.modules.heads import QValueHead, CategoricalHead
from ehp_sn.modules.workspace import WorkspaceLayout, WorkspaceSchema
```

### 5.2 Promotion criteria

A symbol belongs in the root API only when **all** of the following hold:

1. Stable semantic responsibility
2. Expected to be used outside its implementation subpackage
3. Not tied to one model version
4. Constructor and forward contract documented
5. Independent tests exist
6. Renaming would constitute a public compatibility change

Output dataclasses, state types, and configuration objects remain in the
subpackage that owns them unless they are genuinely consumed across multiple
subpackages. Cross-subpackage contracts belong in `ehp_sn.types` or
`ehp_sn.contracts`.

### 5.3 Internal symbols

Keep private:

- helper layers;
- implementation details of one composition;
- experimental ablations;
- model-version-specific classes;
- temporary adapters;
- intermediate configuration objects.

Use underscore-prefixed files for private implementation
(e.g. `modules/memory/_masking.py`).

---

## 6. Naming

Use names that describe the **computation**:

| Good | Weak |
|---|---|
| `AttractorRead` | `MemoryModule` |
| `TwoTimescaleCell` | `NeuralModule` |
| `FrequencyFilter` | `BaseComponent` |

Avoid the suffix `Module` --- `class AssociativeMemoryReader(nn.Module)` is
preferable to `class AssociativeMemoryReaderModule(nn.Module)`.

Do not create model-specific classes for the same computation (e.g.,
`TEMMLP`, `HRMMLP`, `EHPMLP`). Use one `MLP` with model-level configuration.

---

## 7. Summary

```
modules     = reusable neural vocabulary
models      = architectural sentences
controllers = execution policy
functional  = equations
contracts   = shared language
objectives  = learning criteria
```

`ehp_sn.modules` contains the reusable vocabulary: attention, projections,
memory operators, spatial transformations, recurrent cells, embeddings,
workspace machinery, and prediction heads. Subsystem names (HPC, MEC, LEC,
PFC, STR) belong in `modules` only when they refer to narrowly reusable
transformations. When they refer to complete subsystems with lifecycle, phase
semantics, and orchestration, they belong with the model family that defines
them.

The key distinction:

> `nn.Module` is an implementation mechanism.
> `ehp_sn.modules` is an architectural ownership boundary.

---

## Appendix A: Package structure

Promote a category to a subpackage only when it contains multiple stable
implementations with distinct responsibilities. Do **not** create empty
directories.

```
src/ehp_sn/modules/
|-- __init__.py
|-- attention.py
|-- autoencoder.py
|-- mlp.py
|-- transformer.py
|-- token_encoder.py
|-- projection/              (or projection.py until multi-file justified)
|-- memory/                  (AttractorRead, FactorRead, HebbianWrite, EpisodicWrite, PlaceInference)
|-- recurrent/               (TwoTimescaleCell, HighLvRModule, LowLvRModule)
|-- spatial/                 (PathIntegrator, FrequencyFilter, FeatureNorm)
|-- heads/                   (CategoricalHead, QValueHead, ValueHead, ProspectiveFieldHead)
|-- workspace/               (Workspace, WorkspaceSchema, WorkspaceLayout)
```

## Appendix B: Migration strategy

**Phase 1** --- Remove the empty `bg/` package. No structural changes needed.

**Phase 2** --- Create `src/ehp_sn/functional/`. Move pure equations
(`compute_rpe`, Hebbian outer-product, cosine read weights, masking
utilities) into `functional/`. Re-export from original locations with
deprecation warnings.

**Phase 3** --- Promote `Workspace*` from `pfc/workspace.py` to
`modules/workspace/`. Re-export from `pfc/` for backward compatibility.

**Phase 4** --- Relocate subsystem orchestration:

- `HPCBase`, `HPCAttractor`, `HPCAttention`, `ReadCues`, `CueRead`,
  `TargetRead` -> `models.tem/`
- `MECModel`, `LECModel` -> `models.tem/`
- `PFCModel`, `PFCState` -> `models.hrm/`
- `STRModelLinear` -> `models.hrm/`

Each move preserves a backward-compatibility re-export:

```python
# modules/hpc/__init__.py
import warnings
from ehp_sn.models.tem.hpc import TEMHPC as HPCAttractor

warnings.warn(
    "ehp_sn.modules.hpc.HPCAttractor moved to ehp_sn.models.tem.hpc.TEMHPC",
    DeprecationWarning,
    stacklevel=2,
)
```

## Appendix C: Testing standards

### Contract
- Accepted shapes match documented spec
- Returned shapes match documented spec
- Dtype preservation under normal inputs
- Invalid dimensions rejected with clear errors

### State and registration
- `nn.Parameter`s appear in `named_parameters()`
- Buffers appear in `named_buffers()`
- `.to(device)` moves all persistent tensors
- `state_dict()` round-trips correctly
- `train()` / `eval()` propagates to children

### Numerical
- Outputs are finite under normal inputs
- Masked positions follow documented policy
- All-invalid masks produce declared behaviour (not silent `NaN`)
- Gradients are finite
- Small known examples match hand calculations

### Integration
- Usable under `torch.amp.autocast`
- Deterministic under fixed seeds where expected
- Compatible with model checkpoint loading
- No retained activation tensors after forward pass

## Appendix D: Dependency audit (snapshot)

`modules/` currently imports only from `ehc_sn.types`, `ehc_sn.utils`, and
`ehc_sn.activations` --- no `models`, `controllers`, `training`, `evaluation`,
or `logging` imports. The architectural issue is **ownership**, not import
violation.

# Adapter Design Contract

> A narrow composition layer for exactly one supported task–model pairing.

An adapter translates **task-native contracts** into **model-native contracts**
and produces a **stable bridge output** consumed by objectives, controllers,
traces, and evaluators. It is the sole coupling point between tasks and models;
neither side knows about the other.

---

## 1. Architectural definition

```
TaskInput × Model → BridgeOutput
```

Concrete examples:

```
ArenaTaskInput × TEMV1 → ArenaTEMBridgeOutput
MazeHardTaskInput × HRMV1 → MazeHardHRMBridgeOutput
GoaltraceTaskInput × HRMV1 → GoaltraceHRMBridgeOutput
RoutebindTaskInput × HRMV1 → RoutebindHRMBridgeOutput
SeqMazeTaskInput × HRMV1 → SeqMazeHRMBridgeOutput
```

The execution boundary looks like:

```
task-native input
    ↓
prepare_inputs
    ↓
model-native input
    ↓
model step
    ↓
model-native output
    ↓
postprocess
    ↓
task-facing bridge output
```

Recurrent state crosses the boundary separately:

```
model state → model step → next model state
```

The adapter owns:

- task-to-model representation (input encoding);
- model invocation;
- model-to-task representation (output decoding);
- compatibility validation at construction time;
- delegation of model-state lifecycle.

The adapter does **not** own:

- target extraction;
- loss calculation;
- metric computation;
- action selection;
- rollout iteration;
- checkpoint resolution;
- experiment registration;
- trace persistence.

---

## 2. Core responsibilities

An adapter owns exactly five responsibilities.

### 2.1 Validate compatibility

Verify that a concrete task configuration can be paired with a concrete model
configuration — at construction time, not during execution.

```python
adapter = GoaltraceHRMV1BridgeAdapter(
    model=model,
    encoder=encoder,
    decoder=decoder,
    config=config,
)
```

Examples of what to validate:

- observation vocabulary matches decoder output size;
- number of graph nodes fits the HRM token capacity;
- input embedding dimension matches model input dimension;
- requested output heads are available;
- task masks and model sequence lengths are compatible.

Validation errors should identify the pairing, the expected property, the
observed property, and the source configurations:

```
SeqMazeHRMV1BridgeAdapter is incompatible:
required schema slots = max_nodes + max_path_length = 128,
but HRMV1 provides 96.
```

### 2.2 Encode task inputs — `prepare_inputs`

Translate a task-native input into the model's native input contract.
`prepare_inputs` is the only adapter execution method that consumes
task-native input. (Declarative trace observers may also read task records,
but they are not part of the adapter's execution path.)

```python
def prepare_inputs(self, task_input: TaskInputT) -> ModelInputT:
    return self.encoder(task_input)
```

Valid contents:

- token embeddings;
- spatial encodings;
- role embeddings;
- feature projections;
- mask construction for the model;
- sequence packing;
- frequency-band replication for multi-scale codes.

Invalid contents:

- target extraction;
- loss computation;
- optimizer logic;
- metric accumulation;
- evaluation reporting.

### 2.3 Invoke the model

The adapter orchestrates the model call, normalizing minor version-specific
calling conventions, but never reimplementing model logic.

```python
model_output, next_state = self.model(model_input, state)
```

### 2.4 Decode model outputs — `postprocess`

Translate model-native outputs into a task-facing bridge result.

```python
def postprocess(self, model_output: ModelOutputT) -> BridgeOutputT:
    task_output = self.decoder(model_output)
    return GoaltraceHRMV1BridgeOutput(
        task=task_output,
        control=ControlOutput(
            action_logits=model_output.action_logits,
        ),
    )
```

The bridge adapter's `postprocess` may combine decoder output with model-native
control values, but it must not compute loss or select actions.

### 2.5 Expose a stable bridge contract

The adapter produces the canonical surface consumed by objectives, controllers,
traces, and evaluators.

```python
@dataclass(frozen=True)
class GoaltraceHRMV1BridgeOutput:
    task: GoaltraceTaskOutput
    control: ControlOutput
```

Downstream systems consume this bridge output. They must not reach into
model-native outputs.

---

## 3. Bridge output anatomy

The bridge output is a split structure with strict semantics per field:

| Field         | Meaning                                         | Required by         | Optional                                      |
| ------------- | ----------------------------------------------- | ------------------- | --------------------------------------------- |
| `task`        | Prediction defined by the task contract         | Objectives, traces  | Never                                         |
| `control`     | Action logits (ACT regime)                      | Controllers         | Per regime                                    |
| `policy`      | Q-values (RL regime)                            | Controllers         | Per regime                                    |
| `critic`      | State value (RL regime)                         | Controllers         | Per regime                                    |
| `learning`    | Tensors explicitly required by objectives       | Objectives          | Pairing-dependent, but required when declared |
| `diagnostics` | Observability data not required for correctness | Evaluation, figures | Always optional                               |

Do not call objective-required tensors "diagnostics". If removing a field would
make training impossible, it belongs under `learning`. The `learning` field is
only present on pairings whose objectives need non-task intermediate tensors
(e.g., TEM spatial codes). Pairings whose objectives operate purely on
`task` logits (e.g., MazeHard+HRM) omit `learning`.

Do not force all families into one giant output dataclass with mostly optional
fields. Prefer concrete output types per pairing:

```python
# Good — concrete, no invalid states
@dataclass(frozen=True)
class MazeHardHRMV1BridgeOutput:
    task: MazeHardTaskOutput
    control: ControlOutput

@dataclass(frozen=True)
class MazeHardHRMV2BridgeOutput:
    task: MazeHardTaskOutput
    policy: PolicyOutput
    critic: CriticOutput

@dataclass(frozen=True)
class ArenaTEMBridgeOutput:
    task: ArenaTaskOutput
    learning: TEMLearningState
```

```python
# Avoid — invalid states are possible
@dataclass
class GenericBridgeOutput:
    task: TaskOutput | None = None
    control: ControlOutput | None = None
    policy: PolicyOutput | None = None
    critic: CriticOutput | None = None
```

---

## 4. State ownership

The model defines the state type and state semantics. The adapter only delegates
lifecycle operations because the runner needs a uniform entry point.

```python
def init_state(
    self,
    batch_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ModelStateT:
    return self.model.init_state(
        batch_size, device=device, dtype=dtype
    )

def reset_state(
    self,
    state: ModelStateT,
    reset_mask: Tensor,
) -> ModelStateT:
    return self.model.reset_state(state, reset_mask)
```

The adapter must not independently reconstruct model state fields.

The exception is adapter-owned recurrent state (e.g. a decoder with its own
recurrence). In that case, the adapter state must be explicit:

```python
@dataclass(frozen=True)
class AdapterState:
    model: ModelState
    decoder: DecoderState
```

Do not hide mutable recurrent state inside the adapter module.

---

## 5. Typed task input boundary

The adapter must accept a typed task input, not an arbitrary dictionary.

```python
# Preferred
def prepare_inputs(self, task_input: GoaltraceTaskInput) -> HRMInputV1:
    ...

# Avoid
def prepare_inputs(self, batch: Mapping[str, Tensor]) -> HRMInputV1:
    obs_id = batch["observation_id"]
```

The complete batch should distinguish inputs, targets, and metadata:

```python
@dataclass(frozen=True)
class GoaltraceBatch:
    inputs: GoaltraceTaskInput
    targets: GoaltraceTargets
    metadata: GoaltraceMetadata | None = None
```

Execution is then explicit:

```python
output, next_state = adapter(batch.inputs, state)
loss = objective(output, batch.targets)
```

This prevents accidental coupling between adapters and training-only fields.

---

## 6. Diagnostics design

Diagnostics should be observational and detached from core task outputs.

Three conceptual categories:

| Surface                         | Meaning                                                  |
| ------------------------------- | -------------------------------------------------------- |
| `task`                          | Prediction defined by the task contract                  |
| `control` / `policy` / `critic` | Action, halting, or control values                       |
| `learning`                      | Tensors explicitly required by objectives                |
| `diagnostics`                   | Optional observability data not required for correctness |

This distinction matters because diagnostics may later be disabled to reduce
memory. If a tensor is only used for figures, it belongs under `diagnostics`.

---

## 7. Configuration

Each family should have a `config.py` module containing adapter-owned settings.
Configuration describes adapter-owned choices only:

```python
# Good — adapter-owned choice
class GoaltraceHRMAdapterSettings(BaseModel):
    encoder_kind: Literal["learned", "rope"]
    decoder_hidden_size: int
```

```python
# Bad — duplicates model or task spec
class GoaltraceHRMAdapterSettings(BaseModel):
    hrm_num_layers: int        # belongs in HRM model config
    task_num_nodes: int        # belongs in task spec
```

Model architecture belongs in model config. Task dimensions belong in task
specs. The adapter may validate cross-contract compatibility using both, but
should not duplicate their settings.

All adapter settings should be Pydantic `BaseModel` with `extra="forbid"` to
catch misspelled fields at construction time.

---

## 8. Encoder modules

`encoders.py` contains reusable trainable and structural transformations.

```python
class GoaltraceRoPEEncoder(nn.Module):
    def forward(self, task_input: GoaltraceTaskInput) -> HRMInputV1:
        ...
```

An encoder may own:

- embeddings;
- input projections;
- role encoding;
- positional encoding;
- masks required by the model;
- model-native input construction.

An encoder must not own:

- target extraction;
- loss masks whose only purpose is objective computation;
- recurrent state;
- output interpretation.

The distinction is critical for SeqMaze: input masks needed by HRM belong in
the encoder. Output masks used only to determine supervised loss positions
belong in task targets or objective bindings.

---

## 9. Decoder modules

`decoders.py` contains task heads and output conversion.

```python
class GoaltraceMLPDecoder(nn.Module):
    def forward(self, model_output: HRMOutputV1) -> GoaltraceTaskOutput:
        ...
```

A decoder may own:

- task-specific linear or MLP heads;
- selection of model slots;
- reshaping;
- bounded output transforms (e.g. sigmoid);
- task-output construction.

A decoder must not own:

- loss functions;
- target comparison;
- metric computation;
- controller decisions.

---

## 10. Pairing modules (composition roots)

Files such as `goaltrace.py` and `mazehard.py` are composition roots. They
contain the concrete adapter, compatibility validation local to that pairing,
and a builder function.

```python
class GoaltraceHRMV1BridgeAdapter(nn.Module):
    def __init__(
        self,
        *,
        model: HRMV1,
        encoder: GoaltraceTokenEncoder,
        decoder: GoaltraceDecoder,
        config: GoaltraceHRMAdapterSettings,
    ) -> None:
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self._validate_compatibility()

    def prepare_inputs(self, task_input: GoaltraceTaskInput) -> HRMInputV1:
        return self.encoder(task_input)

    def postprocess(self, model_output: HRMOutputV1) -> GoaltraceHRMV1BridgeOutput:
        return GoaltraceHRMV1BridgeOutput(
            task=self.decoder(model_output),
            control=ControlOutput(
                action_logits=model_output.action_logits,
            ),
        )

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> HRMStateV1:
        return self.model.init_state(
            batch_size, device=device, dtype=dtype
        )

    def reset_state(
        self,
        state: HRMStateV1,
        reset_mask: Tensor,
    ) -> HRMStateV1:
        return self.model.reset_state(state, reset_mask)

    def forward(
        self,
        task_input: GoaltraceTaskInput,
        state: HRMStateV1,
    ) -> tuple[GoaltraceHRMV1BridgeOutput, HRMStateV1]:
        model_input = self.prepare_inputs(task_input)
        model_output, next_state = self.model(model_input, state)
        return self.postprocess(model_output), next_state
```

This is intentionally explicit. Do not introduce a generic orchestration
superclass merely to remove these lines. Composition is sufficient.

---

## 11. Builder functions

Each pairing should expose one builder that accepts already-resolved task and
model specifications.

```python
def build_goaltrace_hrm_v1_bridge(
    *,
    model: HRMV1,
    task_spec: GoaltraceTaskSpec,
    config: GoaltraceHRMAdapterSettings,
) -> GoaltraceHRMV1BridgeAdapter:
    encoder = build_goaltrace_encoder(
        task_spec=task_spec,
        model_config=model.config,
        config=config,
    )
    decoder = GoaltraceMLPDecoder(
        hidden_size=model.config.pfc.hidden_size,
        num_observations=task_spec.num_observations,
    )
    return GoaltraceHRMV1BridgeAdapter(
        model=model,
        encoder=encoder,
        decoder=decoder,
        config=config,
    )
```

The builder centralizes:

- dimension derivation;
- encoder selection;
- decoder construction;
- compatibility checks;
- stable defaults.

Experiment modules should not reproduce these details:

```python
# Experiment module — clean
model = build_hrm_v1(model_config)
bridge = build_goaltrace_hrm_v1_bridge(
    model=model,
    task_spec=task_spec,
    config=adapter_config,
)
```

---

## 12. Canonical public API

The stable public surface of every adapter is:

```python
adapter.config        # immutable adapter settings
adapter.model         # the wrapped model

adapter.init_state(batch_size, ...)        # → ModelState
adapter.reset_state(state, reset_mask)     # → ModelState
adapter.prepare_inputs(task_input)         # → ModelInput
adapter.postprocess(model_output)          # → BridgeOutput

adapter(task_input, state)                 # → (BridgeOutput, next_state)
```

Everything else is private implementation or belongs to another subsystem.

---

## 13. Protocol (`contracts.py`)

The top-level `contracts.py` defines the structural protocol that all adapters
satisfy. Concrete adapters remain ordinary `nn.Module` classes — the protocol
is for type-checking runners, controllers, and factories.

```python
from __future__ import annotations

from typing import Protocol, TypeVar

from torch import Tensor
from torch.nn import Module

TaskInputT = TypeVar("TaskInputT", contravariant=True)
ModelInputT = TypeVar("ModelInputT", covariant=True)
ModelOutputT = TypeVar("ModelOutputT", contravariant=True)
ModelStateT = TypeVar("ModelStateT")
BridgeOutputT = TypeVar("BridgeOutputT", covariant=True)

class BridgeAdapter(
    Protocol[TaskInputT, ModelInputT, ModelOutputT, ModelStateT, BridgeOutputT],
):
    """Structural protocol for all task–model bridge adapters."""

    model: Module

    def prepare_inputs(self, task_input: TaskInputT) -> ModelInputT: ...
    def postprocess(self, model_output: ModelOutputT) -> BridgeOutputT: ...
    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> ModelStateT: ...
    def reset_state(
        self,
        state: ModelStateT,
        reset_mask: Tensor,
    ) -> ModelStateT: ...
    def forward(
        self,
        task_input: TaskInputT,
        state: ModelStateT,
    ) -> tuple[BridgeOutputT, ModelStateT]: ...
```

The protocol declares `forward`, not `__call__`, because concrete
implementations are `nn.Module` subclasses where `__call__` is
framework-managed and dispatches through hooks. Typing `forward` better
reflects PyTorch ownership.

Do not create an abstract `BaseBridgeAdapter` unless it contains meaningful
shared behavior. A three-line common `forward()` does not justify inheritance.

---

## 14. Adapter naming

Names encode both sides of the pairing and the model version.
The canonical form uses the full `BridgeAdapter` suffix:

```
ArenaTEMV1BridgeAdapter
MazeHardHRMV1BridgeAdapter
MazeHardHRMV2BridgeAdapter
GoaltraceHRMV1BridgeAdapter
RoutebindHRMV1BridgeAdapter
SeqMazeHRMV1BridgeAdapter
SeqMazeHRMV2BridgeAdapter
```

For subcomponents, include the task and family:

```
GoaltraceHRMEncoder
GoaltraceHRMDecoder
MazeHardEHPEncoder
ArenaTEMDecoder
```

Builder functions use the canonical short form
`build_{task}_{family}_{version}_bridge`:

```
build_goaltrace_hrm_v1_bridge
build_mazehard_hrm_v1_bridge
build_mazehard_hrm_v2_bridge
build_arena_tem_v1_bridge
```

Avoid ambiguous names that lack task and family context:

```
Bridge           — which bridge?
BaseAdapter      — what task? what model?
AdapterOutput    — whose output?
GenericDecoder   — decoding what?
```

---

## 15. Package structure

### Target structure

```
src/ehc_sn/adapters/
├── __init__.py                  # exports BridgeAdapter protocol only
├── contracts.py                 # BridgeAdapter protocol, type vars
├── tem/
│   ├── __init__.py              # family public barrel
│   ├── config.py                # ArenaTEMAdapterSettings
│   ├── contracts.py             # ArenaTEMBridgeOutput, TEMLearningState
│   ├── encoders.py              # ArenaTwoHotEncoder, ArenaTEMInputEncoder
│   ├── decoders.py              # SingleScaleObservationDecoder, ArenaTEMOutputDecoder
│   ├── arena.py                 # ArenaTEMV1BridgeAdapter, ArenaTEMV2BridgeAdapter, builder
│   └── traces.py                # trace field definitions (observer integration)
├── hrm/
│   ├── __init__.py              # family public barrel
│   ├── config.py                # all HRM adapter settings
│   ├── contracts.py             # bridge outputs, control/policy/critic types
│   ├── encoders.py              # all task encoders
│   ├── decoders.py              # all task decoders
│   ├── mazehard.py              # MazeHardHRMV1BridgeAdapter, MazeHardHRMV2BridgeAdapter, builders
│   ├── goaltrace.py             # GoaltraceHRMV1BridgeAdapter, builder
│   ├── routebind.py             # RoutebindHRMV1BridgeAdapter, builder
│   ├── seqmaze.py               # SeqMazeHRMV1BridgeAdapter, SeqMazeHRMV2BridgeAdapter, builders
│   └── traces.py                # trace field definitions (observer integration)
└── ehp/
    ├── __init__.py              # family public barrel
    ├── config.py                # all EHP adapter settings
    ├── contracts.py             # bridge outputs, diagnostics
    ├── encoders.py              # ArenaEHPEncoder, MazeHardEHPEncoder
    ├── decoders.py              # ArenaEHPDecoder, MazeHardEHPDecoder
    ├── arena.py                 # ArenaEHPV1BridgeAdapter, builder
    ├── mazehard.py              # MazeHardEHPV1BridgeAdapter, builder
    └── traces.py                # trace field definitions (observer integration)
```

### `_base.py` deprecation

Do not use `_base.py` as a permanent architecture. It tends to become an
unstructured accumulation point for unrelated abstractions. Prefer separate
`config.py`, `contracts.py`, `encoders.py`, and `decoders.py` modules.

---

## 16. Family `__init__.py` conventions

Family barrels should export complete public pairings and their settings, but
not low-level internals unless those internals are intentionally reusable.

```python
# hrm/__init__.py — good
from .config import (
    GoaltraceHRMAdapterSettings,
    MazeHardHRMAdapterSettings,
    RoutebindHRMAdapterSettings,
    SeqMazeAdapterSettings,
)
from .goaltrace import (
    GoaltraceHRMV1BridgeAdapter,
    build_goaltrace_hrm_v1_bridge,
)

__all__ = [
    "GoaltraceHRMAdapterSettings",
    "GoaltraceHRMV1BridgeAdapter",
    "MazeHardHRMAdapterSettings",
    ...
]
```

Do not export encoder classes, decoder classes, or builder helpers unless
another package is explicitly expected to compose those parts directly.

The top-level package should export only the protocol:

```python
# adapters/__init__.py — good
from .contracts import BridgeAdapter

__all__ = ["BridgeAdapter"]
```

Users import concrete pairings from family barrels:

```python
from ehc_sn.adapters.hrm import GoaltraceHRMV1BridgeAdapter
```

---

## 17. Objective bindings

Objective-specific extraction and target logic does **not** belong in adapters.
It belongs in `ehc_sn.objectives.task/` alongside the task evaluators.

Correct ownership:

| Concern                                  | Location                         |
| ---------------------------------------- | -------------------------------- |
| Adapter protocol                         | `adapters/contracts.py`          |
| Bridge output types                      | `adapters/{family}/contracts.py` |
| Task evaluators (loss, targets, metrics) | `objectives/task/{task}.py`      |
| Composite regime scorers                 | `objectives/composites/`         |

Objective bindings should depend on narrow protocols, not concrete adapter
classes:

```python
# Good — depends on protocol, not implementation
class HasGoaltraceTaskOutput(Protocol):
    task: GoaltraceTaskOutput

class GoaltraceACTBinding:
    def extract_prediction(self, output: HasGoaltraceTaskOutput) -> Tensor:
        return output.task.firing_field
```

---

## 18. Trace placement

`traces.py` lives inside each adapter family package as an **observer
integration module**, not as part of the adapter's execution implementation.
Trace definitions may live in the adapter family package for semantic locality
(because they interpret bridge outputs), but adapters and their contracts must
not depend on the trace subsystem.

Trace fields should depend only on:

- the executed task record (`ctx.record.batch`);
- the public bridge output (`ctx.record.outputs.backbone_output`);
- declared diagnostic contracts.

```python
# Valid
TraceField(
    name="pred/firing_field",
    get=lambda ctx: ctx.record.outputs.backbone_output.task.firing_field,
)
```

```python
# Not valid — reaching into model internals
TraceField(
    name="pfc/internal",
    get=lambda ctx: ctx.adapter.model.blocks[2]._cached_hidden,
)
```

When a trace requires an internal value, expose it deliberately through the
`diagnostics` section of the bridge output.

---

## 19. Registry decision

Do not add a global adapter registry as a default architectural component.

A registry is justified only when **all** of these are true:

1. adapter selection is data-driven (e.g. config file says `task=goaltrace, model=hrm-v1`);
2. pairings are resolved at runtime;
3. multiple callers currently implement the same dispatch logic;
4. import-time registration will not create plugin-ordering problems.

Until then, explicit construction is cleaner:

```python
from ehc_sn.adapters.hrm import build_goaltrace_hrm_v1_bridge

bridge = build_goaltrace_hrm_v1_bridge(
    model=model,
    task_spec=task_spec,
    config=adapter_config,
)
```

If a registry becomes necessary later, register builder functions, not classes,
and keep the registry as a resolver only — it should not own configuration
defaults or model construction.

---

## 20. Import dependency rules

The desired dependency direction uses distinct relationship types:

```
tasks/contracts ──▶ adapters ◀── models/contracts
                         │
                         ▼
              public bridge contracts
                │       │       │
                ▼       ▼       ▼
          objectives controllers traces

runner ──▶ adapter (invocation)
```

- **Adapters import** task contracts and model contracts.
- **Objectives, controllers, and traces import** adapter bridge contracts
  (output types and protocols), but not concrete adapter classes.
- **The runner invokes** the adapter; the adapter does not depend on the runner.

Adapters may import:

- task contracts;
- model contracts and model implementations;
- shared tensor utilities;
- adapter-local configuration.

Adapters must not import:

- objectives;
- controllers;
- evaluation;
- reports;
- experiment modules;
- training runtime.

Objectives may import adapter contracts (bridge output types and protocols),
but ideally not concrete adapter classes.

These rules can be enforced with a simple architectural test that scans for
forbidden import prefixes:

```python
FORBIDDEN_ADAPTER_IMPORT_PREFIXES = (
    "ehc_sn.objectives",
    "ehc_sn.controllers",
    "ehc_sn.eval",
    "ehc_sn.experiments",
    "ehc_sn.training",
)
```

---

## 21. Composition over inheritance

Use composition rather than a hierarchy such as:

```
BaseAdapter → RecurrentAdapter → SpatialAdapter → TEMAdapter → ArenaTEMV1Adapter
```

That hierarchy will become difficult to reason about as pairings grow.

A small protocol plus concrete modules is sufficient:

```python
class ArenaTEMV1BridgeAdapter(nn.Module):
    encoder: ArenaTEMInputEncoder
    model: TEMV1
    decoder: ArenaTEMOutputDecoder
```

Reusable behavior should be extracted into functions or components, not
abstract base classes. A helper function is acceptable when the orchestration
truly is identical:

```python
def run_bridge_step(
    *,
    encoder,
    model,
    decoder,
    task_input,
    state,
):
    model_input = encoder(task_input)
    model_output, next_state = model(model_input, state)
    bridge_output = decoder(model_output)
    return bridge_output, next_state
```

Even this helper may be unnecessary — the three-line orchestration is clearer
than another abstraction.

---

## 22. Final design rule

The cleanest invariant is:

> A **task** defines what data and predictions mean.
> A **model** defines how native representations and state evolve.
> An **adapter** defines how one task is represented to one model and how the
> model result is interpreted for that task.
> A **runner** defines when the adapter is executed.
> An **objective** defines how the result is optimized.
> A **trace profile** defines what is observed.

The adapter is a narrow composition layer. It translates contracts, invokes the
model, and exposes a stable bridge result. Nothing more.

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

Replay controllers operate on source-provided replay rows through controller-
owned slot continuity, not by pushing progression state back into the source or
by storing full trajectories in carry.

**Slot-authoritative invariant (non-negotiable):** carry is the only continuity
authority. Active slots never consult the incoming source batch for trajectory
data. Dataloader batch boundaries and runner chunk boundaries are not reset
boundaries. Trajectory identity changes only on the halted→admitted boundary.

Canonical replay-carry pattern:

```python
@dataclass
class ReplayCarry:
    model_state: object
    cursor: Tensor            # (B,) int64 — per-slot step position
    trajectory_length: Tensor # (B,) int64 — authoritative, set at admission only
    halted: Tensor            # (B,) bool  — slot done; next step will admit
    trajectory_id: Tensor     # (B,) int64 — stable dataset identity; -1 = not yet admitted
    resident_payload: dict[str, Tensor]  # carry-owned trajectory arrays for admitted row
    data: dict[str, Tensor]   # current-step extracted payload (small, included in snapshot)
    task_state: dict[str, Tensor]  # task-local carry state
```

Rules:

- Replay sources own immutable full replay rows and any full batch-major
  replay tensors. They do not own per-slot execution continuity after a batch
  has been emitted.
- Replay carry owns slot-local continuity: recurrent state, cursor, stop facts
  (`trajectory_length`, `halted`), stable identity (`trajectory_id`), and
  carry-owned trajectory arrays (`resident_payload`) for the admitted episode.
- `resident_payload` stores the trajectory columns for the admitted row. It is
  NOT included in `CarrySnapshot` (and thus not cloned into per-step records),
  satisfying the "no large tensors in per-step snapshots" constraint.
- `data` stores the current-step extracted payload (small `(B, ...)` tensors)
  and IS included in `CarrySnapshot.data`.
- Active slots read exclusively from `resident_payload` at the authoritative
  cursor position. They never consult the incoming source batch.
- Halted slots may admit exactly one new candidate row per step, atomically
  replacing `trajectory_id`, `resident_payload`, `cursor`, `trajectory_length`,
  `steps`, `task_state`, and resetting `model_state`.
- `trajectory_id` is fit-path identity only (non-model-visible). It must NOT
  appear in `ArenaTaskInput` or any model input.
- Replay runtime batches are batch-major: `B` is the leading axis and `T` is
  the second axis.
- Current-step replay payloads are derived from the source batch plus the
  carry-owned continuity state. Step records and runner snapshots should
  project only current-step data plus the minimal post-step continuity needed
  downstream.

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

---

## 7 Lightning Evaluation Surface

Every Lightning training family exposes a pair of methods that let the
out-of-band evaluation regime runner drive the family without touching the
fit-path `val_metrics`.

### Protocol

```python
class SupportsEvaluationRegimes(Protocol):
    def build_evaluation_metrics(self, namespace: str) -> MetricCollection: ...
    def execute_evaluation_batch(
        self,
        batch: Batch,
        trace_request: Optional[EvaluationTraceRequest],
    ) -> EvaluationBatchArtifacts: ...
```

Defined in `ehc_sn.lightning.eval.contracts`.

### `build_evaluation_metrics(namespace)`

- Returns a **fresh** `MetricCollection` keyed to the family's episode route
  table and prefixed with `namespace`.
- Must use `.clone(prefix=namespace)` on the same route table used for `val_metrics`.
- The runner calls this once per regime run; the collection is discarded after
  metrics are computed.

### `execute_evaluation_batch(batch, trace_request)`

- Runs a deterministic rollout identical to `validation_step` but **does not**
  update `self.val_metrics`.
- Returns `EvaluationBatchArtifacts` with:
  - `evaluated`: the evaluated chunk from the rollout.
  - `apply_to_metrics`: a closure that stamps the evaluated chunk into any
    `MetricCollection` keyed by the family's route table. The runner calls
    this on the collection returned by `build_evaluation_metrics`.
  - `trace`: populated only when `trace_request.enabled` is `True`.
- Never logs metrics directly; logging is the caller's responsibility.

### Trace request handshake

- `trace_request` is `None` when the regime or runner does not need a trace.
- When non-`None` and `trace_request.enabled` is `True`, the family builds a
  `TraceTree` from `trace_request.key_set()` and includes it in the artifact.
- Families that have a fixed trace spec (e.g. HRM families) may ignore the
  key set and use their canonical spec; this is acceptable for v1.

### Namespace rules

- `val/` — fit-path metrics only (`validation_step`). Never written by regimes.
- `diag/<regime_id>/` — diagnostic regime metrics (out-of-band, driven by
  `EvaluationRegimesCallback`).
- `bench/<regime_id>/` — benchmark regime metrics (reserved for future use).

### Families

All 5 Lightning families implement this surface in parallel with identical
method names:

| Family | Route table          | Module                        |
| ------ | -------------------- | ----------------------------- |
| TEM v1 | `TEM_EPISODE_ROUTES` | `ehc_sn.lightning.tem.tem_v1` |
| TEM v2 | `TEM_EPISODE_ROUTES` | `ehc_sn.lightning.tem.tem_v2` |
| EHC v1 | `EHC_EPISODE_ROUTES` | `ehc_sn.lightning.ehc.ehc_v1` |
| HRM v1 | `ACT_EPISODE_ROUTES` | `ehc_sn.lightning.hrm.hrm_v1` |
| HRM v2 | `RL_EPISODE_ROUTES`  | `ehc_sn.lightning.hrm.hrm_v2` |

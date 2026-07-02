# TODO

## deferred-post-chunk-hebbian-clamp

**Status:** not-started

**Location:** `src/ehp_sn/lightning/modules/variational_replay.py` lines 501–515
(post-chunk hook) and `src/ehp_sn/modules/hpc/modules.py` line 154
(`finalize_memory`).

**Problem:** HPC Hebbian memory matrices are hard-clamped to `[clamp_min,
clamp_max]` **outside** the computation graph, once per BPTT chunk boundary.
The clamp runs after `.detach()` — gradients never see it, creating a
discontinuity at chunk boundaries and breaking gradient-based optimization
of the Hebbian dynamics.

**Long-term fix:** Move clamping **inside** the computation graph, applied
at every step, so that:

- Gradients flow through the bounded operation.
- No discontinuity when the carry crosses a BPTT chunk boundary.
- The model owns its state invariants — no manual post-chunk state surgery.

**Preferred approach:** Replace hard `torch.clamp` with a differentiable
bound (e.g. `tanh`-scaled or soft-clamp) applied per-step inside
`_update_memory_impl` / `clamp_memory`. Remove `finalize_memory` and the
post-chunk hook entirely once the per-step bound is in place.

**Acceptance criteria:**

- `grep -rn "finalize_memory" src/` returns zero matches.
- `grep -rn "Deferred post-chunk Hebbian clamp" src/` returns zero matches.
- HPC memory values stay bounded `[clamp_min, clamp_max]` across BPTT
  boundaries without manual intervention.
- No training regression on mazehard / arena parity benchmarks.

## hpc-prefix-bias-as-instance-context

**Status:** not-started

**Location:**

- `src/ehp_sn/modules/pfc/__init__.py` lines 340–345 (`prefix_bias` on CLS token, currently unused by adapters)
- `src/ehp_sn/adapters/hrm/_base.py` lines 55–130 (token encoder, no instance-level conditioning)
- `src/ehp_sn/adapters/hrm/mazehard.py` lines 65–70 (`_make_input_v1` always passes `prefix_bias=None`)
- `src/ehp_sn/modules/hpc/query_policy.py` (cue-family retrieval operators)
- Legacy reference: `legacy_hrm/models/hrm/hrm_act_v1.py` lines 140–160 (puzzle embedding prepend)

**Problem:** Legacy HRM uses per-instance puzzle embeddings (a learned lookup
table) to inject instance identity into the reasoning module. The modern PFC
has equivalent plumbing — `prefix_bias` on the CLS token — but adapters never
populate it. Without instance-level context, the model must infer which maze it's
solving from tokens alone, burning capacity on identification rather than
reasoning. A learned lookup table (legacy approach) is not a satisfactory
solution: it requires instance IDs at test time and does not generalize to
unseen instances.

**Research direction (target: EHP implementation):** Use HPC episodic retrieval
as the source of `prefix_bias`:

1. Encode the maze observation into an HPC sensory query cue (`"x"` family).
2. Retrieve the most similar episodic memory via attractor or attention read.
3. Feed the recalled place code as `prefix_bias` into `PFCModel.step()`.

This replaces instance-ID lookup with content-addressable memory — the model
retrieves context based on structural similarity, not a hardcoded ID. It
matches the biological HPC→PFC pathway (hippocampal contextual modulation of
prefrontal working memory) and generalizes to unseen instances at test time.

**Key design decisions to resolve:**

- What does the HPC store during training? Raw observations, structural features, or solution-path embeddings?
- Cue encoding: derive query from observation tokens or from an intermediate PFC representation?
- Read policy: single-shot cue read (`CueRead`) vs targeted read with source/target families (`TargetRead`)?
- Does the HPC memory persist across episodes (cumulative experience) or reset per episode?

**Acceptance criteria:**

- `prefix_bias` is populated by HPC retrieval in at least one HRM adapter (mazehard).
- `grep -rn "prefix_bias=None" src/ehp_sn/adapters/` returns zero matches for the adapted task.
- Mazehard training with HPC context matches or exceeds baseline (no prefix_bias) on sequence-exact accuracy.
- The mechanism uses content-addressable retrieval, not instance-ID lookup (no embedding table keyed by puzzle index).

## Phase 1 — Semantic View Extraction (eliminate full-state cloning)

### trace-views-tem-models

**Status:** not-started

**Location:** `src/ehp_sn/models/tem/tem_v1.py` class `TEMModelV1`,
`src/ehp_sn/models/tem/tem_v2.py` class `TEMModelV2`.

**Problem:** Trace fields that need model activations (e.g. `diagnostic/mec/location_mean`,
`diagnostic/hpc/memory`) declare `requires_model_state=True`, which causes the runner to
clone the **entire** model carry — including full HPC `MemoryState` with growing factor-memory
banks — into every per-step `CarrySnapshot`. Memory usage scales as
`O(num_steps × full_state_size)`.

**Fix:** Add a `trace_views(self, required: frozenset[str]) -> dict[str, Tensor]` method
on `TEMModelV1` and `TEMModelV2`. The method returns only the requested semantic views
(e.g. `"mec.cells"`, `"hpc.cells"`, `"hpc.memory"`, `"lec.cells"`, `"lec.filtered"`,
`"lec.sensory_code"`), each detached (`.detach()`). The `"hpc.memory"` view returns
`{"g_cued": dense, "x_cued": dense}` via `.to_dense()` to avoid exposing mutable factor-store
internals. Calling with `frozenset()` returns `{}`.

**Acceptance criteria:**

- `model.trace_views({"mec.cells"})` returns only the `"mec.cells"` key.
- `model.trace_views(frozenset())` returns `{}`.
- All returned tensors are detached (no autograd graph).
- `"hpc.memory"` view returns dense `(B, S, S)` tensors, not `FactorMemoryStore` objects.

### trace-views-hrm-models

**Status:** not-started

**Location:** `src/ehp_sn/models/hrm/` (HRM model classes).

**Problem:** Same as TEM — HRM trace fields (`pfc/z_H`, `pfc/z_L`) use
`requires_model_state=True`, triggering full-state cloning.

**Fix:** Add `trace_views(self, required: frozenset[str]) -> dict[str, Tensor]` to HRM
model classes. Supported view keys: `"pfc.z_H"`, `"pfc.z_L"`. Returns detached tensors;
`frozenset()` returns `{}`.

**Acceptance criteria:**

- `model.trace_views({"pfc.z_H"})` returns only the `"pfc.z_H"` key.
- All returned tensors are detached.
- Unit test mirrors TEM pattern.

### tracefield-dependencies-field

**Status:** not-started

**Location:**

- `src/ehp_sn/traces/observer.py` — `TraceField` dataclass.
- `src/ehp_sn/traces/specs.py` — all `TEM_TRACE_FIELDS`, `HRM_HIDDEN_STATE_FIELDS`, and
  `build_trace_spec`.
- `src/ehp_sn/evaluation/executor.py` — `_trace_request_needs_model_state`.

**Problem:** `TraceField.requires_model_state` is a binary flag. When any field sets it,
the entire `model_state` is cloned into `CarrySnapshot`. There is no way to declare
_which_ views a field needs — just that it needs _something_ from model state.

**Fix:** Add `dependencies: frozenset[str] = frozenset()` to `TraceField`. Keep
`requires_model_state` as deprecated (no warning emitted yet). Add
`TraceSpec.resolved_dependencies() -> frozenset[str]` returning the union of all field
dependencies. Update every `TraceField` in `specs.py` that uses `requires_model_state=True`
to declare precise `dependencies` (e.g. `dependencies=frozenset({"mec.cells"})`).
Update `_trace_request_needs_model_state` in `executor.py` to use `dependencies`.

**Acceptance criteria:**

- `grep -rn "requires_model_state=True" src/ehp_sn/traces/specs.py` returns zero matches.
- `TraceSpec.resolved_dependencies()` returns the correct union for each paradigm.
- `_trace_request_needs_model_state` checks `dependencies`, not `requires_model_state`.

### remove-snapshot-model-state

**Status:** not-started

**Location:**

- `src/ehp_sn/rollouts/runtime.py` — `Runner.run()`, `_snapshot_carry()`, `CarrySnapshot`,
  `SingleStepRunner`, `RecurrentRunner`.
- `src/ehp_sn/evaluation/executor.py` — `execute_replay_evaluation_batch`.
- `src/ehp_sn/lightning/modules/act_supervised.py` — passes `snapshot_model_state=True`.
- `src/ehp_sn/training/rollout.py` — passes `snapshot_model_state=False`.

**Problem:** The `snapshot_model_state` parameter controls whether `_snapshot_carry`
clones the entire `model_state` tree. With view-based extraction, this is no longer
needed — the runner should never clone model state into snapshots.

**Fix:** Remove `snapshot_model_state` parameter from `Runner.run()` protocol and all
implementations. Remove from `_snapshot_carry()`; `CarrySnapshot.model_state` becomes
always `None`. Remove `model_state` field from `CarrySnapshot` (or keep as `None`-only
with a comment explaining it's reserved for future use). Update all call sites:
remove the parameter from `execute_replay_evaluation_batch`, `act_supervised.py`, and
`training/rollout.py`.

**Acceptance criteria:**

- `grep -rn "snapshot_model_state" src/` returns zero matches.
- `CarrySnapshot.model_state` is always `None`.
- Existing evaluation configs produce identical manifest JSON outputs.
- Training path unaffected (already passes `False`).

### stepcontext-with-views

**Status:** not-started

**Location:**

- `src/ehp_sn/traces/observer.py` — `TraceObserver.observe()`.
- `src/ehp_sn/traces/specs.py` — trace field getter callables.
- `src/ehp_sn/evaluation/executor.py` — per-step observer closure.

**Problem:** `TraceField.get` callables receive the full `StepRecord` and access
`ctx.carry.model_state.mec.cells` etc. via `_TEMTraceContext` / `_HRMTraceContext`
protocols. After removing `CarrySnapshot.model_state`, these access paths are dead.

**Fix:** Define a `StepContext` dataclass in `traces/observer.py` (or `evaluation/contracts.py`):
`(index: int, record: StepRecord, views: dict[str, Tensor])`. Update
`TraceObserver.observe()` to accept `StepContext` and pass `ctx.views` to field getters.
Rewrite all trace field getter callables in `specs.py` to read from `ctx.views` instead
of `ctx.carry.model_state.*`. Remove the `_TEMTraceContext` / `_HRMTraceContext` protocols.

**Acceptance criteria:**

- `grep -rn "ctx.carry.model_state" src/ehp_sn/traces/specs.py` returns zero matches.
- `grep -rn "_TEMTraceContext\|_HRMTraceContext" src/ehp_sn/traces/specs.py` returns
  zero matches (or they are repurposed to use `views`).
- Existing trace field keys produce identical extracted values.

### wire-view-extraction-into-executor

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/executor.py` — `execute_replay_evaluation_batch`.

**Problem:** The executor currently decides `snapshot_model_state` based on
`_trace_request_needs_model_state` and passes it to the runner. Views must instead be
extracted per-step from the model and injected into the trace observer.

**Fix:** Before the runner loop (or inside the per-step observer), compute
`required_views` from `trace_request.trace_spec.resolved_dependencies()`. After each
controller step (or from the carry), call `model.trace_views(required_views)`. Build a
`StepContext` with `(index, record, views)` and pass it to the trace observer and any
accumulators. Remove the `snapshot_model_state` logic entirely.

**Acceptance criteria:**

- `execute_replay_evaluation_batch` does not reference `snapshot_model_state`.
- Trace extraction uses `StepContext.views`, not `record.snapshot.model_state`.
- TEM Arena evaluation with trace request produces identical trace tree contents.

## Phase 2 — Online Accumulators

### evaluation-consumer-protocols

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/contracts.py` (new protocols).

**Problem:** No abstraction exists for step-consuming evidence products other than
`TraceSink`. Spatial rate maps, grid scores, and place-field statistics are computed
post-hoc from full temporal traces loaded into memory.

**Fix:** Define two protocols in `evaluation/contracts.py`:

- `EvaluationConsumer`: `name: str`, `required_views: frozenset[str]`,
  `update(context: StepContext) -> None`, `close() -> None`.
- `MergeableAccumulator(EvaluationConsumer)`: `merge(other: Self) -> None`,
  `finalize() -> dict[str, Any]`, `state_dict() -> dict[str, Any]`,
  `load_state_dict(state: dict[str, Any]) -> None`, `reset() -> None`.

**Acceptance criteria:**

- Protocols are importable from `ehp_sn.evaluation.contracts`.
- `EvaluationConsumer` is runtime-checkable.
- `MergeableAccumulator` inherits from `EvaluationConsumer`.

### spatial-rate-map-metric

**Status:** not-started

**Location:** `src/ehp_sn/analysis/spatial/accumulators.py` (new file).

**Problem:** MEC and HPC spatial firing-rate maps are computed by loading the full
`diagnostic/mec/location_mean` trace `(T, U)` into memory, then aggregating per
spatial bin. Memory scales as `O(T × U)`.

**Fix:** Implement `SpatialRateMapMetric(torchmetrics.Metric)` in
`analysis/spatial/accumulators.py`:

- `__init__(height, width, n_units)`: `add_state("activation_sum", [n_bins, n_units])`,
  `add_state("occupancy", [n_bins])`.
- `update(activations, rows, cols, valid)`: scatter-add activations and occupancy into
  bounded states `O(B × U)`.
- `compute() -> dict[str, Tensor]`: returns `{rate_map: [H, W, U], occupancy: [H, W],
visited_mask: [H, W]}`.
- `full_state_update = False`, `dist_reduce_fx="sum"` for DDP safety.
- Uses `torch.float32` for accumulation regardless of input dtype.

**Acceptance criteria:**

- `merge(accumulate(first_half), accumulate(second_half)).compute()` produces identical
  rate maps to `accumulate(all_data).compute()` within float32 addition order.
- Peak memory during accumulation is `O(B × U)`, independent of trajectory length.
- Unit test validates merge correctness and numerical equivalence to post-hoc computation.

### spatial-autocorrelation-metric

**Status:** not-started

**Location:** `src/ehp_sn/analysis/spatial/accumulators.py`.

**Problem:** Grid scores and autocorrelograms are computed from full rate maps loaded
post-hoc. No online accumulator exists for these derived spatial statistics.

**Fix:** Implement a separate TorchMetrics accumulator that accepts pre-computed rate
maps (output of `SpatialRateMapMetric.compute()`) in its `update()`, or implement
`finalize()`-time computation. The autocorrelation is computed on the finalized rate map
(after occupancy normalization), not updated per-step. Reuses existing pure functions in
`analysis/spatial/gridness.py` (`compute_gridness`, `estimate_grid_spacing_orientation`).

**Acceptance criteria:**

- `compute()` returns `{autocorrelation, grid_scores, spacing, orientation}` with correct
  shapes and dtypes.
- Grid scores match post-hoc computation from `analysis/spatial/gridness.py`.
- Autocorrelograms have odd shape `(2H-1, 2W-1)`.

### grouped-accumulator-wrapper

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/accumulators.py` (new file).

**Problem:** Spatial maps must often be grouped by environment or case. Embedding grouping
logic into each accumulator duplicates code and mixes concerns.

**Fix:** Implement `GroupedAccumulator(accumulator_factory, key_fn)`:

- Maintains `dict[key, MergeableAccumulator]`.
- `update(context)`: calls `key_fn(context)` to determine group, routes to correct subgroup.
- `finalize()`: returns `{key: acc.finalize() for key, acc in groups.items()}`.
- `merge(other)`: matches subgroups by key and merges.
- `state_dict()` / `load_state_dict()`: serializes grouped state.

**Acceptance criteria:**

- Groups are created lazily on first `update()` per key.
- `merge()` with mismatched key sets merges overlapping keys and preserves unique keys
  from both sides.
- Unit test with two groups, merge, and finalize round-trip.

### wire-accumulators-into-executor

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/executor.py` — `execute_replay_evaluation_batch`.

**Problem:** The executor currently only supports `TraceSink` for step consumption.

**Fix:** Add `accumulators: Sequence[MergeableAccumulator] = ()` parameter. Compute
`required_views` as union of `trace_request.trace_spec.resolved_dependencies()` and
each accumulator's `required_views`. In the per-step observer, call
`accumulator.update(step_context)` for each accumulator after trace extraction and
scoring. After execution, call `accumulator.finalize()` for each accumulator and attach
results to `EvaluationCaseResult` via a new `accumulator_results: dict[str, Any] | None`
field.

**Acceptance criteria:**

- `execute_replay_evaluation_batch` accepts accumulators and updates them per step.
- `EvaluationCaseResult.accumulator_results` contains finalized accumulator outputs.
- Backward compatible: omitting accumulators produces identical results to current code.

## Phase 3 — Zarr Persistence

### add-zarr-dependency

**Status:** not-started

**Location:** `pyproject.toml` — `[project.dependencies]`.

**Problem:** Dense spatial arrays (rate maps, autocorrelograms) and temporal traces are
currently persisted as `.npz` via `TraceTree`, which requires full-file reads.

**Fix:** Add `"zarr"` to `pyproject.toml` dependencies. Pin minimum version `>=2.17`
for stable v3 API support. Verify import after install.

**Acceptance criteria:**

- `python -c "import zarr; print(zarr.__version__)"` succeeds.
- `zarr.__version__ >= "2.17"`.

### zarr-trace-sink

**Status:** not-started

**Location:** `src/ehp_sn/traces/sink.py`.

**Problem:** `InMemoryTraceSink` accumulates all step payloads in RAM via `TraceTree`.
No disk-backed, chunked alternative exists.

**Fix:** Implement `ZarrTraceSink` conforming to the `TraceSink` protocol:

- Constructor: `ZarrTraceSink(path, schema, chunk_size=256)`.
- `append(payload)`: buffers rows; writes chunk to Zarr group when chunk is full.
- `flush()`: writes partial chunk to disk.
- `finalize()`: flushes and closes the Zarr group, returns path or handle.
- Chunking along time axis: `chunks=(chunk_size,)` for 1-D fields,
  `(chunk_size, -1)` for 2-D fields.
- Compression: `zarr.Blosc(cname="zstd", clevel=3)`.

**Acceptance criteria:**

- `ZarrTraceSink` is `isinstance(TraceSink)`.
- Memory usage is bounded by `chunk_size × trace_payload_size`, independent of total
  steps.
- Written Zarr group can be opened with `zarr.open()` and sliced along the time axis.

### zarr-artifact-writer

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/artifacts.py`.

**Problem:** Accumulator outputs (`activation_sum`, `occupancy`, `rate_map`,
`autocorrelation`, `grid_scores`) need a persistence format that supports chunked
partial reads and compression.

**Fix:** Implement `ZarrArtifactWriter`:

- `write_spatial_metric(name, result: dict, root_path)`: writes each tensor in `result`
  as a Zarr array under `root_path/{name}/`.
- Chunking: spatial dims together, units chunked (e.g. `chunks=(30, 30, 64)` for
  `rate_map [30, 30, 256]`).
- Compression: `zarr.Blosc(cname="zstd", clevel=3)`.
- Writes `.zattrs` with dtype, shape, chunking, and compressor metadata.
- Writes scalar values (grid scores) as small 1-D arrays or JSON metadata.

**Acceptance criteria:**

- `zarr.open("spatial.zarr/mec/rate_map")[:, :, 128:192]` loads only the requested unit
  slice without reading the full array.
- Round-trip: write → read produces identical tensor values.
- `.zattrs` contains `dtype`, `shape`, `chunks`, `compressor` metadata.

### extend-manifest-for-zarr

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/artifacts.py` — `collect_regime_artifact_bundle` and
manifest schema.

**Problem:** The manifest JSON records traces as `.npz` paths and metadata keys. Zarr
artifacts are directory-based groups, not single files.

**Fix:** Add `"zarr_artifacts"` key to the manifest JSON schema. Each entry:
`{path: str, arrays: [{name: str, shape: list[int], chunks: list[int], dtype: str,
compressor: str}]}`. Update `collect_regime_artifact_bundle()` to call
`ZarrArtifactWriter` when accumulator results are present and record them in the manifest.

**Acceptance criteria:**

- `manifest.json` contains a `zarr_artifacts` key when spatial accumulators are used.
- Each Zarr array entry includes shape, chunks, dtype, and compressor.
- The `_SUCCESS` sentinel is written into the Zarr directory root.

## Phase 4 — Parquet + MLflow

### add-parquet-mlflow-dependencies

**Status:** not-started

**Location:** `pyproject.toml`.

**Problem:** `pyarrow` (for Parquet) and `mlflow` are not declared as dependencies.

**Fix:** Add `"pyarrow"` and `"mlflow"` to `[project.optional-dependencies]` under a new
`evaluation` extras group: `evaluation = ["pyarrow", "mlflow", "zarr"]`. Keep them optional so core
library evaluation does not require MLflow or Parquet.

**Acceptance criteria:**

- `pip install -e ".[evaluation]"` installs `pyarrow`, `mlflow`, and `zarr`.
- Core evaluation (alias scripts) works without these packages when not using MLflow.

### parquet-metrics-writer

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/artifacts.py`.

**Problem:** Per-case scalar metrics and per-unit spatial statistics are embedded in
manifest JSON or kept only in memory.

**Fix:** Implement `ParquetMetricsWriter`:

- `write_case_metrics(case_results, path)`: writes `cases.parquet` with columns
  `(case_id, n_steps, loss, completed, halt_step, ...)` plus accumulator-derived
  summaries.
- `write_unit_metrics(accumulator_results, path)`: writes `unit_metrics.parquet` with
  columns `(case_id, region, freq_band, unit_idx, grid_score, spatial_information,
peak_rate, field_area, ...)`.
- Uses `pyarrow.parquet.write_table()`.

**Acceptance criteria:**

- `cases.parquet` can be read with `pd.read_parquet()` or `pq.read_table()`.
- `unit_metrics.parquet` contains one row per (case, region, unit) tuple.
- Column names use snake_case.

### mlflow-registration

**Status:** not-started

**Location:** `scripts/evaluation/_cli.py` (or new `scripts/evaluation/mlflow_register.py`).

**Problem:** Evaluation runs have no experiment-tracking catalog. Provenance is only in
`manifest.json` within the artifact directory.

**Fix:** Add `--mlflow` flag and `--mlflow-experiment` option to the shared CLI factory in `_cli.py`. When set:

- Read `manifest.json`.
- Start MLflow run with `run_name=evaluation_id`.
- Log parameters from manifest: `model_family`, `task`, `checkpoint_digest`,
  `capture_profile`, `dataset_split`, `n_cases`, spatial binning, smoothing config.
- Log scalar metrics from manifest.
- Log all artifacts via `mlflow.log_artifacts(str(output_dir))`.
- Set tags: `git_commit`, `dataset_version`, `artifact_schema_version`, `model_family`,
  `task`.

Or implement as a separate `register` subcommand that registers an already-completed
evaluation artifact directory.

**Acceptance criteria:**

- `python scripts/evaluation/arena/tem_v1.py run ... --mlflow` registers the run without
  errors.
- MLflow UI shows parameters, metrics, and artifact links for the registered run.
- Omitting `--mlflow` produces identical evaluation artifacts as before (no regression).

## Phase 5 — Event Sink (deferred)

### event-sink-protocol

**Status:** not-started

**Location:** `src/ehp_sn/traces/sink.py` (or `src/ehp_sn/evaluation/contracts.py`).

**Problem:** Sparse events (halt, memory write, retrieval failure, NaN detection,
incorrect prediction) are currently captured as dense per-step trace fields or not at
all.

**Fix:** Define `EventSink` protocol: `emit(event: EvaluationEvent) -> None`,
`close() -> None`. Define `EvaluationEvent` dataclass: `step: int`, `case_id: str`,
`event_type: str`, `payload: dict[str, Any]`. Defer implementation of concrete sinks.

**Acceptance criteria:**

- `EventSink` protocol is importable and runtime-checkable.
- `EvaluationEvent` dataclass is JSON-serializable.

### parquet-event-sink

**Status:** not-started

**Location:** `src/ehp_sn/traces/sink.py`.

**Problem:** No concrete event sink exists for sparse event persistence.

**Fix:** Implement `ParquetEventSink`: writes events to `events.parquet` with columns
`(case_id, step, event_type, payload_json)`. Uses `pyarrow.parquet.write_table()` in
append mode or batched writes.

**Acceptance criteria:**

- Events written to `events.parquet` survive process restart.
- Querying `WHERE event_type = 'halt'` returns all halt events across cases.

## Phase 6 — Budget Enforcement (deferred)

### capture-budget-types

**Status:** not-started

**Location:** `src/ehp_sn/evaluation/contracts.py` or `src/ehp_sn/traces/specs.py`.

**Problem:** No mechanism exists to estimate or enforce trace memory budgets before
evaluation starts. Memory exhaustion is a runtime crash, not a validated configuration
error.

**Fix:** Define `CaptureBudget(max_bytes_per_case, max_bytes_per_run,
max_resident_cpu_buffer, max_queued_chunks)`. Define
`TracePlan.estimate_bytes(num_cases, max_steps, model_schema) -> int`. Implement
`validate_budget(plan, budget)` that raises `TraceBudgetExceeded` with downgrade
suggestions (reduce cases, increase stride, sample units, use float16). Defer
integration into the evaluation pipeline.

**Acceptance criteria:**

- `TraceBudgetExceeded` is raised with actionable suggestions when a plan exceeds budget.
- Estimate is within ±20% of actual memory usage for a representative TEM Arena config.
- `validate_budget` is a pure function with no side effects.

---

## Add cli for task and remove scripts/data-gen

---

## Add cli for training and remove scripts/training

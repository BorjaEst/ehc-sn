# EHC-SN Data Contracts Specification

> Non-required companion spec for processed-data format, pipeline contracts, and
> dataset-facing data surfaces.

## 1 Scope

This spec defines the low-frequency reference material for persisted maze data:

- raw → interim → processed pipeline;
- canonical on-disk path grammar (versioned, immutable roots);
- shared-substrate and task-corpus dataset classes;
- authoritative root `manifest.json` and per-sample `index.jsonl`;
- dataset and datamodule output contracts;
- staged CLI semantics;
- task- and adapter-facing consumption notes.

Core ownership and boundary rules remain in `spec/spec-architecture.md` and
`spec/spec-requirements.md`.

---

## 2 Pipeline

```text
Generators (maze-nd, dungeongen)          scripts/data-gen/
    │
    ▼
  data/raw/
    │  (provenance-only: source files, no normalization)
    ▼
  data/interim/   ◄────── source-specific normalization / augmentation scripts
    │
    ▼
  data/processed/<shared-family>/v<N>/    ← shared substrate (immutable version root)
    │  (task-neutral: topology, shared schema, shared manifest)
    ▼
  data/processed/<task-name>/<corpus-name>/v<N>/  ← task-prepared corpus (immutable)
    │  (task-protocol: replay rows, episode schema, task supervision)
    ▼
  ehc_sn.data.datasets / datamodules  ──► tasks/ ──► adapters/
```

Rules:

- `data/raw/` stores untouched source output and is not version controlled.
- `data/interim/` stores source-specific normalized output.
- `data/processed/` stores canonical source-agnostic output consumed by
  dataloaders. All roots under `data/processed/` are immutable versioned leaves.
- `scripts/data-gen/` contains thin CLI entrypoints only; no reusable logic.
- Reusable raw acquisition, canonicalization, and writer logic lives under
  `src/ehc_sn/data/`.
- Build reports and benchmark manifests are **not** canonical dataset contents
  and **must not** live under `data/processed/`. Use `outputs/` or `reports/`.

---

## 3 Canonical Path Grammar

### 3.1 Shared Substrate Root

```text
data/processed/<shared-family>/v<integer>/
```

A **shared substrate** stores task-neutral, provenance-owned channels (at
minimum `topology`). The `<shared-family>` name is owned by the upstream
source, not by any task. Three sources currently registered:

| Source         | Shared family | Upstream raw source                    |
| -------------- | ------------- | -------------------------------------- |
| HuggingFace HF | `maze-nd`     | `flaitenberger/maze_hard_augmented`    |
| dungeongen     | `dungeongen`  | dungeongen library (local generation)  |
| synthetic      | `numberline`  | generated on-the-fly (no external dep) |

### 3.2 Task-Prepared Corpus Root

```text
data/processed/<task-name>/<corpus-name>/v<integer>/
```

A **task corpus** stores task-protocol channels (replay rows, episode data,
task-supervision labels) and carries the shared channels it extends. The
`<task-name>` matches the owning package under `tasks/`. The `<corpus-name>`
is a corpus-specific label (e.g. `default`).

| Task      | Corpus root prefix          | Parent substrate  |
| --------- | --------------------------- | ----------------- |
| mazehard  | `data/processed/mazehard/`  | `maze-nd/v<N>`    |
| dungeon   | `data/processed/dungeon/`   | `dungeongen/v<N>` |
| arena     | `data/processed/arena/`     | `dungeongen/v<N>` |
| countwalk | `data/processed/countwalk/` | `numberline/v<N>` |

### 3.3 Immutability Invariant

A versioned root must not be modified after creation. Rebuilding requires
bumping the version integer. Writers must fail with `FileExistsError` when the
target version root already exists. No overwrite flags exist on canonical roots.

### 3.4 Transactional Build Semantics

Builders write to a temporary sibling directory (`.building-v<N>`) and only
rename it to the final version leaf after structural validation succeeds. If
the build fails for any reason, the final version leaf must not exist. This
ensures that any directory named `v<N>` is complete and valid.

### 3.5 Version Authority Rule

The `v<N>` path leaf is the single authority for the version integer. Builder
functions derive `version` from the path leaf; they do not accept a separate
`version` argument. The CLI derives canonical output roots from
family/task/corpus/version numbers, not from arbitrary path overrides.

### 3.6 Collision Rule

A shared-family name and a task namespace must not be the same string. The
registered shared families (`maze-nd`, `dungeongen`, `numberline`) do not collide with the
task namespaces (`mazehard`, `dungeon`, `arena`).

---

## 4 Root Manifest

### 4.1 Location

Path: `data/processed/<...>/v<N>/manifest.json`

The `manifest.json` at the version root is the authoritative descriptor for
the dataset version. Written once at build time; never overwritten.

### 4.2 Shared Substrate Manifest Fields

| Field                   | Type            | Description                                                                                       |
| ----------------------- | --------------- | ------------------------------------------------------------------------------------------------- |
| `schema_version`        | `int`           | Manifest schema version (currently `1`).                                                          |
| `dataset_class`         | `str`           | Always `"shared_substrate"`.                                                                      |
| `family`                | `str`           | Shared family name (e.g. `"maze-nd"`, `"numberline"`).                                            |
| `version`               | `int`           | Version integer, must match the `v<N>` path leaf.                                                 |
| `channels`              | `list[str]`     | Channel names present in every sample.                                                            |
| `topology_kind`         | `str`           | Topology identifier (e.g. `"grid2d"`, `"line1d"`).                                                |
| `n_states`              | `int`           | Total number of states in the substrate.                                                          |
| `extent`                | `list[int]`     | Dimension sizes (e.g. `[H, W]` for grid2d, `[N]` for line1d).                                     |
| `n_samples`             | `dict[str,int]` | Split → sample count.                                                                             |
| `source_id`             | `str`           | Canonical upstream source identifier.                                                             |
| `builder`               | `str`           | Fully-qualified builder function (e.g. `"ehc_sn.data.substrate.maze_nd.build_shared_substrate"`). |
| `seed`                  | `int`           | Seed used during materialization.                                                                 |
| `normalization_version` | `int`           | Normalization pipeline version (currently `1`).                                                   |
| `shared_schema_version` | `int`           | Shared-substrate schema version (currently `1`).                                                  |
| `stage_params`          | `dict`          | Deterministic parameters bundle passed to the builder.                                            |
| `producer_revision`     | `str`           | `ehc_sn` package version string at build time.                                                    |
| `input_fingerprint`     | `str`           | 16-char hex SHA-256 of `stage_params` (deterministic).                                            |
| `source_revision`       | `str`           | _(optional)_ Upstream source version or commit, when applicable.                                  |

### 4.3 Task Corpus Manifest Fields

Extends shared substrate fields with:

| Field                   | Type  | Description                                                               |
| ----------------------- | ----- | ------------------------------------------------------------------------- |
| `dataset_class`         | `str` | Always `"task_corpus"`.                                                   |
| `task`                  | `str` | Owning task namespace (e.g. `"mazehard"`).                                |
| `corpus`                | `str` | Corpus label (e.g. `"default"`).                                          |
| `parent_substrate`      | `str` | Canonical repo-relative path to the parent shared substrate version root. |
| `parent_family`         | `str` | Shared family name of the parent substrate.                               |
| `parent_version`        | `int` | Version integer of the parent substrate.                                  |
| `task_schema_version`   | `int` | Task-schema version (currently `1`).                                      |
| `task_protocol_version` | `int` | Task-protocol version (currently `1`).                                    |

All task corpora must declare `parent_substrate`. The value is a canonical
repo-relative path (e.g. `data/processed/dungeongen/v1`), never an absolute
filesystem path.

### 4.4 Manifest Identity Rules

- `version` in the manifest **must** equal the integer in the `v<N>` path leaf.
  Builders derive the version from the path; there is no independent version argument.
- `builder` must be a stable, fully-qualified Python dotted path.
- `stage_params` must contain only deterministic, reproducibility-bearing parameters.
- `input_fingerprint` is a 16-character hex prefix of SHA-256 over `stage_params`
  (JSON-serialized, keys sorted).
- Forbidden fields: timestamps, hostnames, absolute paths, machine-local details,
  `created_at`, audit blobs. These belong in `reports/` or `outputs/`.

---

## 5 Canonical On-Disk Layout

### 5.1 Shared Substrate (maze-nd)

```text
data/processed/maze-nd/v1/
├── manifest.json           ← authoritative root descriptor
├── index.jsonl             ← per-sample entries
├── train/
│   ├── dataset.json
│   ├── topology.npy        ← (N, H, W) bool
│   └── mask_valid.npy      ← (N, H, W) bool
├── val/
│   └── ...
└── test/
    └── ...
```

### 5.2 Shared Substrate (dungeongen)

```text
data/processed/dungeongen/v1/
├── manifest.json
├── index.jsonl
├── train/
│   ├── dataset.json
│   ├── topology.npy        ← (N, H, W) bool
│   ├── observations.npy    ← (N, H, W) int32
│   ├── mask_valid.npy      ← (N, H, W) bool
│   ├── regions.npy         ← (N, H, W) int32
│   └── landmarks.npy       ← (N, H, W) int32
├── val/
│   └── ...
└── test/
    └── ...
```

### 5.3 Task Corpus (mazehard)

```text
data/processed/mazehard/default/v1/
├── manifest.json
├── index.jsonl
├── train/
│   ├── dataset.json
│   ├── topology.npy        ← (N, H, W) bool
│   ├── mask_valid.npy      ← (N, H, W) bool
│   ├── start.npy           ← (N, H, W) bool
│   ├── goals.npy           ← (N, H, W) bool
│   └── solution.npy        ← (N, H, W) int32
└── ...
```

### 5.4 Task Corpus (dungeon)

```text
data/processed/dungeon/default/v1/
├── manifest.json
├── index.jsonl
├── train/
│   ├── dataset.json
│   ├── topology.npy                    ← (N, H, W) bool
│   ├── observations.npy               ← (N, H, W) int32
│   ├── mask_valid.npy                  ← (N, H, W) bool
│   ├── regions.npy                     ← (N, H, W) int32
│   ├── landmarks.npy                   ← (N, H, W) int32
│   ├── trajectory_row.npy              ← (N, T) int32
│   ├── trajectory_col.npy              ← (N, T) int32
│   ├── trajectory_previous_action.npy  ← (N, T) int32
│   ├── trajectory_episode_start.npy    ← (N, T) bool
│   ├── trajectory_valid_step.npy       ← (N, T) bool
│   └── trajectory_length.npy           ← (N,) int32
└── ...
```

### 5.5 Task Corpus (arena)

Arena is a task corpus over the `dungeongen` shared substrate. It inherits all
dungeongen spatial channels (topology, observations, mask_valid, regions,
landmarks) and adds Arena-owned trajectory channels.

```text
data/processed/arena/default/v1/
├── manifest.json
├── index.jsonl
├── train/
│   ├── dataset.json
│   ├── topology.npy                    ← (N, H, W) bool
│   ├── observations.npy               ← (N, H, W) int32
│   ├── mask_valid.npy                  ← (N, H, W) bool
│   ├── regions.npy                     ← (N, H, W) int32
│   ├── landmarks.npy                   ← (N, H, W) int32
│   ├── trajectory_row.npy              ← (N, T) int32
│   ├── trajectory_col.npy              ← (N, T) int32
│   ├── trajectory_previous_action.npy  ← (N, T) int32
│   ├── trajectory_episode_start.npy    ← (N, T) bool
│   ├── trajectory_valid_step.npy       ← (N, T) bool
│   └── trajectory_length.npy           ← (N,) int32
└── ...
```

### 5.6 Shared Substrate (numberline)

NumberLine is a purely synthetic 1-D shared substrate. All worlds share the
same bounded integer line topology.

```text
data/processed/numberline/v1/
├── manifest.json
├── index.jsonl
├── train/
│   ├── dataset.json
│   ├── state_ids.npy      ← (N, n_states) int32
│   ├── valid_prev.npy     ← (N, n_states) bool
│   └── valid_next.npy     ← (N, n_states) bool
├── val/  (same layout)
└── test/ (same layout)
```

### 5.7 Task Corpus (countwalk)

Countwalk is a task corpus over the `numberline` shared substrate. It adds
countwalk replay channels including five evaluation buckets (ID / range-OOD /
horizon-OOD / joint-OOD / stretch-OOD).

```text
data/processed/countwalk/default/v1/
├── manifest.json
├── index.jsonl
├── train/
│   ├── dataset.json
│   ├── world_id.npy                      ← (N,) int32
│   ├── cue_surface_id.npy                ← (N,) int32
│   ├── anchor_regime_id.npy              ← (N,) int32
│   ├── eval_bucket_id.npy                ← (N,) int32
│   ├── trajectory_value.npy              ← (N, T_max) int32
│   ├── trajectory_previous_action.npy    ← (N, T_max) int32
│   ├── trajectory_anchor_visible.npy     ← (N, T_max) bool
│   ├── trajectory_cue_tokens.npy         ← (N, T_max, W) int32
│   ├── trajectory_cue_mask.npy           ← (N, T_max, W) bool
│   ├── trajectory_episode_start.npy      ← (N, T_max) bool
│   ├── trajectory_valid_step.npy         ← (N, T_max) bool
│   ├── trajectory_length.npy             ← (N,) int32
│   ├── query_mask.npy                    ← (N, T_max) bool
│   ├── target_digits.npy                 ← (N, DIGIT_WIDTH) int32
│   ├── target_digit_mask.npy             ← (N, DIGIT_WIDTH) bool
│   └── target_value.npy                  ← (N,) int32
├── val/  (same layout; ID bucket only)
└── test/ (same layout; all 5 buckets; T_max = ood_max_steps)
```

---

## 6 Shared Substrate Schema

### 6.1 Mandatory Channel

All shared substrates must expose at least one topology channel. The channel
name and shape depend on the topology kind:

| Topology kind | Topology channel | dtype   | Shape         | Description                                 |
| ------------- | ---------------- | ------- | ------------- | ------------------------------------------- |
| `grid2d`      | `topology`       | `bool`  | `(H, W)`      | Passable cells (`True`) vs walls (`False`). |
| `line1d`      | `state_ids`      | `int32` | `(n_states,)` | Integer state identifiers 0..n_states-1.    |

### 6.2 Permitted Shared Optional Channels

Task-neutral provenance channels derived from the upstream source:

| Channel      | Name           | dtype   | Shape    | Description                     |
| ------------ | -------------- | ------- | -------- | ------------------------------- |
| Valid mask   | `mask_valid`   | `bool`  | `(H, W)` | Largest reachable component.    |
| Observations | `observations` | `int32` | `(H, W)` | Unique observation ID per cell. |
| Regions      | `regions`      | `int32` | `(H, W)` | Room/region ID.                 |
| Landmarks    | `landmarks`    | `int32` | `(H, W)` | Structural landmark IDs.        |

### 6.3 Task-Owned Channels (not in shared substrate)

The following channels belong to task corpora and must not appear in shared
substrate builds:

| Channel    | Name           | Owning task   | Rationale                        |
| ---------- | -------------- | ------------- | -------------------------------- |
| Start      | `start`        | mazehard      | Task-defined episode start cell. |
| Goals      | `goals`        | mazehard      | Task-defined goal cells.         |
| Solution   | `solution`     | mazehard      | Supervised shortest-path labels. |
| Trajectory | `trajectory_*` | dungeon/arena | Replay protocol semantics.       |

---

## 7 Per-Sample Index (index.jsonl)

Path: `<version-root>/index.jsonl`

| Field              | Type              | Description                                                                                                                                             |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | `str`             | Unique sample identifier.                                                                                                                               |
| `source`           | `str`             | Generator or source dataset name.                                                                                                                       |
| `split`            | `str`             | `train`, `val`, or `test`.                                                                                                                              |
| `source_record_id` | `str \| null`     | Stable raw-source record identity (e.g. `"train:12345"`). Null for tasks that generate samples synthetically and have no upstream raw record to rejoin. |
| `shape`            | `tuple[int, int]` | Grid shape `(H, W)`.                                                                                                                                    |
| `channels`         | `list[str]`       | Channel names present in the split.                                                                                                                     |
| `n_observations`   | `int`             | Observation vocabulary size, or `0`.                                                                                                                    |
| `n_goals`          | `int`             | Number of goal cells, or `0`.                                                                                                                           |
| `difficulty`       | `str`             | Optional source-defined difficulty label.                                                                                                               |

---

## 8 Split Metadata (dataset.json)

Path: `<version-root>/<split>/dataset.json`

| Field           | Type        | Description                                  |
| --------------- | ----------- | -------------------------------------------- |
| `source`        | `str`       | Generator or source dataset name.            |
| `split`         | `str`       | Split name.                                  |
| `n_samples`     | `int`       | Number of samples in the split.              |
| `topology_kind` | `str`       | Topology kind (e.g. `"grid2d"`, `"line1d"`). |
| `n_states`      | `int`       | Total number of states.                      |
| `extent`        | `list[int]` | Dimension sizes.                             |
| `channels`      | `list[str]` | Channel names present in the split.          |

---

## 9 Staged CLI Semantics

Each data-gen script owns the stages within its asset slice only.
CLI ownership follows the asset boundary, not the task boundary.

### 9.1 Command definitions

| Command              | Semantic job                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `fetch-raw`          | Download or generate the upstream raw source.                                                |
| `prepare-interim`    | Read raw source, normalize, and write deterministic split files to `data/interim/<family>/`. |
| `materialize-shared` | Build the shared substrate version root.                                                     |
| `materialize-task`   | Build the task corpus version root.                                                          |
| `validate`           | Validate a version root against manifest and schema.                                         |
| `build-all`          | Pure alias — runs all stages within the script's ownership slice, in order.                  |

### 9.2 Asset-first ownership

CLI ownership is defined per asset, not per task pipeline:

| Script                                 | Owned commands                                                                |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| `scripts/data-gen/build-dungeongen.py` | `fetch-raw`, `prepare-interim`, `materialize-shared`, `validate`, `build-all` |
| `scripts/data-gen/build-arena.py`      | `materialize-task`, `validate`, `build-all`                                   |
| `scripts/data-gen/build-dungeon.py`    | `materialize-task`, `validate`, `build-all`                                   |
| `scripts/data-gen/build-numberline.py` | `materialize-shared`, `validate`, `build-all`                                 |
| `scripts/data-gen/build-countwalk.py`  | `materialize-task`, `validate`, `build-all`                                   |

Task CLIs (`build-arena.py`, `build-dungeon.py`) do **not** expose `fetch-raw`,
`prepare-interim`, or `materialize-shared`. Those stages are the exclusive
domain of the shared-family CLI (`build-dungeongen.py`).

### 9.3 Task build-all semantics

`build-all` on a task CLI is a pure alias for `materialize-task` within that
CLI's ownership slice. It does not generate raw, interim, or shared assets.
If the required parent shared substrate does not exist, `build-all` fails fast
with an actionable error directing the user to run
`scripts/data-gen/build-dungeongen.py build-all` or
`scripts/data-gen/build-dungeongen.py materialize-shared`.

### 9.4 validate ownership

Each CLI's `validate` command enforces the asset class it owns:

- `build-dungeongen.py validate` accepts only `shared_substrate` roots with
  `family == "dungeongen"`.
- `build-arena.py validate` accepts only `task_corpus` roots with
  `task == "arena"`.
- `build-dungeon.py validate` accepts only `task_corpus` roots with
  `task == "dungeon"`.
- `build-numberline.py validate` accepts only `shared_substrate` roots with
  `family == "numberline"`.
- `build-countwalk.py validate` accepts only `task_corpus` roots with
  `task == "countwalk"`.

### 9.5 Manifest invariants

Canonical manifests contain only deterministic, identity-bearing fields.
Timestamps, machine-local paths, and audit-only metadata must not appear in
`manifest.json`. Build-environment details belong in `reports/` or `outputs/`.

---

## 10 Dataset and Datamodule Output Contracts

`MazeDataset.__getitem__` returns a model-agnostic per-sample channel mapping.

| Key              | Type     | Shape  | Description                                      |
| ---------------- | -------- | ------ | ------------------------------------------------ |
| `<channel_name>` | `Tensor` | varies | Raw tensor for one stored channel in the sample. |

Rules:

- The dataset layer preserves stored channels for one sample.
- The dataset layer does not synthesize nested task or model payloads.
- Task-local semantic views are built in `tasks/`.
- Model-specific packing is built in `adapters/`.
- DataModules consume only versioned roots under `data/processed/`.

---

## 11 Runtime Consumption Notes

- `data/` owns persisted contracts and static loading only.
- `tasks/` own task schema, task corpus materialization, and task semantics.
- `adapters/` convert task observations into model-native payloads.
- `envs/` house reusable runtime kernels.
- Benchmark manifests and build reports must not live under `data/processed/`.
  Canonical location: `reports/benchmarks/` or `outputs/`.

---

## 12 Dungeongen Raw Snapshot Contract

This section is normative for the dungeongen raw/interim slice.

### 12.1 Overview

The dungeongen raw corpus is a **canonical immutable snapshot** stored as
tar shards containing one NPZ record per sample. It is not a cache and
must not be rewritten in place.

### 12.2 Raw Layout

```text
data/raw/dungeongen/
├── manifest.json
├── train/
│   ├── shard-00000.tar
│   ├── shard-00001.tar
│   └── ...
├── val/
│   ├── shard-00000.tar
│   └── ...
└── test/
    ├── shard-00000.tar
    └── ...
```

### 12.3 Raw Manifest Contract

Path: `data/raw/dungeongen/manifest.json`

The manifest is the authoritative identity descriptor for the snapshot. It is
written once at creation time. It must not be overwritten or modified in place.

| Field                   | Type   | Description                                                                                        |
| ----------------------- | ------ | -------------------------------------------------------------------------------------------------- |
| `schema_version`        | `int`  | Raw manifest schema version (currently `1`).                                                       |
| `snapshot_kind`         | `str`  | Always `"materialized_snapshot"`.                                                                  |
| `source_id`             | `str`  | Always `"dungeongen"` — stable upstream source identifier.                                         |
| `source_revision`       | `str`  | Exact installed dungeongen package version used to generate the snapshot. Must not be `"unknown"`. |
| `record_format`         | `str`  | Always `"tar+npz"`.                                                                                |
| `record_schema_version` | `int`  | NPZ record schema version (currently `1`).                                                         |
| `splits`                | `dict` | Per-split metadata; see 12.3.1.                                                                    |
| `generator_params`      | `dict` | Deterministic generator parameters (at minimum `base_seed`).                                       |
| `producer_revision`     | `str`  | `ehc_sn` package version that wrote the snapshot.                                                  |

#### 12.3.1 Per-split metadata (`splits[<split>]`)

| Field       | Type        | Description                           |
| ----------- | ----------- | ------------------------------------- |
| `n_samples` | `int`       | Total number of samples in the split. |
| `shards`    | `list[str]` | Ordered list of tar shard filenames.  |

Shard filenames follow the pattern `shard-<NNNNN>.tar` (zero-padded to 5 digits).

#### 12.3.2 Identity fields

The following fields are identity-bearing. fetch-raw performs a mismatch check
on all of them:

- `source_revision`
- `record_format`
- `record_schema_version`
- `generator_params`
- per-split `n_samples`

### 12.4 Tar Shard and NPZ Record Structure

Each tar shard contains NPZ members named `sample-<NNNNNN>.npz` (zero-padded
to 6 digits), in insertion order (which is deterministic sample order within
the shard).

Each NPZ record contains:

| Array       | dtype | Shape  | Description                                          |
| ----------- | ----- | ------ | ---------------------------------------------------- |
| `sample_id` | int64 | scalar | Stable sample index within the split.                |
| `seed`      | int64 | scalar | dungeongen seed used to generate this sample.        |
| `topology`  | bool  | (H, W) | Passable cells; native generator bounding-box shape. |
| `regions`   | int32 | (H, W) | Room number per cell; `-1` for non-room cells.       |

`(H, W)` may differ between samples because dungeongen determines its own
bounding box. Callers that need a fixed shape must pad in the interim stage.

### 12.5 fetch-raw Policy (Hard Rules)

1. **No raw root** → create the snapshot, write `manifest.json`.
2. **Manifest exists, identity matches** → no-op.
3. **Manifest exists, identity differs** → fail immediately with an actionable
   error message listing the mismatched fields. Do not modify the existing root.
4. **Raw root exists, no manifest** → fail immediately with an actionable error
   telling the user to delete `data/raw/dungeongen` and rerun fetch-raw.
5. **Legacy loose-file root detected** (any `topology_*.npy` in a split
   subdirectory) → fail immediately with an actionable error telling the user
   to delete `data/raw/dungeongen` and rerun fetch-raw. Do not auto-migrate.

### 12.6 Dungeongen Interim Contract

Path: `data/interim/dungeongen/{train,val,test}.npz`

The interim layer is a **real normalization boundary**. It is materially
different from the raw tar-sharded format:

- No tar packaging.
- No per-sample file fan-out.
- One monolithic NPZ file per split with all samples stacked.
- Arrays are padded to the split's maximum shape with well-defined sentinels.

The interim format is a derived artifact (not an immutable snapshot). Writers
overwrite existing interim files.

#### 12.6.1 Per-split NPZ arrays

| Array       | dtype | Shape             | Description                                     |
| ----------- | ----- | ----------------- | ----------------------------------------------- |
| `sample_id` | int64 | (N,)              | Sample index, matching raw `sample_id`.         |
| `seed`      | int64 | (N,)              | dungeongen seed per sample.                     |
| `height`    | int32 | (N,)              | Native generator height per sample.             |
| `width`     | int32 | (N,)              | Native generator width per sample.              |
| `topology`  | bool  | (N, H_max, W_max) | Padded with `False` to the split maximum shape. |
| `regions`   | int32 | (N, H_max, W_max) | Padded with `-1` to the split maximum shape.    |

`H_max` and `W_max` are the maximum height and width across all samples in the
split. `iter_interim_topologies` reconstructs native-shape arrays by slicing
with the stored `height[i]` and `width[i]` values.

#### 12.6.2 Interim provenance

The interim format does not redefine provenance. It is a normalized access
boundary, not a second source format. Source identity is owned by the raw
manifest (`data/raw/dungeongen/manifest.json`).

#### 12.6.3 Interim consumer contract

`prepare_interim` must consume `iter_raw_topologies(raw_root, split)`
and must not directly inspect raw filenames, shard structure, or tar packaging.

`build_shared_substrate` must consume `iter_interim_topologies(interim_root, split)`
and must not read raw directly.

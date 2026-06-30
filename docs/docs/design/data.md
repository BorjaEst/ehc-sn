---
title: Data Subsystem
description: Two-plane architecture for the ehp-sn data subsystem
---

# Data Subsystem (`ehp_sn.data`)

The data subsystem is responsible for **producing, identifying, validating,
storing, resolving, and reading** canonical dataset representations, and for
converting stored samples into generic runtime batches.

## Two-plane architecture

The package is organised into two conceptual planes:

```
Build plane          → create, validate, version, and publish datasets
Runtime plane        → resolve, open, sample, collate, and load datasets
```

```
sources / generators
    │
    ▼
shared substrates
    │
    ▼
task corpora
    │
    ▼
dataset resolution
    │
    ▼
dataset reader
    │
    ▼
sampler
    │
    ▼
collator
    │
    ▼
batch
    │
    ▼
task/model adapter     ← boundary: data ends here
    │
    ▼
runtime / model       ← ehp_sn.rollouts, ehp_sn.controllers, ehp_sn.models
```

**Ownership boundary**: `ehp_sn.data` owns data semantics up to the point
where a task- or model-specific interpretation begins. It must not import:

- `ehp_sn.models`
- `ehp_sn.controllers`
- `ehp_sn.objectives`
- `ehp_sn.training`
- `ehp_sn.lightning`

A Lightning adapter (`ehp_sn/lightning/data.py`) may depend on
`ehp_sn.data`, but not the reverse.

---

## Package layout

### Conceptual target architecture

```
ehp_sn/data/
├── __init__.py              ← narrow public API
│
├── references.py            ← DatasetRef, DatasetLocation, resolve_dataset
├── manifests.py             ← DatasetManifest, read_manifest, write_manifest
├── indexes.py               ← DatasetIndexEntry, read_index, write_index
├── lineage.py               ← validate_shared_parent
├── errors.py                ← DataError taxonomy
│
├── lifecycle/               ← Transactional build pipeline
│   ├── __init__.py
│   ├── staging.py           ← staging_root context manager
│   ├── publishing.py        ← write_split, create_version_root
│   ├── validation.py        ← validate_version_root
│   └── migrations.py        ← Schema migrations (future)
│
├── sources/                 ← External/procedural source descriptions
│   ├── __init__.py
│   ├── protocols.py
│   ├── specs.py
│   └── generation.py
│
├── substrates/              ← Shared structural representations
│   ├── __init__.py
│   ├── protocols.py
│   ├── grid2d.py            ← Channel constants, validate_grid2d_sample
│   ├── dungeongen.py        ← Pipeline: raw → interim → substrate
│   ├── maze_nd.py           ← Pipeline for HuggingFace maze_hard_augmented
│   ├── dagflow.py           ← Hamiltonian DAG generator
│   └── openfield.py         ← Square/rectangle/hex grid generators
│
├── layouts/                 ← Intermediate layout contracts
│   ├── __init__.py
│   ├── protocols.py         ← SpatialLayout TypedDict
│   ├── spatial.py           ← Geometry + observation placement
│   └── relational.py        ← Goaltrace geometric + DAG weight matrix
│
├── corpora/                 ← Task-ready persistent datasets (future)
├── storage/                 ← Physical backends (future)
│
├── datasets/                ← Runtime record access
│   ├── __init__.py
│   ├── protocols.py         ← MapDataset protocol
│   ├── processed.py         ← ProcessedDataset (mmap-backed)
│   ├── iterable.py
│   └── factory.py           ← open_dataset
│
├── sampling/                ← Record-selection order
│   ├── __init__.py
│   ├── protocols.py         ← Sampler, StatefulSampler protocols
│   ├── sequential.py
│   ├── shuffled.py          ← ShuffledEpisodeSource (EpisodeSource, not Sampler)
│   └── distributed.py
│
├── collators/               ← Sequence[sample] → batch
│   ├── __init__.py
│   ├── protocols.py         ← Collator protocol
│   ├── default.py           ← Default stacking collator
│   ├── padded.py
│   └── recurrent.py
│
├── transforms/              ← Genuine data transformations
│   ├── __init__.py
│   ├── protocols.py         ← SampleTransform protocol
│   ├── compose.py           ← Compose
│   └── spatial.py           ← RandomDihedral
│
└── loaders/                 ← PyTorch DataLoader construction
    ├── __init__.py
    ├── config.py            ← DataLoaderConfig
    └── factory.py           ← build_data_loader, build_sampler
```

### Immediate filesystem

Split into subpackages only when there are multiple implementations, materially
different dependencies, independent testing requirements, or a stable public
abstraction. Until then, a flatter structure is preferred:

```
ehp_sn/data/
├── __init__.py
├── references.py
├── manifests.py
├── indexes.py
├── lineage.py
├── errors.py
├── datasets.py
├── sampling.py
├── collators.py
├── loaders.py
├── transforms.py
├── lifecycle/
├── source/
├── substrate/
├── layout/
└── validation/
```

The Lightning adapter lives in `ehp_sn/lightning/data.py`, **not** inside
`ehp_sn/data/`, so that the dependency direction `lightning adapter →
ehp_sn.data` is physically enforced.

---

## Identity layer

### `DatasetRef`

A logical reference to a versioned dataset root. Configuration should use
this instead of raw filesystem paths.

```python
from dataclasses import dataclass
from typing import Literal

DatasetKind = Literal[
    "shared_substrate",
    "layout_dataset",
    "task_corpus",
]

@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Identifies one versioned dataset root.

    The ref points to the root directory (e.g. ``data/processed/arena/default/v2/``).
    It does **not** include a split — the split is a view within the root.
    """
    kind: DatasetKind
    family: str           # "maze-nd", "dungeongen", "dagflow", "openfield", "arena", ...
    name: str             # "default", "routing", "tem-square", ...
    version: int
```

Splits are selected when opening the dataset, not when identifying the root:

```python
@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """Select a split within a resolved dataset root."""
    dataset: DatasetRef
    split: str            # "train", "val", "test"
```

This keeps root identity and split selection as separate concepts.

Examples:

```python
# A shared substrate root (no split — split is chosen at open time)
DatasetRef(
    kind="shared_substrate",
    family="maze-nd",
    name="default",
    version=1,
)

# Selecting a split within a task corpus
DatasetSelection(
    dataset=DatasetRef(
        kind="task_corpus",
        family="arena",
        name="tem-square",
        version=2,
    ),
    split="train",
)
```

### `DatasetLocation`

Result of resolving a `DatasetRef` to a filesystem path.

```python
@dataclass(frozen=True, slots=True)
class DatasetLocation:
    """Resolved path for a dataset root.  No metadata is loaded yet."""
    ref: DatasetRef
    root: Path
```

### `resolve_dataset`

```python
def resolve_dataset(
    ref: DatasetRef,
    *,
    data_root: Path,
) -> DatasetLocation:
    """Resolve a DatasetRef to a filesystem path.

    Does NOT load the manifest or index.  Use open_dataset() to open.
    """
    ...
```

Manifest and index loading happen inside `open_dataset`, not during
resolution. This keeps resolution cheap (stat check only) and allows
callers to inspect the location before committing to opening.

Uses the regular path grammar:

```
data/interim/<family>/<name>/v<version>/      # shared_substrate, layout_dataset
data/processed/<family>/<name>/v<version>/     # task_corpus
```

---

## Manifest

### `DatasetManifest`

```python
@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_ref: DatasetRef
    record_count: int
    created_at: datetime
    generator: str
    generator_version: Optional[str]
    input_fingerprint: str          # deterministic hash of stage_params
    content_digest: str             # "sha256:<hex>"
    parameters: Mapping[str, JsonValue]
    parent_refs: tuple[DatasetRef, ...] = ()
```

### Manifest authority rules

The manifest is the authoritative declaration. The index enumerates. The
validator cross-checks. The runtime never silently repairs.

| Field               | Authority                                                                        |
| ------------------- | -------------------------------------------------------------------------------- |
| `record_count`      | **Manifest declares**, index confirms                                            |
| `split`             | **Index owns** (per-entry); manifest aggregates counts per split                 |
| `parent_refs`       | **Manifest declares**; lineage checks validate                                   |
| `content_digest`    | **Manifest stores**; `ProcessedDataset` recomputes at open time for verification |
| `input_fingerprint` | **Manifest stores**; computed from `stage_params` at build time                  |
| `generator_version` | **Manifest stores**; cross-referenced at validation time                         |
| `parameters`        | **Manifest stores**; used for reproducibility, never auto-inferred               |

---

## Index

### `DatasetIndexEntry`

```python
@dataclass(frozen=True, slots=True)
class DatasetIndexEntry:
    id: str                       # Stable sample identifier
    split: str                    # "train", "val", "test"
    source_record_id: Optional[str]
    relative_path: str
    length: Optional[int]
    metadata: Mapping[str, JsonValue]
```

### Index responsibilities

- Read / write `index.jsonl`
- Filter by split
- Detect duplicate or missing IDs
- Validate index/manifest consistency
- Expose stable ordering

---

## Build plane

### Sources (`sources/`)

Describes where substrate material comes from. A source is not yet a task
dataset — it may describe:

- A procedural maze generator (`openfield`)
- A downloaded HuggingFace corpus (`maze-nd`)
- A raw graph source (`dagflow`)
- A dungeon generation library (`dungeongen`)

```python
class SourceSpec(Protocol):
    @property
    def source_id(self) -> str: ...
    def fingerprint(self) -> str: ...
```

### Substrates (`substrates/`)

Shared world or structure representations from which task datasets are
derived. A substrate is **reusable across tasks**:

```
maze_nd substrate
    ├── MazeHard task corpus
    ├── Goaltrace task corpus  (future)
    └── Routebind task corpus  (future)
```

Each substrate builder follows the same pipeline:

```
ensure_raw → prepare_interim → build_shared_substrate
```

Existing families: `dungeongen`, `maze_nd`, `dagflow`, `openfield`, `grid2d`.

### Layouts (`layouts/`)

Canonical intermediate structural contracts. The `SpatialLayout` TypedDict
is the source-to-task boundary:

```
source-specific representation
    ↓  (canonical)
SpatialLayout
    ↓  (target generation)
task corpus
```

`SpatialLayout` fields: `layout_id`, `graph_state_count`,
`state_to_row_col`, `observation_id`, `next_state`, `action_valid`,
`action_space`, topology/observation seeds, `observation_vocabulary_size`.

### Corpora (`corpora/`)

Task-ready persistent datasets. This layer turns shared substrates or
layouts into canonical task records with task-specific target channels.

### Lifecycle (`lifecycle/`)

Dataset publication as an atomic, transactional process:

```
begin staging
    ↓
write records (per-split .npy files)
    ↓
write index.jsonl
    ↓
write manifest.json
    ↓
validate complete root
    ↓
atomically rename to v<N>/
```

Key invariants:

- **Immutability**: A versioned root must never be modified after creation.
  Rebuilding requires bumping the version integer.
- **Transactional atomicity**: Writers use a `.building-v<N>/` sibling
  directory. On success the temp dir is atomically renamed; on failure it
  is removed and the final root never exists.
- **Structural validation**: Every published root passes manifest-field
  checks, index/manifest agreement, channel-file presence, and sample-count
  cross-validation before the atomic rename.

---

## Storage plane

Physical backends are hidden behind a repository-owned protocol:

```python
class RecordStore(Protocol[RecordT]):
    def __len__(self) -> int: ...
    def read(self, entry: DatasetIndexEntry) -> RecordT: ...
```

Concrete implementations:

| Backend | Module             | Use case                                |
| ------- | ------------------ | --------------------------------------- |
| NumPy   | `storage/numpy.py` | Per-split `.npy` channel arrays (mmap)  |
| JSONL   | `storage/jsonl.py` | Per-sample index files                  |
| Zarr    | `storage/zarr.py`  | Multiscale arrays, chunked I/O (future) |

Consumers must not depend on backends directly:

```python
# ❌ Wrong — storage detail leaked
group = zarr.open_group(...)
obs = group["observations"][index]

# ✅ Correct — backend is abstracted
sample = dataset[index]
```

---

## Runtime consumption plane

### Datasets (`datasets/`)

```python
SampleT_co = TypeVar("SampleT_co", covariant=True)

class MapDataset(Protocol[SampleT_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> SampleT_co: ...
```

Primary implementation — `ProcessedDataset`:

- Opens manifest and index at construction
- Verifies dataset identity against expected fingerprint
- Maps index entries to stored records
- Decodes one record on `__getitem__`
- Applies genuine sample-level transforms (e.g. `RandomDihedral`)
- Exposes dataset digest

Factory:

```python
def open_dataset(
    ref: DatasetRef,
    *,
    data_root: Path,
    decoder: RecordDecoder[SampleT],
    transform: Optional[SampleTransform[SampleT]] = None,
) -> ProcessedDataset[SampleT]:
    ...
```

### Sampling (`sampling/`)

Record-selection order, not record conversion.

```python
class Sampler(Protocol):
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class StatefulSampler(Protocol):
    def state_dict(self) -> dict[str, JsonValue]: ...
    def load_state_dict(self, state: Mapping[str, JsonValue]) -> None: ...
```

Concrete samplers:

| Sampler                  | Strategy                  | Stateful |
| ------------------------ | ------------------------- | -------- |
| `SequentialSampler`      | `0, 1, 2, ..., N-1`       | No       |
| `RandomSampler`          | Random with optional seed | No       |
| `DistributedSampler`     | Rank-stride partition     | No       |
| `ShuffledEpisodeSampler` | Deterministic coverage    | Yes      |

### ShuffledEpisodeSource (record-emitting, not index-sampling)

The `ShuffledEpisodeSource` emits full **records** (episodes), not integer
indices. This is the correct abstraction for the recurrent replay path
because `DemandDrivenReplaySource` consumes episodes, not raw positions.

There are two related but distinct abstractions:

| Abstraction     | Emits           | Used by                    |
| --------------- | --------------- | -------------------------- |
| `Sampler[int]`  | Integer indices | Standard `DataLoader` path |
| `EpisodeSource` | Full records    | Recurrent replay path      |

`ShuffledEpisodeSource` implements `EpisodeSource`. It is **not** a PyTorch
`Sampler` and should not be forced into that abstraction.

Both are stateful and expose `state_dict()` / `load_state_dict()` for
exact resume.

### Collators (`collators/`)

```python
SampleT = TypeVar("SampleT")
BatchT_co = TypeVar("BatchT_co", covariant=True)

class Collator(Protocol[SampleT, BatchT_co]):
    def __call__(self, samples: Sequence[SampleT]) -> BatchT_co: ...
```

Concrete responsibilities:

- Stacking
- Padding
- Mask construction
- Variable-length packing
- Tensor conversion (numpy → torch)
- Metadata aggregation
- Batch-level validation

### Transforms (`transforms/`)

Genuine sample-level data transformations.

```python
class SampleTransform(Protocol[SampleT]):
    def __call__(self, sample: SampleT) -> SampleT: ...
```

| Transform        | Role                                    |
| ---------------- | --------------------------------------- |
| `RandomDihedral` | D4 spatial augmentation (grid2d only)   |
| `Compose`        | Chain multiple transforms               |
| Coordinate norm  | Normalise spatial coordinates           |
| Stochastic mask  | Task-independent input masking (future) |

A transform must **not**:

- Tokenise specifically for HRM
- Create TEM latent inputs
- Allocate runtime carry
- Move tensors to GPU
- Inspect controller configuration

### Loaders (`loaders/`)

#### `DataLoaderConfig`

```python
@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    drop_last: bool = False
```

#### `build_data_loader`

Uses explicit object composition. No declarative `SamplingConfig` parameter —
the caller constructs the sampler directly.

```python
def build_data_loader(
    dataset: Dataset[SampleT],
    *,
    config: DataLoaderConfig,
    collator: Collator[SampleT, BatchT],
    sampler: Optional[Sampler[int]] = None,
    batch_sampler: Optional[Sampler[list[int]]] = None,
) -> DataLoader[BatchT]:
    ...
```

A separate factory resolves configuration into sampler objects when declarative
configuration is preferred:

```python
def build_sampler(
    dataset: Dataset,
    *,
    strategy: Literal["sequential", "random", "shuffled_episode"],
    seed: int = 0,
) -> Sampler[int]:
    ...
```

This keeps the two mechanisms separate: the `build_sampler` factory for
configuration-driven construction, and explicit sampler injection for
library-level composition.

---

## Three schema levels

Three distinct schema levels exist, each owned by a different subsystem:

| Level             | Owned by          | Examples                                                               |
| ----------------- | ----------------- | ---------------------------------------------------------------------- |
| Persistent data   | `ehp_sn.data`     | `DatasetManifest`, `DatasetIndexEntry`, `SpatialLayout`, `ChannelSpec` |
| Task sample/batch | `ehp_sn.tasks.*`  | `ArenaSample`, `ArenaBatch`, `MazeHardSample`                          |
| Model input       | `ehp_sn.adapters` | `TEMArenaInputs`, `HRMMazeInputs`                                      |

### Task sample schemas (task-owned)

```python
# ehp_sn/tasks/arena/schemas.py
class ArenaSample(TypedDict):
    topology: Tensor
    observations: Tensor
    previous_action: Tensor
    episode_start: Tensor
    mask_valid: Tensor
    trajectory_id: str

@dataclass(frozen=True, slots=True)
class ArenaBatch:
    topology: Tensor          # [B, H, W]
    observations: Tensor      # [B, H, W]
    previous_action: Tensor   # [B, 1]
    episode_start: Tensor     # [B, 1]
    mask_valid: Tensor        # [B, H, W]
    trajectory_ids: tuple[str, ...]
```

```python
# ehp_sn/tasks/mazehard/schemas.py
class MazeHardSample(TypedDict):
    topology: Tensor
    start: Tensor
    goal: Tensor
    solution: Tensor

@dataclass(frozen=True, slots=True)
class MazeHardBatch:
    topology: Tensor          # [B, H, W]
    start: Tensor             # [B, H, W]
    goal: Tensor              # [B, H, W]
    solution: Tensor          # [B, H, W]
```

This replaces the current weak alias:

```python
Batch = dict[str, Tensor]    # ❌ keys discoverable only by convention
```

### Adapter conversion

```
ArenaBatch
    ↓ TEMArenaAdapter.prepare_inputs(batch)
TEMArenaInputs(sensory=..., previous_action=..., reset_mask=..., valid_mask=...)
```

Not:

```
ProcessedDataset                         # ❌ wrong direction
    ↓ model-specific transform
TEMArenaInputs
```

---

## Replay and recurrent runtime boundary

The data layer may own:

- Episode identities
- Episode lengths
- Chunk extraction
- Shuffled episode order
- Reset markers
- Slot-compatible batch shapes
- Stateful sampler position

It must **not** own:

- Live controller carry
- Halted-slot state
- Recurrent memory banks
- Runtime step counters
- Decisions about when a halted slot is replaced
- Active rollout state

### Clean construction

```python
# Assembled by the experiment or training builder (NOT in the DataModule):

dataset = open_dataset(train_ref, data_root=paths.data)

episode_source = ShuffledEpisodeSource(
    dataset,
    seed=training_config.seed,
)

train_source = DemandDrivenReplaySource(
    episode_source=episode_source,
    num_slots=execution_config.replay_slots,
)
```

The DataModule does not discover `train_source` through a private callback
or `attach_source_provider`. The runtime owns `train_source`.

### What NOT to do

```python
# ❌ Boundary inversion — data module reaches into private runtime state
datamodule.attach_source_provider(lambda: module._train_source)
```

### Configuration separation

```toml
# ❌ Wrong — num_slots is a runtime concept
[data]
num_slots = 8

# ✅ Correct — separate concerns
[data.dataset]
kind = "task_corpus"
family = "arena"
name = "default"
version = 1

[data.loader]
batch_size = 8
num_workers = 4
pin_memory = true

[execution.replay]
replay_slots = 8
chunk_length = 25
```

---

## Error taxonomy

```python
class DataError(Exception):
    """Base exception for data-plane failures."""

class DatasetResolutionError(DataError):
    """Could not resolve a DatasetRef to a filesystem path."""

class DatasetNotFoundError(DatasetResolutionError):
    """Dataset root does not exist at the resolved path."""

class ManifestError(DataError):
    """Manifest is missing, corrupt, or incompatible."""

class ManifestVersionError(ManifestError):
    """Manifest schema version is not supported."""

class DatasetValidationError(DataError):
    """Dataset failed structural or semantic validation."""

class DatasetIndexError(DataError):
    """Index is missing, corrupt, or inconsistent with manifest."""

class CorruptRecordError(DataError):
    """A stored record could not be decoded."""

class RecordSchemaError(DataError):
    """Record fields, dtypes, or shapes do not match the expected schema.

    Covers stored-channel mismatches at the persistent-data level:
    missing channel, wrong dtype, wrong rank, missing manifest field.
    Adapter-level incompatibilities belong in ``ehp_sn.adapters``, not here.
    """

class LineageError(DataError):
    """Parent lineage validation failed."""
```

Consumers should be able to distinguish these cases:

```
dataset does not exist                   → DatasetNotFoundError
dataset exists but unsupported schema    → ManifestVersionError
manifest and index disagree              → DatasetValidationError
stored channel has wrong dtype           → RecordSchemaError
record is corrupt                        → CorruptRecordError
adapter received incompatible batch      → AdapterInputError (in ehp_sn.adapters)
```

---

## Public API

Organised by stability tier.

### Stable application API

Types and functions needed by everyday consumers (training scripts, evaluation,
notebooks):

```python
from ehp_sn.data import (
    # Identity and resolution
    DatasetRef,
    DatasetSelection,
    DatasetLocation,
    resolve_dataset,

    # Dataset opening
    open_dataset,

    # Validation
    validate_dataset,

    # Loader construction
    DataLoaderConfig,
    build_data_loader,
    build_sampler,

    # Errors (top-level only)
    DataError,
    DatasetNotFoundError,
    DatasetValidationError,
)
```

### Stable metadata API

Types needed when inspecting or producing manifests and indexes:

```python
from ehp_sn.data import (
    DatasetManifest,
    DatasetIndexEntry,
    read_manifest,
    read_index,
)
```

### Advanced extension API

Protocols and concrete types for extending the data subsystem:

```python
from ehp_sn.data import (
    ProcessedDataset,
)

from ehp_sn.data.collators import Collator
from ehp_sn.data.sampling import Sampler, StatefulSampler, ShuffledEpisodeSampler
from ehp_sn.data.transforms import SampleTransform, RandomDihedral, Compose
from ehp_sn.data.lifecycle import staging_root
```

---

## User-facing workflows

### Open a dataset

```python
from ehp_sn.data import DatasetRef, DatasetSelection, open_dataset

dataset = open_dataset(
    DatasetSelection(
        dataset=DatasetRef(
            kind="task_corpus",
            family="arena",
            name="default",
            version=1,
        ),
        split="train",
    ),
    data_root=paths.data,
)
```

### Validate a dataset

```python
from ehp_sn.data import DatasetRef, resolve_dataset, validate_dataset

location = resolve_dataset(ref, data_root=paths.data)
report = validate_dataset(location)
report.raise_for_errors()
```

### Construct an evaluation loader

```python
loader = build_data_loader(
    dataset,
    config=DataLoaderConfig(
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    ),
    sampler=SequentialSampler(dataset),
    collator=ArenaCollator(),
)
```

### Construct a training replay pipeline

```python
dataset = open_dataset(train_ref, data_root=paths.data)

episode_source = ShuffledEpisodeSource(
    dataset,
    seed=training_config.seed,
)

replay_source = DemandDrivenReplaySource(
    episode_source=episode_source,
    num_slots=execution_config.replay_slots,
)
```

`replay_source` belongs to `ehp_sn.rollouts`, not to `ehp_sn.data`.

### Build a new corpus

```python
with staging_root(target_root) as staging:
    build_arena_corpus(
        substrate=substrate,
        output_root=staging,
        config=config,
    )
    validate_version_root(staging)
```

---

## Naming conventions

| Term      | Meaning                                             |
| --------- | --------------------------------------------------- |
| source    | Raw or procedural origins                           |
| substrate | Shared structural world representation              |
| layout    | Canonical intermediate spatial/relational structure |
| corpus    | Published task-specific dataset                     |
| ref       | Logical identity of a published root                |
| dataset   | Runtime object that retrieves samples               |
| sample    | One decoded logical record                          |
| sampler   | Chooses record indices or order                     |
| collator  | Converts multiple samples into a batch              |
| batch     | Tensor collection consumed by a task adapter        |
| adapter   | Converts task batch into model-specific inputs      |
| loader    | Iteration, workers, prefetching, batching           |
| replay    | Runtime-managed recurrent episode supply            |

Avoid using `source` for both persistent datasets and live replay sources.
A stronger naming distinction:

```
DatasetRecordSource       → data/storage level
EpisodeOrder              → sampling level
ReplaySource              → runtime level
```

---

## Summary of ownership

| Concern                             | Owner                      |
| ----------------------------------- | -------------------------- |
| Dataset identity and resolution     | `references.py`            |
| Manifests, fingerprints, provenance | `manifests.py`             |
| Sample indexes                      | `indexes.py`               |
| Lineage validation                  | `lineage.py`               |
| Error taxonomy                      | `errors.py`                |
| Source specs                        | `sources/`                 |
| Shared substrate generation         | `substrates/`              |
| Layout protocol                     | `layouts/`                 |
| Task corpus builders                | `corpora/`                 |
| Transactional dataset publication   | `lifecycle/`               |
| Physical storage backends           | `storage/`                 |
| Runtime dataset access              | `datasets/`                |
| Sample transforms                   | `transforms/`              |
| Sampling / index selection          | `sampling/`                |
| Collation / batching                | `collators/`               |
| Loader construction                 | `loaders/`                 |
| Lightning integration               | `ehp_sn/lightning/data.py` |
| Task sample/batch schemas           | `ehp_sn.tasks.*`           |
| Model input adaptation              | `ehp_sn.adapters`          |
| Recurrent replay execution          | `ehp_sn.rollouts`          |
| Training regime configuration       | `ehp_sn.experiments.*`     |

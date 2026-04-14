# EHC-SN Data Contracts Specification

> Non-required companion spec for processed-data format, pipeline contracts, and
> dataset-facing data surfaces.

## 1 Scope

This spec defines the low-frequency reference material for persisted maze data:

- raw → interim → processed pipeline;
- canonical processed on-disk format;
- dataset and datamodule output contracts;
- task- and adapter-facing consumption notes.

Core ownership and boundary rules remain in `spec/spec-architecture.md` and
`spec/spec-requirements.md`.

---

## 2 Pipeline

```text
Generators (maze-nd, dungeongen, HF)      scripts/data-gen/
    │
    ▼
  data/raw/
    │
    ▼
  data/interim/  ◄────── source-specific normalization / augmentation scripts
    │
    ▼
  data/processed/ ◄───── canonical split-uniform stacked arrays + index
    │
    ├──► ehc_sn.data.datasets / datamodules  ──► tasks/ ──► adapters/
    │
    └──► ehc_sn.envs.*                       ──► runtime environment kernels
```

Rules:

- `data/raw/` stores untouched source output and is not version controlled.
- `data/interim/` stores source-specific normalized or augmented output.
- `data/processed/` stores canonical source-agnostic output consumed by
  dataloaders.
- Source-specific generation logic lives in `scripts/data-gen/`, not under
  `src/ehc_sn/`.

---

## 3 Canonical Processed On-Disk Format

Processed maze data lives in `data/processed/` as a dataset root with a single
JSONL index and per-split stacked channel arrays. Each split stores one `.npy`
file per declared channel with shape `(N, H, W)`, where all samples in the
split share the same stored spatial shape.

### 3.1 Mandatory Channel

| Channel  | Name       | dtype  | Shape    | Description                                 |
| -------- | ---------- | ------ | -------- | ------------------------------------------- |
| Topology | `topology` | `bool` | `(H, W)` | Passable cells (`True`) vs walls (`False`). |

### 3.2 Optional Channels

| Channel      | Name           | dtype   | Shape    | Description                            | Primary consumers |
| ------------ | -------------- | ------- | -------- | -------------------------------------- | ----------------- |
| Observations | `observations` | `int32` | `(H, W)` | Unique observation ID per cell.        | TEM, EHC          |
| Start        | `start`        | `bool`  | `(H, W)` | Start position(s).                     | HRM, EHC          |
| Goals        | `goals`        | `bool`  | `(H, W)` | Goal position(s).                      | HRM, EHC          |
| Landmarks    | `landmarks`    | `int32` | `(H, W)` | Special object IDs.                    | TEM, EHC          |
| Solution     | `solution`     | `int32` | `(H, W)` | Shortest-path distance or step labels. | HRM supervision   |
| Regions      | `regions`      | `int32` | `(H, W)` | Room or region ID.                     | Future            |
| Valid mask   | `mask_valid`   | `bool`  | `(H, W)` | Explicit legal positions.              | All               |

### 3.3 Split Metadata

Path: `data/processed/<split>/dataset.json`

| Field       | Type        | Description                                          |
| ----------- | ----------- | ---------------------------------------------------- |
| `source`    | `str`       | Generator or source dataset name.                    |
| `split`     | `str`       | Split name.                                          |
| `n_samples` | `int`       | Number of samples in the split.                      |
| `shape`     | `list[int]` | Stored normalized maze shape as `[height, width]`.   |
| `channels`  | `list[str]` | Channel names present for every sample in the split. |

### 3.4 Root JSONL Index

Path: `data/processed/index.jsonl`

| Field            | Type              | Description                                    |
| ---------------- | ----------------- | ---------------------------------------------- |
| `id`             | `str`             | Unique maze identifier.                        |
| `source`         | `str`             | Generator that produced the raw maze.          |
| `split`          | `str`             | Dataset split: `train`, `val`, or `test`.      |
| `shape`          | `tuple[int, int]` | Stored normalized grid shape.                  |
| `channels`       | `list[str]`       | Channel names present for the sample's split.  |
| `n_observations` | `int`             | Observation vocabulary size, or `0` if absent. |
| `n_goals`        | `int`             | Number of goal cells, or `0` if absent.        |
| `difficulty`     | `str`             | Optional source-defined difficulty label.      |

### 3.5 On-Disk Layout

```text
data/
├── raw/
│   ├── maze-nd/
│   ├── dungeongen/
│   └── huggingface/
├── interim/
│   ├── maze-nd/
│   ├── dungeongen/
│   └── huggingface/
└── processed/
    ├── index.jsonl
    ├── train/
    │   ├── dataset.json
    │   ├── topology.npy
    │   ├── observations.npy
    │   └── ...
    ├── val/
    └── test/
```

---

## 4 Dataset and Datamodule Output Contracts

`MazeDataset.__getitem__` returns a model-agnostic per-sample channel mapping.

| Key              | Type     | Shape    | Description                                         |
| ---------------- | -------- | -------- | --------------------------------------------------- |
| `<channel_name>` | `Tensor` | `(H, W)` | Raw tensor for one canonical channel in the sample. |

Rules:

- The dataset layer preserves stored processed channels for one sample.
- The dataset layer does not synthesize nested task or model payloads.
- Task-local semantic views are built in `tasks/`.
- Model-specific packing is built in `adapters/`.
- DataModules consume only `data/processed/`.

---

## 5 Runtime Consumption Notes

- `data/` owns persisted contracts and static loading only.
- `tasks/` own task-local observation and episode semantics.
- `adapters/` convert task observations into model-native payloads.
- `envs/` house reusable runtime kernels; the task meaning attached to those
  kernels belongs in `tasks/`.

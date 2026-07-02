# maze-nd Shared Substrate

## Identity

| Property       | Value                                           |
| -------------- | ----------------------------------------------- |
| Family         | `maze-nd`                                       |
| Topology kind  | `grid2d`                                        |
| Source         | HuggingFace `flaitenberger/maze_hard_augmented` |
| Source ID      | `huggingface/maze_hard_augmented`               |
| CLI script     | `scripts/data-gen/build-maze-nd.py`             |
| Builder module | `ehp_sn.data.substrate.maze_nd`                 |
| Output path    | `data/interim/maze-nd/v<version>/`              |
| Dataset class  | `shared_substrate`                              |

## Purpose and ownership

The maze-nd shared substrate is built from the HuggingFace
`maze_hard_augmented` dataset. Each sample is a 2-D grid maze with walls,
a single start cell, one or more goal cells, and a precomputed solution
path.

The substrate preserves source problem annotations (`start`, `goals`,
`solution`) alongside structural channels (`topology`, `mask_valid`).
These annotations are reusable source facts — they belong to the shared
substrate, not to any task's protocol. Downstream task builders consume
them to construct task-specific episodes and supervision targets.

**Boundary**: maze-nd does **not** own task protocol channels (replay rows,
episode schema, supervision targets, token sequences). Task semantics
belong in the respective task corpus (e.g., mazehard).

## Semantic model

### Grid2d topology

Each maze is a rectangular 2-D grid of `(H, W)` cells. Every cell is either
passable or a wall. The substrate stores two boolean masks:

- `topology` — per-cell passability. `True` for passable cells, `False` for
  walls.
- `mask_valid` — the largest 4-connected component of passable cells. Cells
  outside this component are excluded from downstream navigation.

All spatial channels share the same `(H, W)` shape. The dataset is
homogeneous: all mazes in a versioned root have the same grid dimensions.

### Problem annotations

The HuggingFace source provides three additional channels preserved by the
substrate:

- `start` — exactly one cell marked as the maze entry point.
- `goals` — one or more cells marked as goal positions.
- `solution` — integer path encoding. Positive values `1, 2, …` trace the
  solution from start to the first goal reached. Value `0` marks cells not
  on the solution path.

The solution path is a precomputed shortest path; its semantics
(tie-breaking, goal selection) are owned by the upstream source.

### Split provenance

The raw source provides `train` and `test` splits only. `n_train` records
are sampled deterministically from the training population. `n_val` and
`n_test` are sampled deterministically from the raw test population as
non-overlapping partitions using `numpy.random.SeedSequence`.

## Output artifact

### Channels

| Channel      | Dtype | Shape  | Description                                                          |
| ------------ | ----- | ------ | -------------------------------------------------------------------- |
| `topology`   | bool  | (H, W) | Passable cells (`True`) vs walls (`False`).                          |
| `mask_valid` | bool  | (H, W) | Largest 4-connected component of `topology`.                         |
| `start`      | bool  | (H, W) | Single start cell marker.                                            |
| `goals`      | bool  | (H, W) | One or more goal cell markers.                                       |
| `solution`   | int32 | (H, W) | Integer-encoded solution path. `0` = not on path, `≥1` = step index. |

All spatial channels share the same `(H, W)` shape per sample. Per-sample
validation is provided by `ehp_sn.data.substrate.grid2d.validate_grid2d_sample`.

### On-disk layout

```text
data/interim/maze-nd/v<version>/
├── manifest.json           ← authoritative root descriptor
├── index.jsonl             ← per-sample entries with source_record_id
├── train/
│   ├── dataset.json
│   ├── topology.npy        ← (N, H, W) bool
│   ├── mask_valid.npy      ← (N, H, W) bool
│   ├── start.npy           ← (N, H, W) bool
│   ├── goals.npy           ← (N, H, W) bool
│   └── solution.npy        ← (N, H, W) int32
├── val/
│   └── ...
└── test/
    └── ...
```

## Invariants

- All spatial channels share the same `(H, W)` shape for every sample in the
  versioned root (homogeneous extent).
- `topology` and `mask_valid` are boolean arrays.
- `start` has exactly one `True` cell per sample.
- `goals` has at least one `True` cell per sample.
- `solution` values are non-negative integers: `0` for non-path cells,
  positive step indices `1, 2, …` along the solution.
- `mask_valid` is the largest 4-connected component of `topology`. Start and
  all goal cells lie within it.
- The versioned root is immutable after creation; rebuilding requires bumping
  the version integer.
- `artifact_schema_version = 1` guarantees presence of `topology`, `mask_valid`,
  `start`, `goals`, and `solution` channels.

## Build configuration

`build` parameters:

| Parameter         | Default                             | Description                                           |
| ----------------- | ----------------------------------- | ----------------------------------------------------- |
| `--external-root` | `data/external/maze_hard_augmented` | Raw HuggingFace corpus root.                          |
| `--staging-root`  | `data/raw/maze-nd`                  | Normalized staging root.                              |
| `--interim-root`  | `data/interim/maze-nd`              | Output root for the shared substrate.                 |
| `--version`       | 1                                   | Substrate version integer.                            |
| `--n-train`       | 1000                                | Training samples from raw train population.           |
| `--n-val`         | 40                                  | Validation samples (non-overlapping with `--n-test`). |
| `--n-test`        | 40                                  | Test samples (non-overlapping with `--n-val`).        |
| `--seed`          | 42                                  | Deterministic base seed.                              |
| `--force`         | (flag)                              | Delete existing version root before building.         |

Seeding: a single `--seed` is expanded via
`numpy.random.SeedSequence.spawn(2)` into independent RNG streams for the
train and test populations; val and test are then sampled from the test
stream as non-overlapping partitions.

## CLI

| Command    | Description                                           |
| ---------- | ----------------------------------------------------- |
| `build`    | Fetch raw corpus, normalize, and build the substrate. |
| `validate` | Validate manifest and channel contracts.              |
| `inspect`  | Print a human-readable manifest summary.              |

Usage:

```bash
# Quick local build with defaults (fetch + normalize + build)
python scripts/data-gen/build-maze-nd.py build

# Custom sizes and seed
python scripts/data-gen/build-maze-nd.py build \
    --n-train 4000 --n-val 500 --n-test 500 --seed 7

# Validate an existing root
python scripts/data-gen/build-maze-nd.py validate data/interim/maze-nd/v1

# Inspect a root
python scripts/data-gen/build-maze-nd.py inspect data/interim/maze-nd/v1
```

## Manifest

Root file: `manifest.json`. Dataset class: `shared_substrate`.

Key fields:

| Field                     | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| `dataset_class`           | `"shared_substrate"`                                            |
| `family`                  | `"maze-nd"`                                                     |
| `version`                 | Must match the `v<N>` path leaf.                                |
| `channels`                | `["topology", "mask_valid", "start", "goals", "solution"]`      |
| `topology_kind`           | `"grid2d"`                                                      |
| `n_states`                | Total cells `H · W` in the grid.                                |
| `extent`                  | `[H, W]` — homogeneous across all samples in the root.          |
| `n_samples`               | `{"train": N, "val": N, "test": N}`                             |
| `source_id`               | `"huggingface/maze_hard_augmented"`                             |
| `builder`                 | `"ehp_sn.data.substrate.maze_nd.build_shared_substrate"`        |
| `seed`                    | Deterministic base seed.                                        |
| `artifact_schema_version` | `1` — content-schema version for the shared-substrate protocol. |
| `normalization_version`   | `1`                                                             |
| `stage_params`            | `{"n_train", "n_val", "n_test", "seed"}`                        |
| `input_fingerprint`       | 16-char hex SHA-256 of `stage_params`.                          |

See [spec-data-contracts.md §4.2](../../spec/spec-data-contracts.md) for the
full shared substrate manifest field table.

## Downstream consumers

| Task     | CLI script                           | Parent substrate path        |
| -------- | ------------------------------------ | ---------------------------- |
| mazehard | `scripts/data-gen/build-mazehard.py` | `data/interim/maze-nd/v<N>/` |

Build the task corpus:

```bash
python scripts/data-gen/build-mazehard.py build \
    --substrate-root data/interim/maze-nd/v1
```

## Related

- [Spec: Data Contracts §3.1](../../spec/spec-data-contracts.md)
- [Grid2D Channel Contracts](../../src/ehp_sn/data/substrate/grid2d.py)
- [MazeHard Task Documentation](../tasks/mazehard.md)
- [MazeHard Task Builder](../../scripts/data-gen/build-mazehard.py)

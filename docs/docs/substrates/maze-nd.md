# maze-nd Shared Substrate

## Identity

| Property       | Value                                           |
| -------------- | ----------------------------------------------- |
| Family         | `maze-nd`                                       |
| Topology kind  | `grid2d`                                        |
| Source         | HuggingFace `flaitenberger/maze_hard_augmented` |
| Source ID      | `huggingface/maze_hard_augmented`               |
| CLI script     | `scripts/data-gen/build-maze-nd.py`             |
| Builder module | `ehc_sn.data.substrate.maze_nd`                 |
| Output path    | `data/interim/maze-nd/v<version>/`              |
| Dataset class  | `shared_substrate`                              |

## Description

The maze-nd shared substrate is built from the HuggingFace
`maze_hard_augmented` dataset. Each sample is a 2-D grid maze with walls,
a start cell, goal cells, and a precomputed solution path.

The substrate preserves source problem annotations (`start`, `goals`,
`solution`) alongside structural channels (`topology`, `mask_valid`). These
annotations are reusable source facts — they belong to the shared substrate,
not to any task's protocol. Downstream task builders (e.g., mazehard)
consume them to construct task-specific episodes and supervision targets.

## Channels

| Channel      | Dtype | Shape  | Description                                      |
| ------------ | ----- | ------ | ------------------------------------------------ |
| `topology`   | bool  | (H, W) | Passable cells (`True`) vs walls (`False`).      |
| `mask_valid` | bool  | (H, W) | Largest 4-connected component of `topology`.     |
| `start`      | bool  | (H, W) | Single start cell marker.                        |
| `goals`      | bool  | (H, W) | One or more goal cell markers.                   |
| `solution`   | int32 | (H, W) | Integer-encoded solution path (0 = not on path). |

All spatial channels share the same `(H, W)` shape per sample. See
`ehc_sn.data.substrate.grid2d.validate_grid2d_sample` for the per-sample
contract.

## Pipeline Stages

| Stage                | Command                               | Description                                     |
| -------------------- | ------------------------------------- | ----------------------------------------------- |
| `fetch-raw`          | `build-maze-nd.py fetch-raw`          | Download HuggingFace raw corpus.                |
| `normalize`          | `build-maze-nd.py normalize`          | Validate and write JSONL staging files.         |
| `materialize-shared` | `build-maze-nd.py materialize-shared` | Build versioned substrate root.                 |
| `validate`           | `build-maze-nd.py validate`           | Validate manifest and channel contracts.        |
| `build-all`          | `build-maze-nd.py build-all`          | Run fetch-raw → normalize → materialize-shared. |

The raw HuggingFace source provides `train` and `test` splits only.
`n_train` records are sampled from the train population; `n_val` and
`n_test` are sampled from the test population as non-overlapping partitions.

## Default Parameters

| Parameter   | Default | Description                |
| ----------- | ------- | -------------------------- |
| `--version` | 1       | Substrate version integer. |
| `--n-train` | 1000    | Training samples.          |
| `--n-val`   | 40      | Validation samples.        |
| `--n-test`  | 40      | Test samples.              |
| `--seed`    | 42      | Deterministic base seed.   |

## Usage Examples

```bash
# Quick local build with defaults
python scripts/data-gen/build-maze-nd.py build-all

# Custom sizes and seed
python scripts/data-gen/build-maze-nd.py build-all \
    --n-train 4000 --n-val 500 --n-test 500 --seed 7

# Validate an existing root
python scripts/data-gen/build-maze-nd.py validate data/interim/maze-nd/v1

# Build the MazeHard task corpus from this substrate
python scripts/data-gen/build-mazehard.py materialize-task \
    --parent-substrate data/interim/maze-nd/v1
```

## Downstream Consumers

| Task     | CLI Script                           | Parent Substrate Path        |
| -------- | ------------------------------------ | ---------------------------- |
| mazehard | `scripts/data-gen/build-mazehard.py` | `data/interim/maze-nd/v<N>/` |

## Related

- [Spec: Data Contracts §3.1](../../spec/spec-data-contracts.md)
- [Grid2D Channel Contracts](../../src/ehc_sn/data/substrate/grid2d.py)
- [MazeHard Task Builder](../../scripts/data-gen/build-mazehard.py)

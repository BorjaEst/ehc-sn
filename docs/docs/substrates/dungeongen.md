# dungeongen Shared Substrate

## Identity

| Property       | Value                                            |
| -------------- | ------------------------------------------------ |
| Family         | `dungeongen`                                     |
| Topology kind  | `grid2d`                                         |
| Source         | dungeongen procedural library (local generation) |
| Source ID      | `dungeongen`                                     |
| CLI script     | `scripts/data-gen/build-dungeongen.py`           |
| Builder module | `ehc_sn.data.substrate.dungeongen`               |
| Output paths   | `data/interim/dungeongen/<preset>/v<version>/`   |
| Dataset class  | `shared_substrate` or `layout_dataset`           |

## Description

The dungeongen substrate generates procedurally varied 2-D grid topologies
using the local dungeongen library. Each sample is a variable-size grid
with wall/passable cells, room regions, and a largest-4-connected-component
mask. The substrate assigns random observation IDs and structural landmarks
to each passable cell.

Dungeongen is the canonical substrate for dungeon-navigation tasks
(dungeon, arena, routebind). It is purely structural — no trajectory,
episode, or replay data lives here.

## Channels

| Channel        | Dtype | Shape  | Description                                      |
| -------------- | ----- | ------ | ------------------------------------------------ |
| `topology`     | bool  | (H, W) | Passable cells (`True`) vs walls (`False`).      |
| `observations` | int32 | (H, W) | Unique observation ID per passable cell.         |
| `mask_valid`   | bool  | (H, W) | Largest 4-connected component of `topology`.     |
| `regions`      | int32 | (H, W) | Room/region ID per cell (`-1` for walls).        |
| `landmarks`    | int32 | (H, W) | Structural landmark IDs (binary classification). |

All spatial channels are padded to a uniform `(H, W)` per substrate version.
See `ehc_sn.data.substrate.grid2d.validate_grid2d_sample` for the per-sample
contract.

## Pipeline Stages

| Stage                 | Command                                   | Description                                     |
| --------------------- | ----------------------------------------- | ----------------------------------------------- |
| `generate-topology`   | `build-dungeongen.py generate-topology`   | Ensure raw snapshot + normalize to interim NPZ. |
| `materialize-layouts` | `build-dungeongen.py materialize-layouts` | Build versioned layout dataset root.            |
| `validate`            | `build-dungeongen.py validate`            | Validate manifest and channel contracts.        |
| `build-all`           | `build-dungeongen.py build-all`           | Run generate-topology → materialize-layouts.    |

The raw stage (`ensure_raw_snapshot`) creates a deterministic tar-sharded
snapshot in `data/raw/dungeongen`. The interim stage (`prepare_interim`)
normalizes topologies into per-split NPZ files in `data/interim/dungeongen`.
The materialize stage pads topologies to uniform shape, assigns observations
and landmarks, and writes the versioned layout dataset.

## Default Parameters

| Parameter                       | Default | Description                               |
| ------------------------------- | ------- | ----------------------------------------- |
| `--version`                     | 1       | Substrate version integer.                |
| `--preset`                      | default | Named source preset.                      |
| `--n-train`                     | 250     | Training samples.                         |
| `--n-val`                       | 10      | Validation samples.                       |
| `--n-test`                      | 10      | Test samples.                             |
| `--height`                      | (infer) | Target grid height (inferred if omitted). |
| `--width`                       | (infer) | Target grid width (inferred if omitted).  |
| `--s-size` (`--n-observations`) | 45      | Distinct observation IDs to assign.       |
| `--topology-seed`               | 42      | Base seed for topology generation.        |
| `--n-sensory-instances`         | 1       | Sensory realizations per topology.        |

When `--height` and `--width` are omitted, the builder infers the required
grid dimensions from the maximum topology shape in the selected interim
slice.

## Usage Examples

```bash
# Quick local build (grid shape inferred)
python scripts/data-gen/build-dungeongen.py build-all

# Custom version
python scripts/data-gen/build-dungeongen.py build-all --version 2

# Explicit grid shape override
python scripts/data-gen/build-dungeongen.py build-all \
    --n-train 2000 --n-val 200 --n-test 200 \
    --height 48 --width 48 --s-size 8 --topology-seed 7

# Build an Arena corpus from dungeongen layouts
python scripts/data-gen/build-arena.py materialize-task \
    --layout-root data/interim/dungeongen/default/v1 --corpus dungeons
```

## Downstream Consumers

| Task      | CLI Script                            | Parent Substrate Path                    |
| --------- | ------------------------------------- | ---------------------------------------- |
| dungeon   | `scripts/data-gen/build-dungeon.py`   | `data/interim/dungeongen/<preset>/v<N>/` |
| arena     | `scripts/data-gen/build-arena.py`     | `data/interim/dungeongen/<preset>/v<N>/` |
| routebind | `scripts/data-gen/build-routebind.py` | `data/interim/dungeongen/<preset>/v<N>/` |

## Related

- [Spec: Data Contracts §3.1](../../spec/spec-data-contracts.md)
- [Grid2D Channel Contracts](../../src/ehc_sn/data/substrate/grid2d.py)
- [Dungeon Task Builder](../../scripts/data-gen/build-dungeon.py)
- [Arena Task Builder](../../scripts/data-gen/build-arena.py)

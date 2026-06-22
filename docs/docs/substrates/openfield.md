# openfield Layout Dataset

## Identity

| Property       | Value                                         |
| -------------- | --------------------------------------------- |
| Family         | `openfield`                                   |
| Topology kind  | `grid2d` (square, rectangle, hex)             |
| Source         | synthetic (procedural generation)             |
| Source ID      | `synthetic/openfield`                         |
| CLI script     | `scripts/data-gen/build-openfield.py`         |
| Builder module | `ehc_sn.data.layout.openfield`                |
| Output path    | `data/interim/openfield/<preset>/v<version>/` |
| Dataset class  | `layout_dataset`                              |

## Description

The openfield layout dataset generates legacy-TEM-compatible grid worlds:
square, rectangle, and future hex topologies. Each layout is a
`SpatialLayout` record with graph-state-indexed channels — adjacency,
transition matrix, row/col coordinates, observation IDs, and action space.

Openfield worlds have no walls: every state is valid and reachable from every
other state via 4-direction (or 6-direction for hex) + stay movement.
Sensory assignment is fully random, controlled by a per-layout sensory seed.

## Topology Types

| Type        | Adjacency         | Action Space | Actions |
| ----------- | ----------------- | ------------ | ------- |
| `square`    | 4-neighbor + stay | `grid4_dir`  | 5       |
| `rectangle` | 4-neighbor + stay | `grid4_dir`  | 5       |
| `hex`       | 6-neighbor + stay | `hex6_dir`   | 7       |

For hex grids, the internal graph width is `2*w - 1` and `square2hex` pruning
is applied via `valid_state_mask`.

## Presets

| Preset          | Type      | Grid Count | Widths/Heights                                    |
| --------------- | --------- | ---------- | ------------------------------------------------- |
| `tem-square`    | square    | 16         | [10,10,11,11,8,9,10,11,8,9,10,11,8,8,9,9]         |
| `tem-rectangle` | rectangle | 16         | widths: [11,11,12,12,8,8,9,9,11,11,12,12,8,8,9,9] |
| `tem-hex`       | hex       | 16         | [6,6,7,7,5,5,6,7,5,6,6,7,5,5,6,6]                 |
| `small`         | square    | 4          | [8,8,9,9]                                         |
| `big-square`    | square    | 1          | [30]                                              |

Each preset matches legacy TEM default environment sizes. The `small` preset
is intended for smoke tests; `big-square` for routebind topology generation.

## SpatialLayout Record Fields

Each sample is a `SpatialLayout` TypedDict with these graph-indexed arrays:

| Field                | Type    | Shape  | Description                              |
| -------------------- | ------- | ------ | ---------------------------------------- |
| `layout_id`          | str     | —      | Unique layout instance identifier.       |
| `layout_family`      | str     | —      | Always `"openfield"`.                    |
| `topology_type`      | str     | —      | `"square"`, `"rectangle"`, or `"hex"`.   |
| `graph_state_count`  | int     | —      | Number of graph nodes (W×H).             |
| `valid_state_mask`   | bool    | (S,)   | All `True` for openfield (no walls).     |
| `state_to_row_col`   | int32   | (S, 2) | (row, col) per graph state.              |
| `observation_id`     | int32   | (S,)   | Sensory observation ID per state.        |
| `adjacency`          | bool    | (S, S) | Undirected connectivity matrix.          |
| `action_space`       | dict    | —      | `ActionSpace` with names and deltas.     |
| `transition_matrix`  | float64 | (S, S) | Row-stochastic transition probabilities. |
| `topology_seed`      | int     | —      | Seed for topology generation.            |
| `sensory_seed`       | int     | —      | Seed for observation ID assignment.      |
| `sensory_vocab_size` | int     | —      | Number of distinct observation IDs.      |
| `split`              | str     | —      | `"train"`, `"val"`, or `"test"`.         |

## Pipeline Stages

| Stage                 | Command                                  | Description                                  |
| --------------------- | ---------------------------------------- | -------------------------------------------- |
| `generate-topology`   | `build-openfield.py generate-topology`   | Write per-split source spec JSONL.           |
| `materialize-layouts` | `build-openfield.py materialize-layouts` | Assign sensory IDs, write layout dataset.    |
| `validate`            | `build-openfield.py validate`            | Validate manifest and channel contracts.     |
| `build-all`           | `build-openfield.py build-all`           | Run generate-topology → materialize-layouts. |

The two-stage pipeline supports topology-only generation in stage 1
(`observation_id = -1` sentinel) and sensory enrichment in stage 2. This
allows multiple sensory instances per topology template via
`--n-sensory-instances`.

## Default Parameters

| Parameter               | Default    | Description                          |
| ----------------------- | ---------- | ------------------------------------ |
| `--version`             | 1          | Layout dataset version integer.      |
| `--preset`              | tem-square | Named preset (or custom `--widths`). |
| `--s-size`              | 45         | Sensory vocabulary size.             |
| `--n-sensory-instances` | 1          | Sensory realizations per topology.   |
| `--topology-seed`       | 42         | Base seed for topology generation.   |

When `--preset` is omitted, custom `--widths` (and optionally `--heights`
and `--topology-type`) can be supplied directly.

## Usage Examples

```bash
# Default faithful TEM reproduction (16 square grids)
python scripts/data-gen/build-openfield.py build-all

# Rectangle grids
python scripts/data-gen/build-openfield.py build-all --preset tem-rectangle

# Small smoke test
python scripts/data-gen/build-openfield.py build-all --preset small

# Custom widths
python scripts/data-gen/build-openfield.py build-all \
    --widths 10 --widths 10 --widths 11

# Two-phase pipeline (topology-only then sensory enrichment)
python scripts/data-gen/build-openfield.py generate-topology --preset small
python scripts/data-gen/build-openfield.py materialize-layouts --preset small

# Build an Arena corpus from openfield layouts
python scripts/data-gen/build-arena.py materialize-task \
    --layout-root data/interim/openfield/tem-square/v1 --corpus openfield-tem-square
```

## Downstream Consumers

| Task      | CLI Script                            | Parent Layout Path                      |
| --------- | ------------------------------------- | --------------------------------------- |
| arena     | `scripts/data-gen/build-arena.py`     | `data/interim/openfield/<preset>/v<N>/` |
| routebind | `scripts/data-gen/build-routebind.py` | `data/interim/openfield/<preset>/v<N>/` |

## Related

- [Spec: Data Contracts §3.1](../../spec/spec-data-contracts.md)
- [Spec: Openfield Layout](../../spec/spec-openfield-layout.md)
- [SpatialLayout Protocol](../../src/ehc_sn/data/layout/_protocol.py)
- [Openfield Generator](../../src/ehc_sn/data/layout/openfield.py)
- [Arena Task Builder](../../scripts/data-gen/build-arena.py)

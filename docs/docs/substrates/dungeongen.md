# dungeongen Layout Dataset

## Identity

| Property       | Value                                            |
| -------------- | ------------------------------------------------ |
| Family         | `dungeongen`                                     |
| Topology kind  | `grid2d`                                         |
| Source         | dungeongen procedural library (local generation) |
| Source ID      | `dungeongen`                                     |
| CLI script     | `scripts/data-gen/build-dungeongen.py`           |
| Builder module | `ehp_sn.data.substrate.dungeongen`               |
| Output paths   | `data/interim/dungeongen/<preset>/v<version>/`   |
| Dataset class  | `layout_dataset`                                 |

## Purpose and ownership

Dungeongen generates procedurally varied 2-D grid topologies with walls,
passable cells, room regions, and random sensory assignment. Each layout is
a `SpatialLayout` record — a compact graph-indexed representation of
traversable states. The exported state set is exactly the largest
4-connected component of passable cells.

Dungeongen produces spatial layouts only. It does not produce task queries,
trajectories, targets, episodes, or semantic-transition graphs.

## Semantic model

### Geometry

Each generated dungeon is a rectangular grid with wall and passable cells.
The builder may use padded dense intermediates internally; padding does not
appear as compact graph states in the exported `SpatialLayout`.

Two distinct spatial domains:

| Quantity                        | Key                   | Description                                          |
| ------------------------------- | --------------------- | ---------------------------------------------------- |
| Natural extent                  | `extent` (per layout) | Raster bounds `(H_i, W_i)` of the generated dungeon. |
| Compact traversable-state count | `graph_state_count`   | Number of passable cells (walls excluded).           |

Positions outside a dungeon's natural extent are padding and have no
compact graph representation.

### Observation assignment

Every traversable cell receives an observation ID from `{0, …, s_size-1}`.
IDs may repeat — there are typically more traversable cells than distinct
IDs. Every traversable cell is an observation cell (no FREE cells in the
compact representation).

Walls have no compact state and therefore no `observation_id` entry.
Dense consumers may represent absent positions with a sentinel during
projection.

### Extent heterogeneity

A dungeongen layout dataset may contain layouts with heterogeneous natural
extents. Each layout's declared `extent` is authoritative for that layout.
Downstream tasks read extent from each `SpatialLayout` record, not from
the manifest-level convenience copy.

The manifest reports:

```json
{
  "natural_extent_homogeneous": false,
  "natural_height_range": [20, 32],
  "natural_width_range": [20, 32]
}
```

Dungeongen validation must not reject heterogeneous extents.

### Conversion

Conversion from raw dungeon geometry to `SpatialLayout` is governed by
versioned policy fields (rasterization, component selection, door/corridor
handling) recorded in the manifest `stage_params`.

## Output artifact

| Field                         | Type      | Shape  | Description                                                        |
| ----------------------------- | --------- | ------ | ------------------------------------------------------------------ |
| `layout_id`                   | str       | —      | Unique layout instance identifier.                                 |
| `layout_family`               | str       | —      | Always `"dungeongen"`.                                             |
| `topology_type`               | str       | —      | `"rectangle"` (4-neighbor rectangular grid).                       |
| `topology_kind`               | str       | —      | Always `"grid2d"`.                                                 |
| `graph_state_count`           | int       | —      | Number of traversable graph nodes.                                 |
| `state_to_row_col`            | int32     | (S, 2) | (row, col) per graph state.                                        |
| `observation_id`              | int32     | (S,)   | Observation ID per state (may repeat).                             |
| `extent`                      | (int,int) | —      | Natural canvas dimensions `(height, width)`.                       |
| `next_state`                  | int32     | (S, A) | Destination state index per action; -1 sentinel for invalid moves. |
| `action_valid`                | bool      | (S, A) | True where `next_state != -1`.                                     |
| `action_space`                | dict      | —      | `ActionSpace` with names, deltas, and `movement_kind`.             |
| `topology_seed`               | int       | —      | Seed for topology generation.                                      |
| `observation_seed`            | int       | —      | Seed for observation ID assignment.                                |
| `observation_vocabulary_size` | int       | —      | Cardinality of the observation-ID domain `{0, …, N−1}`.            |
| `split`                       | str       | —      | `"train"`, `"val"`, or `"test"`.                                   |

## Invariants

- The compact graph contains only the largest 4-connected component of passable cells.
- `next_state[i, a]` ∈ {−1} ∪ {0, …, S−1}; −1 indicates an invalid move.
- `action_valid[i, a]` == True iff `next_state[i, a] != -1`.
- `observation_id` values are drawn from `{0, …, observation_vocabulary_size-1}`.
- Walls are absent from `state_to_row_col`; they exist only as missing positions.
- `extent` records the natural bounding box, not the internal storage canvas.

## Presets

| Preset         | Description                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `default`      | General-purpose dungeongen generation with no extent bound. Produces layouts up to ~44×42 cells. |
| `routebind-30` | Routebind-compatible bounded profile. Every exported layout has natural extent ≤ 30×30.          |

### `routebind-30` details

The `routebind-30` preset generates dungeons using source-native compact settings
(DungeonSize.SMALL, cozy room-size bias, high density, no symmetry, passage
width 1). After rasterization, any layout whose natural extent exceeds 30×30 is
deterministically rejected and retried with the next seed in the attempt
sequence. The layout dataset records the accepted attempt index in each
`SpatialLayout` record.

Seeding is deterministic: given the same `--topology-seed` and attempt budget,
the same set of accepted layouts is produced. The observation vocabulary size
defaults to 45 and may be overridden via `--observation-vocabulary-size`.

Usage:

```bash
python build-dungeongen.py build --preset routebind-30
python build-routebind.py build --topology-root data/interim/dungeongen/routebind-30/v1 ...
```

## Build configuration

| Parameter                       | Default | Description                                       |
| ------------------------------- | ------- | ------------------------------------------------- |
| `--version`                     | 1       | Substrate version integer.                        |
| `--preset`                      | default | Named source preset (see `--help` for available). |
| `--n-train`                     | 250     | Training samples.                                 |
| `--n-val`                       | 10      | Validation samples.                               |
| `--n-test`                      | 10      | Test samples.                                     |
| `--observation-vocabulary-size` | 45      | Distinct observation IDs to assign.               |
| `--topology-seed`               | 42      | Base seed for topology generation.                |
| `--n-sensory-instances`         | 1       | Sensory realizations per topology.                |

Seeding: a single `--topology-seed` is expanded per layout via
`numpy.random.SeedSequence`.

## CLI

| Command    | Description                                   |
| ---------- | --------------------------------------------- |
| `build`    | Produce a complete dungeongen layout dataset. |
| `validate` | Validate manifest and channel contracts.      |
| `inspect`  | Print a human-readable manifest summary.      |

Usage:

```bash
python build-dungeongen.py build
python build-dungeongen.py validate data/interim/dungeongen/default/v1
python build-dungeongen.py inspect data/interim/dungeongen/default/v1
```

## Manifest

Root file: `manifest.json`. Dataset class: `layout_dataset`.
Channels: (none — per-layout NPZ files). See `spec-data-contracts.md` §4.5
for the full manifest field table.

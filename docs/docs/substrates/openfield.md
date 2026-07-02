# openfield Layout Dataset

## Identity

| Property       | Value                                         |
| -------------- | --------------------------------------------- |
| Family         | `openfield`                                   |
| Topology kind  | `grid2d` (square, rectangle, hex)             |
| Source         | synthetic (procedural generation)             |
| Source ID      | `synthetic/openfield`                         |
| CLI script     | `scripts/data-gen/build-openfield.py`         |
| Builder module | `ehp_sn.data.layout.openfield`                |
| Output path    | `data/interim/openfield/<preset>/v<version>/` |
| Dataset class  | `layout_dataset`                              |

## Purpose and ownership

Openfield generates grid worlds (square, rectangle, hex) with random sensory
assignments. Each layout is a `SpatialLayout` record — a compact
graph-indexed representation of traversable states with physical movement
connectivity, action space, and observation identities.

Every exported graph state is traversable. Square and rectangle layouts
occupy their complete rectangular extent. Hex layouts may use an internal
rectangular backing raster; invalid backing positions are removed before
export. The compact representation contains only valid states — walls are
absent from the index space.

## Semantic model

### Topology types

| Type        | Movement adjacency | Action space | Actions |
| ----------- | ------------------ | ------------ | ------- |
| `square`    | 4-neighbor         | `grid4_dir`  | 5       |
| `rectangle` | 4-neighbor         | `grid4_dir`  | 5       |
| `hex`       | 6-neighbor         | `hex6_dir`   | 7       |

Three distinct concepts are kept separate:

- **Movement adjacency**: state-changing physical edges only (no self-loops),
  exposed as `next_state` (SxA) + `action_valid` (SxA).
- **Action space**: all movement actions including STAY.
- **Transition matrix**: not stored — row-stochastic transitions are derived
  from `next_state` + `action_valid` + STAY self-loop on demand.

### Observation assignment

Each traversable state receives one random observation ID drawn from
`{0, …, observation_vocabulary_size-1}`. IDs are assigned independently
per layout, controlled by the per-layout `observation_seed`.

### Extent

`extent` is the natural canvas dimensions `(height, width)` derived from the
generated geometry's bounds. For square and rectangle layouts the extent is
exactly filled by traversable states; no wall cells exist within the extent.
Hex layouts may have a larger extent than their exported state count due to
backing-pruning.

## Extent heterogeneity

An openfield layout dataset may contain layouts with heterogeneous natural
extents. Each layout's declared `extent` is authoritative for that layout.
Downstream tasks (arena, routebind) read extent from each `SpatialLayout`
record, not from the manifest-level convenience copy.

Presets such as `big-square` produce a single fixed extent; presets such
as `tem-square` and `small` produce multiple extents. Both patterns
are valid.

The manifest reports:

```json
{
  "natural_extent_homogeneous": false,
  "natural_height_range": [8, 11],
  "natural_width_range": [8, 11]
}
```

Openfield validation must not reject heterogeneous extents.

## Output artifact

Each sample is a `SpatialLayout` record with these graph-indexed arrays:

| Field                         | Type      | Shape  | Description                                                        |
| ----------------------------- | --------- | ------ | ------------------------------------------------------------------ |
| `layout_id`                   | str       | —      | Unique layout instance identifier.                                 |
| `layout_family`               | str       | —      | Always `"openfield"`.                                              |
| `topology_type`               | str       | —      | `"square"`, `"rectangle"`, or `"hex"`.                             |
| `topology_kind`               | str       | —      | Always `"grid2d"`.                                                 |
| `graph_state_count`           | int       | —      | Number of traversable graph nodes.                                 |
| `state_to_row_col`            | int32     | (S, 2) | (row, col) per graph state.                                        |
| `observation_id`              | int32     | (S,)   | Observation ID per state.                                          |
| `extent`                      | (int,int) | —      | Natural canvas dimensions `(height, width)`.                       |
| `next_state`                  | int32     | (S, A) | Destination state index per action; -1 sentinel for invalid moves. |
| `action_valid`                | bool      | (S, A) | True where `next_state != -1`.                                     |
| `action_space`                | dict      | —      | `ActionSpace` with names, deltas, and `movement_kind`.             |
| `topology_seed`               | int       | —      | Seed for topology generation.                                      |
| `observation_seed`            | int       | —      | Seed for observation ID assignment.                                |
| `observation_vocabulary_size` | int       | —      | Cardinality of the observation-ID domain `{0, …, N−1}`.            |
| `split`                       | str       | —      | `"train"`, `"val"`, or `"test"`.                                   |

## Invariants

- `graph_state_count` = W×H for square/rectangle; ≤ W×H for hex.
- `next_state[i, a]` ∈ {−1} ∪ {0, …, S−1}; −1 indicates an invalid move.
- `action_valid[i, a]` == True iff `next_state[i, a] != -1`.
- Every coordinate in `state_to_row_col` satisfies `0 ≤ r < extent[0]`, `0 ≤ c < extent[1]`.
- `observation_id` values are drawn from `{0, …, observation_vocabulary_size-1}`.
- `state_to_row_col` is invertible (no two states share the same row/col).

## Build configuration

| Preset          | Type      | Grid shape(s)                      |
| --------------- | --------- | ---------------------------------- |
| `tem-square`    | square    | 16 grids, widths [8, 9, 10, 11]    |
| `tem-rectangle` | rectangle | 16 grids, mixed widths and heights |
| `tem-hex`       | hex       | 16 grids, widths [5, 6, 7]         |
| `small`         | square    | 4 grids, widths [8, 9]             |
| `big-square`    | square    | 1 grid, width 30                   |

Parameters: `--preset`, `--version`, `--n-train`, `--n-val`, `--n-test`,
`--observation-vocabulary-size`, `--height`, `--width`, `--widths`, `--n-sensory-instances`,
`--topology-seed`, `--raw-root`, `--interim-root`, `--force`.

Seeding: a single `--topology-seed` is expanded per layout using
`numpy.random.SeedSequence` split streams for topology shape and sensory
assignment.

## CLI

| Command    | Description                                            |
| ---------- | ------------------------------------------------------ |
| `build`    | Produce a complete openfield layout dataset.           |
| `validate` | Validate manifest, NPZ files, and dataset constraints. |
| `inspect`  | Print a human-readable manifest summary.               |

Usage:

```bash
python build-openfield.py build --preset small
python build-openfield.py validate data/interim/openfield/small/v1
python build-openfield.py inspect data/interim/openfield/small/v1
```

## Manifest

Root file: `manifest.json`. Dataset class: `layout_dataset`.
Channels: (none — per-layout NPZ files). See `spec-data-contracts.md` §4.5
for the full manifest field table.

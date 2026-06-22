# `routebind` Benchmark Task

## Identity

| Property      | Value                                           |
| ------------- | ----------------------------------------------- |
| Task name     | `routebind`                                     |
| Benchmark     | goal-conditioned spatial route binding          |
| Package path  | `src/ehc_sn/tasks/routebind/`                   |
| CLI script    | `scripts/data-gen/build-routebind.py`           |
| Output path   | `data/processed/routebind/<corpus>/v<version>/` |
| Dataset class | `task_corpus`                                   |

## Purpose and ownership

Routebind is a goal-conditioned spatial prospective-field task. Each sample
supplies a complete two-dimensional layout containing walls, traversable
cells, stable observation identities, one physical start position, and one
semantic goal observation occurring at one or more positions. A fixed
directed graph over observation identities is shared across the corpus but
hidden from the model. The model must learn this graph parametrically,
combine it with visible spatial reachability, select a valid sequence of
observation occurrences, and predict a discounted field over the physical
route from start to goal.

## Semantic model

### Symbol table

| Symbol  | Surface | Description                                        |
| ------- | ------- | -------------------------------------------------- |
| P       | yes     | S = H x W spatial grid positions (row-major)       |
| O       | yes     | stable observation identities, O = {0, ..., N-1}   |
| E_obs   | ---     | directed observation-transition edges (hidden)     |
| G_obs   | ---     | fixed hidden DAG (O, E_obs)                        |
| p_start | yes     | unique physical start position                     |
| o_goal  | ---     | semantic goal observation (via goal_flag)          |
| f_traj  | yes     | spatial trajectory field over S positions (output) |
| f_wp    | yes     | semantic waypoint field over S positions (output)  |

### Hidden semantic graph

One fixed directed acyclic graph per corpus: G_obs = (O, E_obs).
An edge o_i -> o_j means that after semantically accepting o_i, the
route may next accept o_j. Properties:

- Directed, acyclic, fixed across all splits.
- Every non-source node has in-degree >= 1; every non-sink node has out-degree >= 1.
- The underlying undirected graph is connected.
- The graph is never exposed in model inputs.

### Semantic-state initialization

The start position must contain a real observation: o_0 = phi(p_start).
The start observation is already accepted. The first post-start acceptance
must satisfy (o_0, o_1) in E_obs.

### Semantic transition rule

The planning state is (p, o). Two transition types exist:

- Physical movement -- move to an adjacent traversable position q:
  (p, o) -> (q, o) with cost 1.
- Semantic acceptance -- accept the observation at the current position:
  (p, o) -> (p, o') with cost 0, where phi(p) = o' and
  (o, o') in E_obs.

Conventions:

- Semantic acceptance is optional at any observation cell.
- An observation cannot be accepted unless it is a valid DAG successor.
- After accepting the goal observation, the task terminates immediately.
- Physical presence and semantic acceptance are independent. Crossing an
  observation cell without explicit acceptance does not update the semantic
  state.

### Oracle

The oracle is a joint product-state shortest-path search over
X = P_free x O. Goal states are any
(p, o_goal) where phi(p) = o_goal. The optimization
minimizes total physical path length:

pi\* = argmin_pi sum_over_physical_transitions 1

Semantic acceptance transitions constrain validity but do not add cost.
The oracle jointly selects: the observation sequence, concrete occurrences,
physical route between occurrences, and final goal occurrence.

**Three canonical route objects:**

- Product-state path Pi\* = ((p_0, o_0), ..., (p_T, o_T)) --
  contains physical moves and zero-cost semantic acceptances.
- Projected physical route R\* = (p_0, p_1, ..., p_L) --
  physical movement positions only, semantic transitions removed. This is
  what the trajectory field represents.
- Accepted waypoint sequence W\* = ((p_i0, o_0), ..., (p_iM, o_M)) --
  semantic acceptance events. This is what the waypoint field represents.

**Uniqueness**: Two solutions are task-equivalent when they produce the
same projected physical route and the same ordered accepted waypoint events.
The builder enforces strict uniqueness:

1. Exactly one optimal task-equivalence class exists.
2. The projected physical route is simple -- no position is revisited:
   p_i != p_j for all i != j.

Samples violating either constraint are rejected with recorded reasons.

### Outputs

**Primary output -- spatial trajectory field (f_traj):**

For the selected physical route pi\*\_space = (p_0, ..., p_L):

f_traj*(p_k) = gamma_space^k
f_traj*(p) = 0 for p not in pi\*\_space

where gamma_space in (0, 1) is the spatial field-decay factor.

**Structural supervision -- semantic waypoint field (f_wp):**

For accepted observations (p_i0, o_0), ..., (p_iM, o_M) where
o_M = o_goal and m = 0 receives 1.0:

f_wp*(p_im) = gamma_semantic^m
f_wp*(p) = 0 otherwise

Spatial and semantic decay are independent. Spatial decay counts physical
moves; semantic decay counts acceptance events.

**Semantic-length terminology** (waypoint_count includes start):

| Term                        | Definition                        | Value |
| --------------------------- | --------------------------------- | ----- |
| waypoint_count              | total accepted observations       | M+1   |
| semantic_transition_count   | edges traversed                   | M     |
| intermediate_waypoint_count | accepted excluding start and goal | M-1   |

**Auxiliary outputs:**

| Output                  | Shape      | Description                              |
| ----------------------- | ---------- | ---------------------------------------- |
| next_direction_logits   | (B, 4)     | categorical over {UP, RIGHT, DOWN, LEFT} |
| next_observation_logits | (B, N_obs) | categorical over observation vocabulary  |

### Target encoding

| Target            | Source                                               |
| ----------------- | ---------------------------------------------------- |
| target_trajectory | spatial decay over the projected physical route      |
| target_waypoint   | semantic decay over accepted observation occurrences |
| target_next_dir   | first physical step of the optimal route             |
| target_next_obs   | first post-start accepted observation                |

## Parent requirements

### Spatial topology

The spatial parent must provide:

- extent -- declared canvas dimensions (H, W).
- state_to_row_col -- compact state coordinates (N, 2).
- observation_id -- observation identity per traversable state (N,).
- next_state (SxA) with action_valid (SxA) -- four-neighbor physical
  connectivity, validated against `movement_kind == "grid4"`.
- observation_vocabulary_size -- declared observation domain size.
- topology_kind == grid2d and topology_type in {square, rectangle}.
- action_space with `movement_kind == "grid4"` (no hex).

The topology parent must not contain routebind query semantics (hidden DAG,
start/goal selection, waypoints, or routebind targets).

### Semantic graph

The semantic parent must provide a single fixed directed acyclic graph over
exactly the same declared observation vocabulary as the spatial parent:
O_DAG = O_topology. The graph is consumed as hidden
oracle structure; adjacency is never exposed in model inputs.

### Dense canonicalization

Routebind converts each compact graph-indexed SpatialLayout into a dense
fixed-size storage canvas of S = H_store x W_store row-major positions.
The parent layout's natural extent H_i x W_i may be smaller than the
storage canvas. Positions outside the embedded natural extent become
CELL_PAD (storage padding, neither a real wall nor free space).

Within the embedded natural extent:

- Positions with a compact graph state become CELL_OBSERVATION
  (CELL_FREE is reserved for future use; currently every traversable state
  carries an observation).
- Positions without a compact graph state become CELL_WALL.
- Physical-neighbor adjacency is consumed from the parent layout's
  `next_state` + `action_valid` and validated against canonical grid4
  expectations.
- The required action space is four-neighbor undirected movement, no one-way passages.

Each stored sample carries a `spatial_mask` — `True` where the slot
corresponds to a real position in the sample's natural domain, `False`
for storage padding. Losses, metrics, route extraction, and figure
rendering must exclude padding positions.

Four cell classes are distinguished:

| Cell type       | Belongs to layout | Traversable | Observation |
| --------------- | ----------------- | ----------- | ----------- |
| `CELL_PAD (3)`  | No                | No          | No          |
| `CELL_WALL (0)` | Yes               | No          | No          |
| `CELL_FREE (1)` | Yes               | Yes         | No          |
| `CELL_OBS (2)`  | Yes               | Yes         | Yes         |

CELL_PAD is not a synonym for CELL_FREE (that would create artificial
traversable space) and not a synonym for CELL_WALL (walls are real
positions inside the world).

## Output artifact

### Input channels

| Field          | Shape  | Type  | Description                                                       |
| -------------- | ------ | ----- | ----------------------------------------------------------------- |
| cell_type      | (B, S) | int32 | {WALL, FREE, OBSERVATION, PAD}                                    |
| observation_id | (B, S) | int32 | stable observation identity; sentinel for non-observation cells   |
| start_flag     | (B, S) | bool  | True for exactly one traversable position                         |
| goal_flag      | (B, S) | bool  | True for all positions containing the goal observation            |
| spatial_mask   | (B, S) | bool  | True inside the natural spatial domain; False for storage padding |

The task input does not contain the observation-transition adjacency
matrix, a precomputed observation sequence, a distance matrix, path-order
labels, or a sequence of required waypoints.

### Metadata channels (per sample)

| Field          | Shape  | Type  | Description                                                         |
| -------------- | ------ | ----- | ------------------------------------------------------------------- |
| natural_height | scalar | int32 | Natural layout height in cells                                      |
| natural_width  | scalar | int32 | Natural layout width in cells                                       |
| row_offset     | scalar | int32 | Row offset for embedding the natural extent into the storage canvas |
| col_offset     | scalar | int32 | Col offset for embedding the natural extent into the storage canvas |

These metadata channels allow reconstruction of the original natural
geometry from the padded storage tensor.

### Corpus channels

Per-sample channels stored in the task corpus (one NPY array per channel
per split):

cell_type, observation_id, start_flag, goal_flag, spatial_mask,
natural_height, natural_width, row_offset, col_offset,
target_trajectory, target_waypoint, target_next_dir, target_next_obs.

## Invariants

- One fixed DAG per corpus; all samples share (O, E_obs).
- O_DAG = O_topology (declared vocabularies match).
- The start cell is traversable and contains a real observation.
- The goal differs from the start observation and is semantically reachable.
- At least one physical occurrence of the goal is present.
- Exactly one optimal task-equivalence class exists.
- The projected physical route contains no repeated positions.
- Every physical step is a valid four-neighbor non-wall move.
- Every semantic acceptance follows an edge in the hidden DAG.
- Crossed but unaccepted observation cells are excluded from the waypoint field.
- The trajectory and waypoint fields follow their respective decay rules.
- The next-direction target matches the first physical step.
- The next-observation target matches the first post-start acceptance.
- Regeneration with the same substrates, task seed, and query produces
  identical tensors.
- Padding positions (spatial_mask == False) have zero-valued target fields.
- Padding positions do not contribute to losses, metrics, route extraction,
  or figure rendering.

## Build configuration

### Storage policy

Routebind accepts heterogeneous parent natural extents. Every corpus
declares one configured storage extent `[storage_height, storage_width]`.
Each selected parent layout must fit within that storage extent:

    0 < H_i <= H_store    and    0 < W_i <= W_store

Layouts are embedded into the storage canvas via deterministic placement;
a layout larger than the storage extent fails compatibility validation.

The corpus requires homogeneous **storage shape**, not homogeneous natural
shape.

| Policy                 | Value                   |
| ---------------------- | ----------------------- |
| spatial_storage_policy | `pad_to_configured_max` |
| storage_extent         | `[H_store, W_store]`    |
| placement_policy       | `center`                |

Per-sample placement offsets are stored in metadata channels
(`row_offset`, `col_offset`) and computed as:

    row_offset = floor((H_store - H_i) / 2)
    col_offset = floor((W_store - W_i) / 2)

CLI parameters `--storage-height` and `--storage-width` replace the
ambiguous `--canvas-height`/`--canvas-width` aliases.

**Rationale for configured maximum**:

- The corpus schema does not change when one new layout is added.
- The model sequence capacity is predictable across splits.
- Train, validation, and test use the same representation.
- Incompatible layouts are rejected before expensive oracle generation.
- Experiment configuration remains reproducible.

### Routebind presets

| Preset   | Purpose                                                 |
| -------- | ------------------------------------------------------- |
| smoke    | Accept-all; one bucket covering all lengths.            |
| balanced | Broad physical (2-80) and semantic (2-10) distribution. |

Parameters: --preset, --topology-root, --dagflow-root,
--dagflow-graph-id, --corpus, --version, --field-decay-spatial,
--field-decay-semantic, --max-supported-route-length,
--n-queries-per-layout, --seed, --min-route-length,
--max-route-length, --attempt-budget.

Field decay: gamma_space = f_min ^ (1 / L_max).
Example: L_max = 150, f_min = 0.1 gives gamma_space ~ 0.9848.

## CLI

| Command  | Description                                             |
| -------- | ------------------------------------------------------- |
| build    | Materialize a Routebind corpus from topology + dagflow. |
| validate | Verify a corpus against structural and semantic checks. |
| inspect  | Examine corpus metadata, samples, and diagnostics.      |

Usage:

```bash
python build-routebind.py build \
  --topology-root data/interim/openfield/big-square/v1 \
  --dagflow-root data/interim/dagflow/routing/v1 \
  --dagflow-graph-id dagflow-routing-v1-train-000000
python build-routebind.py validate data/processed/routebind/default/v1
python build-routebind.py inspect data/processed/routebind/default/v1 --summary
```

## Manifest

Root file: manifest.json. Key fields:

| Field                        | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| storage_extent               | Storage canvas [height, width].                     |
| num_spatial_slots            | Fixed tensor width S = H_store \* W_store.          |
| spatial_storage_policy       | `"pad_to_configured_max"`                           |
| placement_policy             | `"center"`                                          |
| natural_extent_homogeneous   | Whether all parent layouts share a single extent    |
| natural_height_range         | `[min_height, max_height]` across parent layouts    |
| natural_width_range          | `[min_width, max_width]` across parent layouts      |
| n_observations               | Observation vocabulary cardinality.                 |
| field_decay_spatial          | Spatial field decay factor.                         |
| field_decay_semantic         | Semantic field decay factor.                        |
| max_supported_physical_moves | Maximum route length for terminal activation check. |
| parents.spatial_topology     | Topology parent (family, root, version).            |
| parents.semantic_graph       | Dagflow parent (artifact_id, content_digest).       |

The deprecated n_states key is a synonym for num_spatial_slots.
n_observations must equal topology_observation_vocabulary_size.

## Targets and metrics

| Aspect                          | Value                                              |
| ------------------------------- | -------------------------------------------------- |
| Benchmark track                 | Routebind-Field                                    |
| Claim family                    | goal_conditioned_spatial_route_binding             |
| Execution mode                  | single-step field prediction                       |
| Primary metric                  | valid_semantic_spatial_route_rate (behavioral)     |
| Primary optimality metric       | semantic_spatial_path_cost_ratio                   |
| Primary representational metric | balanced_trajectory_field_error                    |
| Calibration metric              | trajectory_field_mse                               |
| Structural metric               | balanced_waypoint_field_error                      |
| Auxiliary metrics               | next_direction_accuracy, next_observation_accuracy |

valid_semantic_spatial_route_rate: fraction of samples whose extracted
route begins at p_start, uses traversable four-neighbor moves,
contains no repeated positions, terminates at a goal_flag position, and
has a valid semantic acceptance subsequence under the hidden DAG.

semantic_spatial_path_cost_ratio: for valid routes, the ratio of
predicted route cost to oracle optimal cost C(R_hat) / C(R_star).
C(R) = |R| - 1 (number of physical moves). A value of 1.0 indicates
exact optimality.

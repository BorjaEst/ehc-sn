# `routebind` Benchmark Task

## Identity

| Property      | Value                                           |
| ------------- | ----------------------------------------------- |
| Task name     | `routebind`                                     |
| Benchmark     | goal-conditioned spatial route binding          |
| Package path  | `src/ehp_sn/tasks/routebind/`                   |
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

**Canonical oracle truth.** The oracle does NOT commit to a single
reconstructed path. Instead it stores the complete optimal subgraph:

- The **reverse distance table** $d^*(p, o)$ — minimum remaining physical
  moves from product state $(p, o)$ to any terminal state $(q, o_g)$
  where $\phi(q) = o_g$. Computed by one reverse 0-1 BFS per goal
  observation. Well-defined even when many optimal solutions exist.
- The **optimal transition relation** — derived from Bellman equality:
  $x \rightarrow y$ is optimal iff $d^*(x) = c(x,y) + d^*(y)$. For
  physical moves $c = 1$, for semantic acceptances $c = 0$.
- The **query-reachable optimal subgraph** — all product states reachable
  from a given start via optimal transitions.

From the optimal subgraph, four target projections are derived:

- **Trajectory support** $s_{\text{traj}}(p)$ — binary mask: $p$ belongs
  to at least one optimal physical route.
- **Trajectory forward depth** $k_{\min}(p) = \min_{o} D - d^*(p, o)$
  where $D = d^*(p_{\text{start}}, o_{\text{start}})$ is the optimal
  cost. This is the earliest physical step at which $p$ can appear.
- **Waypoint support** $s_{\text{wp}}(p)$ — binary mask: an optimal
  semantic acceptance occurs at $p$ (plus the start position).
- **Waypoint semantic depth** $m_{\min}(p)$ — minimum number of
  acceptance events from start.

The optimal subgraph is traversed once per selected query — no path
enumeration, no uniqueness gating, no arbitrary tie-breaking.

**Three canonical route objects:**

- Product-state path Pi\* = ((p_0, o_0), ..., (p_T, o_T)) --
  contains physical moves and zero-cost semantic acceptances.
- Projected physical route R\* = (p_0, p_1, ..., p_L) --
  physical movement positions only, semantic transitions removed. This is
  what the trajectory field represents.
- Accepted waypoint sequence W\* = ((p_i0, o_0), ..., (p_iM, o_M)) --
  semantic acceptance events. This is what the waypoint field represents.

**Ambiguity is not a rejection criterion.** When multiple optimal
solutions exist (different physical routes or different waypoint
sequences at equal cost), all are represented in the optimal-support
targets. The trajectory field encodes every position reachable on any
optimal route; the auxiliary masks encode every optimal first action.

### Outputs

**Primary output -- spatial trajectory field (f_traj):**

Derived from optimal-subgraph projections:

f_traj*(p) = s_traj(p) * gamma_space^{k_min(p)}
f_traj\*(p) = 0 where s_traj(p) = False

where:

- $s_{\text{traj}}(p)$ is the trajectory support mask (True for any
  position participating in at least one optimal route),
- $k_{\min}(p)$ is the earliest optimal forward depth at position $p$,
- $\gamma_{\text{space}} \in (0, 1)$ is the spatial field-decay factor.

When exactly one optimal route exists, this is identical to the
single-path decay field $\gamma_{\text{space}}^k$ at route position $k$.
When multiple optimal routes exist, every position on any optimal
branch is supported, at its earliest possible depth.

**Structural supervision -- semantic waypoint field (f_wp):**

Derived from optimal-subgraph projections:

f_wp*(p) = s_wp(p) * gamma_semantic^{m_min(p)}
f_wp\*(p) = 0 where s_wp(p) = False

where:

- $s_{\text{wp}}(p)$ is the waypoint support mask (True for any
  position where an optimal semantic acceptance occurs, plus the start),
- $m_{\min}(p)$ is the minimum number of acceptance events from start
  to a waypoint event at $p$,
- $\gamma_{\text{semantic}} \in (0, 1)$ is the semantic field-decay factor.

Spatial and semantic decay are independent. Spatial decay counts physical
moves; semantic decay counts acceptance events.

**Semantic-length terminology** (waypoint_count includes start):

| Term                        | Definition                        | Value |
| --------------------------- | --------------------------------- | ----- |
| waypoint_count              | total accepted observations       | M+1   |
| semantic_transition_count   | edges traversed                   | M     |
| intermediate_waypoint_count | accepted excluding start and goal | M-1   |

**Primary auxiliary output — optimal direction mask:**

| Output                    | Shape  | Description                                         |
| ------------------------- | ------ | --------------------------------------------------- |
| target_optimal_directions | (B, 4) | multi-label bool over {UP, RIGHT, DOWN, LEFT}; True |
|                           |        | for every Bellman-optimal first physical step.      |

**Primary auxiliary output — optimal observation mask:**

| Output                           | Shape      | Description                                        |
| -------------------------------- | ---------- | -------------------------------------------------- |
| target_optimal_next_observations | (B, N_obs) | multi-label bool over observation vocabulary; True |
|                                  |            | for every observation that can be the first        |
|                                  |            | post-start acceptance in an optimal solution.      |

When multiple optimal first actions exist, every optimal action is marked
in the mask. Models should use multi-label binary cross-entropy for
training and multi-label precision/recall/F1 for evaluation — a model is
not penalized for predicting an optimal direction or observation that
differs from an arbitrary tie-break selection.

### Target encoding

| Target                           | Source                                                                      |
| -------------------------------- | --------------------------------------------------------------------------- |
| target_trajectory                | support × γ_space^{forward_depth} over optimal subgraph                     |
| target_waypoint                  | support × γ_semantic^{semantic_depth} over optimal subgraph                 |
| trajectory_support               | bool mask — positions on any optimal route                                  |
| trajectory_forward_depth         | int16 — earliest physical depth per supported position (-1 if unsupported)  |
| trajectory_remaining_cost        | int16 — minimum remaining physical cost over query-reachable optimal        |
|                                  | product states (-1 if unsupported), diagnostic only                         |
| waypoint_support                 | bool mask — waypoint event positions                                        |
| waypoint_semantic_depth          | int16 — earliest acceptance depth per waypoint position (-1 if unsupported) |
| target_optimal_directions        | bool[4] — multi-label optimal first directions                              |
| target_optimal_next_observations | bool[N_obs] — multi-label optimal first observations                        |

The decayed fields (`target_trajectory`, `target_waypoint`) are derived
from the support and depth arrays: `field[p] = support[p] × γ^{depth[p]}`
when `support[p]` is True, 0 otherwise. `trajectory_remaining_cost` is a
diagnostic channel (not used for loss or decoding).

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

**Model input channels:** cell_type, observation_id, start_flag, goal_flag,
spatial_mask.

**Metadata channels:** natural_height, natural_width, row_offset,
col_offset.

**Target channels:** target_trajectory (float32[S]),
target_waypoint (float32[S]), trajectory_support (bool[S]),
trajectory_forward_depth (int16[S], sentinel -1),
trajectory_remaining_cost (int16[S], sentinel -1, diagnostic),
waypoint_support (bool[S]),
waypoint_support (bool[S]),
waypoint_semantic_depth (int16[S], sentinel -1), target_optimal_directions (bool[4]),
target_optimal_next_observations (bool[N_obs]), total_physical_cost (int16 scalar).

## Invariants

- One fixed DAG per corpus; all samples share (O, E_obs).
- O_DAG = O_topology (declared vocabularies match).
- The start cell is traversable and contains a real observation.
- The goal differs from the start observation and is semantically reachable.
- At least one physical occurrence of the goal is present.
- The reverse distance table $d^*(p, o)$ is well-defined for every
  traversable product state, regardless of how many optimal paths exist.
- The optimal-transition relation is derived from Bellman equality:
  $x \rightarrow y$ is optimal iff $d^*(x) = c(x,y) + d^*(y)$.
- Trajectory support includes every position belonging to at least one
  optimal route; trajectory forward depth is the minimum physical depth
  among all optimal routes reaching that position.
- Waypoint support includes every position where an optimal semantic
  acceptance occurs, plus the start; waypoint semantic depth is the
  minimum number of acceptances from start.
- Optimal direction/observation masks include every Bellman-optimal first
  action — no arbitrary tie-breaking among equal-cost alternatives.
- Depth channels use sentinel -1 for unsupported positions:
  `trajectory_forward_depth[p] == -1` iff `trajectory_support[p] == False`;
  `trajectory_remaining_cost[p] == -1` iff `trajectory_support[p] == False`;
  `waypoint_semantic_depth[p] == -1` iff `waypoint_support[p] == False`.
  Supported positions have `depth >= 0`.
- Crossed but unaccepted observation cells are excluded from the waypoint field.
- The trajectory and waypoint fields follow their respective decay rules
  derived from support × γ^{depth}.
- Padding positions (spatial_mask == False) have zero-valued target fields.
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

Each preset is a `RoutebindPreset` composed of three layered contracts:

- **`QuerySelectionProfile`** — pre-oracle, normative. Controls the
  optimal physical route-position distribution by sampling from a
  precomputed product-state oracle query table. The query table records
  the exact minimum physical move count for every possible
  `(start_position, goal_observation)` pair, computed by one reverse
  0-1 BFS per goal observation. Physical route positions =
  `move_count + 1`.
- **`RealizedAdmissionPolicy`** — post-oracle, conditional. Applies
  optional waypoint-count admission quotas. When absent, the builder
  accepts all eligible samples regardless of waypoint count.
  Ambiguity (multiple optimal solutions) is not a rejection criterion —
  the optimal-subgraph targets represent all optimal continuations.
- **`CompletionPolicy`** — artifact validity. `"strict"` requires all
  quotas to be met; `"allow_degraded"` permits corpus emission with
  deficits reported in `capability_report.json`.

The preset table below lists the selection-profile physical-position
bins (pre-oracle, enforced by direct pool sampling). The waypoint-count
admission bins are defined in the preset source
(`builder.py` `ROUTEBIND_PRESETS`).

Conventions:

- `physical_move_count = len(physical\_route) − 1` (number of physical steps).
- `waypoint_count = len(waypoints)` (number of accepted observation events,
  including start and goal).
- Bins below use physical **route positions** (= `move_count + 1`) to match
  the profile API. To convert: subtract 1 to get move-count ranges.

| Preset          | Physical route-position bins (pre-oracle selection)          | Waypoint-count bins (post-oracle admission)      | Hard limits | Purpose                                      |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------ | ----------- | -------------------------------------------- |
| `smoke`         | 2–150 (100%)                                                 | (none — accept all)                              | max 150     | Tests and calibration runs.                  |
| `balanced`      | 2–7 (10%), 8–15 (20%), 16–30 (35%), 31–50 (25%), 51–80 (10%) | 2 (15%), 3 (25%), 4 (25%), 5–6 (25%), 7–10 (10%) | max 80      | Canonical training distribution.             |
| `long-spatial`  | 2–25 (15%), 26–50 (35%), 51–80 (35%), 81–120 (15%)           | 2–3 (45%), 4–5 (40%), 6–10 (15%)                 | max 120     | Emphasize physical planning.                 |
| `long-semantic` | 2–20 (20%), 21–50 (45%), 51–80 (25%), 81–120 (10%)           | 4–5 (20%), 6–8 (50%), 9–12 (25%), 13–15 (5%)     | max 120     | Emphasize DAG composition.                   |
| `joint-hard`    | 20–40 (20%), 41–70 (40%), 71–100 (30%), 101–140 (10%)        | 4–5 (15%), 6–8 (45%), 9–12 (30%), 13–15 (10%)    | max 140     | Jointly long spatial and semantic solutions. |

The `joint-hard-only` variant uses the same bins with `CompletionPolicy(mode="strict")`.

Physical route-position bins are guaranteed by direct pool sampling from the
precomputed query table. Waypoint-count bins are applied as a post-reconstruction
admission filter and may be underfilled when a topology's available queries
produce waypoint counts outside the target range.

**Pre-build capability gate.** When `CompletionPolicy.mode == "strict"`, the
builder raises `ValueError` before writing any corpus sample if any mandatory
joint bin has fewer than `QuerySelectionProfile.minimum_bin_support` (default: 1)
eligible candidates across all processed layouts. The error message lists
per-bin candidate counts. To obtain sufficient support, provide more topology
layouts or choose a different preset.

#### Parent-substrate recommendations

No substrate pairing is canonical for any preset until a capability
calibration artifact exists for the exact tuple:

```text
topology family/version
DAG artifact ID
observation placement policy
query selection profile
oracle version
```

Calibration data from early runs is available under
`artifacts/calibration/` — see the **Capability Calibration**
subsection below. Note: early calibration runs used the legacy
unique-path contract; ambiguity rates there represent the fraction
of candidates that would have been rejected, not task failure.

### Capability Calibration

The following five runs use the `smoke` preset (accept-all,
no admission filtering) to measure substrate capability without
sample-count pressure. All runs use `--seed 42` and
`--target-samples 0`. These runs used the legacy unique-path contract;
under the current optimal-subgraph contract, all eligible candidates
are accepted and ambiguity is measured as an informational statistic
rather than a rejection reason.

| Run | Policy        | Topology  | DAG       | Examined | Eligible | Accepted | Ambig% | Med Route |
| --- | ------------- | --------- | --------- | -------- | -------- | -------- | ------ | --------- |
| A   | dense_uniform | dungeon   | sparse    | 6,291    | 500      | 408      | 92.1%  | 4.0       |
| B   | bounded k=2   | dungeon   | sparse    | 16,011   | 502      | 390      | 96.9%  | 6.0       |
| C   | exactly_one   | openfield | sparse    | 65,390   | 263      | 259      | 99.6%  | 11.0      |
| D   | dense_uniform | dungeon   | branching | 11,329   | 1,004    | 746      | 91.1%  | 5.0       |
| E   | dense_uniform | openfield | sparse    | 107,455  | 2,936    | 2,002    | 97.3%  | 4.0       |

**Key findings:**

1. **Topology structure is the primary ambiguity driver.** The dungeon
   topology (runs A–D) produces consistent ambiguity rates of 91–97%
   regardless of DAG and observation placement. The openfield grid
   (run E) produces 97% ambiguity even with dense observation placement,
   reflecting the large number of equal-cost physical paths in an open
   30×30 grid.

2. **Observation repetition is a secondary factor.** Reducing duplicate
   observations (run B: bounded k=2, median duplicate obs 23.5 vs run A:
   39.3) _increases_ ambiguity (96.9% vs 92.1%) because the sparser
   placement creates more symmetric cell-choice patterns that produce
   additional equal-cost alternatives. Unique landmarks (run C,
   duplicate obs = 0) eliminate occurrence-choice ambiguity, but the
   open grid's physical symmetry sustains 99.6% ambiguity.

3. **DAG shortcut density has a small effect.** The branching DAG
   (run D, 139 edges) has slightly lower ambiguity than the sparse DAG
   (run A, 60 edges): 91.1% vs 92.1%. More DAG alternatives create more
   semantically-distinct solutions, which _increases_ the pool of
   uniquely-optimal eligible starts (1,004 vs 500) by enabling semantic
   differentiation at equal physical cost.

4. **Physical route length increases with unique landmarks.** Run C
   (exactly_one) produces median route length 11.0 vs 4.0 for run E
   (dense openfield). Unique landmarks force the agent to visit
   specific cells rather than whichever occurrence is nearest.

The primary lever for reducing ambiguity is **asymmetric physical
topology** (corridors, bottlenecks) rather than observation placement
density. A future substrate designed for strict uniqueness should
prioritise near-tree physical structure with deterministically unique
shortest paths between semantic landmarks.

Each run's full capability report is stored at
`artifacts/calibration/{A,B,C,D,E}/capability_report.json`.

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
| validate | Full certification: structural + oracle on all samples. |
| inspect  | Examine corpus metadata, samples, and diagnostics.      |

Usage:

```bash
python build-routebind.py build \
  --topology-root data/interim/dungeongen/routebind-30/v1 \
  --dagflow-root data/interim/dagflow/sparse/v1 \
  --dagflow-graph-id dagflow-sparse-v1-train-000000

python build-routebind.py validate data/processed/routebind/default/v1

python build-routebind.py inspect data/processed/routebind/default/v1 --summary
```

## Manifest

Root file: manifest.json. Key fields:

| Field                        | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| target_schema_version        | Schema version: `1`.                                   |
| target_semantics             | `"optimal_subgraph_support"`.                          |
| depth_sentinel               | Sentinel value for unsupported depth positions (`-1`). |
| storage_extent               | Storage canvas [height, width].                        |
| num_spatial_slots            | Fixed tensor width S = H_store \* W_store.             |
| spatial_storage_policy       | `"pad_to_configured_max"`                              |
| placement_policy             | `"center"`                                             |
| natural_extent_homogeneous   | Whether all parent layouts share a single extent       |
| natural_height_range         | `[min_height, max_height]` across parent layouts       |
| natural_width_range          | `[min_width, max_width]` across parent layouts         |
| n_observations               | Observation vocabulary cardinality.                    |
| field_decay_spatial          | Spatial field decay factor.                            |
| field_decay_semantic         | Semantic field decay factor.                           |
| max_supported_physical_moves | Maximum route length for terminal activation check.    |
| parents.spatial_topology     | Topology parent (family, root, version).               |
| parents.semantic_graph       | Dagflow parent (artifact_id, content_digest).          |

The deprecated n_states key is a synonym for num_spatial_slots.
n_observations must equal topology_observation_vocabulary_size.

### Validation

The `validate` command runs two layers on every sample:

- **Structural validation** — bidirectional support/depth/field algebra,
  sentinel consistency (-1 for unsupported), traversability and
  observation-cell constraints, padding zeros, mask shape/dtype/non-empty,
  start depth-zero invariants.
- **Oracle recomputation validation** — recomputes the optimal product-state
  subgraph from the parent topology and DAG via
  `compute_goal_distance_table` + `derive_optimal_transition_masks` +
  `traverse_optimal_subgraph`, then compares every stored support/depth/mask
  channel exactly against the recomputed projection.

Corpora not declaring `target_semantics: "optimal_subgraph_support"` are
rejected with `target_semantics_manifest_mismatch`.

## Targets and metrics

| Aspect                          | Value                                          |
| ------------------------------- | ---------------------------------------------- |
| Benchmark track                 | Routebind-Field                                |
| Claim family                    | goal_conditioned_spatial_route_binding         |
| Execution mode                  | single-step field prediction                   |
| Primary metric                  | valid_semantic_spatial_route_rate (behavioral) |
| Primary optimality metric       | semantic_spatial_path_cost_ratio               |
| Primary representational metric | balanced_trajectory_field_error                |
| Calibration metric              | trajectory_field_mse                           |
| Structural metric               | balanced_waypoint_field_error                  |
| Auxiliary metrics               | next_direction_precision/recall/F1,            |
|                                 | next_observation_precision/recall/F1           |

valid_semantic_spatial_route_rate: fraction of samples whose extracted
route begins at p_start, uses traversable four-neighbor moves,
contains no repeated positions, terminates at a goal_flag position, and
has a valid semantic acceptance subsequence under the hidden DAG.

semantic_spatial_path_cost_ratio: for valid routes, the ratio of
predicted route cost to oracle optimal cost C(R_hat) / C(R_star).
C(R) = |R| - 1 (number of physical moves). A value of 1.0 indicates
exact optimality. When multiple optimal routes exist, any optimal
cost (cost = d\*(start)) satisfies the ratio test.

Multi-label auxiliary metrics evaluate the model's ability to
identify all equally-optimal first actions. A model should not be
penalized for predicting an optimal direction or observation that
differs from an arbitrary tie-break selection.

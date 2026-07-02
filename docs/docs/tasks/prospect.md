# `prospect` Benchmark Task

## Identity

| Property      | Value                                          |
| ------------- | ---------------------------------------------- |
| Task name     | `prospect`                                     |
| Benchmark     | memory-conditioned spatial route binding       |
| Package path  | `src/ehp_sn/tasks/prospect/`                   |
| CLI script    | `scripts/data-gen/build-prospect.py`           |
| Output path   | `data/processed/prospect/<corpus>/v<version>/` |
| Dataset class | `task_corpus`                                  |

## Purpose and ownership

Prospect is a memory-conditioned spatial prospective-field task. Each
sample supplies one start position, a spatial mask identifying all physical
occurrences of the selected semantic goal observation, a spatial slot mask,
and a reference to a separately acquired EC–HPC memory state. The scalar
semantic goal observation identity is not exposed in v1; goal localization
has been resolved by the dataset builder. Unlike `routebind`, the model
does **not** receive the spatial topology, wall positions, or
observation-to-position bindings in the current input.

The scientific question is:

> Can an EHP model use an acquired EC–HPC state to solve novel start–goal
> route queries without receiving the spatial topology or observation bindings
> in the current input?

The following conditions hold:

- The arena was experienced before the Prospect query.
- Memory acquisition is not performed inside a Prospect task step.
- The route query may be novel (unseen start–goal pair over the same topology).
- The complete route must be reasoned, not retrieved as a memorized start–goal
  answer.
- Prospect does not expose the spatial topology directly.
- Prospect does not train or generate the memory during corpus generation.
- The memory is a separately produced, topology-specific resource.

## Semantic model

### Symbol table

| Symbol  | Persisted | Resolved | Model | Description                                                  |
| ------- | --------- | -------- | ----- | ------------------------------------------------------------ |
| P       | yes       | yes      | no    | S = H x W spatial grid positions (row-major)                 |
| O       | ---       | ---      | ---   | stable observation identities, O = {0, ..., N-1}             |
| E_obs   | ---       | ---      | ---   | directed observation-transition edges (hidden)               |
| G_obs   | ---       | ---      | ---   | fixed hidden DAG (O, E_obs)                                  |
| M       | no        | yes      | no    | acquired EC–HPC memory state (resolved resource)             |
| p_start | yes       | yes      | no    | unique physical start position (via start_flag)              |
| o_goal  | ---       | ---      | ---   | semantic goal observation identity (not exposed in v1 input) |
| f_traj  | yes       | yes      | yes   | spatial trajectory field over S positions (output)           |
| f_wp    | yes       | yes      | yes   | semantic waypoint field over S positions (output)            |

- **Persisted**: stored in the corpus record (NPY arrays).
- **Resolved**: available at task-input construction after memory-provider resolution.
- **Model**: surfaced in the model-facing representation after adapter encoding.
- **(—)**: model-internal structure, not part of any data contract surface.

### Information regime

The defining epistemic boundary is:

| Provided to model                        | Withheld from model                        |
| ---------------------------------------- | ------------------------------------------ |
| `start_flag` — one unique start position | `cell_type` — wall/free/observation labels |
| `goal_flag` — all goal-observation cells | `observation_id` — identity per position   |
| `spatial_mask` — real vs. padding slots  | Spatial adjacency / connectivity           |
| resolved acquired-memory resource        | Precomputed route or waypoint sequence     |
|                                          | Oracle target fields                       |
|                                          | Observation-transition adjacency matrix    |
|                                          | `memory_id` string (storage metadata only) |

The `memory_id` string reference is persisted in the corpus record but resolved
to a typed `AcquiredMemoryV1` object before the adapter sees it. The model never
receives `memory_id` directly.

The model must determine traversability (which positions are walls vs. free)
solely from the acquired memory state. The `spatial_mask` identifies which
positions correspond to real cells in the natural domain, but it does not
encode traversability.

### Memory resource

Prospect does not own memory acquisition. The acquired EC–HPC memory state is
a separately produced, topology-specific resource:

```text
checkpoint + acquisition history
    → memory-bank entry (exported, qualified, versioned)
```

Key invariants:

- The memory is topology-compatible: `memory.spatial_layout_id == query.spatial_layout_id`
  and `memory.topology_digest == query.topology_digest`.
- The memory was acquired before the Prospect query; acquisition did not
  consume query-specific route targets or oracle solutions.
- One memory entry may serve many start–goal queries over the same topology.
- Memory acquisition, export, and qualification are separate pipeline stages
  (Arena training → memory export). They do not occur during Prospect corpus
  generation or training.

### Semantic graph

The hidden semantic DAG (G_obs = (O, E_obs)) is shared across the corpus but
never exposed in model inputs. The DAG must be learned parametrically by the
model from route supervision across the Prospect corpus, not from the acquired
memory. The acquired EC–HPC memory supplies the spatial topology and
observation-to-position bindings; the semantic DAG is a separate corpus-level
construct from dagflow.

If the design instead intended the semantic DAG to be stored in HPC memory,
the Arena acquisition task would need to expose semantic-acceptance transitions,
which it currently does not. That would be a different acquisition task and a
different scientific claim.

The semantic parent must provide a single fixed directed acyclic graph over
exactly the same declared observation vocabulary as the spatial parent:
O_DAG = O_topology. The graph is consumed as hidden
oracle structure; adjacency is never exposed in model inputs.

### Semantic-state initialization and transition rule

The planning state is (p, o). The start position must contain a real
observation: o_0 = phi(p_start). The start observation is already accepted;
the first post-start acceptance must satisfy (o_0, o_1) in E_obs.

Two transition types:

- **Physical movement** — move to an adjacent traversable position q:
  (p, o) → (q, o) with cost 1.
- **Semantic acceptance** — accept the observation at the current position:
  (p, o) → (p, o') with cost 0, where phi(p) = o' and (o, o') in E_obs.

Conventions:

- Semantic acceptance is optional at any observation cell.
- An observation cannot be accepted unless it is a valid DAG successor.
- After accepting the goal observation, the task terminates immediately.
- Physical presence and semantic acceptance are independent. Crossing an
  observation cell without explicit acceptance does not update the semantic
  state.

### Oracle

The oracle is a joint product-state shortest-path search over
X = P_free × O. Goal states are any (p, o_goal) where phi(p) = o_goal.
The optimization minimizes total physical path length:

π\* = argmin_π sum_over_physical_transitions 1

Semantic acceptance transitions constrain validity but do not add cost.
The oracle jointly selects: the observation sequence, concrete occurrences,
physical route between occurrences, and final goal occurrence.

**Three canonical route objects:**

- Product-state path Pi\* = ((p_0, o_0), ..., (p_T, o_T))
- Projected physical route R\* = (p_0, p_1, ..., p_L)
- Accepted waypoint sequence W\* = ((p_i0, o_0), ..., (p_iM, o_M))

**Uniqueness**: Exactly one optimal task-equivalence class exists. The
projected physical route is simple (no repeated positions). Samples violating
either constraint are rejected with recorded reasons.

**Targets are computed from the source topology and semantic graph, never from
the acquired memory state.** A degraded memory must not alter ground truth.

### Outputs

**Primary output — spatial trajectory field (f_traj):**

For the selected physical route pi\*\_space = (p_0, ..., p_L):

f_traj*(p_k) = gamma_space^k
f_traj*(p) = 0 for p not in pi\*\_space

where gamma_space in (0, 1) is the spatial field-decay factor.

**Structural supervision — semantic waypoint field (f_wp):**

For accepted observations (p_i0, o_0), ..., (p_iM, o_M) where
o_M = o_goal and m = 0 receives 1.0:

f_wp*(p_im) = gamma_semantic^m
f_wp*(p) = 0 otherwise

Spatial and semantic decay are independent. Spatial decay counts physical
moves; semantic decay counts acceptance events.

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

### Semantic-length terminology

| Term                        | Definition                        | Value |
| --------------------------- | --------------------------------- | ----- |
| waypoint_count              | total accepted observations       | M+1   |
| semantic_transition_count   | edges traversed                   | M     |
| intermediate_waypoint_count | accepted excluding start and goal | M-1   |

## Parent requirements

### Spatial topology

The oracle topology parent must provide:

- extent — declared canvas dimensions (H, W).
- state_to_row_col — compact state coordinates (N, 2).
- observation_id — observation identity per traversable state (N,).
- next_state (N, A) with action_valid (N, A) — four-neighbor physical
  connectivity, validated against `movement_kind == "grid4"`.
- observation_vocabulary_size — declared observation domain size.
- topology_kind == grid2d and topology_type in {square, rectangle}.
- action_space with `movement_kind == "grid4"` (no hex).

### Dense canonicalization

Prospect converts each compact graph-indexed SpatialLayout into a dense
fixed-size storage canvas of S = H_store × W_store row-major positions.
The parent layout's natural extent H_i × W_i may be smaller than the
storage canvas. Positions outside the embedded natural extent become
CELL_PAD (storage padding, neither a real wall nor free space).

Within the embedded natural extent:

- Positions with a compact graph state become CELL_OBSERVATION.
- Positions without a compact graph state become CELL_WALL.
- Physical-neighbor adjacency is consumed from the parent layout's
  `next_state` + `action_valid` and validated against canonical grid4
  expectations.
- The required action space is four-neighbor undirected movement, no
  one-way passages.

Each stored sample carries a `spatial_mask` — True where the slot
corresponds to a real position in the sample's natural domain, False
for storage padding. Losses, metrics, route extraction, and figure
rendering must exclude padding positions.

Four cell classes are distinguished:

| Cell type       | Belongs to layout | Traversable | Observation |
| --------------- | ----------------- | ----------- | ----------- |
| `CELL_PAD (3)`  | No                | No          | No          |
| `CELL_WALL (0)` | Yes               | No          | No          |
| `CELL_FREE (1)` | Yes               | Yes         | No          |
| `CELL_OBS (2)`  | Yes               | Yes         | Yes         |

CELL_PAD positions (spatial_mask == False) have zero-valued target fields
and do not contribute to losses, metrics, route extraction, or figure
rendering.

### Memory compatibility

Every Prospect sample resolves to exactly one compatible memory entry. The
compatibility relation is memory ↔ spatial topology only:

- `query.spatial_layout_id == memory.spatial_layout_id`
- `query.topology_digest == memory.topology_digest`
- Spatial extent and slot schema match.
- Observation vocabulary cardinality matches.
- Coordinate convention matches.

The memory-bank artifact must not depend on the Prospect semantic DAG or
on query-specific route targets. The query ↔ semantic-graph relation is
separate from memory compatibility.

## Output artifact

### Input channels

| Field        | Shape  | Type | Description                                                       |
| ------------ | ------ | ---- | ----------------------------------------------------------------- |
| start_flag   | (B, S) | bool | True for exactly one traversable position                         |
| goal_flag    | (B, S) | bool | True for all positions containing the goal observation            |
| spatial_mask | (B, S) | bool | True inside the natural spatial domain; False for storage padding |
| memory_id    | scalar | str  | Reference to a compatible acquired memory entry                   |

The task input does not contain `cell_type`, `observation_id`, spatial
adjacency, wall mask, precomputed route, oracle waypoint sequence, or
target fields.

### Metadata channels (per sample)

| Field          | Shape  | Type  | Description                                                         |
| -------------- | ------ | ----- | ------------------------------------------------------------------- |
| natural_height | scalar | int32 | Natural layout height in cells                                      |
| natural_width  | scalar | int32 | Natural layout width in cells                                       |
| row_offset     | scalar | int32 | Row offset for embedding the natural extent into the storage canvas |
| col_offset     | scalar | int32 | Col offset for embedding the natural extent into the storage canvas |

### Corpus channels

Per-sample channels stored in the task corpus (one NPY array per channel
per split):

start_flag, goal_flag, spatial_mask, memory_id,
natural_height, natural_width, row_offset, col_offset,
target_trajectory, target_waypoint, target_next_dir, target_next_obs.

### Resolved runtime input

The persisted corpus record differs from the runtime model input. At load time,
`memory_id` is resolved through a memory provider to a typed memory object:

```text
Persisted record:
    memory_id (string reference)

Resolved task input:
    memory (AcquiredMemoryV1 — typed resolved object)
```

The resolution is performed by the data-loading layer, not by the adapter.
The adapter receives resolved memory and constructs model representations.

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
- The trajectory and waypoint fields follow their respective decay rules.
- The next-direction target matches the first physical step.
- The next-observation target matches the first post-start acceptance.
- Padding positions (spatial_mask == False) have zero-valued target fields.
- Padding positions do not contribute to losses, metrics, route extraction,
  or figure rendering.
- Regeneration with the same route-query artifact and build configuration
  produces identical query and target tensors. Changing the compatible
  memory-bank parent may change only memory references and lineage, not
  oracle targets.
- Targets are computed from the source topology and semantic graph, not from
  the acquired memory. A degraded or substituted memory must not alter
  ground truth.
- One memory entry may serve many start–goal queries.
- Memory acquisition did not consume query-specific oracle targets.
- Prospect test start–goal pairs must be held out from acquisition
  trajectories.
- Every sample carries `acquisition_exact_route_seen` (bool) and
  `acquisition_fragment_coverage` (float) diagnostic fields.
- The evaluator may access source topology and the hidden semantic graph
  for route-validity computation. These are privileged evaluation resources
  and are never exposed to the model.
- The observation vocabulary cardinality and index domain are corpus-level
  schema parameters available at model construction, although per-position
  observation identities are withheld from the query input.

## Build configuration

### Storage policy

Prospect accepts heterogeneous parent natural extents. Every corpus declares
one configured storage extent `[storage_height, storage_width]`. Each
selected parent layout must fit within that storage extent:

    0 < H_i <= H_store    and    0 < W_i <= W_store

Layouts are embedded into the storage canvas via deterministic centering.

| Policy                 | Value                   |
| ---------------------- | ----------------------- |
| spatial_storage_policy | `pad_to_configured_max` |
| storage_extent         | `[H_store, W_store]`    |
| placement_policy       | `center`                |

### Memory bank requirement

Prospect corpora require a separately produced, validated memory-bank artifact.
The memory-bank entry must satisfy the qualification thresholds declared by the
experiment configuration.

### Routebind preset compatibility

Prospect may reuse the same route-query sampling configuration as `routebind`
for target generation. The route targets are computed identically; the input
contract is what differs.

## CLI

| Command  | Description                                                |
| -------- | ---------------------------------------------------------- |
| build    | Materialize a Prospect corpus from route queries + memory. |
| validate | Verify a corpus against structural and semantic checks.    |
| inspect  | Examine corpus metadata, samples, and diagnostics.         |

Usage:

```bash
python scripts/data-gen/build-prospect.py build \
    --corpus default --version 1 \
    --topology-root data/interim/openfield/big-square/v1 \
    --dagflow-root data/interim/dagflow/routing/v1 \
    --memory-bank data/interim/memorybank/tem-v1/v1 \
    --field-decay-spatial 0.9848 \
    --field-decay-semantic 0.8 \
    --n-train 4000 --n-val 500 --n-test 500 \
    --seed 42
python scripts/data-gen/build-prospect.py validate data/processed/prospect/default/v1
python scripts/data-gen/build-prospect.py inspect data/processed/prospect/default/v1 --summary
```

## Manifest

Root file: manifest.json. Key fields:

| Field                        | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| storage_extent               | Storage canvas [height, width].                     |
| num_spatial_slots            | Fixed tensor width S = H_store \* W_store.          |
| spatial_storage_policy       | `"pad_to_configured_max"`                           |
| placement_policy             | `"center"`                                          |
| n_observations               | Observation vocabulary cardinality.                 |
| field_decay_spatial          | Spatial field decay factor.                         |
| field_decay_semantic         | Semantic field decay factor.                        |
| max_supported_physical_moves | Maximum route length for terminal activation check. |
| parents.spatial_topology     | Topology parent (family, root, version).            |
| parents.semantic_graph       | Dagflow parent.                                     |
| parents.memory_bank          | Memory-bank parent (root, version, producer).       |

## Targets and metrics

| Aspect                          | Value                                                                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Benchmark track                 | Prospect-Field                                                                                                          |
| Claim family                    | memory_conditioned_spatial_route_binding                                                                                |
| Execution mode                  | field prediction from a static query (model may perform internal recurrent deliberation and multiple memory retrievals) |
| Primary metric                  | valid_semantic_spatial_route_rate (behavioral)                                                                          |
| Primary optimality metric       | semantic_spatial_path_cost_ratio                                                                                        |
| Primary representational metric | balanced_trajectory_field_error                                                                                         |
| Calibration metric              | trajectory_field_mse                                                                                                    |
| Structural metric               | balanced_waypoint_field_error                                                                                           |
| Auxiliary metrics               | next_direction_accuracy, next_observation_accuracy                                                                      |
| Diagnostic metrics              | wrong_memory_validity_delta, memory_zero_ablation_field_error_delta                                                     |

- `valid_semantic_spatial_route_rate`: fraction of samples whose extracted
  route begins at p_start, uses traversable four-neighbor moves, contains no
  repeated positions, terminates at a goal_flag position, and has a valid
  semantic acceptance subsequence under the hidden DAG.

- `semantic_spatial_path_cost_ratio`: for valid routes, the ratio of predicted
  route cost to oracle optimal cost C(R_hat) / C(R_star). A value of 1.0
  indicates exact optimality.

- `wrong_memory_validity_delta`: change in `valid_semantic_spatial_route_rate`
  when the acquired memory is substituted with a memory from a different
  topology. A large negative delta indicates causal reliance on the correct
  memory. This is the primary diagnostic for memory-usage attribution.

- `memory_zero_ablation_field_error_delta`: increase in
  `trajectory_field_mse` when memory tensors are zeroed. Separates
  reliance on memory content from reliance on memory presence.

## Deferred features

The following capabilities are deliberately deferred to later Prospect versions
and are not part of the initial v1 contract:

- **Semantic goal-to-location retrieval**: v1 supplies `goal_flag` directly.
  A future version will require the model to retrieve goal-position bindings
  from memory given only a goal observation identity.
- **Online end-to-end Arena→Prospect training**: v1 uses a frozen, pre-exported
  memory bank. Joint acquisition-and-reasoning is deferred.
- **Semantic goal cue encoding via LEC**: v1 supplies `goal_flag` as a
  spatial-position cue. Encoding a raw observation ID into a sensory
  representation is deferred.
- **Multiple retrievals during deliberation**: v1 does not prescribe the
  number, timing, or form of memory retrieval operations. Retrieval control
  belongs to the EHP model and experiment. A single-retrieval baseline may
  be defined in the experiment specification, not in the task contract.

## Pipeline overview

Prospect does not own the full EHP pipeline. The canonical stages are:

```text
Stage A: Build task-neutral environment artifacts
    topology generator → spatial layouts
    dagflow → semantic graph
    spatial layouts + semantic graph → canonical route-query artifact

Stage B: Build Arena acquisition corpus
    spatial layouts + acquisition policy → Arena trajectories

Stage C: Train TEM / EC–HPC
    Arena corpus + TEM config → Arena checkpoint

Stage D: Export memory states
    Arena checkpoint + acquisition trajectories → memory-bank artifact

Stage E: Materialize Prospect corpus
    canonical route queries + memory-bank manifest → Prospect corpus

Stage F: Train EHP
    Prospect corpus + memory bank + EHP model → EHP checkpoint
```

Prospect owns Stage E. Stage F (training the EHP model) is owned by the
Prospect/EHP experiment layer — task packages must not own model training
per the repository architecture. Stages A–D are owned by upstream data
pipelines, the Arena task, and the TEM training pipeline, respectively.

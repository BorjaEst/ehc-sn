# `routebind` Benchmark Task

## Task identity and overview

Task name: `routebind`

Benchmark family: goal-conditioned spatial route binding

| Symbol                     | Surface | Description                                                                        |
| -------------------------- | ------- | ---------------------------------------------------------------------------------- |
| $P$                        | yes     | 900 spatial grid positions (30×30, row-major)                                      |
| $O$                        | yes     | stable observation identities ($N_{\text{obs}}$ in v1)                             |
| $E_{\text{obs}}$           | —       | directed observation-transition edges (hidden; never supplied to the model)        |
| $G_{\text{obs}}$           | —       | fixed observation-transition graph $(O, E_{\text{obs}})$ (hidden corpus structure) |
| $p_{\text{start}}$         | yes     | unique physical start position                                                     |
| $o_{\text{goal}}$          | —       | semantic goal observation (not a direct scalar input; exposed via `goal_flag`)     |
| $\mathbf{f}^{\text{traj}}$ | yes     | spatial prospective trajectory field over 900 positions (model output)             |
| $\mathbf{f}^{\text{wp}}$   | yes     | semantic waypoint field over 900 positions (model output)                          |
| $z_H, z_L$                 | —       | HRM high-level and low-level recurrent states (model-internal)                     |

_Surface_ symbols appear in the task input/output contract.
_Model-internal_ symbols (—) are emergent representations the model learns
but are not part of the task-level data contract.

Canonical package path:

```text
src/ehc_sn/tasks/routebind/
```

`routebind` is a goal-conditioned spatial prospective-field task. Each sample
supplies a complete two-dimensional layout containing walls, free cells,
stable observation identities, one physical start position, and one semantic
goal observation that may occur at several positions. A fixed directed graph
over observation identities is shared across the corpus but hidden from the
model. HRM must learn this graph parametrically, combine it with visible
spatial reachability, select a valid sequence of observation occurrences, and
predict a discounted field over the complete physical route from the start to
the selected goal occurrence.

The core computation is:

```text
spatial layout + start position + goal observation
    + learned observation-transition structure
    → goal-directed spatial trajectory field
```

The task combines two forms of structure — visible spatial and hidden
semantic — that must be jointly reasoned over. The observation identities
($O$) are visible in the input; the transition relation ($E_{\text{obs}}$)
is hidden and must be learned. The model does not receive the
observation-transition adjacency matrix, a precomputed observation sequence,
or waypoint labels as input. The selected waypoint field is available only
as a training target.

---

## Relationship to other tasks

`routebind` is intentionally designed at the intersection of two existing
task families.

### Goaltrace

[Goaltrace](goaltrace.md) tests whether HRM can learn a fixed DAG over
observation identities and produce a prospective field over those nodes.
The model receives oracle-quality current-location and relational-weight
signals. No spatial structure is present.

```text
goaltrace:
  current + goal + relational weights → prospective field over N nodes
```

`routebind` inherits the hidden DAG learning and field-prediction
supervision from Goaltrace, but replaces the oracle weight signals with
a visible spatial layout. The model must derive reachability from the
layout and bind observations to concrete positions.

### MazeHard

MazeHard uses the same 30×30 grid, row-major slot ordering, and
spatial-position-indexed tokens. The model navigates a fixed grid with
visible walls, start, and goal — reasoning purely over spatial
traversability.

```text
mazehard:
  start + goal + walls → spatial route
```

`routebind` adds a second, hidden structure over observation identities.
A physically nearby observation may be invalid because it does not follow
the learned observation-transition graph. Conversely, a semantically valid
observation may be unreachable because it is separated by walls.

### Design summary

| Aspect           | Goaltrace              | MazeHard                                                           | routebind                     |
| ---------------- | ---------------------- | ------------------------------------------------------------------ | ----------------------------- |
| Schema slots     | N observation nodes    | 900 spatial positions                                              | 900 spatial positions         |
| Slot semantics   | Observation identity   | Grid-cell position                                                 | Spatial grid position         |
| Hidden structure | Fixed DAG over obs IDs | Fixed implicit: four-neighbor movement rule; sample-specific: none | Fixed DAG over obs IDs        |
| Oracle           | Dijkstra over N nodes  | BFS over 900 cells                                                 | Product-state search          |
| Target           | Field over N nodes     | Per-cell categorical labels                                        | Two fields over 900 positions |

---

## Scientific purpose

`routebind` tests whether HRM can separate and recombine:

1. an abstract learned transition structure over observation identities;
2. a concrete spatial arrangement supplied in the current sample;
3. a physical start position;
4. a semantic goal cue;
5. a prospective route representation.

The central question is:

> Can HRM learn a reusable latent representation of the hidden
> observation-transition constraints sufficient to bind semantic routes
> to unseen spatial layouts?

Recovering the exact hidden DAG structure requires separate probing or
causal tests. Behavior alone establishes that the model learned a function
consistent with using that DAG.

More specifically, the task tests whether HRM produces behavior consistent
with having learned to:

- distinguish semantic identity from physical position;
- interpret a complete spatial layout;
- bind abstract observations to concrete locations;
- choose among multiple occurrences of the same observation;
- combine semantic validity with physical reachability;
- select a unique goal-reaching route;
- produce a distributed prospective field over the chosen route;
- use recurrent deliberation to refine that field.

`routebind` does **not** test:

- episodic retrieval or hippocampal binding;
- online navigation or action execution;
- environment interaction or state transitions;
- learning a new observation-transition graph per sample;
- dynamic hidden-graph changes;
- memory-derived observation-to-position binding.

The task provides all current spatial bindings directly. It tests whether
HRM can apply one learned abstract directed structure to many concrete
spatial arrangements.

---

## Input and output

### Execution mode

`routebind` uses **single-step field prediction with fixed-depth recurrent
deliberation**. The model does not navigate, execute actions, or interact
with a runtime loop. There is no physical movement, no episode horizon, and
no state transition during the task step. It may predict the oracle's first
movement direction as an auxiliary supervised output, but this does not
constitute action selection in an interactive environment.

1. The adapter encodes each of the 900 spatial positions into one HRM latent
   vector, composing cell type, observation identity, start/goal roles, and
   2D positional information.
2. HRM deliberates (recurrent $z_H$, $z_L$ cycles; fixed depth $K$ in v1,
   no ACT halting). The layout and start position remain constant throughout
   deliberation.
3. Per-slot heads decode the two fields (trajectory and waypoint) from all
   900 latent states. Start-conditioned heads decode the next direction and
   next observation from the start-slot latent state $h_{\\text{start}}$.
4. Fields are compared against oracle targets for supervision.

### Tensor shape contract

| Parameter        | Description               | Owner                |
| ---------------- | ------------------------- | -------------------- |
| $S = 900$        | number of spatial slots   | task schema          |
| $N_{\text{obs}}$ | number of observation IDs | corpus + task config |
| $D$              | model dimension           | model config         |

### Schema-token layout

```text
schema_tokens: (B, 900, D)

positions [0 : 900):
  spatial-position embeddings in row-major order

  slot(r, c) = 30·r + c
  r = ⌊i / 30⌋,   c = i mod 30
```

One slot per grid position. Row-major ordering is a hard data contract.
The adapter must preserve this ordering and the decoder must return one
output per slot in the same ordering.

### Input channels

| Field            | Shape      | Type  | Description                                                                                   |
| ---------------- | ---------- | ----- | --------------------------------------------------------------------------------------------- |
| `cell_type`      | `(B, 900)` | int32 | `{WALL, FREE, OBSERVATION}`                                                                   |
| `observation_id` | `(B, 900)` | int32 | stable observation identity; sentinel for non-observation cells                               |
| `start_flag`     | `(B, 900)` | bool  | `True` for exactly one traversable position                                                   |
| `goal_flag`      | `(B, 900)` | bool  | `True` for all positions containing the goal observation                                      |
| `cell_mask`      | `(B, 900)` | bool  | `True` for all 900 positions in fixed-size v1; walls are meaningful cells and remain unmasked |

Start and goal are independent roles, not exclusive cell types. A start
position may contain an observation while also being the start. A goal
observation may occur at multiple positions.

**Cell invariants**: A spatial cell contains at most one observation
identity. Observation-bearing cells are traversable unless independently
marked as walls; in v1, wall and observation cell types are mutually
exclusive. Non-observation cells use a dedicated sentinel observation ID.
The presence of FREE cells depends on the topology source: `openfield`
assigns observations to every traversable cell (no FREE cells), while
`dungeongen` may include traversable cells without observations.

**Observation presence per layout**: Observation identities are assigned
by the topology interim and consumed as-is by routebind. Not every DAG
observation identity must appear in every layout. The start observation
appears at the start cell; the goal observation appears at least once;
all observations needed by the selected oracle solution appear; other
DAG observations may be absent; distractor observations may appear zero,
one, or several times. The hidden DAG is corpus-wide; each sample supplies
a partial spatial instantiation of it. Routebind selects a query from the
supplied layout; it does not modify the layout to make a query valid.

The task input does **not** contain:

- the observation-transition adjacency matrix;
- a precomputed optimal observation sequence;
- a distance matrix;
- path-order labels;
- a sequence of required waypoints.

### Adapter equation

For cell $p = (r, c)$, the adapter constructs:

```text
e_p =
    E_cell(t_p)
  + E_obs(o_p)
  + s_p · r_start
  + g_p · r_goal
  + E_row(r)
  + E_col(c)
```

where:

- $t_p$ is the cell type;
- $o_p$ is the observation identity or sentinel;
- $s_p \in \{0, 1\}$ is the start role flag;
- $g_p \in \{0, 1\}$ is the goal role flag;
- $E_{\text{row}}, E_{\text{column}}$ provide explicit 2D spatial identity.

The adapter outputs:

```text
E ∈ ℝ^{B × 900 × D}
```

The adapter only performs representation conversion. It must not calculate
semantic reachability, spatial routes, waypoints, or graph transitions.
HRM's internal RoPE mechanism additionally supplies its standard sequence-
position encoding; the 2D row/column embeddings are the primary spatial
prior. This redundancy is deliberate for v1 — row/column embeddings encode
physical grid coordinates while RoPE preserves the standard HRM sequence-
processing contract. No claim is made that flattened index proximity alone
represents grid adjacency.

### Hidden semantic structure

One fixed directed acyclic graph per corpus, shared across all samples:

```text
G_obs = (O, E_obs)
|O| = N_obs
```

An edge $o_i \rightarrow o_j$ means that after semantically accepting
observation $o_i$, the route may next accept $o_j$.

Properties:

- directed and acyclic (v1);
- fixed across train, validation, and test for a given corpus;
- observation IDs are stable but assigned independently of topological order;
- every non-source observation has at least one incoming edge;
- every non-sink observation has at least one outgoing edge;
- the underlying undirected graph is connected;
- every observation participates in at least one reachable ordered pair
  (it can appear as start or goal of some valid query);
- selected queries use only reachable start–goal pairs;
- the graph is not supplied in the input — the DAG is never exposed as
  adjacency. It is learned indirectly from route-related task supervision,
  including spatial fields, semantic waypoints, and the next-observation
  auxiliary target.

v1 recommended profile:

```text
N_obs:              16
mean out-degree:    1.5–2.5
maximum out-degree: 4
semantic path length (accepted observations): 2–6
```

**Absent observations vs. invalid transitions**. The DAG defines semantic
validity corpus-wide. A transition $o_a \rightarrow o_b$ that exists in
$E_{\text{obs}}$ is semantically valid. However, if no occurrence of $o_b$
exists in the current layout, that transition is not sample-realizable. The
product-state oracle already distinguishes these cases. The task-level
conceptual distinction is:

```text
semantic validity:     edge exists in hidden DAG
sample realizability:  at least one appropriate spatial occurrence exists
                       and is reachable
```

### Semantic-state initialization

The start position must contain a real observation identity. Let:

```text
p_0 = p_start
o_0 = φ(p_0)
```

Then $o_0$ initializes the semantic state. The start observation is
considered already accepted. The first post-start semantic acceptance
must satisfy $(o_0, o_1) \in E_{\text{obs}}$.

The route succeeds when $o = o_{\text{goal}}$ at a position containing
that observation.

### Semantic transition rule

Traversing an observation-bearing cell does not automatically update the
semantic state. The semantic state only updates through an explicit
acceptance transition, which is valid only when the hidden DAG permits it.

The planning state is $(p, o)$ where:

- $p$ is the physical position;
- $o$ is the most recently accepted semantic observation.

Two transition types exist:

**Physical movement** — move to an adjacent traversable position $q$:

```text
(p, o) → (q, o)       cost: c_move = 1
```

**Semantic acceptance** — accept the observation at the current position:

```text
(p, o) → (p, o')      cost: c_accept = 0
```

when position $p$ contains observation $o'$ and $(o, o') \in E_{\text{obs}}$.

This contract ensures:

- irrelevant observation cells do not behave as walls;
- the route may physically pass an observation without selecting it;
- the route may cross the same observation identity multiple times
  without accepting it;
- the route may pass a future waypoint without accepting it and later
  accept a different occurrence of the same observation identity;
- repeated observations are manageable;
- semantic progression is explicit;
- the oracle can jointly optimize spatial movement and semantic choices.

**Acceptance conventions**:

- Semantic acceptance is optional at any observation cell. It occurs only
  when the oracle explicitly chooses it.
- An observation cannot be accepted unless it is a valid successor under
  the hidden DAG.
- After accepting the goal observation, the task terminates immediately.
  No further movement or semantic acceptance is part of the target.
- Physical presence and semantic acceptance are independent. Any
  traversable observation cell may be crossed without state change.
  Only explicit acceptance creates a semantic waypoint.

### Oracle

The oracle is a **joint product-state shortest-path search** over:

```text
S = P_free × O
```

where $P_{\text{free}}$ contains all non-wall positions. With 900 positions
and 16 observations the maximum state count is approximately 14,400.

Goal states are any $(p, o_{\text{goal}})$ where $\phi(p) = o_{\text{goal}}$.

The optimization minimizes total physical path length:

```math
π* = argmin_π  Σ_{physical transitions} 1
```

Semantic acceptance transitions constrain validity but do not independently
add cost. The oracle jointly selects: the semantic observation sequence,
the concrete occurrence of every selected observation, the physical route
between occurrences, and the final goal occurrence.

A two-stage decomposition (first select the observation sequence, then
realize it spatially) can produce incorrect targets. A semantically shorter
observation sequence may require longer physical segments. The product-state
search correctly resolves all decisions together.

**Three canonical route objects**. The oracle solution defines three
distinct objects:

- **Product-state path** $\Pi^* = ((p_0, o_0), \ldots, (p_T, o_T))$:
  contains physical moves and zero-cost semantic-acceptance transitions.
  Consecutive product states may share the same physical position when a
  semantic acceptance occurs without movement.

- **Projected physical route** $R^* = (p_0, p_1, \ldots, p_L)$:
  contains only physical movement positions, with zero-cost semantic
  transitions removed. This is what the trajectory field represents.

- **Accepted waypoint sequence**
  $W^* = ((p_{i_0}, o_0), \ldots, (p_{i_M}, o_M))$:
  the selected semantic acceptance events. This is what the waypoint
  field represents.

Two oracle solutions are **task-equivalent** when they produce the same
projected physical route and the same ordered accepted waypoint events.
Uniqueness is defined over these task-equivalence classes, not over raw
search predecessor chains.

For v1 the builder enforces **strict uniqueness**:

1. Exactly one optimal task-equivalence class exists, defined by the
   projected physical route and ordered semantic acceptance events. Two
   solutions are task-equivalent when they produce the same projected
   physical route and the same ordered accepted waypoint events. Internal
   queue or predecessor differences that yield the same projected route
   and accepted events are not distinct.
2. The projected physical route is **simple** — no position is revisited:

```math
p_i ≠ p_j  for all i ≠ j
```

Samples violating either constraint are rejected with recorded reasons
(`multiple_optimal_solutions`, `non_simple_spatial_projection`).

**Scope boundary**: Routebind v1 does not represent plans requiring
physical revisitation. Such samples are outside the v1 task domain.
A later version may support revisitation through richer outputs
(time-indexed path sequence, per-cell directional policy, or
multi-channel visit-order representation).

Deterministic queue ordering (by physical slot index, then semantic
observation ID) ensures reproducible search traversal but is not a semantic
tie-breaking rule. When the search discovers alternative equal-cost solutions
the sample is rejected, not lexicographically resolved.

### Outputs

**Primary output — spatial trajectory field** (`trajectory_field`):

For the selected physical route $\pi^*_{\text{space}} = (p_0, p_1, \ldots, p_L)$:

```math
\mathbf{f}^{\text{traj}*}(p_k) = \gamma_{\text{space}}^k
\mathbf{f}^{\text{traj}*}(p) = 0  \text{ for } p \notin \pi^*_{\text{space}}
```

where $\gamma_{\text{space}} \in (0, 1)$ is the spatial field-decay factor.
Because the route is simple and $0 < \gamma_{\text{space}} < 1$, each on-route
activation corresponds monotonically to distance from the start. The target
field therefore encodes: route membership, route ordering, and approximate
distance from the start. This field includes every traversed grid cell. It
answers: "Where should the route proceed physically?"

**Structural supervision — semantic waypoint field** (`waypoint_field`):

Let the selected semantic acceptance events be:

```text
(p_{i_0}, o_0), (p_{i_1}, o_1), ..., (p_{i_M}, o_M)
```

where $o_M = o_{\text{goal}}$. $m = 0$ corresponds to the already accepted
start observation and receives $1.0$. Then:

```math
\mathbf{f}^{\text{wp}*}(p_{i_m}) = \gamma_{\text{semantic}}^m
\mathbf{f}^{\text{wp}*}(p) = 0  \text{ otherwise}
```

This includes the start observation occurrence, every accepted intermediate
observation, and the selected goal occurrence. It excludes: free transit
cells, unselected observation occurrences, and observation cells physically
crossed but not semantically accepted.

**Scope of waypoint supervision**: The waypoint field directly supervises
the selected accepted sequence for each sample. It does not supervise
unused outgoing edges, unreachable DAG regions, edges never required by
sampled queries, or transitive versus direct relations outside selected
solutions. Complete edge recovery requires sufficient query coverage and
separate graph probing; the waypoint field alone does not guarantee that
the full hidden DAG has been internalized.

**Spatial and semantic decay are independent**: when two semantic
acceptances are spatially adjacent (no free transit cell between them),
the spatial trajectory field advances by one physical step while the
semantic waypoint field advances by one acceptance step. Spatial decay
counts physical moves; semantic decay ($\gamma_{\text{semantic}}$) counts
acceptance events.

**Auxiliary outputs:**

| Output                    | Shape        | Description                                 |
| ------------------------- | ------------ | ------------------------------------------- |
| `next_direction_logits`   | `(B, 4)`     | categorical over `{UP, DOWN, LEFT, RIGHT}`  |
| `next_observation_logits` | `(B, N_obs)` | categorical over the observation vocabulary |

Both auxiliary heads are derived from the latent state at the start
position $h_{\text{start}} = H[b, p_{\text{start}}]$. HRM self-attention
and recurrence allow the start-slot state to integrate the complete layout,
so no `[CLS]` token is required. The next-direction target is the first
physical step of the optimal route. The next-observation target is the
first semantically accepted observation after the start observation.

### Task output struct

```text
RoutebindTaskOutput:
    trajectory_field:        FloatTensor[B, 900]  ∈ [0, 1] via sigmoid
    waypoint_field:          FloatTensor[B, 900]  ∈ [0, 1] via sigmoid
    next_direction_logits:   FloatTensor[B, 4]
    next_observation_logits: FloatTensor[B, N_obs]
```

Fields are multi-label continuous representations — each component is an
independent activation. No softmax is applied across positions. The
auxiliary heads use standard softmax cross-entropy.

### Target encoding

Targets are derived from the oracle solution:

| Target              | Source                                               |
| ------------------- | ---------------------------------------------------- |
| `target_trajectory` | spatial decay over the projected physical route      |
| `target_waypoint`   | semantic decay over accepted observation occurrences |
| `target_next_dir`   | first physical step of the optimal route             |
| `target_next_obs`   | first post-start accepted observation                |

Path selection and field representation are kept separate — the oracle
selects the route, then the decay rule encodes it. Decay factors are
corpus constants (see [Corpus manifest](#corpus-manifest)).

---

## Corpus and data generation

### Data pipeline

`routebind` is a **dual-substrate derived task**. It consumes two
versioned, read-only parent artifacts:

```text
data/interim/<topology_family>/<preset>/v<N>/        (spatial topology + observation IDs)
                                                         │
                                                         ├── data/processed/routebind/<corpus>/v<N>/
                                                         │
data/interim/dagflow/<preset>/v<N>/                   (hidden semantic DAG)
```

The topology parent supplies the complete visible spatial world,
including traversability, spatial adjacency, coordinates, and the
observation identity assigned to each position. Routebind preserves
these channels exactly — it does not place, remove, duplicate, replace,
or remap observations.

The dagflow parent supplies one fixed directed graph over exactly the
same public observation vocabulary used by the topology substrate.
Routebind consumes this graph as hidden oracle structure; the adjacency
is never exposed in model inputs.

Routebind always consumes one versioned dagflow graph artifact. It does
not generate a DAG. The selected dagflow graph and the topology substrate
must use exactly the same declared public observation vocabulary:

$$
O_{\mathrm{DAG}} = O_{\mathrm{topology}}
= \{0, \ldots, N_{\mathrm{obs}}-1\}
$$

The topology substrate provides the observation-to-position mapping
$\phi: P \rightarrow O \cup \{\varnothing\}$. The dagflow substrate
provides the hidden transition relation $E_{\mathrm{obs}} \subseteq O \times O$.
Routebind computes over $(P, \phi, E_{\mathrm{obs}})$ without modifying
either $\phi$ or $E_{\mathrm{obs}}$.

### Spatial layout substrate

Routebind consumes existing spatial topology datasets (`openfield` or
`dungeongen`) that provide the `SpatialLayout` protocol
(see `spec/spec-openfield-layout.md`). The topology supplies the complete
visible world:

- `valid_state_mask`: traversable positions (`True` = traversable).
- `observation_id`: observation identity per traversable position,
  or a sentinel for positions without observations.
- `state_to_row_col`: row-major positional mapping.
- `adjacency`: four-neighbor connectivity.

Routebind derives `cell_type` from these fields — it does not place
observations, assign cell types, or invent cell semantics. For every
spatial slot $p$:

$$
\operatorname{cell_type}(p) =
\begin{cases}
\text{WALL}, & \text{inaccessible}\\
\text{OBSERVATION}, & \text{traversable and has an observation ID}\\
\text{FREE}, & \text{traversable and no observation ID}
\end{cases}
$$

The distribution of cell types is determined by the chosen topology source.
`openfield` assigns an observation to every traversable cell, producing
~0\% FREE cells. `dungeongen` may produce walls, FREE cells, and
observation-bearing cells.

For v1 the recommended profile is 30×30 square grids (900 positions,
row-major ordering). The topology source owns observation identities;
routebind must not replace them. The topology must not contain routebind
query semantics: the hidden DAG, start/goal selection, waypoints, or
routebind targets.

### Three-layer preset architecture

Routebind separates configuration into three independent preset families
that together determine the difficulty distribution of the generated
corpus:

```text
topology preset       dagflow preset       routebind preset
    │                      │                       │
    │  (physical world)    │  (semantic graph)     │  (retained distribution)
    │                      │                       │
    └──────────────────────┴───────────────────────┘
                                │
                     builder reconciles them
                     through deficit-driven
                     query selection
```

**Topology preset** controls the physical world: wall density, connected-
component structure, corridor widths, bottlenecks, and observation
placement. Examples: `openfield`, `sparse-dungeon`,
`balanced-dungeon`.

**Dagflow preset** controls semantic structure: number of observations,
branch and merge structure, shortcuts, semantic diameter. Examples:
`routing`, `chain16`, `branching16`.

**Routebind preset** controls the retained task-query
distribution: physical route length, semantic waypoint count, start/goal
observation coverage, and edge coverage. Selection is driven by
deficit-aware goal and start prioritisation rather than random sampling.
The preset declares target proportions over difficulty dimensions
and the builder fills underfilled buckets first.

Concrete presets are registered in
`ROUTEBIND_PRESETS` in the builder module.

| Preset     | Purpose                                                 |
| ---------- | ------------------------------------------------------- |
| `smoke`    | Accept-all; one bucket covering all lengths.            |
| `balanced` | Broad physical (2–80) and semantic (2–10) distribution. |

### Deficit-driven query generation

The builder does not randomly select goals and starts and accept whatever
distribution emerges. Instead:

1. It inspects current joint-bucket deficits for the split.
2. It selects candidate goals most likely to fill underfilled buckets.
3. For each goal, it computes the reverse distance table once and bins all
   eligible start positions by physical-distance bucket (`_bin_starts_by_distance`).
4. It iterates buckets in deficit-priority order.
5. It reconstructs only candidates from underfilled buckets.
6. It checks the semantic-length bucket post-reconstruction before accepting.
7. It places accepted samples into the joint `(physical_bucket, semantic_bucket)`
   bin and updates deficit scores.

Candidates that fail the uniqueness check (`opt_count >= 2`) are counted
in the **generation funnel** rather than silently excluded.

### Generation phases

**Phase A — Load the hidden semantic DAG.** One fixed DAG per corpus,
consumed from a versioned dagflow substrate. The dagflow artifact
expresses edges in the same public observation IDs used by the topology.
Validate vocabulary identity ($O_{\text{DAG}} = O_{\text{topology}}$),
in-degree/out-degree distributions, reachable-pair count, semantic
shortest-path distribution, and graph connectivity. Reject graphs with
insufficient route diversity. Routebind does not generate the DAG.

**Phase B — Load the topology interim.** Read one versioned spatial
layout from the selected topology family (e.g. `openfield` or
`dungeongen`). Each layout supplies traversability, per-position
observation identities, and spatial adjacency. Validate grid dimensions
and connectivity.

**Phase C — Derive routebind spatial channels.** From the topology record,
derive `cell_type`, `observation_id`, and `cell_mask`:

- inaccessible $\rightarrow$ `WALL`;
- traversable with observation $\rightarrow$ `OBSERVATION`, passthrough
  the topology's observation ID;
- traversable without observation $\rightarrow$ `FREE`, sentinel
  observation ID.

Build occurrence lists for every observation identity. Reject a layout
that lacks observations required by the DAG. No observations are placed,
moved, or invented here.

**Phase D — Select start and goal.** Choose a reachable semantic
start–goal pair from the precomputed reachable-pair relation of the hidden
DAG. The start cell must contain the start observation. The goal must
differ from the start observation. Mark all physical occurrences of the
goal via `goal_flag`. Reject the sample if no valid semantic query can be
formed on this layout.

**Phase E — Run the joint oracle.** Execute product-state shortest-path
search over the topology-supplied layout and selected query. Recover: the
full product-state path, projected spatial route, accepted semantic
observation sequence, selected occurrence of each waypoint, selected goal
occurrence, first physical direction, and next observation. Reject the
sample if no suitable route exists.

**Phase F — Enforce uniqueness.** Verify exactly one optimal
task-equivalence class exists and the projected physical route is simple. Reject samples that
fail either check. Monitor rejection rates by reason.

**Phase G — Encode targets.** Generate trajectory field, waypoint field,
next-direction target, and next-observation target. Apply the declared
spatial and semantic decay factors.

### Key invariants

- One fixed DAG per corpus; all samples share the same $(O, E_{\text{obs}})$.
- The DAG vocabulary equals the topology observation vocabulary
  ($O_{\text{DAG}} = O_{\text{topology}}$). Both use the same declared
  public observation IDs; a layout may contain a subset of those IDs,
  but the declared vocabularies must match exactly.
- Observation IDs are stable and do not encode graph order.
- Cell types and observation IDs are derived from the topology interim;
  routebind does not generate or replace them.
- The hidden graph is never exposed in model inputs.
- The start cell is traversable and contains a real observation.
- The goal differs from the start observation and is semantically reachable.
- At least one physical occurrence of the goal is present.
- Exactly one optimal task-equivalence class exists.
- The projected physical route contains no repeated positions.
- Every physical step is a valid four-neighbor non-wall move.
- Every semantic acceptance follows an edge in the hidden DAG.
- Crossed but unaccepted observation cells are excluded from the waypoint
  field.
- The trajectory and waypoint fields follow their respective decay rules.
- The next-direction target matches the first physical step.
- The next-observation target matches the first post-start acceptance.
- Regeneration with the same substrates, task seed, and query produces
  identical tensors.

### Field decay

Spatial routes are substantially longer than Goaltrace's observation-node
paths. The decay factor must be chosen relative to the expected physical
route length. For a minimum terminal activation $f_{\min}$ at maximum
route length $L_{\max}$:

```math
\gamma_{\text{space}} = f_{\min}^{1 / L_{\max}}
```

Example: $L_{\max} = 150$, $f_{\min} = 0.1$ gives $\gamma_{\text{space}} \approx 0.9848$.

The semantic decay $\gamma_{\text{semantic}}$ operates over accepted
observation steps and may use a smaller factor (e.g., $0.8$, as in
Goaltrace) because semantic sequences are short (2–6 steps).

### Corpus manifest

```json
{
  "task": "routebind",
  "corpus": "default",
  "version": 1,
  "manifest_schema_version": 2,
  "parents": {
    "spatial_topology": {
      "family": "openfield",
      "root": "data/interim/openfield/big-square/v1",
      "version": 1
    },
    "semantic_graph": {
      "family": "dagflow",
      "root": "data/interim/dagflow/balanced/v1",
      "version": 1,
      "artifact_id": "dagflow-balanced-v1-train-000000",
      "content_digest": "sha256:..."
    }
  },
  "n_observations": 45,
  "topology_observation_vocabulary_size": 45,
  "grid_shape": [30, 30],
  "field_decay_spatial": 0.9848,
  "field_decay_semantic": 0.8,
  "max_supported_route_length": 150,
  "minimum_terminal_activation": 0.1
}
```

`parents.spatial_topology` records the topology substrate that supplies
the complete visible spatial world, including per-position observation
identities. `parents.semantic_graph` records the dagflow graph artifact
that supplies the hidden semantic DAG. Both parents are immutable,
versioned artifacts; the routebind corpus is derived from their exact
content, not merely their generation parameters.

The dagflow parent includes an `artifact_id` identifying the specific
graph instance and a `content_digest` verifying its exact content. The
same graph may be shared by a Goaltrace corpus for cross-task transfer
experiments.

`n_observations` and `topology_observation_vocabulary_size` must be
equal ($O_{\text{DAG}} = O_{\text{topology}}$). `field_decay_spatial` is
the authoritative decay parameter. `minimum_terminal_activation` and
`max_supported_route_length` are derived consistency checks.

### Profile coupling

| Checkpoint         | Validation                                                              |
| ------------------ | ----------------------------------------------------------------------- |
| Training startup   | `model.num_schema_slots == 900`                                         |
| Training startup   | `adapter.num_observation_embeddings == n_observations + 1`              |
| Training startup   | `n_observations == topology_observation_vocabulary_size`                |
| Training startup   | dagflow parent `artifact_id` and `content_digest` match training corpus |
| Evaluation startup | checkpoint profile agrees with eval corpus on all coupled fields        |
| Evaluation startup | topology family and version match between training and eval corpora     |
| Evaluation startup | dagflow graph identity matches between training and eval corpora        |

### Data generation

```bash
# Build the dagflow graph first:
python scripts/data-gen/build-dagflow.py build --preset balanced --version 1

# Build the routebind corpus consuming both parents (default preset: balanced):
python scripts/data-gen/build-routebind.py materialize-task \
    --topology-root data/interim/openfield/big-square/v1 \
    --dagflow-root data/interim/dagflow/balanced/v1 \
    --dagflow-graph-id dagflow-balanced-v1-train-000000 \
    --corpus default --version 1 \
    --field-decay-spatial 0.9848 --field-decay-semantic 0.8 \
    --seed 42

# Build with an explicit preset:
python scripts/data-gen/build-routebind.py materialize-task \
    --topology-root data/interim/openfield/big-square/v1 \
    --dagflow-root data/interim/dagflow/routing/v1 \
    --dagflow-graph-id dagflow-routing-v1-train-000000 \
    --corpus default --version 1 \
    --preset balanced \
    --seed 42

# Build with a smoke preset (accept-all, for calibration runs):
python scripts/data-gen/build-routebind.py materialize-task \
    --topology-root data/interim/openfield/big-square/v1 \
    --dagflow-root data/interim/dagflow/routing/v1 \
    --dagflow-graph-id dagflow-routing-v1-train-000000 \
    --corpus calibration --version 1 \
    --preset smoke \
    --seed 42

# Override individual preset fields from the CLI:
python scripts/data-gen/build-routebind.py materialize-task \
    --topology-root data/interim/openfield/big-square/v1 \
    --dagflow-root data/interim/dagflow/routing/v1 \
    --dagflow-graph-id dagflow-routing-v1-train-000000 \
    --corpus long --version 1 \
    --preset balanced \
    --min-route-length 30 \
    --max-route-length 120 \
    --attempt-budget 20 \
    --seed 42
```

The dagflow graph artifact is built separately by `build-dagflow.py`.
Routebind reads the selected graph by stable artifact ID. Parameters
such as `--n-observations` and `--max-out-degree` are owned by dagflow;
routebind reads them from the dagflow manifest and validates
compatibility.

The `--preset` option selects a named `RoutebindSamplingProfile`
from `ROUTEBIND_PRESETS`. Individual fields may be overridden
with `--min-route-length`, `--max-route-length`, and
`--attempt-budget`. When omitted the default is `balanced`.

Output path: `data/processed/routebind/<corpus>/v<version>/`

### Dataset splits

The primary split unit is the **complete spatial layout**. All queries
derived from one layout belong to exactly one split.

| Split                   | Layouts | Start-goal pairs                         | Purpose                             |
| ----------------------- | ------- | ---------------------------------------- | ----------------------------------- |
| Standard generalization | unseen  | may overlap training                     | Application to new spatial bindings |
| Pair generalization     | unseen  | held-out from training                   | Composition within the learned DAG  |
| Binding challenge       | unseen  | familiar pairs, new duplicate placements | Identity-to-position binding        |
| Strongest               | unseen  | unseen pairs, longer sequences           | Challenge track                     |

The strongest generalization split should be a challenge track, not the
only test.

### Curriculum profiles

The task should be introduced progressively. Profiles are expressed as
topology-source presets or sample-selection constraints — routebind does
not control observation occurrence patterns:

| Profile | Name                              | Topology properties                                                                     |
| ------- | --------------------------------- | --------------------------------------------------------------------------------------- |
| 1       | Sparse observations, open layouts | each DAG observation appears few times; low wall density; short semantic sequences      |
| 2       | Semantic branching                | branching DAG; one valid route; moderate route lengths                                  |
| 3       | Repeated observations             | selected observations appear at multiple positions; binding choices                     |
| 4       | Obstacles and bottlenecks         | higher wall density; longer paths; spatially close but semantically invalid distractors |
| 5       | Full task                         | branching + repeated observations + complex layouts + unseen configurations             |

---

## Training supervision

The primary loss is over the trajectory field with region-balanced
normalization. A plain MSE over 900 positions is unsuitable because most
targets are zero.

### Trajectory loss

```text
L_trajectory =
    λ_route · L_route
  + λ_off   · L_off
  + λ_start · L_start
```

where:

```text
L_route = (1 / |π*|)  Σ_{p ∈ π*}  (\hat{\mathbf{f}}^{\text{traj}}(p) − \mathbf{f}^{\text{traj}*}(p))²
L_off   = (1 / |P \ π*|)  Σ_{p ∉ π*}  \hat{\mathbf{f}}^{\text{traj}}(p)²
```

$L_{\text{start}}$ supervises the start position toward $1.0$.

### Waypoint loss

```text
L_waypoint =
    λ_wp-on  · L_wp-on
  + λ_wp-off · L_wp-off
```

Normalizing selected waypoints and non-waypoint positions separately.

### Auxiliary losses

```text
L_next-dir = CrossEntropy(ℓ_dir, dir_target)
L_next-obs = CrossEntropy(ℓ_obs, obs_target)
```

### Total training objective

```text
L = L_trajectory + α · L_waypoint + β · L_next-dir + η · L_next-obs
```

Conceptual priority:

```text
trajectory field:      primary
waypoint field:        strong structural supervision
next direction:        modest auxiliary
next observation:      modest auxiliary
```

Exact coefficients are configuration fields, not embedded task semantics.
The first experiment should compare: (a) trajectory field only,
(b) trajectory plus waypoint field, (c) all four targets.

---

## Benchmark and evaluation

### Routebind-Field track

| Aspect                          | Value                                                       |
| ------------------------------- | ----------------------------------------------------------- |
| Benchmark track                 | Routebind-Field                                             |
| Claim family                    | `goal_conditioned_spatial_route_binding`                    |
| Execution mode                  | single-step field prediction after fixed-depth deliberation |
| Primary metric                  | `valid_semantic_spatial_route_rate` (behavioral)            |
| Primary optimality metric       | `semantic_spatial_path_cost_ratio`                          |
| Primary representational metric | `balanced_trajectory_field_error`                           |
| Calibration metric              | `trajectory_field_mse` (raw full-grid)                      |
| Structural metric               | `balanced_waypoint_field_error`                             |
| Auxiliary metrics               | `next_direction_accuracy`, `next_observation_accuracy`      |

### Primary behavioral metric: `valid_semantic_spatial_route_rate`

Fraction of samples for which the extracted route satisfies both checks
below. The checks are computed by the evaluator from the extracted spatial
route and the hidden DAG; they do not depend on the model's waypoint
activations crossing a threshold.

**Spatial validity** — the extracted route must:

- begin at the declared start position $p_{\text{start}}$;
- use only traversable four-neighbor moves (no wall crossings);
- contain no repeated positions (simple path);
- terminate at a position whose `goal_flag` is `True`.

**Semantic realizability** — given the sequence of observation-bearing
cells encountered along the extracted route and the hidden DAG, there must
exist an accepted observation subsequence:

```text
o_0, o_1, ..., o_M = o_goal
```

such that:

- $o_0$ is the observation at the start position;
- each $o_m$ ($m \ge 1$) occurs at the corresponding route position;
- every transition satisfies $(o_m, o_{m+1}) \in E_{\text{obs}}$.

This check is computed by the evaluator, not by the model. A route that
terminates at a goal-flagged cell without a valid semantic acceptance
history is not semantically realizable.

### Primary optimality metric: `semantic_spatial_path_cost_ratio`

For routes that pass both validity checks, the ratio of predicted route
cost to oracle optimal cost:

```math
\text{path-cost-ratio} = \frac{C(\hat{R})}{C(R^*)}
```

where $C(R) = |R| - 1$ (number of physical moves). A value of $1.0$
indicates exact optimality; larger values indicate longer-than-optimal
routes. Routes that fail validity are excluded. The companion metric
`semantic_spatial_path_regret` reports $C(\hat{R}) - C(R^*)$.

**Behavioral interpretation**:

```text
validity:    can the model produce a usable route?
optimality:  how close is that route to the task objective?
exactness:   did the model recover the unique oracle route?
```

### Primary representational metric: `balanced_trajectory_field_error`

Error over the trajectory field with separate normalization for on-route
and off-route cells:

```text
balanced_error = (L_route + L_off) / 2
```

where $L_{\text{route}}$ and $L_{\text{off}}$ are defined as in the
training loss. This prevents the trivial all-zero predictor from obtaining
a deceptively low score on sparse grids.

### Calibration metric: `trajectory_field_mse`

Raw full-grid MSE, retained for cross-task comparability:

```text
trajectory_field_mse = (1 / (B·900)) Σ_i Σ_p (\hat{\mathbf{f}}^{\text{traj}}(p) − \mathbf{f}^{\text{traj}*}(p))²
```

### Route extraction

The predicted trajectory field is converted to a discrete route by a
canonical greedy decoder. Behavioral route metrics evaluate the field
through this decoder; they measure the combination of field quality and
this decoding rule. The decoder proceeds as follows:

1. Begin at the known start position $p_0 = p_{\text{start}}$.
2. At each step, examine the four neighbor cells of the current position.
   Select the traversable neighbor with the highest predicted trajectory
   activation that has not been visited in the current extracted route.
3. When two or more unvisited traversable neighbors have equal predicted
   activation, break ties by a fixed deterministic order: UP, RIGHT,
   DOWN, LEFT.
4. Stop when the route length reaches a fixed maximum (default $L_{\max}$),
   when no valid unvisited neighbor exists (failed extraction), or when
   the extracted route reaches a goal-flagged cell AND a semantic
   realizability check confirms a valid acceptance history ending at
   that cell.
5. The extraction must not cross walls or revisit cells.

The primary extraction uses only the trajectory field. The
`next_direction_logits` auxiliary head is evaluated separately through
`next_direction_accuracy` and does not affect the primary behavioral
metric. An optionally reported `auxiliary_assisted_route_rate` may use
next-direction logits for the first step, but it must not replace the
primary metric.

Physical arrival at a goal occurrence does not automatically terminate
the route unless the semantic realizability check succeeds there. The
evaluator performs this check independently (see [Primary behavioral
metric](#primary-behavioral-metric-valid_semantic_spatial_route_rate)).

### Waypoint extraction and evaluation

Waypoint extraction serves two distinct purposes, which must not be
conflated:

**1. Route semantic realizability** (evaluator-side): Given the extracted
spatial route and the hidden DAG, the evaluator determines whether an
accepted observation subsequence exists satisfying the semantic constraints.
This check does NOT use the predicted waypoint field. It is part of the
primary behavioral metric.

**2. Waypoint prediction agreement** (model-side): How accurately the
predicted waypoint field identifies the oracle's accepted waypoint
occurrences. This is a representational quality metric, not a validity gate.

For waypoint prediction agreement, accepted waypoints are identified from
the waypoint field along the extracted physical route:

1. Force the start position to be the first waypoint candidate.
2. Force the route endpoint (terminal position) to be the terminal waypoint
   candidate.
3. Apply a fixed threshold (default $0.1$) to intermediate
   observation-bearing cells only.

The metric `balanced_waypoint_field_error` compares the predicted waypoint
field against the oracle waypoint field with separate normalization for
waypoint and non-waypoint cells (analogous to the balanced trajectory
error). Raw `waypoint_field_mse` is retained as calibration.

The threshold-based extraction above is used only for discrete waypoint
metrics (`waypoint_occurrence_precision`, `waypoint_occurrence_recall`,
`waypoint_sequence_exact`). Continuous structural metrics use the full
field without thresholding.

### Semantic sequence metrics

For behavioral evaluation, the evaluator derives a minimal accepted
subsequence $\hat{W}_{\text{valid}}$ from the extracted spatial route
and the hidden DAG by a deterministic algorithm (shortest valid acceptance
sequence). This evaluator-derived sequence is compared against the oracle
accepted waypoint sequence $W^*$ independently of the predicted waypoint
field:

| Metric                          | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- | ------- | --- | ---- | ----------------------------------- |
| `accepted_sequence_exact`       | evaluator-derived sequence matches oracle exactly             |
| `accepted_transition_precision` | fraction of evaluator-derived transitions in $E_{\text{obs}}$ |
| `accepted_transition_recall`    | fraction of oracle transitions recovered                      |
| `semantic_acceptance_regret`    | $                                                             | \hat{W} | -   | W^\* | $ (extra acceptances beyond oracle) |

### Failure taxonomy

The primary behavioral metric collapses different failures into one
invalid outcome. The evaluation should separately report:

| Failure category                    | Description                                                      |
| ----------------------------------- | ---------------------------------------------------------------- |
| `dead_end`                          | no valid unvisited neighbor exists before reaching any goal      |
| `maximum_length_reached`            | extraction reached $L_{\max}$ without reaching a goal            |
| `wall_or_invalid_move`              | extraction attempted a non-traversable move                      |
| `premature_goal_without_acceptance` | reached a goal cell but no valid acceptance history exists       |
| `semantic_unrealizability`          | route reached a goal cell but no valid subsequence exists        |
| `wrong_goal_occurrence`             | reached a goal cell that differs from the oracle goal occurrence |
| `valid_but_suboptimal`              | route is valid but longer than the oracle optimal route          |

### Secondary and diagnostic metrics

| Metric                          | Description                                                             |
| ------------------------------- | ----------------------------------------------------------------------- |
| `semantic_spatial_path_regret`  | $C(\hat{R}) - C(R^*)$ for valid routes                                  |
| `exact_route_agreement`         | extracted route matches oracle route exactly                            |
| `balanced_waypoint_field_error` | balanced MSE over waypoint field (waypoint vs non-waypoint)             |
| `waypoint_field_mse`            | raw MSE over waypoint field (calibration)                               |
| `waypoint_occurrence_precision` | fraction of thresholded waypoint predictions matching oracle waypoints  |
| `waypoint_occurrence_recall`    | fraction of oracle waypoints recovered by thresholded predictions       |
| `waypoint_sequence_exact`       | thresholded waypoint sequence matches oracle exactly                    |
| `accepted_sequence_exact`       | evaluator-derived acceptance sequence matches oracle                    |
| `accepted_transition_precision` | fraction of evaluator-derived transitions in $E_{\text{obs}}$           |
| `accepted_transition_recall`    | fraction of oracle transitions recovered                                |
| `route_cell_error`              | MSE restricted to cells on the oracle route                             |
| `off_route_suppression`         | mean predicted activation outside the selected route                    |
| `start_activation`              | predicted activation at the physical start position                     |
| `goal_activation`               | predicted activation at the selected goal position                      |
| `field_decay_correlation`       | Pearson $r$ between predicted and target activation along route         |
| `next_direction_accuracy`       | whether the predicted first movement matches oracle                     |
| `next_observation_accuracy`     | whether the predicted next observation matches oracle                   |
| `goal_occurrence_selection_acc` | whether the extracted route endpoint matches the oracle goal occurrence |

### Score accumulation

All metrics are mean-aggregated over samples. Subfield metrics (route-cell,
off-route) use masked aggregation over the relevant position subsets.

### Evidence contract

Strong benchmark scores alone do not prove the hidden DAG was learned.
The evidence contract distinguishes four levels of claim:

1. **Behavioral competence**: valid goal-reaching routes in unseen layouts.
2. **Representation of semantic transitions**: above-chance next-observation
   prediction and correct waypoint field.
3. **Exact reconstruction**: exact route agreement and graph probing tests
   confirming the model internalized specific transition edges.
4. **Causal use**: the model changes its predicted route when observation
   identities are swapped while layout geometry is preserved.

The benchmark must separately establish each level.

**Mandatory control conditions**. The benchmark must include at least:

**Spatial-only control**: Remove or randomize observation identities while
preserving walls, start position, and goal-position mask. This measures
how much performance can be achieved from geometry alone.

**Semantic-use control**: Use open layouts or matched spatial distances
while varying observation identities and hidden-DAG validity. This tests
whether the model learned semantic transitions rather than merely spatial
route geometry.

**Generalization scope**. Routebind evaluates generalization across spatial
layouts and queries within one learned corpus-specific DAG. The same DAG is
shared across training, validation, and test. Routebind does not evaluate:
learning a new DAG at inference time, transferring to unseen DAG structures,
graph induction from few samples, or universal graph-planning across graph
instances. The scientific claim is:

> HRM learns a corpus-specific hidden observation-transition structure and
> reuses it across unseen spatial instantiations of that same structure.

**Positive-query scope**. Routebind v1 is a positive-query benchmark. Every
accepted sample has a valid semantic-spatial solution. The task does not
require detecting that no solution exists. A future negative-query track
could include semantically unreachable goals or spatially blocked waypoints.

**Intervention validity criteria**. The identity-swap causal test applies
only to interventions satisfying:

```text
original sample solvable
intervened sample solvable
oracle target changes materially (R* or W* differs)
spatial geometry unchanged
start and semantic goal contract preserved
```

The model prediction must be compared against the recomputed oracle target,
not against the pre-swap target. Unchanged predictions are correct when the
oracle target also remains unchanged.

**Coverage-aware graph probing**. Graph probing must distinguish:

```text
seen positive edges         (appeared in training waypoint sequences)
held-out positive edges     (in E_obs but never in training waypoints)
hard negative non-edges     (not in E_obs; both identities appear in training)
transitive but non-direct   (A ⇝ B in DAG but (A,B) ∉ E_obs)
direction-reversed pairs    ((B,A) where (A,B) ∈ E_obs)
```

A complete probe evaluates both $(o_i, o_j) \in E_{\text{obs}}$ and
$(o_i, o_j) \notin E_{\text{obs}}$. The distinction between a direct edge
$A \rightarrow B$ and transitive reachability $A \leadsto B$ is critical.

**Corpus query-coverage requirements**. The data-generation contract must
ensure minimum supervision coverage of the hidden DAG. At minimum:

```text
every DAG edge intended to be learned appears in ≥1 training oracle waypoint
every observation identity appears as start in ≥N_start samples
every observation identity appears as goal in ≥N_goal samples
semantic path lengths cover a declared range
reachable start-goal pairs cover a declared fraction of all reachable pairs
```

Configurable minimum counts (not hardcoded): `min_samples_per_obs`,
`min_samples_per_edge`, `min_samples_per_path_length_bucket`,
`min_samples_per_reachable_pair_bucket`.

**Controlled semantic distractors**. A meaningful fraction of samples must
contain at least one spatially competitive but semantically invalid
alternative: a cell that is physically close to the optimal route,
traversable, and contains a distractor observation whose transition from
the current semantic state is NOT in $E_{\text{obs}}$. Without controlled
hard distractors, the task may be solvable largely through geometry and
goal proximity.

The strongest intervention: keep walls, start, goal mask shape, and
observation positions fixed; swap two non-goal observation identities;
recompute the oracle; verify that the predicted route changes appropriately.
A model relying only on spatial layout should fail. A model using the
learned observation DAG should respond correctly.

---

## Open questions

1. **Deliberation depth**: How many internal iterations $K$ are needed for
   the model to jointly resolve semantic and spatial constraints? Does a
   single forward pass suffice, or does product-space reasoning demand
   explicit recurrent steps?

2. **Hidden DAG learning from spatial signal**: Can the model learn the
   hidden observation-transition graph from the spatial trajectory field
   alone, or is the waypoint field essential? The first experiment should
   compare training with and without waypoint supervision.

3. **Decay factor sensitivity**: How does the spatial decay factor affect
   learning? A factor too close to 1.0 may provide weak positional signal;
   a factor too small may make long-route terminal activations
   indistinguishable from zero.

4. **Simple-route assumption**: How frequently does the product-state oracle
   produce routes with revisited physical positions? If the simple-route
   rejection rate is high, the layout generator must be tuned or the output
   contract must support revisitation through richer representations.

5. **Scaling observation count**: Goaltrace uses up to 45 observations.
   Routebind starts at 16 because the product-state space grows with
   $|P_{\text{free}}| \times N_{\text{obs}}$. How does performance scale
   as $N_{\text{obs}}$ increases?

6. **2D vs. 1D positional encoding**: Does the explicit 2D row/column
   embedding provide measurable benefit over HRM's standard 1D RoPE for
   grid-based spatial reasoning?

7. **Causal intervention sensitivity**: At what training stage does the
   model become sensitive to observation-identity swaps? This reveals when
   the hidden DAG is internalized.

8. **Transfer from Goaltrace**: Does a model pre-trained on Goaltrace
   (hidden DAG learning without spatial structure) learn routebind faster
   than a randomly initialized model? This tests whether the DAG-learning
   capability transfers.

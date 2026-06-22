# dagflow Layout Dataset

## Identity

| Property       | Value                                       |
| -------------- | ------------------------------------------- |
| Family         | `dagflow`                                   |
| Topology kind  | `dag` (Directed Acyclic Graph)              |
| Source         | synthetic (procedural generation)           |
| Source ID      | `synthetic/dagflow`                         |
| CLI script     | `scripts/data-gen/build-dagflow.py`         |
| Builder module | `ehc_sn.data.substrate.dagflow`             |
| Output path    | `data/interim/dagflow/<preset>/v<version>/` |
| Dataset class  | `layout_dataset`                            |

## Description

Dagflow generates immutable DAG artifacts over a public observation
vocabulary. Each graph is first constructed over a private total rank
order. A mandatory Hamiltonian rank backbone guarantees reachability
from every lower-ranked node to every higher-ranked node. Additional
forward edges control branching and shortcut structure. A random
bijection then maps private ranks to public observation IDs, preventing
numerical IDs from revealing graph order. The interim substrate
preserves both rank and public identity for validation and provenance,
while downstream tasks consume only the published public-ID graph.

Dagflow is a _semantic-graph substrate_ — it defines what semantic
transitions are permitted, not where observations are spatially located,
which query is asked, or what target representation a task should produce.
Internal construction indices (topological ranks) are private generation
machinery; published structural channels and canonical graph edges are
expressed in public observation IDs whose numerical order does not encode
graph rank.

The public observation vocabulary is `{0, …, N−1}` where `N = n_max`.
Every graph uses exactly `n_max` nodes (fixed-size regime;
`fixed_n_actual=True`, non-configurable). An edge `u → v` means that
after observation `u` has been accepted, observation `v` may be accepted
next.

Task protocol channels (target path, path length, edge labels, weights,
targets) belong in the downstream task corpora (seqmaze, goaltrace,
routebind), not in this layout dataset. Specifically:

- Dagflow guarantees $\text{rank}(a) < \text{rank}(b) \Rightarrow a \leadsto b$
  but does **not** select $(a, b)$.
- Goaltrace, SeqMaze, and Routebind each own query selection, query
  balancing, difficulty stratification, and target generation.

## Generation Algorithm

1. Create internal topological ranks $r_0, \dots, r_{N-1}$ with edges only
   forward in rank.
2. Build the mandatory Hamiltonian backbone $r_i \to r_{i+1}$ for all
   non-terminal nodes, guaranteeing full reachability.
3. Add extra forward shortcut edges at controlled semantic spans (short,
   medium, long) up to `max_out_degree` per node, governed by
   `extra_edge_density` and the `span_profile` preset field.
4. Permute public observation IDs via a random bijection
   $\pi: r_i \mapsto o_{\pi(i)}$, so that public IDs are uninformative
   about topological rank.
5. Publish edges in public ID space: $\pi(r_i) \to \pi(r_j)$.
6. Write all `N` node rows (no variable-size padding).

**Fixed-size contract.** Every graph uses exactly `n_max` nodes. The
`node_mask` channel is all-`True` for every sample and `N_\text{obs} = N`.
Padding machinery is present for schema uniformity and future variable-size
compatibility but is never exercised in the current regime.

## Channels

| Channel             | Dtype | Shape  | Description                                                              |
| ------------------- | ----- | ------ | ------------------------------------------------------------------------ |
| `node_rank`         | int32 | (N,)   | Topological rank per row. Padded entries would be `N` (sentinel).        |
| `node_obs_id`       | int32 | (N,)   | Public observation ID per row. Padded entries would be `N` (sentinel).   |
| `rank_to_obs_id`    | int32 | (N,)   | Public observation ID for each rank. Strict bijection.                   |
| `obs_id_to_rank`    | int32 | (N,)   | Rank for each public observation ID. Inverse of `rank_to_obs_id`.        |
| `successor_indices` | int32 | (N, K) | Public observation IDs of successors. Padded entries are `N` (sentinel). |
| `successor_mask`    | bool  | (N, K) | `True` where a successor exists in that slot.                            |
| `node_mask`         | bool  | (N,)   | `True` for actual (non-padded) nodes. Currently always all-`True`.       |

- `N` = `n_max` (all nodes are actual; no padding in the current regime).
- `K` = `max_out_degree` (maximum successors per node).
- Sentinel value for padding: `N` (outside the public observation domain
  `{0, …, N-1}`).
- Nodes are stored in **rank-indexed order** — row `r` contains rank `r`.
  This means `node_rank[r] == r` for all valid rows. The `node_rank`
  channel is therefore **redundant with the row index** in v1. It is
  retained as explicit structural metadata, a self-asserting contract,
  debugging aid, and future-compatibility guard against storage-order
  changes.
- Every non-terminal node has out-degree ≥ 1 (no stranded nodes except the
  sink). Every non-source node has in-degree ≥ 1 (no unreachable nodes).
- Published edges are expressed in **public observation IDs**, not
  storage-row indices. All downstream task builders receive edges in
  public ID domain.

### Mixed representation: row indexing vs. successor values

Rows are indexed by rank. Successor values are public observation IDs.
This is a common source of confusion. Example for a graph with `N = 45`:

```text
row 3:
    node_rank    = 3
    node_obs_id  = 11
    successor_indices = [5, 2, N, N]   (K = 4, two slots padded)
    successor_mask    = [True, True, False, False]
```

This means:

```text
obs_11 → obs_5
obs_11 → obs_2
```

not `rank_3 → rank_5` or `rank_3 → rank_2`. To interpret an edge in rank
space, look up `obs_id_to_rank[5]` and `obs_id_to_rank[2]`.

## Graph Artifact Identity

Each graph in a dagflow dataset is identified by a stable, deterministic
`artifact_id` formed as:

```text
dagflow-{preset}-v{version}-{split}-{idx:06d}
```

for example `dagflow-balanced-v1-train-000042`.

Every sample carries a `content_digest` — a SHA-256 hash of the canonical
graph representation (see `canonical_dag_digest` in `graph.py`). The
digest covers:

- schema version (`ehc-sn.dag.v1`);
- number of nodes;
- public observation IDs in node-index order;
- directed edges sorted lexicographically by `(source_obs, dest_obs)`.

The digest does **not** depend on edge-array padding order, storage row
order, Python object order, or task-specific metadata. Rank mapping is
**not** part of the public graph digest — it belongs to construction
provenance only.

This matters because Goaltrace and Routebind need to select exactly the
same graph artifact, not merely the same dagflow root. The index entry
for each sample includes `artifact_id`, `split`, and `content_digest`.

## Rank Exposure Rules

| Layer               | `node_rank`, `rank_to_obs_id`, `obs_id_to_rank` |
| ------------------- | ----------------------------------------------- |
| dagflow interim     | Visible (validation and provenance)             |
| Task builder/oracle | Usable for query selection and validation       |
| Model-facing input  | **Forbidden**                                   |

The validator at the dagflow level permits these channels. Downstream
task builders (seqmaze, goaltrace, routebind) **must not** copy rank
channels into the task corpus. Model-facing inputs must contain only
the public-ID graph (`node_obs_id`, `successor_indices`,
`successor_mask`, `node_mask`).

## CLI Commands

| Command    | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `build`    | Generate graph layouts and write a versioned layout dataset. |
| `validate` | Validate a version root's manifest, channels, and data.      |
| `inspect`  | Print a human-readable summary of a version root manifest.   |

## Build Parameters

| Parameter              | Type  | Default                | Description                                                           |
| ---------------------- | ----- | ---------------------- | --------------------------------------------------------------------- |
| `--preset`             | str   | _(required)_           | Named generation preset (see Presets below).                          |
| `--version`            | int   | _(required)_           | Layout dataset version integer. Must be ≥ 1.                          |
| `--n-max`              | int   | _(preset)_             | Maximum candidate nodes N. Overrides preset.                          |
| `--extra-edge-density` | float | _(preset)_             | Fraction of possible extra forward edges in [0, 1]. Overrides preset. |
| `--max-out-degree`     | int   | _(preset)_             | Maximum out-degree K. Overrides preset.                               |
| `--span-profile`       | str   | _(preset)_             | Rank-span profile: `local`, `balanced`, `heavy`, `uniform`.           |
| `--n-train`            | int   | 4000                   | Number of training graph artifacts.                                   |
| `--n-val`              | int   | 500                    | Number of validation graph artifacts.                                 |
| `--n-test`             | int   | 500                    | Number of test graph artifacts.                                       |
| `--seed`               | int   | 42                     | Deterministic base seed.                                              |
| `--output-root`        | path  | `data/interim/dagflow` | Root path for interim files.                                          |
| `--force`              | flag  | false                  | Delete existing version root before building.                         |

## Presets

Each preset resolves to an explicit `target_edges` value — the desired
total edge count (backbone + optional). All predefined presets are
verified feasible under their degree caps. The `--extra-edge-density`
CLI flag remains available as an advanced override with strict feasibility
validation.

| Preset    | `n_max` | `max_out_degree` | `target_edges` | `span_profile` | Purpose                                                               |
| --------- | ------- | ---------------- | -------------- | -------------- | --------------------------------------------------------------------- |
| `small`   | 8       | 3                | 11             | `balanced`     | Smoke tests and rapid iteration.                                      |
| `sparse`  | 45      | 4                | 139            | `local`        | Sparse 45-node graphs; mostly backbone with few shortcuts.            |
| `routing` | 45      | 4                | 80             | `local`        | Canonical routing graph (low extra edges, semantic paths 5–10).       |
| `chain16` | 16      | 3                | 18             | `local`        | Long-composition stress preset (near-chain, semantic paths up to 15). |

All presets use a mandatory Hamiltonian backbone and permuted public
observation IDs (`public_id_policy: "permuted"`). These are
non-configurable invariants.

### Density and target edges

Predefined presets specify an explicit `target_edges` count. The
`--extra-edge-density` CLI parameter overrides the preset's edge count
and is validated against the degree cap at build time.

The density definition underlying `--extra-edge-density` is:

$$\rho_{\text{extra}} = \frac{|E| - (N-1)}{\frac{N(N-1)}{2} - (N-1)}$$

where:

- $|E|$ = total edge count;
- $N-1$ mandatory edges belong to the Hamiltonian backbone;
- the denominator counts all possible forward edges excluding the backbone.

Then:

- $\rho = 0.0$ means backbone only;
- $\rho = 1.0$ means complete forward DAG (every possible forward edge);
- intermediate values apply only to **optional** edges beyond the backbone.

When used as an override, `--extra-edge-density` is converted to
`target_edges = (N-1) + round(ρ × max_extra)` and checked against the
degree cap. Invalid combinations are rejected with a clear error
message and the maximum feasible density for the requested settings.

### Graph structural diagnostics

During `build`, the first generated graph for each run is validated
against the preset's structural acceptance profile. Diagnostics
including edge counts, degree distributions, rank-span histogram,
branch/merge counts, shortest-path-length distribution, semantic
diameter, and rank-to-public-ID correlation are computed and recorded
in the manifest. If a generated graph fails the preset's acceptance
profile, generation retries (up to 50 attempts) before raising a
`RuntimeError`. This ensures that every named preset produces graphs
consistent with its described regime.

### Span Profiles

For each candidate edge $r_i \to r_j$, define $\Delta r = j - i$.
Span bands are scaled as a function of $N$:

| Profile    | Short $\Delta r$           | Medium $\Delta r$                           | Long $\Delta r$               | Short prob | Medium prob | Long prob |
| ---------- | -------------------------- | ------------------------------------------- | ----------------------------- | ---------- | ----------- | --------- |
| `local`    | $[2,\ \lfloor N/8\rfloor]$ | $[\text{short\_hi}+1,\ \lfloor N/3\rfloor]$ | $[\text{medium\_hi}+1,\ N-2]$ | 0.70       | 0.25        | 0.05      |
| `balanced` | same band boundaries       | same                                        | same                          | 0.45       | 0.40        | 0.15      |
| `heavy`    | same band boundaries       | same                                        | same                          | 0.25       | 0.40        | 0.35      |
| `uniform`  | same band boundaries       | same                                        | same                          | 0.33       | 0.34        | 0.33      |

For $N=45$:

- short: $\Delta r \in [2, 5]$
- medium: $\Delta r \in [6, 15]$
- long: $\Delta r \in [16, 43]$

For $N=8$ (the `small` preset):

- short: $\Delta r \in [2, 2]$
- medium: $\Delta r \in [3, 3]$
- long: $\Delta r \in [4, 6]$

Per-band probabilities are independent across bands — they need not sum to
1.0. Each eligible edge independently competes with its band's probability.

### Seed Derivation

A single `--seed` parameter is expanded into independent RNG streams via
`numpy.random.SeedSequence.spawn`:

```text
master seed (--seed)
  ├── graph structure stream (per-sample via rng.integers)
  ├── public-ID permutation stream (per-sample via rng.integers)
  └── split identity stream (train / val / test via spawn)
```

This ensures that changing one component (e.g., edge sampling) does not
change the permutation sequence for unrelated samples.

## Usage Examples

```bash
# Build the canonical routing graph
python scripts/data-gen/build-dagflow.py build --preset routing --version 1

# Build with the sparse preset (45-node graphs)
python scripts/data-gen/build-dagflow.py build --preset sparse --version 1

# Custom node count and density override (must be feasible)
python scripts/data-gen/build-dagflow.py build \
    --preset sparse --version 1 \
    --n-max 64 --extra-edge-density 0.10

# Long-composition stress preset
python scripts/data-gen/build-dagflow.py build --preset chain16 --version 1

# Small smoke test
python scripts/data-gen/build-dagflow.py build \
    --preset small --version 1

# Force-rebuild an existing root
python scripts/data-gen/build-dagflow.py build \
    --preset sparse --version 1 --force

# Validate an existing root
python scripts/data-gen/build-dagflow.py validate data/interim/dagflow/routing/v1

# Inspect manifest
python scripts/data-gen/build-dagflow.py inspect data/interim/dagflow/routing/v1

# Build a SeqMaze task corpus from dagflow layouts
python scripts/data-gen/build-seqmaze.py materialize-task \
    --layout-root data/interim/dagflow/sparse/v1
```

## Downstream Consumers

| Task      | CLI Script                            | Parent Layout Path                    | Owns                                                          |
| --------- | ------------------------------------- | ------------------------------------- | ------------------------------------------------------------- |
| seqmaze   | `scripts/data-gen/build-seqmaze.py`   | `data/interim/dagflow/<preset>/v<N>/` | query selection, target generation, difficulty stratification |
| goaltrace | `scripts/data-gen/build-goaltrace.py` | `data/interim/dagflow/<preset>/v<N>/` | query selection, target generation, query balancing           |

Routebind consumes dagflow graph artifacts as a **secondary** semantic
constraint on top of spatial topology (openfield / dungeongen). Its
primary substrate is spatial, not dagflow — see the Routebind docs for
details. The `routing` preset is the canonical graph profile for
Routebind v1 experiments and shared Goaltrace–Routebind transfer studies.
The `chain16` preset is a long-composition stress profile for evaluating
recurrent depth and curriculum training — it is **not** the canonical
Routebind v1 profile.

Rank channels (`node_rank`, `rank_to_obs_id`, `obs_id_to_rank`) are
substrate metadata. They are visible in the dagflow interim for
validation and provenance. Downstream task builders must **not** copy
them into the task corpus. Model-facing inputs contain only the
public-ID graph (`node_obs_id`, `successor_indices`, `successor_mask`,
`node_mask`).

## Related

- [Spec: Data Contracts §3.1, §4.5](../../spec/spec-data-contracts.md)
- [Graph utilities](../../src/ehc_sn/utils/graph.py) — `generate_hamiltonian_dag`,
  `canonical_dag_digest`, `remap_obs_ids`
- [SeqMaze Task Builder](../../scripts/data-gen/build-seqmaze.py)
- [GoalTrace Task Builder](../../scripts/data-gen/build-goaltrace.py)

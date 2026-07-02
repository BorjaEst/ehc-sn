# dagflow Semantic Graph Dataset

## Identity

| Property       | Value                                       |
| -------------- | ------------------------------------------- |
| Family         | `dagflow`                                   |
| Topology kind  | `dag` (Directed Acyclic Graph)              |
| Source         | synthetic (procedural generation)           |
| Source ID      | `synthetic/dagflow`                         |
| CLI script     | `scripts/data-gen/build-dagflow.py`         |
| Builder module | `ehp_sn.data.substrate.dagflow`             |
| Output path    | `data/interim/dagflow/<preset>/v<version>/` |
| Dataset class  | `semantic_graph_dataset`                    |

## Purpose and ownership

Dagflow generates immutable DAG artifacts over a public observation
vocabulary. Each graph defines what semantic transitions are permitted:
an edge $u \to v$ means that after observation $u$ has been accepted,
observation $v$ may be accepted next.

Dagflow is a semantic-graph substrate. It does not define where observations
are spatially located, which query is asked, or what target representation
a downstream task should produce. Architectural indices (topological ranks)
are private generation machinery; published edges use public observation
IDs whose numerical order does not encode graph rank.

## Semantic model

### Graph structure

Each graph is a fixed-size DAG over $N = \text{n\_nodes}$ public
observation IDs $\{0, \dots, N-1\}$. Construction:

1. Create internal topological ranks $r_0, \dots, r_{N-1}$ with edges only
   forward in rank.
2. Mandatory Hamiltonian backbone $r_i \to r_{i+1}$ guarantees full
   reachability: $\text{rank}(a) < \text{rank}(b) \Rightarrow a \leadsto b$.
3. Extra forward shortcut edges at controlled semantic spans up to
   `max_out_degree` per node, governed by `target_edges`.
4. A random bijection $\pi: r_i \mapsto o_{\pi(i)}$ maps private ranks to
   public observation IDs so numerical IDs are uninformative about rank.
5. Edges are published in public ID space: $\pi(r_i) \to \pi(r_j)$.

### Fixed-size contract

Every graph uses exactly `n_nodes` nodes. `node_mask` is all-True.
$N_\text{obs} = N$.

### Graph artifact identity

Each graph is identified by a deterministic `artifact_id`:
`dagflow-{preset}-v{version}-{split}-{idx:06d}`.

Every sample carries a `content_digest` (SHA-256 of the canonical public
graph representation): schema version, node count, sorted public
observation IDs, and directed edges sorted lexicographically by
`(source_obs, dest_obs)`. Two DAGs produce the same `content_digest`
iff they have the same public observation IDs and the same directed edges.

## Output artifact

### Public observation-ID channels (consumer API)

Rows are stored in rank-indexed order; successor values are public
observation IDs.

| Channel             | Dtype | Shape  | Description                                                  |
| ------------------- | ----- | ------ | ------------------------------------------------------------ |
| `node_obs_id`       | int32 | (N,)   | Public observation ID per row (rank-indexed storage).        |
| `successor_indices` | int32 | (N, K) | Public observation IDs of successors (`N` sentinel for pad). |
| `successor_mask`    | bool  | (N, K) | `True` where a successor exists.                             |
| `node_mask`         | bool  | (N,)   | `True` for actual (non-padded) nodes. Always all-True.       |

- $N$ = `n_nodes`, $K$ = `max_out_degree`.
- Sentinel value for padding: $N$ (outside `{0, …, N-1}`).
- Every non-terminal node has out-degree ≥ 1; every non-source node has in-degree ≥ 1.

### Provenance metadata (rank channels)

| Channel          | Dtype | Shape | Description                                         |
| ---------------- | ----- | ----- | --------------------------------------------------- |
| `node_rank`      | int32 | (N,)  | Topological rank per row. Redundant with row index. |
| `rank_to_obs_id` | int32 | (N,)  | Public obs ID for each rank. Strict bijection.      |
| `obs_id_to_rank` | int32 | (N,)  | Rank for each public obs ID. Inverse mapping.       |

Rank channels are for validation and provenance only; they must not be
included in learning inputs whose objective is to infer public graph structure.

## Invariants

- All edges are forward in rank space: $\text{rank}(u) < \text{rank}(v)$ for
  every edge $u \to v$.
- The Hamiltonian backbone $r_i \to r_{i+1}$ is present for every non-terminal node.
- $\text{node\_mask}$ sums to $N$.
- `successor_indices` contains values in $\{0, \dots, N-1\}$ for active slots, $N$ for padding.
- The bijection between ranks and public observation IDs is complete and invertible.
- `content_digest` changes iff the public edge relation or public observation ID set changes.

## Build configuration

| Preset      | `n_nodes` | `max_out_degree` | `target_edges` | `span_profile` | Purpose                                        |
| ----------- | --------- | ---------------- | -------------- | -------------- | ---------------------------------------------- |
| `small`     | 8         | 3                | 11             | `balanced`     | Smoke tests.                                   |
| `routing`   | 45        | 4                | 80             | `local`        | Canonical routing profile, moderate branching. |
| `sparse`    | 45        | 4                | 60             | `local`        | Sparse routing: 44 backbone + 16 extra edges.  |
| `branching` | 45        | 4                | 139            | `local`        | High shortcut density, multi-path stress.      |
| `chain16`   | 16        | 3                | 18             | `local`        | Long composition, limited shortcuts.           |

> **Edge count convention**: `target_edges` is the **total** number of directed edges,
> including the mandatory Hamiltonian backbone of `n_nodes − 1` edges. Extra shortcut
> edges = `target_edges − (n_nodes − 1)`. For example, `sparse` with
> `target_edges=60`, n=45 has 44 backbone + 16 extra (~1.7% extra-edge density).

Edge density for `--extra-edge-density` override:

$$\rho_{\text{extra}} = \frac{|E| - (N-1)}{\frac{N(N-1)}{2} - (N-1)}$$

Parameters: `--preset`, `--version`, `--n-nodes`, `--n-max` (deprecated),
`--extra-edge-density`, `--max-out-degree`, `--span-profile`, `--n-train`,
`--n-val`, `--n-test`, `--seed`, `--output-root`, `--force`.

Seeding: a single `--seed` is expanded into independent RNG streams via
`SeedSequence.spawn` for graph structure, permutation, and split identity.

## CLI

| Command    | Description                                           |
| ---------- | ----------------------------------------------------- |
| `build`    | Generate graph layouts and write a versioned dataset. |
| `validate` | Validate manifest, channels, and data.                |
| `inspect`  | Print a human-readable manifest summary.              |

Usage:

```bash
python build-dagflow.py build --preset routing --version 1
python build-dagflow.py validate data/interim/dagflow/routing/v1
python build-dagflow.py inspect data/interim/dagflow/routing/v1
```

## Manifest

Root file: `manifest.json`. Dataset class: `semantic_graph_dataset`.
Channels: `node_obs_id`, `successor_indices`, `successor_mask`, `node_mask`,
`node_rank`, `rank_to_obs_id`, `obs_id_to_rank`.

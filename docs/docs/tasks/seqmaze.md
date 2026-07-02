# `seqmaze` Benchmark Task

## Task identity and overview

Task name: `seqmaze`

Benchmark family: sequence reasoning / transition-graph inference

| Symbol   | Surface | Description                                                  |
| -------- | ------- | ------------------------------------------------------------ |
| $obs[t]$ | yes     | observation identifier at time t                             |
| $x'[t]$  | yes     | observation token at time t (decoded, including metadata)    |
| $x[t]$   | —       | embedding for token node at time t (model-internal encoding) |

_Surface_ symbols appear in the task input/output contract.
_Model-internal_ symbols (—) are emergent representations the model learns
but are not part of the task-level data contract.
$g[t]$, $p[t]$, $M$, and $a[t]$ are genuinely absent — `seqmaze` has no
spatial component, no actions, and no episodic memory.

Canonical package path:

```text
src/ehp_sn/tasks/seqmaze/
```

`seqmaze` tests whether a model can reason about sequences and transitions
without any spatial grounding. The model receives a set of observation-node
tokens and the valid successor transitions between them. It must infer the
shortest valid observation sequence from a start token to a goal token, purely
from the transition-graph structure. No spatial coordinates, grid cells, or
location encodings are provided — the transition structure is encoded entirely
inside the node tokens.

The core problem is:

```text
given the start token (obs[0]), the goal token (obs[n]),
a candidate set of observation tokens (obs[0:n+m]),
and valid successor transitions for each token (obs[i] → obs[j]),
predict the shortest valid observation sequence from start to goal
(obs[0] → obs[1] → ... → obs[n]).
```

The task forces the model to _infer_ the path rather than _recall_ it.
Each sample is generated with sample-local structure (permuted candidates,
remapped ids, per-sample successor graph) so the model cannot memorize
fixed transitions.

---

## Relationship to other tasks

### MazeHard analogy

In `mazehard`, each token corresponds to a fixed grid cell with strong spatial
grounding — the embedding carries both "what" (wall, open, start, goal) and
"where" (cell location). The model navigates a known spatial layout.

```text
MazeHard:
  cell tokens + wall/open status + start + goal → spatial route
```

In `seqmaze`, each token corresponds to an observation node with no spatial
anchor:

```text
seqmaze:
  observation-node tokens + valid successor information + start + goal
  → token route (no spatial coordinates)
```

### Relationship to `arena`

`arena` provides spatial grounding — the model learns grid-cell-like encodings
($g[t]$) and observation-location bindings ($M$) through structural learning
(see `docs/docs/tasks/arena.md`). `seqmaze` deliberately removes all spatial
information to isolate pure transition-graph reasoning. The two tasks bookend
the spatial-reasoning spectrum: arena is grounded, seqmaze is abstract.

### Relationship to `goalchain`

`goalchain` (see `docs/docs/tasks/goalchain.md`) combines spatial memory from
arena with goal-conditioned reasoning — it uses episodic memory ($M$) and
location belief ($g$) to plan shortest-walk navigation. `seqmaze` isolates the
reasoning component alone, without memory or space, serving as a potential
pre-training step for the PFC-like inference that `goalchain` demands.

### Anti-memorization contract

The model must not solve the task by memorizing fixed transition patterns
(e.g., _obs_5 always goes to obs_3_). Every sample uses sample-local structure:

- candidate order is permuted
- observation ids may be remapped
- successor graph is generated per sample
- shortest path is computed offline and guaranteed unique

---

## Scientific purpose

`seqmaze` tests whether a model can perform structured reasoning over a graph
of tokens without spatial grounding or memorization. The model must learn to
read the transition graph and infer the correct path from start to goal.

The relevant computation is not:

```text
read this graph → memorize/recall the path
```

It is:

```text
read this graph → infer the path
```

This isolates the PFC-like reasoning process — inferring the correct sequence
of steps toward a goal from structured transition knowledge — without
confounding it with spatial memory, location encoding, or episodic recall.

Interpretation:

| Component              | Role in `seqmaze`                                                       |
| ---------------------- | ----------------------------------------------------------------------- |
| PFC / reasoning module | infers the correct sequence from the transition graph via deliberation. |
| Token structure        | encodes valid successors inside node-token metadata (adapter-owned).    |
| Deliberation           | recurrent HRM processing (ACT-compatible); parallel path prediction.    |

### Why this is reasoning, not recall

In a standard supervised task, the model learns a mapping $x \rightarrow y$
and stores it in its weights. At inference, one forward pass produces the
answer from parametric memory.

In `seqmaze`, the transition graph is novel per sample. The model
received no training example of this specific graph. The answer cannot be
retrieved from weights — it must be computed from the input data
through recurrent deliberation over the packed graph-path workspace.
The model runs multiple internal reasoning cycles (HRM $z_H$/$z_L$,
with or without ACT halting) and then emits the full path sequence in parallel.
This multi-step inference-time computation over novel structured input is
the operational definition of reasoning that `seqmaze` tests.

---

## Input and output

### Execution mode

`seqmaze` v1 uses **parallel path prediction after recurrent deliberation**.
The model does not generate autoregressively one token per step. Instead:

1. The adapter packs graph nodes and learned path-query slots into a single
   fixed-length schema-token sequence.
2. The HRM model deliberates over the packed workspace (recurrent $z_H$/$z_L$
   cycles; ACT halting is supported but not required).
3. The decoder reads all path-region states simultaneously and produces
   logits over all $T_{max}$ output positions in one shot.

This is a **packed-sequence, deliberative, parallel-prediction** contract,
not a language-model-style autoregressive generation contract.

### Tensor shape contract

`seqmaze` uses separate graph and path axes packed into one HRM-compatible
schema-token sequence of fixed length $S$:

$$S = N_{max} + T_{max}$$

| Parameter | Description                          | Owner                                  |
| --------- | ------------------------------------ | -------------------------------------- |
| $N_{max}$ | maximum candidate nodes per batch    | corpus + task config                   |
| $T_{max}$ | maximum generated path length        | corpus + task config                   |
| $S$       | total schema-token sequence length   | model config (`pfc.seq_length`)        |
| $K$       | maximum out-degree per node          | corpus                                 |
| $V$       | path vocabulary size = $N_{max} + 2$ | corpus (candidate indices + EOS + PAD) |

The invariant $S = N_{max} + T_{max}$ is enforced at:

- **Corpus manifest**: records $N_{max}$, $T_{max}$, $S$, $K$, $V$.
- **Model/training config**: `pfc.seq_length` must equal $S$.
- **Training startup**: fails fast if model config and corpus manifest disagree.
- **Evaluation startup**: fails fast if checkpoint profile disagrees with evaluation corpus.
- **Adapter construction**: fails fast if $N_{max} + T_{max} \neq S$.

A model checkpoint is tied to a specific $(N_{max}, T_{max}, S)$ profile.
Common profiles would be named:

```text
seqmaze-n32-t32-s64
seqmaze-n48-t16-s64
seqmaze-n40-t40-s80
```

All three values must match exactly — not just $S$. The allocation between
graph capacity and path capacity changes the semantics of the schema slots.

### Schema-token layout

```text
schema_tokens: (B, S, D)

positions [0 : N_max):
  graph region — node embeddings for the candidate graph

positions [N_max : N_max + T_max):
  path region — learned query embeddings for output positions
```

### Adapter input (task-data → model)

The task exposes structured graph fields. The adapter translates them into
HRM-compatible schema tokens.

**Raw task fields** (per sample, before packing):

| Field               | Shape       | Description                                         |
| ------------------- | ----------- | --------------------------------------------------- |
| `obs_id`            | `(B, N)`    | unique observation identifier (remapped per sample) |
| `candidate_index`   | `(B, N)`    | index of the token in the candidate set (0 to N−1)  |
| `start_flag`        | `(B, N)`    | `True` if this token is the start token             |
| `goal_flag`         | `(B, N)`    | `True` if this token is the goal token              |
| `successor_indices` | `(B, N, K)` | indices of valid successor tokens (PAD for unused)  |
| `successor_mask`    | `(B, N, K)` | binary mask marking valid successor slots           |
| `node_mask`         | `(B, N)`    | `True` for valid nodes (mask for N < N_max)         |

**Adapter-packed model input**:

```text
SeqMazeHRMInput:
  schema_tokens:       FloatTensor[B, S, D]
  schema_mask:         BoolTensor[B, S]
  graph_region_mask:   BoolTensor[B, S]
  path_region_mask:    BoolTensor[B, S]
```

No `successor_indices` or `successor_mask` are passed as separate tensors
to the model in v1. They are encoded into the graph-region node embeddings
by the adapter (see [Adapter specification](#adapter-specification)).

### Graph region packing (positions `[0 : N_max)`)

For each candidate node $i$:

```text
h_i^0 = E_obs(obs_id_i)               content: what observation
      + E_candidate(candidate_index_i)  identity: which slot
      + E_start(start_flag_i)           flag: is this the start?
      + E_goal(goal_flag_i)             flag: is this the goal?
      + edge_embedding_i                transition structure (see below)

node_embedding_i = h_i^0 + E_region("graph")
```

The edge encoding (v1 default: `successor_index_embedding`):

```text
edge_embedding_i =
    Pool_k [ E_successor_slot(k) + E_candidate_index(successor_indices[i, k]) ]
    masked by successor_mask[i, k]
```

This embeds the discrete adjacency list into each node token without
performing graph message-passing. The model must learn to map embedded
successor indices to the corresponding schema-slot positions.

Padded nodes ($i \ge N$) receive zero embedding and are excluded via
`node_mask` and `schema_mask`.

### Path region packing (positions `[N_max : N_max + T_max)`)

For each output position $t$:

```text
path_query_t = E_path_query
             + E_path_position(t)
             + E_region("path")
```

Path queries are **learned embeddings**, not teacher-forced previous tokens.
The model must fill all path slots after deliberation without seeing the
target sequence.

### Adapter output (model → evaluation)

**Model output**:

```text
SeqMazeTaskOutput:
  path_logits:  FloatTensor[B, T_max, N_max + 2]
  halt_probs:   FloatTensor[B, steps] (optional; present when ACT controller is used)
```

The decoder reads:

```text
path_states = hrm_output[:, N_max : N_max + T_max, :]
decoder_input_t = path_states[:, t, :] + E_decode_position(t)
path_logits = Linear(D, N_max + 2)(decoder_input)
```

$E_{decode\\_position}$ shares weights with $E_{path\\_position}$ by default
(`share_path_position_embeddings = true`). This guards against slot-symmetry
failures where the model cannot distinguish which output slot should emit EOS.

**Local vocabulary**:

```text
0 .. N_max-1    candidate node indices
N_max           EOS
N_max + 1       PAD
```

**Target path**:

```text
SeqMazeTargets:
  path_index:   LongTensor[B, T_max]
  path_mask:    BoolTensor[B, T_max]
  path_length:  LongTensor[B]
```

### Example

```text
candidate tokens (N=6):
  obs_5, obs_3, obs_4, obs_0, obs_1, obs_2
  indices:  0, 1, 2, 3, 4, 5

start token: candidate index 3 (obs_0)
goal token:  candidate index 1 (obs_3)

valid transitions:
  candidate 3 → candidate 2   (obs_0 → obs_4)
  candidate 0 → candidate 2   (obs_5 → obs_4)
  candidate 1 → candidate 5   (obs_3 → obs_2)
  candidate 2 → candidate 4   (obs_4 → obs_1)
  candidate 2 → candidate 1   (obs_4 → obs_3)

shortest path (indices): [3, 2, 1]

target_path (T_max=8):  [3, 2, 1, EOS, PAD, PAD, PAD, PAD]
path_mask:               [1, 1, 1, 1,   0,   0,   0,   0]
path_length: 3

schema_tokens (N_max=32, T_max=32, S=64):
  positions [0:6):   graph node embeddings (nodes 0–5)
  positions [6:32):  zero + mask (padding graph slots)
  positions [32:64): path query embeddings (T_max=32 slots)
```

---

## Corpus and data generation

Each sample is a self-contained directed acyclic graph (DAG) problem. The
corpus stores the node fields, start/goal flags, successor adjacency, and
the precomputed unique shortest-path target sequence.

### Graph topology

v1 uses **DAGs** exclusively. DAGs guarantee:

- Unique shortest paths are well-defined (BFS solves exactly).
- No cycles to create ambiguous tie-breaking.
- Directional transition structure is cleaner for initial reasoning experiments.

Cyclic or undirected graphs may be added in a future protocol version.

### Key invariants

- Every sample has a **unique shortest path** from start to goal.
  Multiple valid solutions may be added in a future protocol version.
- Transition graphs are generated per sample; no global transition table
  is shared across samples.
- Observation ids are remapped per sample to prevent memorization.
- Candidate order is permuted per sample.
- Graphs are DAGs (v1).

### Corpus manifest

Each versioned corpus root includes a `manifest.json`:

```json
{
  "task": "seqmaze",
  "corpus": "default",
  "version": 1,
  "n_max": 32,
  "t_max": 32,
  "seq_length": 64,
  "max_out_degree": 4,
  "path_vocab_size": 34,
  "special_tokens": {
    "eos": 32,
    "pad": 33
  }
}
```

The manifest is the authoritative source for profile parameters. Model
configs, benchmark configs, and adapters must agree with it at startup.

### Profile coupling

A model checkpoint is bound to a specific $(N_{max}, T_{max}, S)$ profile.
The following must be validated at startup:

| Checkpoint         | Validation                                        |
| ------------------ | ------------------------------------------------- |
| Training startup   | `pfc.seq_length == manifest.seq_length`           |
|                    | `adapter.n_max == manifest.n_max`                 |
|                    | `adapter.t_max == manifest.t_max`                 |
| Evaluation startup | `checkpoint.seq_length == eval_corpus.seq_length` |
|                    | `checkpoint.n_max == eval_corpus.n_max`           |
|                    | `checkpoint.t_max == eval_corpus.t_max`           |

All three values must match exactly — not just $S$. A model trained on
`n32_t32_s64` must not silently evaluate on `n40_t24_s64`.

### Build configuration

| Parameter          | Default      | Description                          |
| ------------------ | ------------ | ------------------------------------ |
| `--layout-root`    | _(required)_ | Path to dagflow layout dataset root. |
| `--corpus`         | `default`    | Corpus label.                        |
| `--n-max`          | 32           | Maximum candidate nodes $N$.         |
| `--t-max`          | 32           | Maximum path length $T$.             |
| `--max-out-degree` | 4            | Maximum out-degree $K$ per node.     |
| `--n-train`        | 4000         | Number of training samples.          |
| `--n-val`          | 500          | Number of validation samples.        |
| `--n-test`         | 500          | Number of test samples.              |
| `--version`        | 1            | Task corpus version integer.         |
| `--seed`           | 42           | Deterministic base seed.             |

### CLI

| Command    | Description                                           |
| ---------- | ----------------------------------------------------- |
| `build`    | Build a SeqMaze task corpus over a dagflow substrate. |
| `validate` | Validate a SeqMaze task-corpus version root.          |
| `inspect`  | Print a human-readable summary of a version root.     |

Usage:

```bash
# Build
python scripts/data-gen/build-seqmaze.py build \
    --layout-root data/interim/dagflow/sparse/v1

# Validate
python scripts/data-gen/build-seqmaze.py validate \
    data/processed/seqmaze/default/v1

# Inspect
python scripts/data-gen/build-seqmaze.py inspect \
    data/processed/seqmaze/default/v1 --summary
```

### Prerequisites

A dagflow shared substrate must exist before running. Build it first:

```bash
python scripts/data-gen/build-dagflow.py build --preset branching --version 1
```

### Utilities

Graph generation and validation utilities live in
`src/ehp_sn/tasks/seqmaze/_graph_utils.py`:

```python
generate_transition_dag(n_nodes, max_out_degree, seed) → adjacency
shortest_path(adjacency, start, goal) → list[candidate_index]
count_shortest_paths(adjacency, start, goal) → int
validate_unique_shortest_path(adjacency, start, goal) → bool
remap_obs_ids(n_nodes, seed) → list[obs_id]
permute_candidates(node_fields, seed) → permuted_fields
```

These are task-internal utilities, not a shared substrate. `seqmaze` has no
`data/interim/` layer. The versioned immutable data root is:

```text
data/processed/seqmaze/<corpus>/v<version>/
  train/
  val/
  test/
  manifest.json
  index.jsonl

Per split, each channel (``obs_id``, ``candidate_index``, ``start_flag``,
``goal_flag``, ``successor_indices``, ``successor_mask``, ``node_mask``,
``path_index``, ``path_mask``, ``path_length``) is stored as a separate
``{channel}.npy`` file.
```

### Build

```bash
python scripts/data-gen/build-seqmaze.py build \
    --layout-root data/interim/dagflow/sparse/v1 \
    --corpus default --version 1 \
    --n-max 32 --t-max 32 --max-out-degree 4 \
    --n-train 4000 --n-val 500 --n-test 500 \
    --seed 42
```

---

## Benchmark and evaluation

### SeqMaze-Reason track

| Aspect            | Value                                                        |
| ----------------- | ------------------------------------------------------------ |
| Benchmark track   | SeqMaze-Reason                                               |
| Claim family      | `sequence_reasoning`                                         |
| Execution mode    | deliberative graph-conditioned parallel sequence prediction  |
| Primary metric    | `sequence_exact`                                             |
| Secondary metrics | `path_position_accuracy`, `valid_transition_rate`,           |
|                   | `reaches_goal`, `path_length_regret`, `eos_accuracy`         |
| Controller        | HRM deliberation (ACT-compatible; pure supervised supported) |

### Primary metric: `sequence_exact`

The predicted sequence must exactly match the target sequence after
**EOS canonicalization**.

**Canonicalization rule**: given the argmax predicted path, keep all tokens
up to and including the first EOS; replace all following positions with PAD.
If no EOS is predicted, use the full $T_{max}$ sequence without canonicalization.

```text
target:          [3, 2, 1, EOS, PAD, PAD, PAD, PAD]
pred (argmax):   [3, 2, 1, EOS, 7,   14,  EOS, PAD]
canonicalized:   [3, 2, 1, EOS, PAD, PAD, PAD, PAD]
exact match?     yes
```

Canonicalization ensures the evaluation does not penalize post-EOS noise
that the training loss never supervises (see [Training loss](#training-loss)).

### Secondary metrics

| Metric                   | Description                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `path_position_accuracy` | fraction of non-PAD target positions predicted correctly, including EOS                     |
| `valid_transition_rate`  | proportion of adjacent candidate-token pairs (before first EOS) that are valid graph edges  |
| `reaches_goal`           | proportion of sequences where the goal token appears before the first EOS                   |
| `path_length_regret`     | extra tokens beyond the shortest-path length in the candidate-token prefix before first EOS |
| `eos_accuracy`           | fraction of samples where the first predicted EOS position matches the target EOS position  |

All secondary metrics are computed on the canonicalized prediction
(truncated after first EOS).

### EOS canonicalization for metrics

All evaluation metrics apply the same canonicalization:

1. Find the first EOS token in the prediction.
2. Truncate all positions after it (replace with PAD).
3. Compare the canonicalized prediction against the canonical target.

The only exception is `reaches_goal`: it checks whether the goal token
appears anywhere in the canonicalized prefix (before first EOS).

### Training loss

Cross-entropy over path positions, masked by `path_mask`:

```text
loss = CE(path_logits, target_path)
masked where path_mask == True
```

The loss mask includes EOS and excludes PAD:

```text
target_path:  [3, 2, 1, EOS, PAD, PAD]
path_mask:    [1, 1, 1, 1,   0,   0  ]
```

This trains the model to emit the correct chain and terminate with EOS,
but does not supervise post-EOS padding. Evaluation canonicalization
handles the mismatch between training supervision and exact-match strictness.

Teacher forcing is applied only to the **loss**, not to the model input.
Path query slots are learned embeddings, not previous-token predictions.

---

## Adapter specification

### Boundary

The adapter owns **task-to-latent translation**, not reasoning.

| Adapter owns                                         | Model owns                                       |
| ---------------------------------------------------- | ------------------------------------------------ |
| Packing task fields into HRM `schema_tokens`         | Recurrent processing ($z_H$, $z_L$)              |
| Region embeddings (`graph` / `path`)                 | Halting control (ACT or fixed-step)              |
| Node and path-slot embeddings                        | Graph-reasoning operations                       |
| Edge encoding (discrete adjacency → embeddings)      | Any learned propagation over successor structure |
| Local-vocabulary decoding ($N_{max} + 2$)            | Deliberation dynamics                            |
| Task masks (`schema_mask`, `node_mask`, `path_mask`) |                                                  |

The adapter does **not** perform:

- Shortest-path computation
- Graph message-passing or successor-node pooling (by default)
- Graph search of any kind

### Edge encoding modes

The edge encoding is the mechanism by which transition structure is injected
into graph-node embeddings. It is a configurable experimental axis.

```toml
[adapter.seqmaze]
edge_encoding = "successor_index_embedding"   # v1 default
```

| Mode                        | Description                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `successor_index_embedding` | **v1 default.** Embeds adjacency list as `E_slot(k) + E_index(succ[i,k])`, pooled per node. Model must learn index-to-slot mapping.                                     |
| `none`                      | Negative control. Removes all transition information. Model should fail except for dataset biases.                                                                      |
| `successor_node_pool`       | Ablation. Pools successor node base embeddings, injecting one-hop structural summary. Tests whether explicit structure helps. Removes the index-indirection bottleneck. |

The v1 default (`successor_index_embedding`) was chosen because:

1. It works with HRM's single `schema_tokens` input — no model architecture changes needed.
2. It encodes transition rules in a latent form the model must learn to use, rather than pre-solving the graph.
3. It mirrors the EHP design principle: the PFC receives structured state information and must infer the chain, rather than receiving a pre-computed solution.

If `successor_index_embedding` fails (model cannot learn index-to-slot mapping),
`successor_node_pool` is available as a fallback to isolate the bottleneck.

### Node composition

Content and structure embeddings are combined additively with a region tag:

```text
node_embedding_i = content_embedding_i + edge_embedding_i + E_region("graph")
```

where:

```text
content_embedding_i =
    E_obs(obs_id_i)
  + E_candidate(candidate_index_i)
  + E_start(start_flag_i)
  + E_goal(goal_flag_i)
```

The additive composition is the v1 default. Future ablations may add
concat-MLP, gated, bilinear, or low-rank outer-product composition modes.
This makes `seqmaze` a controlled testbed for content-structure binding
mechanisms relevant to broader EHP architecture questions.

### Decoder position signal

The decoder applies an explicit output-position embedding before the
linear head:

```text
decoder_input_t = path_states[:, t, :] + E_decode_position(t)
path_logits = Linear(D, N_max + 2)(decoder_input)
```

`E_decode_position` shares weights with `E_path_position` by default:

```toml
[adapter.seqmaze]
share_path_position_embeddings = true
```

This guards against slot-symmetry failure (all $T_{max}$ output slots are
otherwise identical, making EOS placement ambiguous).

---

## Open questions

1. **Graph topology beyond DAGs**: v1 uses DAGs exclusively. When should
   cyclic or undirected graphs be introduced? Cyclic graphs require
   visited-set tracking and complicate the uniqueness guarantee.

2. **Multiple valid solutions**: The current contract requires a unique
   shortest path. How should the task handle graphs with multiple equally
   valid shortest paths? Accept any valid solution, or require a specific
   tie-breaking convention?

3. **Scaling graph size**: How does performance degrade as $N_{max}$ grows
   relative to $S$? Is there a phase transition where graph-reasoning
   capacity saturates? The packed-sequence design lets us trade $N_{max}$
   against $T_{max}$ while keeping $S$ fixed (e.g., `n48-t16-s64` vs.
   `n32-t32-s64`).

4. **Transfer from arena**: Does spatial structural knowledge acquired during
   `arena` training transfer to improved `seqmaze` reasoning, even though
   `seqmaze` has no spatial component? This would test whether structural
   learning produces general-purpose reasoning improvements.

5. **Content-structure composition**: Does additive composition suffice, or
   do more expressive binding mechanisms (gated, bilinear, outer-product)
   improve graph-reasoning performance? `seqmaze` is a controlled testbed
   for this question before it matters in the full EHP architecture.

6. **Autoregressive decoding path**: When should `seqmaze` add an
   autoregressive decoding variant (iterative next-token with deliberation
   at each step)? This is a natural extension but adds training and
   evaluation complexity. v1 starts with parallel prediction to establish
   the baseline.

7. **Goalchain relationship**: `seqmaze` provides reusable graph-reasoning
   utilities and adapter patterns. `goalchain` uses arena-derived spatial
   structure plus goal-rule logic, not abstract seqmaze graph files. The
   shared abstraction is the transition-rule substrate contract (nodes,
   edges, start, goal, shortest-path solver), not concrete graph data.

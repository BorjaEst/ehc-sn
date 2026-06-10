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
src/ehc_sn/tasks/seqmaze/
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

| Component              | Role in `seqmaze`                                             |
| ---------------------- | ------------------------------------------------------------- |
| PFC / reasoning module | infers the correct sequence from the transition graph.        |
| Token structure        | encodes valid successors entirely within node-token metadata. |
| Generation             | autoregressive token prediction with EOS termination.         |

### Why this is reasoning, not recall

In a standard supervised task, the model learns a mapping $x \rightarrow y$
and stores it in its weights. At inference, one forward pass produces the
answer from parametric memory.

In `seqmaze`, the transition graph is novel per sample. The model
received no training example of this specific graph. The answer cannot be
retrieved from weights — it must be computed from the input data
through multiple autoregressive steps, each conditioned on the previous.
This multi-step inference-time computation over novel structured input is
the operational definition of reasoning that `seqmaze` tests.

---

## Input and output

### Adapter input (task-data → model)

The model receives a set of structured node tokens describing the full
transition graph. Each token carries its own successor information.

| Variable    | Description                                                 |
| ----------- | ----------------------------------------------------------- |
| $x'[0:n+m]$ | candidate set of observation tokens (start + goal + others) |

Each node token encodes:

| Attribute           | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `obs_id`            | unique observation identifier (e.g., `obs_5`)                    |
| `candidate_index`   | index of the token in the candidate set (0 to N−1)               |
| `start_flag`        | `True` if this token is the start token                          |
| `goal_flag`         | `True` if this token is the goal token                           |
| `successor_indices` | list of indices of valid successor tokens (PAD for unused slots) |
| `successor_mask`    | binary mask marking valid successor slots                        |

### Adapter output (model → evaluation)

| Variable  | Description                                                 |
| --------- | ----------------------------------------------------------- |
| $x'[0:T]$ | predicted sequence of observation tokens (start → … → goal) |

The output is a variable-length token sequence terminated by EOS, padded with
PAD tokens. The model generates autoregressively, one token per step.

### Example

```text
candidate tokens:
  obs_5, obs_3, obs_4, obs_0, obs_1, obs_2

start token:
  obs_0

goal token:
  obs_3

valid transitions:
  obs_0 → obs_4
  obs_5 → obs_4
  obs_3 → obs_2
  obs_4 → obs_1
  obs_4 → obs_3

target sequence:
  obs_0 → obs_4 → obs_3, EOS, PAD, PAD, ...
```

---

## Corpus and data generation

Each sample is a self-contained transition-graph problem. The corpus stores
the token array, start/goal indices, successor adjacency, and the precomputed
shortest-path target sequence.

### Key invariants

- Every sample has a **unique shortest path** from start to goal.
  Multiple valid solutions may be added in a future protocol version.
- Transition graphs are generated per sample; no global transition table
  is shared across samples.
- Observation ids are remapped per sample to prevent memorization.

### Build

```bash
python scripts/data-gen/build-seqmaze.py build-all
```

Output path: `data/processed/seqmaze/<corpus>/v<version>/`

---

## Benchmark and evaluation

### SeqMaze-Reason track

| Aspect             | Value                                           |
| ------------------ | ----------------------------------------------- |
| Benchmark track    | SeqMaze-Reason                                  |
| Claim family       | `sequence_reasoning`                            |
| Execution mode     | generation (autoregressive, teacher-forced)     |
| Primary metric     | `sequence_exact`                                |
| Secondary metrics  | `next_token_accuracy`, `valid_transition_rate`, |
|                    | `reaches_goal`, `path_length_regret`            |
| Supported families | HRM v1, HRM v2 (current)                        |
| Readiness          | `design`                                        |

### Primary metric: `sequence_exact`

The generated sequence must exactly match the target sequence, including
token order, EOS placement, and PAD placement. This is a strict sequence-level
exact-match metric, analogous to LLM generation evaluation.

### Secondary metrics

| Metric                  | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `next_token_accuracy`   | per-step accuracy of predicting the next token     |
| `valid_transition_rate` | proportion of generated transitions that are valid |
| `reaches_goal`          | proportion of sequences that reach the goal token  |
| `path_length_regret`    | extra steps beyond the shortest-path length        |

### Training loss

Teacher-forced sequence cross-entropy with padding masks. Loss is computed
only on non-PAD, non-EOS positions.

---

## Open questions

1. **Transition-graph generation**: Should graphs be purely random, or should
   they follow specific structures (tree, DAG, cyclic) to study how graph
   topology affects reasoning capability?

2. **Multiple valid solutions**: The current contract requires a unique
   shortest path. How should the task handle graphs with multiple equally
   valid shortest paths? Accept any valid solution, or require a specific
   tie-breaking convention?

3. **Scaling graph size**: How does performance degrade as the number of
   candidate tokens (N) grows? Is there a phase transition where inference
   breaks down?

4. **Transfer from arena**: Does spatial structural knowledge acquired during
   `arena` training transfer to improved `seqmaze` reasoning, even though
   `seqmaze` has no spatial component? This would test whether structural
   learning produces general-purpose reasoning improvements.

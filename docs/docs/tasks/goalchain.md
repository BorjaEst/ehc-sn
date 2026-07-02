# `goalchain` Benchmark Task

## Task identity and overview

Task name: `goalchain`

Benchmark family: goal-conditioned sequence planning with episodic memory

| Symbol   | Surface | Description                                                        |
| -------- | ------- | ------------------------------------------------------------------ |
| $obs[t]$ | yes     | observation identifier at time t                                   |
| $x'[t]$  | yes     | observation token at time t (decoded, including metadata)          |
| $x[t]$   | —       | latent "what" sensory state at time t (LEC embedding sensory)      |
| $g'[t]$  | yes     | location identifier at time t (decoded topology)                   |
| $g[t]$   | —       | latent "where" location state at time t (MEC embedding topology)   |
| $a[t]$   | yes     | action at time t                                                   |
| $p[t]$   | —       | latent conjunctive embedding at time t (HPC encoded state)         |
| $M$      | yes     | memory state containing observation-location bindings (task input) |
| $goal'$  | yes     | goal observation cue (decoded, task input)                         |
| $goal$   | —       | latent goal cue (PFC encoded goal cue, model-internal)             |

_Surface_ symbols appear in the task input/output contract.
_Model-internal_ symbols (—) are emergent representations the model learns
but are not part of the task-level data contract.
$M$ is an exception: it is a model-internal from `arena`'s perspective, but
a task input from `goalchain`'s perspective — the data-generation pipeline
extracts it from a pre-trained arena model and provides it as conditioning.

Canonical package path:

```text
src/ehp_sn/tasks/goalchain/
```

`goalchain` tests whether a model can use a learned episodic memory state
($M$) to solve a goal-conditioned sequence-reasoning problem over a spatial
layout. The model receives a layout-matched memory state produced during
`arena` structural-learning pre-training, a goal observation cue ($goal'$),
and its current believed location ($g'$). It must infer the correct sequence
of observations ($x'[0:n]$) and locations ($g'[0:n]$) needed to navigate from
the current location to the goal via the shortest walk.

The core problem is:

```text
given the episodic memory (M) from structural learning,
the current believed location (g'),
and a final goal observation cue (goal'),
infer the shortest-walk observation sequence (x'[0:n])
and the corresponding location sequence (g'[0:n]).
```

The task does not ask the model to learn the spatial layout from scratch.
It assumes the model already has access to the memory state for the same
layout — each `goalchain` episode must be paired with the correct memory
state for its layout ($L_i$).

---

## Relationship to other tasks

### Structural-learning prerequisite: `arena`

`goalchain` depends on `arena` (see `docs/docs/tasks/arena.md`) as a hard
pre-training requirement. During `goalchain` data generation, a pre-trained
arena model is run over a layout to produce the episodic memory state ($M$)
and latent representations ($g[t]$, $p[t]$). `goalchain` then tests whether
the model can use that memory to solve goal-conditioned reasoning.

```text
arena (structural learning)
    → model trained to encode layouts and form observation-location bindings
    → produces emergent memory state ($M$) and latent representations
    → goalchain data-gen extracts these as task inputs
    → goalchain model conditions on $M$, $g'$, $goal'$ to infer sequences
```

The critical contract: **each goalchain episode layout must be paired with
the correct memory state for that layout.** A sample from layout $L_i$ must
receive the memory state produced by the arena model on layout $L_i$.

### Relationship to `seqmaze`

`seqmaze` (see `docs/docs/tasks/seqmaze.md`) isolates pure transition-graph
reasoning without spatial grounding or episodic memory. `goalchain` adds both:
it combines the spatial memory from `arena` with the sequence-reasoning
challenge that `seqmaze` isolates. Where `seqmaze` asks "can the model infer
a path from a graph?", `goalchain` asks "can the model infer a path from its
memory of a spatial layout?"

---

## Scientific purpose

`goalchain` tests whether a model can use a learned memory state to solve a
goal-conditioned shortest-walk planning problem. The relevant EHP computation
is no longer:

```text
learn the map
```

It is:

```text
use the learned map-memory to infer the correct sequence toward a goal
```

Interpretation:

| Component              | Role in `goalchain`                                                   |
| ---------------------- | --------------------------------------------------------------------- |
| HPC / episodic memory  | provides remembered structure and observation-location bindings.      |
| PFC / reasoning module | infers the correct observation sequence and location targets.         |
| MEC-like structure     | provides location encoding for path integration and target selection. |
| LEC-like content       | provides observation identifiers used to ground the sequence.         |

The benchmark isolates whether the reasoning module can use memory to produce
the right goal-directed chain, without confounding the evaluation with the
model's ability to learn the layout in the first place.

### Why HRM architecture for PFC reasoning

The reasoning process requires tracking the global solution (the sequence of
target locations) in a high-level control loop ($z_H$) while evaluating
candidate next steps in a low-level loop ($z_L$). This dual-loop structure
is a natural fit for the HRM architecture: $z_H$ maintains the global plan
and $z_L$ evaluates candidate next steps. See
`jolicoeur-martineau_less_2025`.

---

## Input and output

### Adapter input (task-data → model)

The model receives the memory state, a goal cue, and its believed location.
The memory state is produced by the `goalchain` data-gen pipeline from a
pre-trained arena model and must correspond to the same layout.

| Variable         | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| $M$              | memory state with observation-location bindings and layout |
| $goal'$ (obs_id) | the final goal observation identifier (decoded goal cue)   |
| $g'$             | the model's current believed location ("where" state)      |

### Adapter output (model → evaluation)

The model must produce two aligned sequences: the observation chain and the
location chain needed to reach the goal via the shortest walk.

| Variable           | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| $x'[0:n]$ sequence | predicted sequence of observation identifiers to reach the goal |
| $g'[0:n]$ sequence | predicted sequence of location identifiers to reach the goal    |

### Example

```text
Initial location:
    g'_7

Goal cue:
    obs_id = 5

Memory state:
    learned structure + observation-location bindings for layout L_i

Target observation sequence:
    x'_0 → x'_1 → x'_2 → x'_3 → x'_4 → x'_5

Target location sequence:
    g'_0 → g'_1 → g'_2 → g'_3 → g'_4 → g'_5
```

---

## Corpus and data generation

Each `goalchain` sample consists of a layout-matched memory state ($M$), a
start location ($g'$), a goal observation cue ($goal'$), and precomputed
target sequences ($x'[0:n]$, $g'[0:n]$) representing the shortest walk.

### Data pipeline

1. A pre-trained arena model is run over a layout to produce the memory
   state ($M$) and latent representations.
2. A start location and goal observation are selected.
3. The shortest-walk observation and location sequences are computed offline
   from the layout graph.
4. The memory state, start location, goal cue, and target sequences are
   packaged as one sample.

### Build

```bash
# Requires a trained arena model checkpoint:
python scripts/data-gen/build-goalchain.py build-all \
    --arena-checkpoint artifacts/models/tem-v1-arena/run-000001
```

Output path: `data/processed/goalchain/<corpus>/v<version>/`

---

## Benchmark and evaluation

### GoalChain-Nav track

| Aspect            | Value                                       |
| ----------------- | ------------------------------------------- |
| Benchmark track   | GoalChain-Nav                               |
| Claim family      | `memory_conditioned_reasoning`              |
| Execution mode    | generation (autoregressive, teacher-forced) |
| Primary metric    | `observation_sequence_exact`                |
| Secondary metrics | `location_sequence_exact`,                  |
|                   | `next_observation_accuracy`,                |
|                   | `next_location_accuracy`,                   |
|                   | `path_length_regret`                        |

### Evaluation axes

| Axis                          | Description                                                    |
| ----------------------------- | -------------------------------------------------------------- |
| Observation-sequence accuracy | Can the model infer the correct chain of observations?         |
| Location-sequence accuracy    | Can the model select the correct target location at each step? |
| Path efficiency               | Does the predicted path match the shortest-walk length?        |

The observation-sequence evaluation is pure reasoning — it tests whether the
model can infer the correct observation chain from memory, not whether it can
recall the exact location of a specific observation.

### Score accumulation

Sequence-level exact match is the primary metric. Per-step accuracy metrics
(`next_observation_accuracy`, `next_location_accuracy`) are accumulated
additively across batches (sum correct / sum total) to avoid averaging errors.

---

## Open questions

1. **Latent ↔ decoded location autoencoder**: The model must translate between
   latent locations ($g[t]$) and decoded topology locations ($g'[t]$). Should
   arena training include an auxiliary autoencoder loss for this mapping, or
   should a separate encoder-decoder be trained post-hoc on arena-produced
   latent states?

2. **Pure reasoning pre-training**: The observation-sequence problem is a pure
   reasoning task. Needed separate pre-training task with e.g. `seqmaze` that
   isolates the observation-chain inference without spatial structure.

3. **Transition information in $p[t]$**: The latent conjunctive state ($p[t]$)
   may need to encode valid transitions between observations, not just the
   observation identifier. Can the HPC recall cue provide enough transition
   structure by itself, or does $p[t]$ need to carry explicit successor
   information?

4. **TEM vs HRM token design**: HRM MazeHard embeddings carry both "what" and
   "where" in a single token. TEM separates them into a two-hot observation
   vector and a separate location belief. How should TEM tokens be extended
   to carry the transition information that PFC-level reasoning requires?

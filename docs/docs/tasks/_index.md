# Benchmark Tasks

Tasks in the EHP-SN benchmark suite, ordered by increasing architectural
demand.

## Task list

| Task        | Document                     | Type                    | Core claim                                                         |
| ----------- | ---------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `arena`     | [arena.md](arena.md)         | structural learning     | acquires spatial representations from sequential experience        |
| `mazehard`  | (external)                   | spatial navigation      | navigates a fixed grid with blocked cells                          |
| `goaltrace` | [goaltrace.md](goaltrace.md) | prospective field (HRM) | produces a goal-conditioned prospective field from oracle inputs   |
| `routebind` | [routebind.md](routebind.md) | spatial route binding   | binds hidden obs-transition graph to new spatial layouts           |
| `prospect`  | [prospect.md](prospect.md)   | prospective field (EHP) | produces a goal-conditioned prospective field from episodic memory |

## Task relationships

### Task decomposition

| Task        | Model         | Task supplies                                               | Model must produce or retrieve                            |
| ----------- | ------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| `arena`     | TEM           | sensory observations and transitions through an environment | EC/HPC representations and episodic memory                |
| `mazehard`  | HRM           | grid layout, start, goal, walls                             | spatial route (token sequence)                            |
| `goaltrace` | HRM           | $g_t$, $x_{\text{goal}}$, $\mathbf{w}_t$                    | goal-conditioned prospective field $\mathbf{f}_t$         |
| `routebind` | HRM           | spatial layout, start, goal obs, hidden obs-transition DAG  | spatial trajectory + waypoint fields over 900 positions   |
| `prospect`  | EHP (TEM+HRM) | $o_{\text{goal}}$ + environmental experience                | memory-derived $g_t$, $\mathbf{r}_t$, then $\mathbf{f}_t$ |

```text
arena:     world → memory
mazehard:  grid → spatial route
goaltrace: memory-like evidence → prospective field
routebind: grid + hidden DAG → spatial route with semantic binding
prospect:  world memory + goal → prospective field
```

### `arena` ↔ `goaltrace`

`arena` trains TEM to learn structural representations ($g_t$, $p_t$) and
eventual relational retrieval. `goaltrace` receives approximations of those
signals as oracle inputs, so HRM can be trained in isolation. The two tasks
train complementary systems that `prospect` later integrates.

### `arena` ↔ `prospect`

`prospect` depends on `arena` for TEM pretraining. The TEM modules (LEC,
MEC, HPC, and memory store $M$) must be pretrained on structural exposure
before `prospect` training begins. During `prospect` training, TEM weights
may be frozen or fine-tuned depending on the training strategy.

During `prospect` data generation, a pretrained arena TEM model is run over
a layout to produce the episodic memory state ($M$) and latent representations
($g_t$, $p_t$). `prospect` then tests whether the model can use that memory
to solve goal-conditioned prospective-field problems.

### `goaltrace` ↔ `prospect`

`goaltrace` and `prospect` share the same output semantics (a goal-conditioned
prospective field) but differ in input provenance:

|                    | `goaltrace`                     | `prospect`                            |
| ------------------ | ------------------------------- | ------------------------------------- |
| Current location   | task-supplied $g_t$             | MEC-derived from experience           |
| Goal cue           | task-supplied $x_{\text{goal}}$ | LEC-encoded from $o_{\text{goal}}$    |
| Relational weights | task-supplied $\mathbf{w}_t$    | HPC-retrieved $\mathbf{r}_t$ from $M$ |
| Model              | HRM only                        | EHP (TEM + HRM)                       |
| Claim              | PFC computation                 | memory-guided PFC computation         |

An HRM pretrained on `goaltrace` (oracle weights) may transfer to `prospect`
(memory-derived evidence). This transfer is itself a scientific test of
whether the PFC computation generalizes across input quality.

`goaltrace` cannot support claims about memory retrieval, localization, or
episodic binding. Those claims belong to `prospect`.

### `goaltrace` ↔ `seqmaze`

`seqmaze` provides a novel graph per sample and requires full-path output.
`goaltrace` uses one fixed DAG learned parametrically and requires a distributed
field, not a discrete path. `seqmaze` tests novel-graph interpretation;
`goaltrace` tests parametric topology knowledge applied under changing local
conditions.

### `arena` ↔ `seqmaze`

`seqmaze` isolates pure transition-graph reasoning without spatial grounding.
Arena provides the spatial grounding that `seqmaze` deliberately removes.
The two tasks bookend the structural-learning-to-reasoning spectrum.

### Why `prospect` replaces `goalchain`

`goalchain` required an explicit ordered chain output
($v_0 \rightarrow v_1 \rightarrow \cdots \rightarrow v_g$). `prospect`
requires a distributed prospective field that:

- is anchored at the current location;
- activates goal-relevant prospective states;
- can represent multiple plausible branches;
- can be updated after movement;
- need not commit immediately to one complete path.

A path can emerge from the field over physical time, but the model is not
required to output the entire chain at once.

### Fixed topology contract

`goaltrace` and `prospect` v1 share a common fixed-topology contract:

```text
G = (V, E)
V = {obs_0, obs_1, ..., obs_{N-1}}
```

One fixed DAG per corpus; all samples share the same $(V, E)$. Observation
IDs have stable identities. Numerical order does not imply graph order.
Edges are directed and constant across all samples. The model learns the
topology parametrically from task supervision.

# `goaltrace` Benchmark Task

## Task identity and overview

Task name: `goaltrace`

Benchmark family: goal-conditioned prospective field prediction

| Symbol            | Surface | Description                                                             |
| ----------------- | ------- | ----------------------------------------------------------------------- |
| $g_t$             | yes     | current location latent (MEC-like; supplied by task)                    |
| $x_{\text{goal}}$ | yes     | goal observation identifier (supplied by task)                          |
| $w_{t,j}$         | yes     | relational weight from $g_t$ to candidate $j$ (supplied by task)        |
| $\mathbf{f}_t$    | yes     | goal-conditioned prospective firing field over $N$ nodes (model output) |
| $z_H$             | —       | HRM/PFC recurrent state (model-internal)                                |

_Surface_ symbols appear in the task input/output contract.
_Model-internal_ symbols (—) are emergent representations the model learns
but are not part of the task-level data contract.

Canonical package path:

```text
src/ehp_sn/tasks/goaltrace/
```

`goaltrace` is the **isolated HRM/PFC training task**. It trains HRM to transform
a current-location representation, a goal observation, and state-dependent
relational weights into a goal-conditioned prospective firing field over the
nodes of a fixed learned DAG.

The task isolates the PFC computation by providing — as oracle task inputs —
the signals that TEM/HPC would eventually supply in the integrated EHP
architecture. HRM does not retrieve a map, infer its current location, or
reconstruct episodic bindings. It learns only the transformation:

```text
current state + goal + current relational field → goal-directed prospective field
```

The core computation is:

```text
g_t, x_goal, w_t  ⟶  HRM  ⟶  f_t
```

where $\mathbf{f}_t \in [0,1]^N$ anchors the current location at $1$ and
assigns decreasing activation to other nodes according to their discounted
prospective relevance for reaching the goal.

The model does not receive a new graph in every sample. It learns one stable
DAG over a fixed set of observation IDs. Each sample varies the current
location, goal observation, and state-dependent weight vector.

---

## Fixed topology

One fixed DAG per corpus; all samples share the same $(V, E)$:

```text
G = (V, E)
V = {obs_0, obs_1, ..., obs_{N-1}}
```

Observation IDs have stable identities. **v1 uses a dense DAG** in which
every forward pair $(i, j)$ with $i < j$ is a directed edge (990 total edges
for $N = 45$). This is guaranteed by a Hamiltonian backbone chain
$0 \to 1 \to 2 \to \dots \to N-1$ plus the rank-space adjacency where
$\operatorname{adj}[i] = \{i+1, \dots, N-1\}$. Every node can reach every
later node; there are no stranded nodes.

The topology is not supplied in the input; the model learns it parametrically
from the field prediction loss.

---

## Scientific purpose

`goaltrace` tests whether HRM can:

1. learn a fixed directed topology parametrically from field supervision;
2. anchor computation on a current location $g_t$;
3. condition the prospective representation on a goal $x_{\text{goal}}$;
4. integrate continuous state-dependent relational weights $\mathbf{w}_t$;
5. use recurrent deliberation to refine the field;
6. produce a field that correctly identifies which nodes lie on viable
   goal-reaching continuations.

The relevant computation is:

```text
current anchor + goal cue + relational evidence
    → recurrent deliberation
    → goal-conditioned prospective field
```

Interpretation:

| Component       | Role in `goaltrace`                                                     |
| --------------- | ----------------------------------------------------------------------- |
| HRM / PFC       | constructs queries, integrates evidence, refines prospective field      |
| Token structure | encodes observation identity, current/goal flags, and relational weight |
| Deliberation    | recurrent HRM processing; $g_t$ remains fixed throughout                |
| Parametric DAG  | learned transition structure stored in model parameters                 |

The central question is:

> Can HRM use a learned fixed topology and state-dependent relational
> weights to produce a goal-conditioned prospective field that correctly
> identifies which observations lie on viable paths to the goal?

`goaltrace` does **not** test:

- hippocampal episodic retrieval;
- MEC-based self-localization;
- causal use of memory;
- interactive navigation or action selection.

Those capabilities are tested by downstream tasks that consume the
prospective field.

---

## Input and output

### Execution mode

`goaltrace` uses **single-step field prediction with optional recurrent
deliberation**. The model does not navigate, select actions, or interact with
a runtime loop. There is no physical movement, no episode horizon, and no
state transition.

1. The adapter packs $g_t$, $x_{\text{goal}}$, and $\mathbf{w}_t$ into a
   fixed-length schema-token sequence.
2. HRM deliberates (recurrent $z_H$ cycles; ACT halting supported but not
   required). $g_t$ is fixed throughout deliberation.
3. A decoder reads all node states and produces $\hat{\mathbf{f}}_t \in [0,1]^N$
   via sigmoid activation per node.
4. The field is compared against the target field for supervision.

### Tensor shape contract

| Parameter | Description                     | Owner                |
| --------- | ------------------------------- | -------------------- |
| $N$       | number of fixed observation IDs | corpus + task config |
| $D$       | model dimension                 | model config         |

### Schema-token layout

```text
schema_tokens: (B, N, D)

positions [0 : N):
  observation-node embeddings for the fixed graph
```

One slot per observation ID. The graph topology is not supplied in the input;
it is learned parametrically from the field prediction loss.

### Adapter input (task-data → model)

| Field            | Shape    | Description                                       |
| ---------------- | -------- | ------------------------------------------------- |
| `observation_id` | `(B, N)` | stable observation identity                       |
| `weight`         | `(B, N)` | relational weight from $g_t$ to candidate $j$     |
| `current_flag`   | `(B, N)` | `True` for the current location $g_t$             |
| `goal_flag`      | `(B, N)` | `True` for the goal observation $x_{\text{goal}}$ |
| `node_mask`      | `(B, N)` | `True` for valid observation slots                |

v1 has **no padding slots** ($N_{\text{pad}} = N_{\text{actual}} = 45$).
The `padding_obs_id` adapter setting is not needed.

Two additional channels are stored in the corpus for evaluation diagnostics
but are **excluded from the model input surface**:

| Field               | Shape       | Description                            |
| ------------------- | ----------- | -------------------------------------- |
| `successor_indices` | `(B, N, K)` | successor node indices per observation |
| `successor_mask`    | `(B, N, K)` | validity mask for successor slots      |

where $K = N-1$ (full forward cone; every later node is a successor in the
dense DAG).

For observation $j$, the adapter constructs:

```text
e_{t,j} =
    E_obs(obs_id_j)
  + f_weight(w_{t,j})
  + E_current([j = g_t])
  + E_goal([j = x_goal])
```

where $f_{weight}$ is a linear projection or small MLP. The weight remains a
continuous scalar; attention receives only the resulting continuous embedding.

### Relational weight

The model receives a universal relational weight per candidate:

```text
w_{t,j} ∈ [0, 1]
```

with a stable orientation:

```text
0.0 = unavailable / blocked / minimally supported
1.0 = maximally available / certain / most supported
```

The adapter always performs the same embedding; no semantics-specific
transformation is required inside the model. The meaning of the weight
is declared in the corpus manifest and used only by the oracle.

### Oracle semantics

The corpus manifest declares how the universal weight is interpreted:

| Manifest key       | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| `weight_range`     | always `[0.0, 1.0]`                                 |
| `oracle_semantics` | `"reliability"`, `"linear_cost"`, or `"preference"` |

The oracle converts $w_e$ into an edge cost $c_s(w_e)$ according to the
declared semantics, then selects the optimal path:

```text
π_t* = argmin_π Σ_{e∈π} c_s(w_e)
```

| Semantics     | Edge cost $c_s(w)$       | Path objective                           |
| ------------- | ------------------------ | ---------------------------------------- |
| `reliability` | $-\log(w + \varepsilon)$ | min $\sum -\log w_e$ (≡ max $\prod w_e$) |
| `linear_cost` | $1 - w$                  | min $\sum (1 - w_e)$                     |
| `preference`  | $-w + \lambda$           | min $\sum (-w_e + \lambda)$              |

`reliability` unifies probability and log-cost. Use `linear_cost` or
`preference` when path length should interact non-trivially with the
objective. For `preference`, $\lambda \ge 0$ is a per-step penalty
(`preference_step_penalty` in the manifest).

The oracle does **not** embed the weight into the target field values.
Path selection and field representation are kept separate.

### Adapter output (model → evaluation)

```text
ObsNavStepOutput:
  firing_field:  FloatTensor[B, N]  ∈ [0, 1] via sigmoid
```

The field is a multi-label continuous representation — each component
$\hat{f}_t(j)$ is an independent activation. It is not a categorical
distribution; no softmax is applied. The field satisfies:

```text
f_t(i_t) = 1                           current location at maximum
f_t(j) ∈ [0, 1) for j ≠ i_t            decays over prospective states
f_t(j) = 0 for nodes off viable goal-reaching paths
```

### Target

```text
ObsNavTargets:
  target_field:  FloatTensor[B, N]  ∈ [0, 1]
```

The oracle constructs the target field in two clean stages:

```text
w_t  →  (oracle semantics)  →  π_t*  →  (field encoding)  →  f_t*
```

**Stage 1 — Path selection.** The oracle converts weights to edge costs
via the declared semantics and selects the optimal path $\pi_t^*$ from
$i_t$ to the goal.

**Stage 2 — Field encoding.** Once the path is selected, the field is
constructed from pure discounted distance along that path:

$$
f_t^*(i) =
\begin{cases}
\gamma^{d_{\pi_t^*}(i_t, i)}, & i \in \pi_t^* \\[4pt]
0, & \text{otherwise}
\end{cases}
$$

where $d_{\pi_t^*}(i_t, i)$ is the number of edges from the current node
$i_t$ to node $i$ along the optimal path, and $\gamma \in (0,1)$ is the
decay factor.

The weight does **not** multiply the field values. The weight determines
_which route is intended_; the decay rule determines _how that route is
represented_. This keeps route quality and trajectory position disentangled
in the output.

Nodes not on the optimal path receive zero, even if they are reachable
with high direct weight.

### Example

```text
Dense DAG (all forward pairs are edges, N=5 for illustration):
  obs_0 → obs_1, obs_2, obs_3, obs_4
  obs_1 → obs_2, obs_3, obs_4
  obs_2 → obs_3, obs_4
  obs_3 → obs_4
  obs_4  (terminal)

Sample:
  current: obs_0
  goal:    obs_4
  oracle semantics: reliability
  decay: γ = 0.8

Weights are derived from a hidden spatial substrate (see Generation pipeline).
For this example, suppose the geometric kernel assigns these edge costs:
  obs_0 → obs_1:  w=0.90  →  c = -log(0.90) = 0.105
  obs_0 → obs_2:  w=0.10  →  c = 2.303
  obs_0 → obs_3:  w=0.70  →  c = 0.357
  obs_0 → obs_4:  w=0.05  →  c = 2.996
  obs_1 → obs_2:  w=0.85  →  c = 0.163
  obs_1 → obs_3:  w=0.60  →  c = 0.511
  obs_1 → obs_4:  w=0.30  →  c = 1.204
  obs_2 → obs_3:  w=0.95  →  c = 0.051
  obs_2 → obs_4:  w=0.80  →  c = 0.223
  obs_3 → obs_4:  w=0.40  →  c = 0.916

Oracle: Dijkstra on the dense graph finds the minimum-cost path.
  Optimal route: obs_0 → obs_1 → obs_2 → obs_3 → obs_4  (Σc = 1.235)
  (The direct hop obs_0→obs_3→obs_4 would cost 1.273 — slightly worse.)

Target field:
  obs_0  = 1.000   (γ^0, current location)
  obs_1  = 0.800   (γ^1)
  obs_2  = 0.640   (γ^2)
  obs_3  = 0.512   (γ^3)
  obs_4  = 0.410   (γ^4)
```

Note: in the dense DAG every forward pair is reachable, so path selection is
purely determined by the geometric weight costs. Nodes with high direct weight
may be skipped if a multi-hop route has lower total cost.

---

## Corpus and data generation

Each corpus is built around one fixed dense DAG shared across all samples.
Samples vary the current observation, goal observation, and weight vector.
The weight vector is a row of a static relational weight matrix $W$ derived
from a hidden geometric substrate (anchors on a $20 \times 30$ grid, BFS
distances, truncated exponential kernel with $\tau = 8.0$, masked by the DAG
adjacency). v1 has no padding slots ($N_{\text{pad}} = N_{\text{actual}} = 45$).

### Frozen v1 channels

```text
observation_id              — (B, N) int32, stable observation identity
weight                      — (B, N) float32, relational weight from current
                              to candidate j, ∈ [0, 1]
current_flag                — (B, N) bool,  True at current location g_t
goal_flag                   — (B, N) bool,  True at goal observation
node_mask                   — (B, N) bool,  True for valid observation slots
target_field                — (B, N) float32, target prospective field ∈ [0, 1]
successor_indices           — (B, N, K) int32, successor node indices
successor_mask              — (B, N, K) bool,  validity mask for successor slots
```

$K = N - 1$ (full forward cone; every later node is a successor in the dense
DAG). `successor_indices` / `successor_mask` are persisted for evaluation
diagnostics but excluded from the model input surface.

### Data generation

```bash
# Requires dagflow substrate first:
python scripts/data-gen/build-dagflow.py build --preset branching --version 1
python scripts/data-gen/build-goaltrace.py build \
    --layout-root data/interim/dagflow/branching/v1

# All flags have sensible defaults.  Explicit:
python scripts/data-gen/build-goaltrace.py build \
    --layout-root data/interim/dagflow/branching/v1 \
    --static-weights --min-optimality-margin 0.0 \
    --distance-tau 8.0 --distance-max 0 \
    --grid-width 20 --grid-height 30 \
    --n-train 500 --n-val 250 --n-test 240 \
    --version 1
```

Output path: `data/processed/goaltrace/<corpus>/v<version>/`

### Train/test split

The split is over **(current, goal) pairs** (500 / 250 / 240), not over graph
structures. All splits share the same fixed DAG and static weight matrix $W$.
Pairs are assigned by identity so a specific (current, goal) pair never appears
in more than one split.

---

## Benchmark and evaluation

### Goaltrace-Field track

| Aspect            | Value                                                        |
| ----------------- | ------------------------------------------------------------ |
| Benchmark track   | Goaltrace-Field                                              |
| Claim family      | `goal_conditioned_prospective_field`                         |
| Execution mode    | single-step field prediction after recurrent deliberation    |
| Primary metric    | `field_mse`                                                  |
| Secondary metrics | `current_accuracy`, `successor_accuracy`, `goal_activation`, |
|                   | `off_path_suppression`, `field_decay_correlation`            |

### Primary metric: `field_mse`

Mean squared error between predicted and target firing field, averaged over
all $N$ observation slots and all samples:

```text
field_mse = (1 / (B·N)) Σ_i Σ_j (f̂_t^{(i)}(j) − f_t^{*(i)}(j))²
```

### Secondary metrics

| Metric                    | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| `current_accuracy`        | how close $\hat{f}_t(i_t)$ is to $1.0$                                  |
| `successor_accuracy`      | MSE restricted to direct successors on the optimal path                 |
| `goal_activation`         | mean predicted firing at the goal observation                           |
| `off_path_suppression`    | mean predicted firing at observations not on viable goal-reaching paths |
| `field_decay_correlation` | Pearson $r$ between predicted and target activation profile along path  |

### Score accumulation

`field_mse` and component metrics are mean-aggregated over samples. Subfield
metrics (successor, off-path) use masked aggregation over the relevant node
subsets.

---

## Training supervision

The primary loss is mean squared error over the firing field:

```text
L_field = (1/N) Σ_j (f̂_t(j) − f_t^*(j))²
```

The target is dense: all $N$ observations receive a target value (nonzero for
observations on viable goal-reaching paths, zero otherwise). The current
location is supervised toward $1.0$.

Binary cross-entropy per node is an alternative:

```text
L_field = (1/N) Σ_j BCE(f̂_t(j), f_t^*(j))
```

The loss does not require action labels, path sequences, or EOS tokens. The
model is supervised directly on the quality of its prospective representation.

---

## Open questions

1. **Deliberation depth**: How many internal iterations $K$ are needed for
   field convergence? Can a single forward pass suffice, or does the model
   benefit from explicit recurrent steps? ACT halting may reveal whether
   harder (current, goal) pairs require more deliberation.

2. **Field decay factor**: Should $\gamma$ be a fixed corpus constant, or
   should the model learn to infer the appropriate decay from the weight
   distribution? A learned decay would make the field adaptive to cost
   magnitude.

3. **Multiple optimal paths**: When multiple equally optimal paths exist,
   how should the target field be defined? Options include merging
   activations across all optimal paths (sum or max of discounted
   occupancies) or restricting generation to unique solutions.

4. **Forward–backward decomposition**: The target field combines forward
   accessibility and backward goal relevance. Should these be supervised as
   separate auxiliary outputs, or is the combined field sufficient?

5. **Scaling graph size**: How does field accuracy degrade as $N$ grows?
   The field has $N$ outputs; the DAG topology has $O(N \cdot d)$ edges
   to internalize. Is there a phase transition where parametric graph
   memory saturates?

6. **Recurrence necessity**: Does HRM's recurrence provide measurable benefit
   over a single-pass baseline for this task? A non-recurrent control (K=1)
   would test whether iterative refinement is needed for field computation.

7. **Input quality robustness**: Does the PFC computation generalize when
   oracle-quality weights are replaced by noisier, memory-derived relational
   evidence? This tests whether the learned transformation tolerates
   degraded input signals.

8. **Goal-conditioning mechanism**: How does the goal cue modulate the
   field? Is it sufficient to provide the goal as a flag on one token, or
   does the model need a separate goal-query pathway?

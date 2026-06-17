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
src/ehc_sn/tasks/goaltrace/
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

Observation IDs have stable identities. Numerical order does not imply graph
order. Edges are directed and constant across all samples. The topology is
not supplied in the input; the model learns it parametrically from the field
prediction loss.

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
| `node_mask`      | `(B, N)` | `True` for valid nodes (masks padding)            |

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
Fixed DAG:
  obs_7  → obs_2
  obs_7  → obs_11
  obs_2  → obs_5
  obs_11 → obs_5
  obs_5  → obs_3

Sample:
  current: obs_7  (i_t = 7)
  goal:    obs_3
  oracle semantics: reliability
  decay: γ = 0.8

Oracle: reliability uses c_e = −log(w_e).
  obs_7 → obs_2:  w=0.3  →  c=1.20
  obs_7 → obs_11: w=0.8  →  c=0.22
  obs_2 → obs_5:  w=0.4  →  c=0.92
  obs_11 → obs_5: w=0.9  →  c=0.11
  obs_5 → obs_3:  w=0.2  →  c=1.61

Optimal route (min Σ c): obs_7 → obs_2 → obs_5 → obs_3  (Σc = 3.73)
  (obs_7 → obs_11 → obs_5 → obs_3 would be cheaper at Σc = 1.94,
   but obs_11 cannot reach the goal in this DAG)

Target field:
  obs_7  = 1.000   (γ^0, current location)
  obs_2  = 0.800   (γ^1)
  obs_5  = 0.640   (γ^2)
  obs_3  = 0.512   (γ^3)
  obs_11 = 0.000   (off optimal path, despite high direct weight)
```

Note: `obs_11` receives zero target activation because it does not lie on a
viable path to the goal, even though it is directly reachable with high weight.
The field is goal-conditioned, not merely proximity-based.

---

## Corpus and data generation

Each corpus is built around one fixed DAG shared across all samples.
Samples vary the current observation, goal observation, and weight vector.

### Graph topology

v1 uses **DAGs** exclusively. The DAG is fixed per corpus. Observation IDs
are stable and assigned independently of topological order.

### Key invariants

- One fixed DAG per corpus; all samples share the same $(V, E)$.
- Observation IDs are stable and do not encode graph order.
- Only the current-state weight vector is exposed (not the full $N \times N$
  weight matrix).
- Weight values are in $[0, 1]$ with the orientation $0$ = blocked,
  $1$ = maximally supported, and sampled continuously to prevent
  categorical-threshold memorization.
- The optimal path is unique under the declared oracle semantics.
- The target field encodes only the selected path via pure decay; weights
  do not multiply field values.

### Corpus manifest

```json
{
  "task": "goaltrace",
  "corpus": "default",
  "version": 1,
  "n_observations": 32,
  "max_out_degree": 4,
  "weight_range": [0.0, 1.0],
  "oracle_semantics": "reliability",
  "field_decay": 0.8
}
```

`preference_step_penalty` (default `0.0`) is required when
`oracle_semantics` is `"preference"`.

### Profile coupling

| Checkpoint         | Validation                                                |
| ------------------ | --------------------------------------------------------- |
| Training startup   | `model.n_observations == manifest.n_observations`         |
| Evaluation startup | `checkpoint.n_observations == eval_corpus.n_observations` |

### Data generation

```bash
python scripts/data-gen/build-goaltrace.py build-all \
    --corpus default --version 1 \
    --n-observations 32 --max-out-degree 4 \
    --oracle-semantics reliability --field-decay 0.8 \
    --n-train 4000 --n-val 500 --n-test 500 \
    --seed 42
```

Output path: `data/processed/goaltrace/<corpus>/v<version>/`

### Train/test split

The split is over **(current, goal, weight) configurations**, not over graph
structures. All splits share the same fixed DAG. The strongest generalization
split holds out specific (current, goal) pairs entirely.

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

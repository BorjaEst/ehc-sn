# `prospect` Benchmark Task

## Task identity and overview

Task name: `prospect`

Benchmark family: memory-derived goal-conditioned prospective field prediction

| Symbol            | Surface | Description                                                             |
| ----------------- | ------- | ----------------------------------------------------------------------- |
| $o_{\text{goal}}$ | yes     | goal observation as sensory cue (task boundary input)                   |
| $x_{\text{goal}}$ | —       | LEC-encoded goal sensory state (model-internal)                         |
| $g_t$             | —       | MEC current location state (model-internal, derived from experience)    |
| $p_t$             | —       | HPC conjunctive state (model-internal)                                  |
| $\mathbf{r}_t$    | —       | relational evidence retrieved from HPC memory (model-internal)          |
| $\mathbf{f}_t$    | yes     | goal-conditioned prospective firing field over $N$ nodes (model output) |
| $z_H$             | —       | HRM/PFC recurrent state (model-internal)                                |
| $M$               | —       | episodic memory store (TEM-derived, model-internal)                     |

_Surface_ symbols appear in the task input/output contract.
_Model-internal_ symbols (—) are emergent representations the model learns
but are not part of the task-level data contract.

Canonical package path:

```text
src/ehc_sn/tasks/prospect/
```

`prospect` is the **integrated EHP training task**. It tests whether the
combined TEM (EC/HPC memory) and HRM (PFC deliberation) system can produce
a goal-conditioned prospective firing field when the task provides only a
sensory goal cue and ongoing environmental experience — not oracle positions
or weights.

The task requires a distributed prospective field that can represent
multiple viable branches and need not commit to a single path — in contrast
to tasks that require an explicit ordered chain output
($v_0 \rightarrow v_1 \rightarrow \cdots \rightarrow v_g$).

The integrated computation is:

```text
o_goal + current EC/HPC state
    → LEC encodes x_goal
    → HPC retrieves relational evidence r_t from M
    → MEC provides current location g_t
    → HRM deliberates over (g_t, x_goal, r_t)
    → produce prospective field f_t
```

Crucially, the task does **not** supply $g_t$ or $\mathbf{w}_t$ as oracle
inputs. These must emerge from TEM/HPC memory. This is the integration test:
can memory-derived relational evidence substitute for oracle-quality
position and weight signals?

---

## Scientific purpose

`prospect` tests whether EHP can:

1. encode a sensory goal cue via LEC into $x_{\text{goal}}$;
2. maintain a current location state $g_t$ via MEC from ongoing experience;
3. retrieve relational evidence $\mathbf{r}_t$ from HPC memory $M$ given
   $g_t$ and $x_{\text{goal}}$;
4. use HRM deliberation to transform $(g_t, x_{\text{goal}}, \mathbf{r}_t)$
   into a goal-conditioned prospective field $\mathbf{f}_t$;
5. demonstrate that $\mathbf{r}_t$ causally shapes $\mathbf{f}_t$ (ablating
   or corrupting retrieval should degrade the field).

The relevant computation is:

```text
sensory goal + environmental experience
    → memory encoding + retrieval
    → recurrent PFC deliberation over memory-derived evidence
    → goal-conditioned prospective field
```

Interpretation:

| Component | Role in `prospect`                                                                |
| --------- | --------------------------------------------------------------------------------- |
| LEC       | encodes sensory goal observation into $x_{\text{goal}}$                           |
| MEC       | tracks current structural location $g_t$ from movement history                    |
| HPC       | stores episodic bindings in $M$; retrieves $\mathbf{r}_t$ given cues              |
| HRM / PFC | deliberates over $(g_t, x_{\text{goal}}, \mathbf{r}_t)$ to produce $\mathbf{f}_t$ |

The central question is:

> Can EHP derive a goal-conditioned prospective field from episodic memory,
> without oracle access to the current location or transition weights?

Success demonstrates that memory causally guides deliberation — not merely
that a feed-forward network can compute distances on a memorized graph.

---

## Input and output

### Execution mode

`prospect` uses **single-step field prediction after memory retrieval and
recurrent deliberation**. The model does not navigate or select actions.

1. The task provides $o_{\text{goal}}$ (sensory goal observation) and the
   current environmental state (observation, action history).
2. TEM processes the inputs: LEC encodes $o_{\text{goal}} \rightarrow x_{\text{goal}}$,
   MEC tracks $g_t$, HPC retrieves $\mathbf{r}_t$ from $M$.
3. HRM deliberates over $(g_t, x_{\text{goal}}, \mathbf{r}_t)$ for $K$
   recurrent iterations. $g_t$ is fixed throughout deliberation.
4. A decoder reads HRM node states and produces $\hat{\mathbf{f}}_t \in [0,1]^N$
   via sigmoid activation per node.
5. The field is compared against the target field for supervision.

### Task boundary contract

The task provides:

| Field               | Description                                          |
| ------------------- | ---------------------------------------------------- |
| $o_{\text{goal}}$   | goal observation as sensory identifier               |
| environmental state | current observation, action history, episode context |

The task does **not** provide:

| Withheld                | Must be derived by                                       |
| ----------------------- | -------------------------------------------------------- |
| $g_t$                   | MEC from movement history and structural knowledge       |
| $x_{\text{goal}}$       | LEC from $o_{\text{goal}}$                               |
| $\mathbf{r}_t$          | HPC retrieval from $M$ given $g_t$ and $x_{\text{goal}}$ |
| $\mathbf{w}_t$ (oracle) | not available; replaced by memory-derived $\mathbf{r}_t$ |

### Adapter input (task-data → model)

The adapter provides the environmental context needed for memory retrieval
and current-state inference. The exact fields depend on the TEM architecture
and training strategy (frozen vs fine-tuned).

### Adapter output (model → evaluation)

```text
GoalFieldStepOutput:
  firing_field:  FloatTensor[B, N]  ∈ [0, 1] via sigmoid
```

The output is a continuous firing field over $N$ nodes. The field satisfies:

```text
f_t(i_t) = 1                           current location at maximum
f_t(j) ∈ [0, 1) for j ≠ i_t            decays over prospective states
f_t(j) = 0 for nodes off viable goal-reaching paths
```

### Target

```text
GoalFieldTargets:
  target_field:  FloatTensor[B, N]  ∈ [0, 1]
```

The target is the discounted prospective relevance from the current location
toward the goal, restricted to nodes on viable goal-reaching continuations
(see `goaltrace.md` for the full target formula).

---

## Training strategy

### Pretraining requirements

`prospect` requires pretrained components:

| Component                | Pretrained on                      | Frozen during `prospect`?      |
| ------------------------ | ---------------------------------- | ------------------------------ |
| TEM (LEC, MEC, HPC, $M$) | structural exposure (replay)       | Frozen (v1) or fine-tuned (v2) |
| HRM                      | oracle field prediction (optional) | Trainable                      |

The recommended v1 strategy is **frozen TEM + trainable HRM**:

1. Pretrain TEM on structural exposure.
2. Optionally pretrain HRM on oracle-quality field prediction.
3. Train on `prospect` with TEM frozen, HRM trainable.

This tests whether HRM can adapt from oracle-quality relational evidence to
memory-derived relational evidence without retraining TEM.

### Loss

The primary loss is mean squared error over the firing field:

```text
L_field = (1/N) Σ_j (f̂_t(j) − f_t^*(j))²
```

### Causal memory test

To verify that $\mathbf{r}_t$ causally shapes $\mathbf{f}_t$:

- Ablate $\mathbf{r}_t$ (set to zero or shuffle) and measure field degradation.
- If the field quality does not degrade, HRM is solving the task through
  parametric shortcuts rather than using memory-derived evidence.

---

## Corpus and data generation

`prospect` corpora require a pretrained TEM checkpoint from structural
exposure training. The data generation pipeline runs TEM over layouts to
produce the memory state $M$ and latent representations.

### Data pipeline

1. A pretrained TEM model is run over a layout to produce $M$, $g_t$,
   and $p_t$.
2. Start locations and goal observations are sampled.
3. The oracle computes the target field $\mathbf{f}_t^*$ from the optimal
   remaining path.
4. The environmental state, $o_{\text{goal}}$, $M$, and target field are
   packaged as one sample.

### Graph topology

v1 uses fixed DAGs. The DAG is fixed per corpus. Observation IDs are stable.
The topology class matches the isolated field-prediction task for
cross-task parity.

### Data generation

```bash
python scripts/data-gen/build-prospect.py build-all \
    --corpus default --version 1 \
    --arena-checkpoint checkpoints/arena/tem-v1-weights-only.pt \
    --n-observations 32 --max-out-degree 4 \
    --oracle-semantics reliability --field-decay 0.8 \
    --n-train 4000 --n-val 500 --n-test 500 \
    --seed 42
```

Output path: `data/processed/prospect/<corpus>/v<version>/`

---

## Benchmark and evaluation

### Prospect-Mem track

| Aspect            | Value                                                                          |
| ----------------- | ------------------------------------------------------------------------------ |
| Benchmark track   | Prospect-Mem                                                                   |
| Claim family      | `memory_derived_prospective_field`                                             |
| Execution mode    | single-step field prediction after memory retrieval + deliberation             |
| Primary metric    | `field_mse`                                                                    |
| Secondary metrics | `current_accuracy`, `successor_accuracy`, `goal_activation`,                   |
|                   | `off_path_suppression`, `field_decay_correlation`                              |
| Diagnostic metric | `retrieval_ablation_delta` (field_mse increase when $\mathbf{r}_t$ is ablated) |

### Primary metric: `field_mse`

Mean squared error between predicted and target field, as in the isolated
field-prediction task.

### Diagnostic metric: `retrieval_ablation_delta`

The increase in `field_mse` when $\mathbf{r}_t$ is zeroed or shuffled.
A large delta indicates causal reliance on memory retrieval. A near-zero
delta indicates the model ignores HPC output and uses parametric shortcuts.

### Score accumulation

`field_mse` and component metrics are mean-aggregated over samples. The
ablation delta is computed as `field_mse(ablated) − field_mse(clean)`.

---

## Open questions

1. **Frozen vs fine-tuned TEM**: Does freezing TEM preserve structurally
   learned representations, or does fine-tuning cause representational
   collapse (e.g., MEC learning to output oracle-like $g_t$ directly)?

2. **HRM transfer from oracle pretraining**: Does an HRM pretrained on
   oracle-quality weights transfer to memory-derived evidence without
   retraining? If not, how much integrated training is needed to adapt?

3. **Form of $\mathbf{r}_t$**: What shape and semantics should the retrieved
   relational evidence have? A per-node vector (matching oracle weight
   format) enables direct transfer; a richer representation may improve
   field quality at the cost of compatibility.

4. **Causal sufficiency of $\mathbf{r}_t$**: Is memory-derived $\mathbf{r}_t$
   sufficient to replace oracle-quality weights, or does HRM need additional
   structural information from TEM (e.g., $p_t$, successor embeddings)?

5. **Multiple retrievals per deliberation step**: Should HRM query HPC
   multiple times during deliberation ($\mathbf{r}_t^{(k)}$ varies with $k$),
   or is a single retrieval before deliberation sufficient?

6. **Scaling to spatial layouts**: The initial v1 uses abstract DAGs for
   cross-task parity. When should the task transition to spatial layouts
   (dungeon graphs, openfield grids) where MEC spatial representations
   provide genuine localization?

7. **Goal cue format**: Should $o_{\text{goal}}$ be a raw observation ID
   (requiring LEC encoding) or a richer cue (image, description) that
   exercises LEC's sensory encoding capabilities?

8. **Oracle baseline comparison**: What is the performance gap between
   memory-derived $\mathbf{r}_t$ and an oracle-weight baseline on the same
   DAG? This gap quantifies the cost of replacing oracle evidence with
   memory retrieval.

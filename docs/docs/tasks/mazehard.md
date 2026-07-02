# `mazehard` Benchmark Task

## Identity

| Property      | Value                                          |
| ------------- | ---------------------------------------------- |
| Task name     | `mazehard`                                     |
| Benchmark     | dense spatial classification over maze grids   |
| Package path  | `src/ehp_sn/tasks/mazehard/`                   |
| CLI script    | `scripts/data-gen/build-mazehard.py`           |
| Output path   | `data/processed/mazehard/<corpus>/v<version>/` |
| Dataset class | `task_corpus`                                  |

## Purpose and ownership

MazeHard is a fully observed, supervised, fixed-grid transformation task:
$f_\theta : \{0,\dots,V-1\}^{900} \to \{0,\dots,V-1\}^{900}$. The
model receives a complete maze layout (walls, empty cells, start, goal) and
must produce the reference shortest path from start to goal as a parallel
spatial classification over every cell.

MazeHard is **not** an RL environment, an online navigation episode, an
action-sequence prediction task, or a memory-retrieval task. It is a dense,
static, fully supervised spatial reasoning benchmark.

**Boundary**: MazeHard owns the classification protocol, spatial vocabulary,
and evaluation semantics. It does **not** own the maze generation (that is
maze-nd), the model architecture, or the deliberation/control strategy.
Token coercion from spatial channels is task-owned
(`runtime.coerce_maze_hard_batch`); the model receives only `(input_ids,
labels)` pairs.

## Semantic model

### Underlying problem

Convert the maze into an undirected graph $G = (V, E)$ where each non-wall
cell is a vertex and edges connect 4-adjacent traversable cells with unit
cost. Let $s$ be the start cell and $g$ the goal. The target route is:

$$P^* = \arg\min_{P: s \leadsto g} |P|$$

The model is not given the BFS frontier, predecessor table, distance map,
or intermediate search states. It must learn an internal computation that
maps the raw maze directly to the final route.

### Why the task is difficult

A cell's correct label may depend on global connectivity across the entire
maze — a corridor near the start may lead into a dead end hundreds of steps
away. Local pattern recognition is insufficient; the model must propagate
information over potentially long spatial distances, distinguish viable
routes from dead ends, and select the optimal path.

### Spatial-to-token vocabulary

Each maze grid of `(H, W)` cells is flattened to a 1-D sequence of
`S = H · W` positions. Each position is assigned a token ID from the
MazeHard semantic vocabulary:

| Token ID | Symbol  | Meaning                                   | Appears in `input_ids` | Appears in `labels` |
| -------- | ------- | ----------------------------------------- | ---------------------- | ------------------- |
| 0        | `PAD`   | Padding (not used; all S positions valid) | No                     | No                  |
| 1        | `WALL`  | Impassable wall cell                      | Yes                    | Yes (ignored)       |
| 2        | `EMPTY` | Passable empty cell                       | Yes                    | Yes                 |
| 3        | `START` | Start cell                                | Yes                    | Yes                 |
| 4        | `GOAL`  | Goal cell                                 | Yes                    | Yes                 |
| 5        | `PATH`  | Solution path overlay                     | **No**                 | **Yes**             |

The `PATH` token (5) is the critical distinction: it appears only in
`labels`, overlaid onto cells that lie on the solution path. The model
receives `input_ids` without path information and must predict `PATH`
tokens at the correct positions.

### Token coercion

At runtime, `coerce_maze_hard_batch` converts raw spatial channels into
the canonical token pair:

```text
channels → grid → input_ids (spatial vocabulary)
channels → solution_mask → labels (spatial vocabulary + PATH overlay)
```

- `input_ids`: topology-derived base tokens (`WALL`, `EMPTY`, `START`, `GOAL`).
- `labels`: identical to `input_ids`, then `PATH` is written over cells
  where `solution > 0`.

### Ignore label

`MAZE_HARD_IGNORE_LABEL_ID = -100` (diverges from upstream HRM convention of
`0`). Only non-ignored positions contribute to the loss.

## Parent requirements

MazeHard requires a maze-nd shared substrate with `artifact_schema_version = 1`
providing all five substrate channels:

| Channel      | Required | Purpose                                            |
| ------------ | -------- | -------------------------------------------------- |
| `topology`   | Yes      | Wall vs passable cell mask.                        |
| `mask_valid` | No\*     | Largest connected component (not used at runtime). |
| `start`      | Yes      | Start cell marker for token overlay.               |
| `goals`      | Yes      | Goal cell marker for token overlay.                |
| `solution`   | Yes      | Solution path for `PATH` label overlay.            |

\*`mask_valid` is carried in the corpus for provenance but not consumed by
the token coercion path.

The parent substrate must have homogeneous `(H, W)` extent across all samples.
The task corpus carries the parent channels verbatim — no transformation is
applied at corpus build time. Tokenization happens at runtime.

## Output artifact

### Corpus channels

The task corpus stores the same five spatial channels as the parent substrate,
sampled deterministically from the parent's splits:

| Channel      | Dtype | Shape  | Description                    |
| ------------ | ----- | ------ | ------------------------------ |
| `topology`   | bool  | (H, W) | Wall mask.                     |
| `mask_valid` | bool  | (H, W) | Largest 4-connected component. |
| `start`      | bool  | (H, W) | Single start cell.             |
| `goals`      | bool  | (H, W) | Goal cell markers.             |
| `solution`   | int32 | (H, W) | Integer-encoded solution path. |

### Runtime batch

After coercion via `coerce_maze_hard_batch`, each sample becomes:

| Field       | Shape | Dtype | Description                                        |
| ----------- | ----- | ----- | -------------------------------------------------- |
| `input_ids` | (S,)  | int64 | Spatial vocabulary IDs (WALL/EMPTY/START/GOAL).    |
| `labels`    | (S,)  | int64 | Same as `input_ids` with PATH overlay on solution. |

Where `S = H · W`. Stacked batches use shape `(B, S)`.

### Adapter input (task-data → model)

| Field       | Shape    | Description                                     |
| ----------- | -------- | ----------------------------------------------- |
| `input_ids` | `(B, S)` | Token IDs for the maze layout (no path tokens). |

### Adapter output (model → evaluation)

| Field         | Shape       | Description                         |
| ------------- | ----------- | ----------------------------------- |
| `task_logits` | `(B, S, V)` | Logits over vocabulary of size `V`. |

Where `V = 6` (PAD, WALL, EMPTY, START, GOAL, PATH).

### Targets

| Field    | Shape    | Description                                                 |
| -------- | -------- | ----------------------------------------------------------- |
| `labels` | `(B, S)` | Token labels with PATH overlay; ignored positions = `-100`. |

## Invariants

- All spatial channels in the corpus share the same `(H, W)` shape.
- `input_ids` never contains `PATH` (5); `labels` overlays `PATH` from `solution > 0`.
- `start` has exactly one `True` cell; `goals` has at least one.
- `solution` values are non-negative integers; `0` means not on path.
- The task corpus is a sampled subset of the parent substrate — channels are
  copied verbatim, no transformation at build time.
- Token coercion is deterministic: same channels always produce the same
  `(input_ids, labels)` pair.
- `MAZE_HARD_IGNORE_LABEL_ID = -100` positions are excluded from loss and
  accuracy computation.
- `PATH` tokens receive 2× loss weight via `build_mazehard_weights`.
- Regeneration with the same parent substrate and seed produces identical
  tensors.

## Build configuration

| Parameter          | Default      | Description                                        |
| ------------------ | ------------ | -------------------------------------------------- |
| `--substrate-root` | _(required)_ | Path to maze-nd shared substrate version root.     |
| `--corpus`         | `default`    | Corpus label.                                      |
| `--n-train`        | 1000         | Training samples (capped by parent's train split). |
| `--n-val`          | 40           | Validation samples (capped by parent's val split). |
| `--n-test`         | 40           | Test samples (capped by parent's test split).      |
| `--version`        | 1            | Task corpus version integer.                       |
| `--seed`           | 42           | Deterministic sampling seed.                       |

The parent substrate must have `artifact_schema_version = 1`. Rebuild with:

```bash
python scripts/data-gen/build-maze-nd.py build
```

## CLI

| Command    | Description                                              |
| ---------- | -------------------------------------------------------- |
| `build`    | Build the MazeHard task corpus from a maze-nd substrate. |
| `validate` | Validate a MazeHard task-corpus version root.            |
| `inspect`  | Inspect corpus metadata, samples, diagnostics, figures.  |

Usage:

```bash
python scripts/data-gen/build-mazehard.py build \
    --substrate-root data/interim/maze-nd/v2
python scripts/data-gen/build-mazehard.py validate \
    data/processed/mazehard/default/v1
python scripts/data-gen/build-mazehard.py inspect \
    data/processed/mazehard/default/v1

# Inspect with figures
python scripts/data-gen/build-mazehard.py inspect \
    data/processed/mazehard/default/v1 --summary
python scripts/data-gen/build-mazehard.py inspect \
    data/processed/mazehard/default/v1 \
    --sample-figure --sample-index 0
python scripts/data-gen/build-mazehard.py inspect \
    data/processed/mazehard/default/v1 --gallery 5
```

### Inspect options

| Option            | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `--summary`       | Display corpus summary with statistics.                              |
| `--split`         | Split name for sample inspection (default: train).                   |
| `--sample-index`  | Specific sample index to inspect.                                    |
| `--sample-figure` | Render `task_overview_mazehard` for the selected sample.             |
| `--gallery`       | Number of overview figures to produce.                               |
| `--selection`     | Sample selection policy: random, stratified.                         |
| `--figure-seed`   | RNG seed for figure selection.                                       |
| `--json-out`      | Write inspection JSON diagnostics to path.                           |
| `--output-dir`    | Output directory for figures (default: outputs/mazehard-inspection). |
| `--show`          | Display figures interactively.                                       |

### Figures

`task_overview_mazehard` (registry name) — three-panel horizontal figure showing
the input token grid, the target solution path, and task metadata summary.
Consumes only meta keys (`input_ids`, `target/prediction_overlay`) and works
directly from the corpus without requiring an evaluation trace.

## Manifest

Root file: `manifest.json`. Dataset class: `task_corpus`.

Key fields:

| Field                      | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| `task`                     | `"mazehard"`                                               |
| `corpus`                   | Corpus label (e.g. `"default"`).                           |
| `dataset_class`            | `"task_corpus"`                                            |
| `version`                  | Must match the `v<N>` path leaf.                           |
| `channels`                 | `["topology", "mask_valid", "start", "goals", "solution"]` |
| `topology_kind`            | `"grid2d"`                                                 |
| `n_states`                 | `H · W` — total cells in the grid.                         |
| `extent`                   | `[H, W]` from the parent substrate.                        |
| `task_schema_version`      | `1`                                                        |
| `task_protocol_version`    | `1`                                                        |
| `parents.shared_substrate` | `{"family": "maze-nd", "root": "...", "version": N}`       |

## Targets and metrics

| Aspect            | Value                                  |
| ----------------- | -------------------------------------- |
| Benchmark track   | MazeHard                               |
| Claim family      | `dense_spatial_reasoning`              |
| Execution mode    | single-shot dense grid classification  |
| Primary metric    | `sequences_exact`                      |
| Secondary metrics | `token_accuracy`, `sequences_accuracy` |

### Primary metric: `sequences_exact`

Fraction of samples where **all** $S$ output tokens match:

$$\text{exact}(x) = \mathbb{1}\left[\forall p,\ \hat y_p = y_p\right]$$

This is stricter than token accuracy. A prediction with 899/900 cells
correct has $\text{token\_accuracy} = 99.89\%$ but $\text{exact} = 0$
for that sample. One incorrect route cell may break connectivity, enter a
wall, or produce a non-optimal route.

### Secondary metrics

| Metric               | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| `token_accuracy`     | Fraction of non-ignored tokens predicted correctly.             |
| `sequences_accuracy` | Mean per-sequence token accuracy over all supervised positions. |

### Score accumulation

Token-level counts (`correct`, `total`) are summed across batches; accuracy
scalars are derived from totals. Per-sequence metrics are mean-aggregated.

### Loss weighting

`PATH` tokens (label 5) receive 2× loss weight. Ignored positions
(`label == -100`) receive 0× weight. All other positions receive 1× weight.

## Related

- [maze-nd substrate documentation](../substrates/maze-nd.md)
- [Spec: Data Contracts §3.2](../../spec/spec-data-contracts.md)
- [MazeHard Task Builder](../../scripts/data-gen/build-mazehard.py)

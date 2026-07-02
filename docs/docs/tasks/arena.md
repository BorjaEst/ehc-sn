# `arena` Benchmark Task

## Identity

| Property      | Value                                                    |
| ------------- | -------------------------------------------------------- |
| Task name     | `arena`                                                  |
| Benchmark     | structural representation / structural-navigation replay |
| Package path  | `src/ehp_sn/tasks/arena/`                                |
| CLI script    | `scripts/data-gen/build-arena.py`                        |
| Output path   | `data/processed/arena/<corpus>/v<version>/`              |
| Dataset class | `task_corpus`                                            |

## Purpose and ownership

Arena is the structural-learning task family. The model traverses spatial
layouts via random-walk replay and learns to predict the observation at each
step from its preceding internal location belief and action. The surface task
is observation prediction; the scientific purpose is acquiring structured
environmental representations — spatial encodings, sensory codes, conjunctive
bindings, and episodic memory — that downstream reasoning tasks consume.

## Semantic model

### Recurrent step contract

```text
predict: input = (obs[t-1], action[t-1], recurrent_state[t-1]) → obs_logits[t]
update:  feedback = obs[t]  (provided after prediction for state update;
                             not a predictive input)
```

The current observation `obs[t]` is provided as feedback for learning and
state update. It is not a predictive input to the model. The predictive
pathway is from `(obs[t-1], action[t-1], state[t-1])` to `obs_logits[t]`.

### Revisit semantics

A step is a **revisit** when the current spatial position has been visited
earlier in the same episode. `is_revisit` is `True` for every step after the
first visit to that position, including subsequent visits. The primary metric
`accuracy_revisit` isolates episodic memory recall from the confound of
exploration: on first visits the model has no basis for prediction beyond the
prior distribution; on revisits it must recall what observation was previously
at that location.

## Parent requirements

Arena requires a spatial-layout parent artifact providing:

- compact physical states with action-conditioned transitions (`next_state` +
  `action_valid`);
- observation identity per state;
- episode-compatible movement semantics (`movement_kind == "grid4"`,
  4-neighbor + stay).

The layout parent is the only required upstream artifact. Arena is
layout-agnostic: any source that satisfies these requirements is compatible.

## Output artifact

### Adapter input (task-data → model)

Arena replay v1 is topology-free. The model receives only step-level
identifiers and flags:

| Variable          | Description                              | Shape / dtype  |
| ----------------- | ---------------------------------------- | -------------- |
| `observation_id`  | current observation identifier           | `(B, 1)` int64 |
| `previous_action` | action that produced the current state   | `(B, 1)` int64 |
| `landmark_id`     | optional landmark / shiny-cue identifier | `(B, 1)` int64 |
| `step_count`      | transitions taken so far in the episode  | `(B, 1)` int32 |
| `episode_start`   | `True` on the first step of an episode   | `(B,)` bool    |

`landmark_id` is `None` when the layout source does not provide landmarks.

### Adapter output (model → evaluation)

| Variable     | Description                  | Shape          |
| ------------ | ---------------------------- | -------------- |
| `obs_logits` | predicted observation logits | `(B, obs_dim)` |

### Frozen corpus channels

The arena replay corpus stores nine frozen channels per sample, precomputed
at build time from layout walks:

```text
trajectory_row              — (N, T_max) int32, -1 sentinel on padded steps
trajectory_col              — (N, T_max) int32, -1 sentinel on padded steps
trajectory_observation_id   — (N, T_max) int32
trajectory_previous_action  — (N, T_max) int32, step 0 = STAY, -1 on padded
trajectory_landmark_id      — (N, T_max) int32, -1 when absent or padded
trajectory_is_revisit       — (N, T_max) bool,  False on padded steps
trajectory_episode_start    — (N, T_max) bool,  True only at step 0
trajectory_valid_step       — (N, T_max) bool,  = (t < trajectory_length)
trajectory_length           — (N,)       int32
```

Spatial arrays (`topology`, `observations`, `mask_valid`) are also stored
per-episode so evaluation providers never need external parent resolution.

### Targets

| Variable         | Description                                   |
| ---------------- | --------------------------------------------- |
| `observation_id` | ground-truth observation id for this step     |
| `is_revisit`     | `True` if this step revisits a prior location |

## Invariants

- `observation_id` values are drawn from `{0, …, vocab_size-1}`.
- `previous_action` at step 0 is always STAY.
- `is_revisit` is computed from the full episode trajectory at build time.
- The task is layout-agnostic: no spatial coordinates, valid-action mask,
  or location id are exposed as model inputs.
- Arena replay v1 provides no explicit topology information to the model.

## Build configuration

| Parameter       | Default      | Description                            |
| --------------- | ------------ | -------------------------------------- |
| `--layout-root` | _(required)_ | Path to layout dataset root.           |
| `--corpus`      | `default`    | Corpus label.                          |
| `--walk-policy` | `angle_bias` | Walk policy for trajectory generation. |
| `--n-episodes`  | 4            | Number of episodes per layout.         |
| `--max-steps`   | 2000         | Maximum steps per episode.             |
| `--version`     | 1            | Task corpus version integer.           |
| `--seed`        | 45           | Deterministic base seed.               |

## CLI

| Command    | Description                                       |
| ---------- | ------------------------------------------------- |
| `build`    | Build an Arena task corpus over a layout dataset. |
| `validate` | Validate an Arena task-corpus version root.       |
| `inspect`  | Print a human-readable manifest summary.          |

Usage:

```bash
python build-arena.py build --layout-root data/interim/openfield/tem-square/v1
python build-arena.py validate data/processed/arena/default/v1
python build-arena.py inspect data/processed/arena/default/v1
```

## Manifest

Root file: `manifest.json`. Channels: `trajectory_row`, `trajectory_col`,
`trajectory_observation_id`, `trajectory_previous_action`,
`trajectory_landmark_id`, `trajectory_is_revisit`,
`trajectory_episode_start`, `trajectory_valid_step`, `trajectory_length`,
`topology`, `observations`, `mask_valid`.

## Targets and metrics

| Aspect             | Value                                              |
| ------------------ | -------------------------------------------------- |
| Benchmark track    | Arena-Struct                                       |
| Claim family       | `structural_representation`                        |
| Execution mode     | replay                                             |
| Primary metric     | `accuracy_revisit`                                 |
| Secondary metrics  | `accuracy_all`, `correct_all`, `count_all`,        |
|                    | `correct_revisit`, `count_revisit`                 |
| Diagnostic metrics | `accuracy_path_revisit`, `accuracy_recall_revisit` |

Score accumulation uses additive accumulation: raw counts (`correct_*`,
`count_*`) are summed across batches; accuracy scalars are derived from
totals. This avoids numerical error from averaging per-batch accuracies.

# Configurations

Canonical TOML configuration files for the EHC-SN project. Each subdirectory
covers one config category; configs are composed at runtime by training
entrypoints, evaluation scripts, and benchmark runners.

## Benchmark Configs

**Directory:** `config/benchmarks/`

Benchmark track definitions, artifacts manifests, recipes, and suites.
Consumed by benchmark runners under `scripts/benchmarks/`.

| Path                  | Content                                       |
| --------------------- | --------------------------------------------- |
| `arena-struct.toml`   | Arena-Struct benchmark launcher defaults      |
| `mazehard-delib.toml` | MazeHard-Delib benchmark launcher defaults    |
| `b0-mazehard.toml`    | b0-MazeHard alias launcher                    |
| `tracks/`             | Track metadata (primary metric, claim family) |
| `manifests/`          | Artifact manifests (checkpoint + config refs) |
| `recipes/`            | Comparison protocol and execution policy      |
| `suites/`             | Curated suite compositions for paper runs     |

Internal TOML schema details are owned by `spec/spec-benchmark-configuration-contracts.md`.

---

## Debug Configs

**Directory:** `config/debug/`

Training-time debug configurations for short low-resource runs. These are
full training configs (same schema as `config/training/`) with reduced
`max_steps`, `global_batch_size`, and `num_workers`, and include
`[[eval_regimes.regimes]]` blocks that schedule in-training evaluation
via the `EvaluationRegimesCallback` during Lightning `Trainer.fit()`.

| File                | Model  | Purpose                           |
| ------------------- | ------ | --------------------------------- |
| `hrm-v1-debug.toml` | HRM v1 | Low-resource debug run (32 steps) |

---

## Model Configs

**Directory:** `config/models/`

Pure architecture TOML files describing model hyperparameters. These do not
contain training or evaluation settings. They are referenced by training
configs, evaluation configs, and benchmark manifests via `model_config_path`.

| File               | Model  |
| ------------------ | ------ |
| `tem-v1-base.toml` | TEM v1 |
| `tem-v2-base.toml` | TEM v2 |
| `hrm-v1-base.toml` | HRM v1 |
| `hrm-v2-base.toml` | HRM v2 |

---

## Training Configs

**Directory:** `config/training/`

Canonical default training configurations for each model family. These are
the full configs loaded by training entrypoints under `scripts/training/`.
They contain model architecture, adapter, controller, objective, optimizer,
scheduler, runtime, logger, and checkpoint settings. Evaluation configs
under `config/evaluation/` are derived from these by stripping
training-orchestration fields.

| File                           | Model  | Task     |
| ------------------------------ | ------ | -------- |
| `tem-v1-default-vram8gib.toml` | TEM v1 | Arena    |
| `tem-v2-default-vram8gib.toml` | TEM v2 | Arena    |
| `hrm-v1-default-vram8gib.toml` | HRM v1 | MazeHard |
| `hrm-v2-default-vram8gib.toml` | HRM v2 | MazeHard |

---

## Evaluation Executor Configs

**Directory:** `config/evaluation/`

Each TOML file reconstructs one model-family executor for offline evaluation
via `run_offline_eval()`. Files are derived from the canonical training configs
under `config/training/` by stripping all training-orchestration fields
(optimizer, scheduler, data loading, logging, checkpoints, trainer strategy,
figure generation).

Files are named `{family}-{task}.toml`.

| File                   | Model  | Task     | Source training config                         |
| ---------------------- | ------ | -------- | ---------------------------------------------- |
| `tem-v1-arena.toml`    | TEM v1 | Arena    | `config/training/tem-v1-default-vram8gib.toml` |
| `tem-v2-arena.toml`    | TEM v2 | Arena    | `config/training/tem-v2-default-vram8gib.toml` |
| `hrm-v1-mazehard.toml` | HRM v1 | MazeHard | `config/training/hrm-v1-default-vram8gib.toml` |
| `hrm-v2-mazehard.toml` | HRM v2 | MazeHard | `config/training/hrm-v2-default-vram8gib.toml` |
| `ehc-v1-arena.toml`    | EHC v1 | Arena    | `config/training/ehc-v1-spatial-vram8gib.toml` |
| `ehc-v1-mazehard.toml` | EHC v1 | MazeHard | `config/training/ehc-v1-reason-vram8gib.toml`  |

---

## Reporting Configs

**Directory:** `config/reporting/`

Each TOML file is a `ReportSpec` that selects existing eval artifacts and
declares report outputs (metrics, figures, rendered formats). Consumed by
`scripts/reporting/run_report.py` or
`python -m ehc_sn.reporting.run_report build`.

Files are named `{family}_{task}_n{cases}.toml`.

| File                      | Model  | Task     | Cases | Figures                    |
| ------------------------- | ------ | -------- | ----: | -------------------------- |
| `tem_v1_arena_n4.toml`    | TEM v1 | Arena    |     4 | `arena_prediction_overlay` |
| `tem_v2_arena_n4.toml`    | TEM v2 | Arena    |     4 | `arena_prediction_overlay` |
| `hrm_v1_mazehard_n4.toml` | HRM v1 | MazeHard |     4 | `overlay`                  |
| `hrm_v2_mazehard_n4.toml` | HRM v2 | MazeHard |     4 | `overlay`                  |

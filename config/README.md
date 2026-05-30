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

### Checkpoint Format

`run_offline_eval()` requires a **weights-only** checkpoint. Full Lightning
trainer-resume checkpoints (containing `optimizer_states`, `lr_schedulers`,
`loops`) are rejected.

Existing `eval-weights-only.pt` files are already present for some families
under `checkpoints/<family>/`. To convert a full Lightning checkpoint:

```python
import torch
ckpt = torch.load("checkpoints/<family>/last.ckpt", map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)
torch.save(state_dict, "checkpoints/<family>/eval-weights-only.pt")
```

### Usage

```bash
# TEM v1 Arena full diagnostic (4 cases, with traces)
python scripts/evaluation/run_eval.py \
    --model-family tem-v1 \
    --checkpoint checkpoints/tem-v1/eval-weights-only.pt \
    --config config/evaluation/tem-v1-arena.toml \
    --task arena \
    --provider-ref ehc_sn.tasks.arena.providers.ArenaReplayProvider \
    --provider-settings '{"dataset_path": "data/processed/arena/default/v1", "split": "test", "n_cases": 4}' \
    --regime-id arena_struct_4 \
    --regime-kind diagnostic \
    --output artifacts/evaluation/tem_v1/arena_n4 \
    --trace-keys "pred/observation,mec/g_2d,hpc/p" \
    --device cpu

# HRM v1 MazeHard full diagnostic (4 case, with traces)
python scripts/evaluation/run_eval.py \
    --model-family hrm-v1 \
    --checkpoint checkpoints/hrm-v1/eval-weights-only.pt \
    --config config/evaluation/hrm-v1-mazehard.toml \
    --task mazehard \
    --provider-ref ehc_sn.tasks.mazehard.providers.MazeHardReplayProvider \
    --provider-settings '{"dataset_path": "data/processed/mazehard/default/v1", "split": "test", "n_cases": 4}' \
    --regime-id mazehard_reason_4 \
    --regime-kind diagnostic \
    --output artifacts/evaluation/hrm_v1/mazehard_4n \
    --trace-keys "pred/solution_overlay,act/halted" \
    --device cpu
```

### Key Differences from Training Configs

| Difference                     | Reason                                                                      |
| ------------------------------ | --------------------------------------------------------------------------- |
| `exploration_prob = 0.0` (HRM) | Training uses 0.1 ε-greedy; eval must be deterministic for stable metrics   |
| `global_batch_size = 1` (HRM)  | Required by `HRMV1ModelConfig`/`HRMV2ModelConfig`; truthful for single-case |
| `runtime.*` preserved          | Memory dynamics (TEM) and rollout caps (HRM/EHC) affect inference behavior  |
| `mode` in EHC configs          | Required by `parse_ehc_v1_config()` dispatch logic                          |

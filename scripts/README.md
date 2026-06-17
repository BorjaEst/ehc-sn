# Scripts

CLI entry points for the EHP-SN project. Scripts call library APIs from
`ehc_sn.*` — they contain no business logic, metric computation, adapter
dispatch, or figure rendering.

## Evaluation Runner

**Script:** `scripts/evaluation/run_eval.py`

Runs offline evaluation from a checkpoint against a task provider, producing
a persisted eval artifact (v3) compatible with the reporting system.

Uses `ehc_sn.eval.offline.run_offline_eval`.

### Evaluation Usage

```bash
# TEM v1 Arena diagnostic (1 case, with traces for overlays)
python scripts/evaluation/run_eval.py \
    --model-family tem-v1 \
    --checkpoint models/tem-v1/best.pt \
    --config config/evaluation/tem-v1-arena.toml \
    --task arena \
    --provider-ref ehc_sn.tasks.arena.providers.ArenaReplayProvider \
    --provider-settings '{"dataset_path": "data/processed/arena/default/v1", "split": "test", "n_cases": 1}' \
    --regime-id arena_n1 --regime-kind diagnostic \
    --output artifacts/evaluation/tem_v1/arena_n1 --device cpu \
    --trace-keys '["diagnostic/mec/location_mean","diagnostic/hpc/location_mean","diagnostic/lec/cells","diagnostic/lec/filtered","diagnostic/lec/sensory_code","world_step/observation","pred/observation_id/post","pred/observation_id/recall","pred/observation_id/path","lec/filter/alpha_sigmoid","lec/w_f_sigmoid"]'

# TEM v2 Arena diagnostic (1 case, with traces for overlays)
python scripts/evaluation/run_eval.py \
    --model-family tem-v2 \
    --checkpoint best/tem-v2/best.pt \
    --config config/evaluation/tem-v2-arena.toml \
    --task arena \
    --provider-ref ehc_sn.tasks.arena.providers.ArenaReplayProvider \
    --provider-settings '{"dataset_path": "data/processed/arena/default/v1", "split": "test", "n_cases": 1}' \
    --regime-id arena_n1 --regime-kind diagnostic \
    --output artifacts/evaluation/tem_v2/arena_n1 --device cpu \
    --trace-keys '["diagnostic/mec/location_mean","diagnostic/hpc/location_mean","diagnostic/lec/cells","diagnostic/lec/filtered","diagnostic/lec/sensory_code","world_step/observation","pred/observation_id/post","pred/observation_id/recall","pred/observation_id/path","lec/filter/alpha_sigmoid","lec/w_f_sigmoid"]'

# HRM v1 MazeHard full diagnostic (1 case, with traces)
python scripts/evaluation/run_eval.py \
    --model-family hrm-v1 \
    --checkpoint best/hrm-v1/best.pt \
    --config config/evaluation/hrm-v1-mazehard.toml \
    --task mazehard \
    --provider-ref ehc_sn.tasks.mazehard.providers.MazeHardReplayProvider \
    --provider-settings '{"dataset_path": "data/processed/mazehard/default/v1", "split": "test", "n_cases": 1}' \
    --regime-id mazehard_n1 \
    --regime-kind diagnostic \
    --output artifacts/evaluation/hrm_v1/mazehard_n1 \
    --trace-keys "pred/solution_overlay,act/halted" \
    --device cpu

# HRM v2 MazeHard diagnostic (1 cases, with traces)
python scripts/evaluation/run_eval.py \
    --model-family hrm-v2 \
    --checkpoint best/hrm-v2/best.pt \
    --config config/evaluation/hrm-v2-mazehard.toml \
    --task mazehard \
    --provider-ref ehc_sn.tasks.mazehard.providers.MazeHardReplayProvider \
    --provider-settings '{"dataset_path": "data/processed/mazehard/default/v1", "split": "test", "n_cases": 1}' \
    --regime-id mazehard_n1 \
    --regime-kind diagnostic \
    --output artifacts/evaluation/hrm_v2/mazehard_n1 \
    --trace-keys "pred/solution_overlay,act/halted" \
    --device cpu
```

### Evaluation Key Flags

| Flag                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `--model-family`      | One of `tem-v1`, `tem-v2`, `hrm-v1`, `hrm-v2`, `ehp-v1`        |
| `--checkpoint`        | Path to a **weights-only** `.pt` file                          |
| `--config`            | Path to an executor TOML from `config/evaluation/`             |
| `--task`              | `arena` or `mazehard`                                          |
| `--provider-ref`      | Dotted path to an `EvaluationSourceProvider` class             |
| `--provider-settings` | JSON dict forwarded to the provider constructor                |
| `--regime-id`         | Free-form string; must match `regime_ids` in the report config |
| `--output`            | Target directory for the eval artifact                         |
| `--trace-keys`        | Comma-separated trace keys for figure materialization          |
| `--device`            | `cpu` or `cuda`                                                |

---

## Report Runner

**Script:** `scripts/reporting/run_report.py`

Assembles a report run from existing eval artifacts using a `ReportSpec`
TOML config. Calls `ehc_sn.reporting.builder.build_report_run`.

### Report Usage

```bash
# Assemble a report from existing eval artifacts
python scripts/reporting/run_report.py \
    --config config/reporting/tem_v1_arena_n1.toml

# Override the output directory
python scripts/reporting/run_report.py \
    --config config/reporting/hrm_v2_mazehard_n1.toml \
    --output /tmp/my_custom_report

# Skip figure rendering (metrics only, faster)
python scripts/reporting/run_report.py \
    --config config/reporting/tem_v2_arena_n1.toml \
    --no-figures

# Skip metric normalization
python scripts/reporting/run_report.py \
    --config config/reporting/tem_v2_arena_n1.toml \
    --no-metrics
```

### Report Key Flags

| Flag           | Description                                          |
| -------------- | ---------------------------------------------------- |
| `--config`     | Path to a `ReportSpec` TOML from `config/reporting/` |
| `--output`     | Override `output_dir` from the config                |
| `--no-metrics` | Skip metric normalization                            |
| `--no-figures` | Skip figure rendering                                |

### What It Produces

```text
<output_dir>/
├── report_manifest.json      # authoritative artifact list
├── provenance.json           # creation metadata
├── config.resolved.yaml      # resolved copy of the input spec
├── metrics.records.json      # normalized metric rows
├── figures/
│   └── index.json            # figure index
├── rendered/                 # markdown output
└── _SUCCESS                  # completion sentinel
```

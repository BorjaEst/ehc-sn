## Run Cookbook

This page provides copy-pasteable run recipes using repository entrypoints.

## Inspect Entrypoint Surface

```bash
python scripts/training/tem_v1_baseline.py --help
python scripts/training/tem_v2_softmax.py --help
python scripts/training/hrm_v1_baseline.py --help
python scripts/training/hrm_v2_rl-striatum.py --help
python scripts/training/ehc_v1_pretraining.py --help
python scripts/data-gen/build-maze-nd.py --help
python scripts/data-gen/build-mazehard.py --help
python scripts/benchmarks/b0-mazehard.py --help
python scripts/diagnostics/collect_artifact_bundle.py --help
```

## Data Generation

Generate datasets with the scripts in `scripts/data-gen/`.

```bash
# Build shared substrates first
python scripts/data-gen/build-maze-nd.py build-all

# Then build task corpora
python scripts/data-gen/build-mazehard.py materialize-task \
    --parent-substrate data/interim/maze-nd/v1
python scripts/data-gen/build-arena.py materialize-task ...
python scripts/data-gen/build-numberline.py build-all
python scripts/data-gen/build-dungeon.py materialize-task ...
```

Generated data is typically stored under `data/processed/` and can be consumed by
training and evaluation entrypoints.

## Training Recipes

The training entrypoints read configuration from environment variables.
Use the defaults in `config/training/` or override with your own config file.

### TEM v1 baseline

```bash
export TEM_V1_CONFIGURATION_PATH=config/training/tem-v1-arena-vram8gib.toml
python scripts/training/tem_v1_baseline.py
```

### TEM v2 softmax

```bash
export TEM_V2_CONFIGURATION_PATH=config/training/tem-v2-arena-vram8gib.toml
python scripts/training/tem_v2_softmax.py
```

### HRM v1 baseline

```bash
export HRM_V1_CONFIGURATION_PATH=config/training/hrm-v1-mazehard-vram8gib.toml
python scripts/training/hrm_v1_baseline.py
```

### HRM v2 RL-Striatum

```bash
export HRM_V2_CONFIGURATION_PATH=config/training/hrm-v2-mazehard-vram8gib.toml
python scripts/training/hrm_v2_rl-striatum.py
```

### EHP v1 spatial pretraining

```bash
export EHP_V1_CONFIGURATION_PATH=config/training/ehp-v1-spatial-vram8gib.toml
python scripts/training/ehc_v1_pretraining.py
```

### EHP v1 reason pretraining

```bash
export EHP_V1_CONFIGURATION_PATH=config/training/ehp-v1-reason-default.toml
python scripts/training/ehc_v1_pretraining.py
```

## Benchmark Recipes

Run benchmark wrapper scripts directly.

```bash
python scripts/benchmarks/b0-mazehard.py
python scripts/benchmarks/mazehard-delib.py
```

These wrappers use benchmark configuration and evaluation flows intended for
comparing model and training variants.

## Evaluation and Figure Extraction

Use diagnostics scripts to collect evaluation artifacts and render reports.

```bash
python scripts/diagnostics/collect_artifact_bundle.py --output outputs/figures/bundle
python scripts/diagnostics/render_report_figures.py --input outputs/figures/bundle
```

Evaluation artifacts are typically stored under `outputs/` for later analysis and
reporting.

## Checkpoints and Model Selection

Training scripts save model checkpoints and experiment artifacts to
`checkpoints/`.

- Keep one directory per model family or experiment.
- Use explicit config names to capture the model profile.
- Record the `export ... && python ...` command used for each run.

Example checkpoint flow:

```bash
mkdir -p checkpoints/tem_v1
export TEM_V1_CONFIGURATION_PATH=config/training/tem-v1-arena-vram8gib.toml
python scripts/training/tem_v1_baseline.py --checkpoint-dir checkpoints/tem_v1
```

To select a model family, choose the matching training entrypoint and config.
For example, use `tem_v1_baseline.py` for TEM v1, `hrm_v1_baseline.py` for HRM
v1, and `ehc_v1_pretraining.py` for EHP pretraining.

## Common Notes

- Use `--help` on any script to inspect runtime options.
- Default train configs live under `config/training/` and may include resource-
  specific variants (for example, `-vram8gib` profiles).
- Keep data generation output, checkpoints, and report artifacts separated by
  model family and experiment.

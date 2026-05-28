## Getting Started

This page provides a minimal clean-start workflow for local development and
first execution.

## Prerequisites

- Linux or macOS shell environment
- Python 3.12+

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Verify Package Import

```bash
python -c "import ehc_sn; print('ehc_sn import OK')"
```

## Inspect Available Training Entrypoints

```bash
python scripts/training/tem_v1_baseline.py --help
python scripts/training/tem_v2_softmax.py --help
python scripts/training/hrm_v1_baseline.py --help
python scripts/training/hrm_v2_rl-striatum.py --help
python scripts/training/ehc_v1_pretraining.py --help
```

## Inspect Benchmark Entrypoints

```bash
python scripts/benchmarks/mazehard-delib.py --help
python scripts/benchmarks/b0-mazehard.py --help
```

## Configuration Paths

Training scripts load TOML defaults from config/training and support
environment-variable overrides, for example:

- TEM_V1_CONFIGURATION_PATH
- TEM_V2_CONFIGURATION_PATH
- HRM_V1_CONFIGURATION_PATH
- HRM_V2_CONFIGURATION_PATH
- EHC_V1_CONFIGURATION_PATH

## Data Layout Constraints

The canonical pipeline is:

- data/raw
- data/interim
- data/processed

Do not place benchmark reports in data/processed. Use reports/benchmarks.

See [Data And Paths](data-and-paths.md) for details.

## Development

This page describes the local setup, code quality checks, documentation workflow,
and the main script-driven workflows for datasets, training, benchmarks, and
evaluation.

## Local Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install the package with development extras:

```bash
pip install -U pip
pip install -e ".[dev]"
```

3. Verify the package imports cleanly:

```bash
python -c "import ehc_sn; print('ehc_sn import OK')"
```

## Code Quality and Tests

Run the full test suite:

```bash
pytest
```

Run a focused test file:

```bash
pytest tests/test_data.py
```

Formatting and linting:

```bash
black .
isort .
flake8 .
```

Static analysis guidance is configured in `pyproject.toml` with:

- `black`
- `isort`
- `flake8`
- `mypy`
- `ruff`

## Documentation Workflow

The site source is in `docs/`, with content under `docs/docs/` and site config
in `docs/mkdocs.yml`.

Build the documentation locally:

```bash
mkdocs build -f docs/mkdocs.yml
```

Serve the site for local review:

```bash
mkdocs serve -f docs/mkdocs.yml
```

Docs content should be updated for any workflow, benchmark, configuration, or
user-facing change.

## Script Workflows

### Data generation

Generate datasets using the scripts under `scripts/data-gen/`.

Example:

```bash
# Build the shared substrate
python scripts/data-gen/build-maze-nd.py --help
python scripts/data-gen/build-maze-nd.py build-all

# Then build the task corpus
python scripts/data-gen/build-mazehard.py --help
python scripts/data-gen/build-mazehard.py materialize-task \
    --parent-substrate data/interim/maze-nd/v1
```

Other generators include:

- `scripts/data-gen/build-arena.py`
- `scripts/data-gen/build-numberline.py`
- `scripts/data-gen/build-countwalk.py`
- `scripts/data-gen/build-dungeon.py`
- `scripts/data-gen/build-dungeongen.py`

### Training

Training entrypoints live in `scripts/training/`.

Example:

```bash
python scripts/training/tem_v1_baseline.py --help
python scripts/training/tem_v1_baseline.py --config path/to/config.yaml
```

Common training scripts:

- `scripts/training/tem_v1_baseline.py`
- `scripts/training/tem_v2_softmax.py`
- `scripts/training/hrm_v1_baseline.py`
- `scripts/training/hrm_v2_rl-striatum.py`
- `scripts/training/ehc_v1_pretraining.py`

### Benchmarks

Benchmark wrappers are in `scripts/benchmarks/`.

Examples:

```bash
python scripts/benchmarks/b0-mazehard.py --help
python scripts/benchmarks/mazehard-delib.py --help
```

### Evaluation and figures

Use diagnostics scripts to export figures and reports.

Example:

```bash
python scripts/diagnostics/collect_artifact_bundle.py --help
python scripts/diagnostics/render_report_figures.py --help
```

## Checkpoints and Artifacts

Checkpoints and evaluation artifacts are typically written to `checkpoints/` and
`outputs/`.

When running experiments:

- Keep checkpoints in a dedicated directory for the model family.
- Use meaningful names for saved artifacts.
- Record the training command and config used for each run.

## Canonical References

Keep implementation, runtime contracts, and documentation aligned with:

- `spec/spec-manifest.toml`
- `spec/spec-architecture.md`
- `spec/spec-requirements.md`
- `spec/spec-standards.md`
- `docs/docs/specs-and-governance.md`

## Contribution Process

Follow the root `CONTRIBUTING.md` guidance for PR expectations, commit style,
and issue reporting.

## Docs Maintenance

Docs updates should accompany any change that affects user workflows, model
profiles, configuration, or evaluation behavior.

- Review `docs/docs/` pages for accuracy when you change code or configs.
- Keep doc examples, recipe commands, and config references aligned with the
  current repository layout.
- Treat doc updates as part of the same PR when public-facing behavior changes.

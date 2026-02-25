# EHC-SN

**Entorhinal-Hippocampal Complex — Spatial Navigation**

A research library for biologically-inspired spatial navigation models,
built on PyTorch and Lightning. EHC-SN implements neural architectures
that capture how the hippocampal formation and prefrontal cortex interact
for goal-directed generalization in maze-solving tasks.

Inspired by Zheng, Wolf, Ranganath, O'Reilly & McKee — _"Flexible
Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed
Generalization."_

## Models

| Model      | Description                                                                                       | Status  |
| ---------- | ------------------------------------------------------------------------------------------------- | ------- |
| **TEM v1** | Tolman-Eichenbaum Machine — composes LEC, MEC, and HPC modules for multi-scale spatial memory     | Active  |
| **HRM v1** | Hierarchical Reasoning Model — PFC-based recurrent reasoning with Adaptive Computation Time (ACT) | Active  |
| **TRM v1** | Tiny Reasoning Model                                                                              | Planned |

## Architecture

The library is organized around neuroscience-grounded circuit modules:

- **LEC** — Sensory encoding and temporal frequency filtering
- **MEC** — Path integration, grid cells, object-vector cells
- **HPC** — Hebbian memory, attractor dynamics, place codes
- **PFC** — Working memory and ACT-based hierarchical reasoning

Models compose these modules as `LightningModule` wrappers with explicit
recurrent state management via dataclasses.

See [spec/spec-architecture.md](spec/spec-architecture.md) for full
component taxonomy and design decisions.

## Installation

Requires **Python ≥ 3.12**.

```bash
# Clone the repository
git clone https://github.com/BorjaEst/ehc-sn.git
cd ehc-sn

# Install in development mode
pip install -e ".[dev]"
```

PyTorch must be installed separately with the appropriate CUDA version
for your system. See [pytorch.org](https://pytorch.org/get-started/).

## Quick Start

### Data Generation

Generate and process maze datasets:

```bash
# Process raw mazes into training data
python scripts/data-gen/build_mazes.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --subset train
```

### Running an Experiment

```bash
# Run the baseline HRM experiment
python experiments/exp01_baseline.py \
    --architecture.embedding_dim 128 \
    --architecture.num_layers 4
```

Experiment configuration uses Pydantic settings with CLI overrides.
Static defaults live in `config/defaults_ehc.toml`.

### Examples

```bash
# Data pipeline visualization
python examples/data_datamodule.py

# Model rollout and trace collection
python examples/model_rollout.py
```

## Project Layout

```
src/ehc_sn/          Main package
  modules/           Brain-region modules (LEC, MEC, HPC, PFC)
  models/            Composed LightningModules (TEM, HRM, TRM)
  loss/              Per-model loss computation
  training/          Step loop, optimizers, schedulers
  data/              DataModules and datasets
  metrics/           TorchMetrics evaluation
  rollouts/          Trace collection and rollout data
  figures/           Publication-ready plotting
  callbacks/         Lightning callbacks
  logging/           TensorBoard logger wrapper
  activations/       Custom activations (stablemax)
  utils/             Cross-cutting helpers
src/mazes/           Auxiliary maze environment package
config/              TOML configuration defaults
experiments/         Training experiment scripts
examples/            Runnable usage examples
scripts/data-gen/    Data processing CLI tools
data/                On-disk data (raw/interim/processed)
spec/                Canonical architecture and requirements specs
docs/                MkDocs documentation site
```

## Configuration

All configuration uses Pydantic models with `extra="forbid"` for
strictness. Experiment entry points use `pydantic_settings.BaseSettings`
with CLI argument parsing, so any config field can be overridden from
the command line:

```bash
python experiments/exp01_baseline.py --optimizer.lr 1e-4
```

## Dependencies

Core: `torch`, `lightning`, `gymnasium`, `pydantic`, `pydantic_settings`
| Compute: `scipy` | Visualization: `matplotlib`, `SciencePlots`,
`pub-ready-plots` | Logging: `tensorboard`, `rich` | Optimization:
`adam-atan2-pytorch`

## Documentation

- **Architecture spec**: [spec/spec-architecture.md](spec/spec-architecture.md)
- **Requirements spec**: [spec/spec-requirements.md](spec/spec-requirements.md)
- **Standards spec**: [spec/spec-standards.md](spec/spec-standards.md)
- **MkDocs site**: [docs/](docs/) (build with `mkdocs serve` from
  the `docs/` directory)

## License

[GNU General Public License v3.0](LICENSE)

# ehc-sn

[![Python >= 3.12](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-not%20configured-lightgrey.svg)](#)

Entorhinal-Hippocampal Circuit (EHC) Spatial Navigation library.

## Overview

ehc-sn is a research library for biologically inspired spatial cognition and
navigation models built on PyTorch.

The codebase supports multiple model families, tasks, and adapters under a
single canonical namespace: ehc_sn.

## Installation

### Requirements

- Python 3.12 or newer

### Install from source

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Verify installation

```bash
python -c "import ehc_sn; print('ehc_sn import OK')"
```

## Quick Start

The repository provides thin training entrypoints under scripts/training.

```bash
python scripts/training/tem_v1_baseline.py --help
python scripts/training/tem_v2_softmax.py --help
python scripts/training/hrm_v1_baseline.py --help
python scripts/training/hrm_v2_rl-striatum.py --help
python scripts/training/ehc_v1_pretraining.py --help
```

Benchmark wrappers live under scripts/benchmarks.

```bash
python scripts/benchmarks/mazehard-delib.py --help
python scripts/benchmarks/b0-mazehard.py --help
```

## Documentation

- Project docs site source: docs/
- Developer onboarding and workflow guide: docs/docs/development.md
- Local docs guide: docs/README.md
- Hosted docs URL (project metadata): https://ehc-sn.readthedocs.io

## Canonical Specifications

- Manifest and governance: spec/spec-manifest.toml
- Architecture: spec/spec-architecture.md
- Requirements: spec/spec-requirements.md
- Standards: spec/spec-standards.md
- Docs governance: docs/docs/specs-and-governance.md

## Legacy Context

Historical references used for migration and parity checks:

- legacy_tem/README.md
- legacy_hrm/README.md

## Citation

If you use this repository in research, cite the relevant foundational and
model-family papers from article/references.bib, including:

- whittington_tolman-eichenbaum_2020
- whittington_relating_2022
- whittington_how_2022
- wang_hierarchical_2025
- zheng_flexible_2025

## License

This project is licensed under GNU General Public License v3.0.
See LICENSE.

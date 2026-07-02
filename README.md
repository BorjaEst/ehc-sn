# ehp-sn

[![Python >= 3.12](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-not%20configured-lightgrey.svg)](#)

Entorhinal-Hippocampal Circuit (EHP) Spatial Navigation library.

> todo: Rename to EHP — Entorhinal–Hippocampal–Prefrontal Model

## Overview

ehp-sn is a research library for biologically inspired spatial cognition and
navigation models built on PyTorch.

The codebase supports multiple model families, tasks, and adapters under a
single canonical namespace: ehp_sn.

## Architecture

### Papers

| Family              | Citation key                       | Claim                                                                                                                                       |
| ------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| TEM / HPC           | whittington_tolman-eichenbaum_2020 | Unifies space and relational memory through generalization; MEC encodes structural knowledge, HPC binds it to sensory representations.      |
| TEM / Transformer   | whittington_relating_2022          | Transformers with recurrent position encodings reproduce hippocampal-formation spatial representations (place, grid cells).                 |
| TEM / Review        | whittington_how_2022               | Organizes cognitive-map models into a clear ontology bridging hippocampal-cortical understanding.                                           |
| HRM / PFC           | wang_hierarchical_2025             | Hierarchical recurrent model with slow (abstract planning) and fast (detailed computation) modules for one-pass reasoning without CoT data. |
| HPC-PFC interaction | zheng_flexible_2025                | PFC top-down query-key control over HPC episodic memory enables goal-directed generalization across novel situations.                       |

### Models

| Family  | Versions | Brain region           | Computation style                                                                                      | Training mode         |
| ------- | -------- | ---------------------- | ------------------------------------------------------------------------------------------------------ | --------------------- |
| **TEM** | v1, v2   | LEC ↔ MEC ↔ HPC        | Predictive cognitive map; grid transitions, attractor retrieval, Hebbian memory                        | Replay                |
| **HRM** | v1, v2   | PFC                    | Hierarchical recurrent ACT (Adaptive Computation Time); actor-critic deliberation with Q-value halting | Deliberation          |
| **EHP** | v1, v2   | Unified EC ↔ HPC ↔ PFC | Subsumes TEM patterns; supports spatial replay and reasoning deliberation                              | Replay & deliberation |

### Tasks

| Task          | Execution mode                | evaluation surface                                       | Readiness  |
| ------------- | ----------------------------- | -------------------------------------------------------- | ---------- |
| **Arena**     | Replay (structural exposure)  | `ArenaScoreReport` — accuracy_all, accuracy_revisit      | Production |
| **MazeHard**  | Deliberation (puzzle batches) | `MazeHardScoreReport` — sequences_exact, tokens_accuracy | Production |
| **routebind** |                               |                                                          |            |

### Adapters

Adapters are the only canonical seam between models and tasks.

|         | Arena                          | MazeHard                          |
| ------- | ------------------------------ | --------------------------------- |
| **TEM** | `adapters/arena/tem/` (v1, v2) | —                                 |
| **EHP** | `adapters/arena/ehp/` (v1)     | `adapters/mazehard/ehp/` (v1)     |
| **HRM** | —                              | `adapters/mazehard/hrm/` (v1, v2) |

### Benchmarks

| Track                    | Claim family              | Primary metric                          | Current support   | Readiness |
| ------------------------ | ------------------------- | --------------------------------------- | ----------------- | --------- |
| **Arena-Struct**         | structural_representation | accuracy_revisit                        | TEM v1/v2, EHP v1 | ready     |
| **MazeHard-Delib**       | deliberative_reasoning    | sequences_exact                         | HRM v1/v2, EHP v1 | ready     |
| **CrossTask-Transfer**   | cross_task_transfer       | Target track primary + delta vs scratch | EHP v1            | partial   |
| **OneShot-Relocation**   | one_shot_adaptation       | success_rate                            | —                 | blocked   |
| **Dungeon-Nav**          | navigation                | success_rate                            | —                 | blocked   |
| **Interference-Control** | interference_control      | Delta success_rate                      | —                 | blocked   |
| **Memory-FixedExposure** | episodic_memory           | TBD                                     | —                 | blocked   |
| **Countwalk-Regression** | development_regression    | value_accuracy                          | —                 | blocked   |

### Supported Model–Task Pairs

| Model      | Arena replay | MazeHard deliberation | Dungeon live nav | Cross-task transfer |
| ---------- | ------------ | --------------------- | ---------------- | ------------------- |
| **TEM v1** | ✅           | ✗                     | ✗                | ✗                   |
| **TEM v2** | ✅           | ✗                     | ✗                | ✗                   |
| **EHP v1** | ✅           | ✅                    | —                | 🟡                  |
| **EHP v2** | —            | —                     | —                | —                   |
| **HRM v1** | ✗            | ✅                    | ✗                | ✗                   |
| **HRM v2** | ✗            | ✅                    | ✗                | ✗                   |

✅ Wired today. ✗ Not applicable to family. 🟡 Architecturally feasible, not wired. — Not yet implemented.

## Reports and Notebooks

The repository uses artifact-backed reports for scientific evaluation. Reports
are organized around paper-level scientific questions, not around individual
checkpoints, tasks, or models.

A report run consumes existing evaluation artifacts, assembles normalized
metrics, tables, figures, and provenance, and optionally renders notebook,
HTML, or manuscript-style outputs. Notebooks are interactive report viewers and
interpretation layers; they should not contain task execution logic, adapter
dispatch, metric computation, or figure-generation internals.

The intended flow:

```text
checkpoint + report config
        ↓
evaluation artifact(s)
        ↓
ReportRun
        ↓
metrics, tables, figures, provenance, rendered outputs
        ↓
notebook inspection
```

### Report principles

- Reports are specific to a scientific mechanism or paper-level claim.
- Models and tasks are selected as evidence for that question.
- Adapters remain the canonical model–task seam.
- Evaluation artifacts remain the raw evidence source.
- ReportRuns are the normalized report artifacts.
- Notebooks inspect, compare, and present ReportRuns.

### Notebook portfolio

| Notebook                            | Scientific question                                                                              | Models                 | Task / benchmark              | Primary metric                        | Status        |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------- | ----------------------------- | ------------------------------------- | ------------- |
| `01_structural_memory_report.ipynb` | Do TEM-style and EHP-style systems learn reusable structural representations under Arena replay? | TEM v1, TEM v2, EHP v1 | Arena / Arena-Struct          | `accuracy_revisit`                    | First target  |
| `02_working_memory_report.ipynb`    | Do HRM-style and EHP-style systems solve deliberative sequence problems reliably?                | HRM v1, HRM v2, EHP v1 | MazeHard / MazeHard-Delib     | `sequences_exact`                     | Second target |
| `03_flexible_learning_report.ipynb` | Does EHP bridge TEM-style structural memory and HRM-style deliberative control?                  | EHP v1, TEM v2, HRM v2 | Arena-Struct + MazeHard-Delib | `accuracy_revisit`, `sequences_exact` | Third target  |

### `01_structural_memory_report.ipynb`

This report evaluates the structural-memory claim associated with TEM-style
hippocampal-entorhinal computation.

**Question:**

```text
Do TEM v1, TEM v2, and EHP v1 learn reusable structural representations under Arena replay?
```

**Models:** TEM v1, TEM v2, EHP v1

**Evidence:**

```text
Task: Arena
Benchmark: Arena-Struct
Execution mode: replay / structural exposure
Primary metric (task): accuracy_revisit
Primary metric (TEM evidence): accuracy_path_revisit
Secondary metric: accuracy_all
```

This notebook should compare whether EHP v1 preserves TEM-like structural-memory
behavior, whether TEM v2 improves over TEM v1, and whether revisit-specific
accuracy differs from overall accuracy.

**Legal model–task bindings:**

```text
TEM v1 × Arena → adapters/arena/tem/
TEM v2 × Arena → adapters/arena/tem/
EHP v1 × Arena → adapters/arena/ehp/
```

### `02_working_memory_report.ipynb`

This report evaluates the deliberative-reasoning claim associated with HRM-style
PFC computation.

**Question:**

```text
Do HRM v1, HRM v2, and EHP v1 solve MazeHard through reliable hierarchical deliberation?
```

**Models:** HRM v1, HRM v2, EHP v1

**Evidence:**

```text
Task: MazeHard
Benchmark: MazeHard-Delib
Execution mode: deliberation / puzzle batches
Primary metric: sequences_exact
Secondary metric: tokens_accuracy
```

This notebook should compare whether HRM v2 improves over HRM v1, whether EHP v1
participates in deliberative reasoning, and whether token-level accuracy and
sequence-level correctness tell the same story.

**Legal model–task bindings:**

```text
HRM v1 × MazeHard → adapters/mazehard/hrm/
HRM v2 × MazeHard → adapters/mazehard/hrm/
EHP v1 × MazeHard → adapters/mazehard/ehp/
```

### `03_flexible_learning_report.ipynb`

> TODO This report evaluates whether EHP can improve generalization and
> flexible learning by bridging TEM-style structural memory and HRM-style
> deliberative control.

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
python -c "import ehp_sn; print('ehp_sn import OK')"
```

## Quick Start

The repository provides thin training entrypoints under scripts/training.

```bash
python scripts/training/tem_v1_arena.py --help
python scripts/training/tem_v2_arena.py --help
python scripts/training/hrm_v1_mazehard.py --help
python scripts/training/hrm_v2_mazehard.py --help
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
- Hosted docs URL (project metadata): https://ehp-sn.readthedocs.io

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

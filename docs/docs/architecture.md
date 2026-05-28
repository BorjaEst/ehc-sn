## Architecture

This page summarizes canonical boundaries from spec/spec-architecture.md.

## Project Identity

ehc-sn is a multi-task, multi-model research library.

- Models own architecture.
- Tasks own semantics.
- Adapters are the canonical model-task seam.

## Canonical Namespace

Use ehc_sn as the only first-party import namespace.

## Layered Dependency Model

| Layer | Components                                                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 4     | scripts/training, scripts/haicore, scripts/benchmarks                                                                                |
| 3     | models, tasks, adapters, benchmarks, lightning                                                                                       |
| 2     | modules, controllers, objectives, policies, training, loss, metrics, rollouts, traces, eval, figures, callbacks, logging, data, envs |
| 1     | activations, utils, types.py                                                                                                         |

Internal imports must only flow downward or within the same layer.

## Ownership Invariants

- Models are task-agnostic.
- Tasks own observation/action/reward and episode semantics.
- Adapters own binding, packing, decoding, and task-model bridge logic.
- Callbacks own scheduling and routing, not reusable evaluation contracts.
- Benchmarks own benchmark semantics.

## Component Taxonomy

Top-level package families under src/ehc_sn:

- activations
- adapters
- benchmarks
- callbacks
- controllers
- data
- eval
- envs
- figures
- lightning
- logging
- loss
- metrics
- models
- modules
- objectives
- policies
- rollouts
- tasks
- traces
- training
- utils

## Related Specs

- ../../spec/spec-architecture.md
- ../../spec/spec-model-interfaces.md
- ../../spec/spec-controller-runtime-contracts.md

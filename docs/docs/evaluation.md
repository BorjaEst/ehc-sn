## Evaluation

Evaluation is organized as reusable contracts plus callback-owned scheduling.

## Core Contracts

Defined in src/ehc_sn/eval/contracts.py:

- EvaluationCaseBatch
- EvaluationTraceRequest
- EvaluationCaseResult
- EvaluationRegimeResult
- EvaluationSourceProvider protocol
- EvaluationExecutor protocol

## Execution Helpers

src/ehc_sn/eval/executor.py provides reusable helpers:

- execute_replay_evaluation_batch
- iter_evaluation_regime

These helpers run provider batches, execute family seams, score chunks, and
optionally materialize traces.

## Named Regime Callback

src/ehc_sn/callbacks/evaluation.py owns:

- cadence scheduling (step and epoch)
- provider resolution
- per-regime trace requests
- optional online diagnostic figure routing
- artifact persistence and namespaced metrics

This callback is orchestration-only. Reusable execution remains in eval/.

## Figure Bundle Persistence

src/ehc_sn/eval/artifacts.py owns persisted evaluation-artifact contracts and
loading helpers.

Supported artifact families include:

- ehp-v1
- hrm-v1
- hrm-v2
- tem-v1
- tem-v2

## Offline Report Rendering

src/ehc_sn/eval/reports.py renders report-kind figures from persisted
run artifacts.

This is the standard report surface for post-hoc figure generation.

## Related Specs

- ../../spec/spec-architecture.md
- ../../spec/spec-model-interfaces.md
- ../../spec/spec-controller-runtime-contracts.md

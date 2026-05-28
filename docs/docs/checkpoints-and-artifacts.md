## Checkpoints And Artifacts

This page documents training checkpoint and evaluation artifact behavior.

## Training Checkpoints

Checkpoint settings are defined in src/ehc_sn/callbacks/checkpoint.py via:

- CheckpointSettings
- CheckpointCallback

Key behaviors:

- supports metric-monitored top-k retention
- supports step or epoch cadence
- supports save_last and save_weights_only

## Eval Weights Companion Artifact

On checkpoint save, the callback also writes a sibling artifact:

- eval-weights-only.pt

This artifact stores deduplicated weights-only state for evaluation workflows.

## Evaluation Figure Bundles

Evaluation regime persistence is handled by src/ehc_sn/eval/artifacts.py.

Bundle artifacts include:

- persisted case trace payloads
- summary rows
- manifest metadata

Offline report figures can be rendered from these persisted runs using
src/ehc_sn/eval/reports.py.

## Benchmark Reports

Benchmark wrappers write normalized reports to:

- reports/benchmarks/<track>-<model>.json

## Artifact Placement Guidance

- Keep training checkpoints under checkpoints/.
- Keep benchmark outputs under reports/benchmarks.
- Keep task/shared datasets under data/processed only.

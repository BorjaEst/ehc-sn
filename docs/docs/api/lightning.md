## Lightning Surface

This page renders the public Lightning package surface.

::: ehc_sn.lightning

## Regime Modules

`lightning/modules/` provides reusable LightningModules organized by
training regime, not by task or model family. Each module is parameterized
by component classes injected via a `Components` bundle.

::: ehc_sn.lightning.modules

## Experiment Compositions

`experiments/` provides explicit composition modules — one per
`(task, family, version)` triple. Each module wires concrete model,
adapter, controller, and objective classes into a regime module.

::: ehc_sn.experiments

## Runtime Notes

- Lightning modules may expose `execute_evaluation_batch` seams for
  replay-evaluation integration.
- Training callbacks schedule logging, checkpoints, figures, and evaluation
  regimes.
- Training scripts import from `experiments/`, not from `lightning/`
  directly.

## Related Modules

- `ehc_sn.callbacks`
- `ehc_sn.eval`
- `ehc_sn.experiments`
- `ehc_sn.objectives`
- `ehc_sn.controllers`
- `ehc_sn.training`

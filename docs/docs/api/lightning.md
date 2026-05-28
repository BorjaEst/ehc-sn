## Lightning Surface

This page renders the public Lightning package surface.

::: ehc_sn.lightning

## Selected Lightning Modules

The Lightning family modules provide executable training orchestration and
connect model families to adapters.

::: ehc_sn.lightning.tem.tem_v1

::: ehc_sn.lightning.hrm

::: ehc_sn.lightning.ehc

## Runtime Notes

- Lightning modules may expose `execute_evaluation_batch` seams for
  replay-evaluation integration.
- Training callbacks schedule logging, checkpoints, figures, and evaluation
  regimes.

## Related Modules

- `ehc_sn.callbacks`
- `ehc_sn.eval`
- `ehc_sn.objectives`
- `ehc_sn.controllers`

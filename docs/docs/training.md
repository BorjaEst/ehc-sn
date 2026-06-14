## Training

Training entrypoints are thin wrappers under scripts/training.

## Available Entrypoints

- scripts/training/tem_v1_baseline.py
- scripts/training/tem_v2_softmax.py
- scripts/training/hrm_v1_baseline.py
- scripts/training/hrm_v2_rl-striatum.py
- scripts/training/ehc_v1_pretraining.py

Inspect any script surface with:

```bash
python scripts/training/<entrypoint>.py --help
```

## Default Config Loading

Each entrypoint loads TOML defaults from config/training and supports
environment-variable override of the config path.

Examples:

```bash
export TEM_V1_CONFIGURATION_PATH=config/training/tem-v1-arena-vram8gib.toml
python scripts/training/tem_v1_baseline.py
```

```bash
export HRM_V1_CONFIGURATION_PATH=config/training/hrm-v1-mazehard-vram8gib.toml
python scripts/training/hrm_v1_baseline.py
```

## Distributed Training Notes

Training scripts validate global batch size divisibility against effective world
size and choose trainer strategy accordingly.

## Logging And Checkpoints

Training scripts integrate callback-managed logging and checkpointing through
Lightning callbacks in src/ehc_sn/callbacks.

## Constraints

- scripts/training must remain thin entrypoints.
- Reusable training logic belongs under src/ehc_sn.

## Related Specs

- ../../spec/spec-requirements.md
- ../../spec/spec-architecture.md

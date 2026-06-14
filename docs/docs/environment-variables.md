## Environment Variables

This page lists high-impact environment variables used by entrypoints.

## Training Config Path Overrides

| Variable                  | Default                                       |
| ------------------------- | --------------------------------------------- |
| TEM_V1_CONFIGURATION_PATH | config/training/tem-v1-arena-vram8gib.toml    |
| TEM_V2_CONFIGURATION_PATH | config/training/tem-v2-arena-vram8gib.toml    |
| HRM_V1_CONFIGURATION_PATH | config/training/hrm-v1-mazehard-vram8gib.toml |
| HRM_V2_CONFIGURATION_PATH | config/training/hrm-v2-mazehard-vram8gib.toml |
| EHP_V1_CONFIGURATION_PATH | config/training/ehp-v1-spatial-vram8gib.toml  |

Example:

```bash
export HRM_V1_CONFIGURATION_PATH=config/training/hrm-v1-mazehard-vram8gib.toml
python scripts/training/hrm_v1_baseline.py
```

## Distributed Training Context

When using torchrun or cluster launchers, standard environment variables such
as LOCAL_RANK and WORLD_SIZE can affect device and strategy selection inside
training scripts.

## Recommendation

Keep environment-variable overrides in launcher scripts and commit canonical
TOML defaults under config/training.

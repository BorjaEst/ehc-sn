# Scripts

CLI entry points for the EHP-SN project. Scripts call library APIs from
`ehc_sn.*` — they contain no business logic, metric computation, adapter
dispatch, or figure rendering.

## Evaluation

**Script:** `scripts/evaluation/run_eval.py` (generic) + per-task aliases

The generic CLI exposes three subcommands: `run`, `inspect`, `benchmark`.
Per-task aliases bind a default config for convenience.

All evaluation logic lives in `src/ehc_sn.eval` — scripts are thin wrappers.

### Run — execute a checkpoint against an experiment config

```bash
# Generic (any experiment):
python scripts/evaluation/run_eval.py run \
    --config config/evaluation/hrm-v1-mazehard.toml \
    --checkpoint checkpoints/hrm-v1/best.ckpt \
    --output artifacts/eval/mazehard-example \
    --device cpu

# Per-task alias (same, without --config):
python scripts/evaluation/mazehard/hrm_v1.py run \
    --checkpoint checkpoints/hrm-v1/best.ckpt \
    --output artifacts/eval/mazehard-example \
    --device cpu
```

### Inspect — read a completed evaluation artifact

```bash
python scripts/evaluation/run_eval.py inspect \
    artifacts/eval/mazehard-example \
    --case 0 --list-fields
```

### Benchmark — not yet implemented

```bash
python scripts/evaluation/run_eval.py benchmark placeholder
```

### Available aliases

| Command | Default config |
|---|---|
| `scripts/evaluation/mazehard/hrm_v1.py run` | `config/evaluation/hrm-v1-mazehard.toml` |
| `scripts/evaluation/mazehard/hrm_v2.py run` | `config/evaluation/hrm-v2-mazehard.toml` |
| `scripts/evaluation/goaltrace/hrm_v1.py run` | `config/evaluation/hrm-v1-goaltrace.toml` |
| `scripts/evaluation/routebind/hrm_v1.py run` | `config/evaluation/hrm-v1-routebind.toml` |
| `scripts/evaluation/seqmaze/hrm_v1.py run` | `config/evaluation/hrm-v1-seqmaze.toml` |
| `scripts/evaluation/seqmaze/hrm_v2.py run` | `config/evaluation/hrm-v2-seqmaze.toml` |
| `scripts/evaluation/arena/tem_v1.py run` | `config/evaluation/tem-v1-arena.toml` |
| `scripts/evaluation/arena/tem_v2.py run` | `config/evaluation/tem-v2-arena.toml` |

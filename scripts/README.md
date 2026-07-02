# Scripts

CLI entry points for the EHP-SN project. Scripts call library APIs from
`ehp_sn.*` — they contain no business logic, metric computation, adapter
dispatch, or figure rendering.

## Evaluation

Evaluation is managed through a single CLI entry point installed as `ehp`:

```bash
ehp evaluation run ALIAS --model MODEL_REF [OPTIONS]
```

The alias selects a registered task--model evaluation recipe. See
`ehp evaluation run --help` for all supported options.

### Run — execute a checkpoint against a recipe

```bash
ehp evaluation run arena-tem-v1 \
    --model models/tem-v1-arena/best.pt \
    --device auto

ehp evaluation run mazehard-hrm-v1 \
    --model checkpoints/hrm-v1-mazehard/best.ckpt \
    --split test \
    --count 64
```

### Inspect — read a completed evaluation artifact

```bash
ehp evaluation inspect artifacts/evaluation/arena-tem-v1-20260627/

ehp evaluation inspect artifacts/evaluation/mazehard-hrm-v1/ --gallery
```

### Supported aliases

```
arena-tem-v1
arena-tem-v2
goaltrace-hrm-v1
mazehard-hrm-v1
mazehard-hrm-v2
routebind-hrm-v1
seqmaze-hrm-v1
seqmaze-hrm-v2
```

Legacy experiment IDs (e.g. `tem-v1-arena`) are accepted with a deprecation
warning.

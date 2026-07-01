# Evaluate Models — `ehp-sn evaluate`

The `evaluate` command group resolves one evaluation specification, applies it
to one immutable model reference and one immutable dataset reference, and
commits one provenance-complete artifact set.

```
ehp-sn evaluate [OPTIONS] COMMAND [ARGS]...
```

**Design principle**: `evaluate` owns **production** of an immutable evaluation
artifact set. It does not own interactive inspection, result comparison, figure
experimentation, or reporting. Those are downstream concerns that consume the
artifact set through separate APIs.

---

## Commands

| Command | Purpose |
|---|---|
| [`run`](#run) | Execute an evaluation and commit its artifact set |
| [`plan`](#plan) | Resolve and display an evaluation without executing it |
| [`validate`](#validate) | Preflight validation of evaluation inputs and compatibility |
| [`recipes`](#recipes) | Discover registered evaluation recipes |

---

## `evaluate run`

```
ehp-sn evaluate run RECIPE --model MODEL_REF [OPTIONS]
```

Execute a resolved evaluation recipe and commit a canonical artifact set.

### Positional arguments

| Argument | Description |
|---|---|
| `RECIPE` | Registered evaluation recipe name (e.g. `arena-tem-v1`, `mazehard-hrm-v1`) |

### Principal options

| Option | Description | Default |
|---|---|---|
| `--model` `MODEL_REF` | **Required.** Checkpoint path, run URI, or registered model reference | — |
| `--dataset` `DATASET_REF` | Override the dataset declared by the recipe | recipe default |
| `--split` `SPLIT` | Override the evaluation split | recipe default |
| `--output` / `-o` `DIR` | Destination for the committed evaluation artifact set | `artifacts/evaluations/<alias>/<evaluation_id>/` |

### Execution policy options

| Option | Description | Default |
|---|---|---|
| `--regime` `NAME` | Run only the named regime. Repeatable (`--regime diagnostic --regime test`) | all regimes |
| `--device` `DEVICE` | Execution device or accelerator selector | `auto` |
| `--precision` `PRECISION` | Runtime precision override (e.g. `16-mixed`, `bf16-mixed`) | recipe default |
| `--seed` `INT` | Evaluation seed override | recipe default |
| `--determinism` `MODE` | Deterministic-execution enforcement level. One of `off`, `warn`, `strict` | `warn` |
| `--capture-profile` `PROFILE` | Override the recipe trace-capture profile | recipe default |
| `--max-cases` `INT` | Bound the number of evaluated cases (≥ 1) | recipe default |
| `--existing` `POLICY` | Policy when the target evaluation already exists. One of `error`, `resume`, `replace` | `error` |
| `--publish` / `--no-publish` | Publish the committed artifact set to configured tracking | `True` |

### Output formatting options

| Option | Description | Default |
|---|---|---|
| `--format` `FORMAT` | Terminal result format. One of `text`, `json` | `text` |

### Examples

```bash
# Standard execution
ehp-sn evaluate run arena-tem-v1 --model best.ckpt

# Explicit dataset and runtime placement
ehp-sn evaluate run arena-tem-v1 \
    --model runs:/01J.../checkpoints/best.ckpt \
    --dataset arena:v3@test \
    --device cuda

# Bounded evaluation with explicit output directory
ehp-sn evaluate run mazehard-hrm-v1 \
    --model artifacts/models/mazehard-hrm-v1/best.ckpt \
    --max-cases 32 \
    --output data/evaluations/mazehard-hrm-v1/ablations/032

# JSON output for programmatic consumption
ehp-sn evaluate run arena-tem-v1 \
    --model best.ckpt \
    --format json
```

### Responsibilities

1. **Resolve** the evaluation recipe, model checkpoint, and dataset.
2. **Construct** the evaluation runtime (executor, provider, controllers, objectives).
3. **Execute** inference and metric aggregation under `torch.inference_mode()`.
4. **Capture** configured traces and cases via the `EvaluationConsumer` lifecycle.
5. **Materialise** figures only when the recipe declares them.
6. **Write** manifests, metrics, provenance, diagnostics, and completion state.
7. **Publish** the committed artifact set to MLflow (unless `--no-publish`).

### Must not

- compare two evaluations;
- generate reports;
- provide general-purpose trace exploration;
- mutate the checkpoint;
- silently infer a test dataset;
- silently overwrite a completed evaluation.

---

## `evaluate plan`

```
ehp-sn evaluate plan RECIPE --model MODEL_REF [OPTIONS]
```

Resolve everything without loading the full model or executing inference.
Displays or emits the effective evaluation configuration as a stable,
inspectable object.

### Options

Accepts the same `--model`, `--dataset`, `--split`, `--regime`, `--seed`,
`--capture-profile`, `--max-cases`, and `--device` options as [`run`](#run).

| Option | Description | Default |
|---|---|---|
| `--format` `FORMAT` | Output format. One of `text`, `json` | `text` |

### Plan contents

| Field | Description |
|---|---|
| Recipe ID and version | Canonical recipe alias and schema version |
| Model reference and digest | Resolved path, model family, checkpoint digest |
| Dataset URI, split, and digest | Resolved dataset identity |
| Evaluation regimes | Regime IDs, kinds, and capture profiles |
| Metric set and primary metric | Full metric key set plus designated primary metric |
| Case-selection policy | Split, count, seed, batch size |
| Trace-capture profile | Profile name, field list, max cases |
| Expected output location | Target artifact directory |
| Collision policy | What happens if `--existing` triggers |
| Reproducibility warnings | Missing seeds, non-deterministic ops, etc. |

### Examples

```bash
# Human-readable plan
ehp-sn evaluate plan arena-tem-v1 --model best.ckpt

# Machine-readable plan
ehp-sn evaluate plan arena-tem-v1 \
    --model best.ckpt \
    --format json
```

---

## `evaluate validate`

```
ehp-sn evaluate validate RECIPE --model MODEL_REF [OPTIONS]
```

Perform preflight validation without executing the evaluation. Answers the
question: **can this evaluation run correctly?**

Accepts the same `--model`, `--dataset`, `--split`, `--regime`, `--seed`,
`--capture-profile`, and `--device` options as [`run`](#run).

### Validation scope

| Category | Checks |
|---|---|
| **Recipe** | Schema validity, registration completeness |
| **Model** | Checkpoint readability, metadata compatibility, model-family match |
| **Task** | Task/model/evaluator compatibility, adapter resolution |
| **Dataset** | Existence, schema, split availability, digest match |
| **Trace** | Required trace fields, capture profile validity |
| **Metrics** | Metric availability, scoring-spec compatibility |
| **Figures** | Figure-plan compatibility, registered figure names |
| **Runtime** | Device availability, output writability, dependency availability |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All checks passed |
| `2` | CLI usage or parsing error |
| `3` | Configuration or recipe error |
| `4` | Reference resolution error |
| `5` | Compatibility or validation failure |

### Example

```bash
# CI preflight
ehp-sn evaluate validate arena-tem-v1 --model best.ckpt
```

---

## `evaluate recipes`

Discover and describe registered evaluation recipes.

### `evaluate recipes list`

```
ehp-sn evaluate recipes list [OPTIONS]
```

List all registered evaluation recipe aliases.

```bash
ehp-sn evaluate recipes list

# arena-tem-v1       arena × tem-v1
# arena-tem-v2       arena × tem-v2
# mazehard-hrm-v1    mazehard × hrm-v1
# mazehard-hrm-v2    mazehard × hrm-v2
# routebind-hrm-v1   routebind × hrm-v1
# seqmaze-hrm-v1     seqmaze × hrm-v1
# seqmaze-hrm-v2     seqmaze × hrm-v2
# goaltrace-hrm-v1   goaltrace × hrm-v1
```

| Option | Description | Default |
|---|---|---|
| `--format` `FORMAT` | Output format. One of `text`, `json` | `text` |

### `evaluate recipes show`

```
ehp-sn evaluate recipes show RECIPE [OPTIONS]
```

Show full details of one registered evaluation recipe.

```bash
ehp-sn evaluate recipes show arena-tem-v1
```

| Option | Description | Default |
|---|---|---|
| `--format` `FORMAT` | Output format. One of `text`, `json` | `text` |

### Recipe display contents

| Field | Description |
|---|---|
| `alias` | Canonical recipe alias |
| `task` | Task-family identifier |
| `model_family` | Model-family identifier |
| `primary_metric` | Canonical primary metric name |
| `required_capabilities` | Model capabilities the artifact must satisfy |
| `cases` | Default case-selection policy (split, count, seed) |
| `dataset` | Default dataset URI |
| `evaluation` | Task-specific evaluation parameters |
| `capture` | Trace-capture profile name and limits |
| `figures` | Default inspection figure names |

---

## Configuration precedence

```
recipe TOML defaults  <  invocation TOML  <  CLI flags  <  --set overrides
```

- **Recipe defaults** — baked into `config/evaluation/recipes/<alias>.toml`.
- **Invocation TOML** — optional file providing overrides for a single run.
- **CLI flags** — override both recipe and invocation file values.
- **`--set` overrides** — generic key=value mechanism for exploratory
  exceptions. Every `--set` override is recorded verbatim in provenance.

CLI options may alter execution placement or select a bounded subset; they must
not silently redefine what the evaluation means.

---

## Artifact lifecycle

```
resolve → validate → stage → execute → finalize → commit
```

1. **Resolve** recipe, model, dataset, and runtime parameters.
2. **Validate** compatibility and availability.
3. **Stage** — write into a staging directory (`<output>/.tmp/...`).
4. **Execute** inference, metric aggregation, trace capture, figure rendering.
5. **Finalize** consumer artifacts, analysis specs, and manifests.
6. **Commit** — atomically finalise the output directory and write `_SUCCESS`.

### On-disk layout

```
artifacts/evaluations/<alias>/<evaluation_id>/
├── manifest.json               ← EvaluationManifest (identity, provenance, regimes)
├── resolved-recipe.toml         ← Canonical resolved recipe with all defaults
├── provenance.json              ← Model ref, dataset ref, CLI overrides, timestamps
├── metrics.json                 ← Scalar summary metrics
├── diagnostics.json             ← Structured failure or warning record
├── <regime_kind>/               ← e.g. "diagnostic"
│   ├── manifest.json            ← RegimeArtifactManifest
│   ├── metrics.json             ← Per-regime metrics
│   ├── traces/                  ← Zarr archives + trace_index.json
│   ├── aggregates/              ← Zarr groups (spatial populations, etc.)
│   ├── figures/                 ← PNG / PDF outputs
│   └── _SUCCESS
└── _SUCCESS                     ← Commit sentinel (empty file)
```

### Failure semantics

On failure:

- retain a structured failure record when useful;
- never make a partial directory look completed;
- return a nonzero exit code;
- include the staging path in diagnostic output;
- do not register the run as successful.

---

## Stable evaluation identity

A canonical evaluation ID derives from the semantic inputs, not from a
mutable timestamp or alias:

```
evaluation_id = hash(
    recipe_digest,
    model_digest,
    dataset_digest,
    split,
    regimes,
    semantic_overrides,
    evaluator_version,
)
```

| Input change | Identity change |
|---|---|
| Different model checkpoint | Different ID |
| Different dataset version | Different ID |
| Different dataset split | Different ID |
| Device change (cuda:0 → cuda:1) | **Same** ID (device is placement, not semantics) |
| Capture profile change | Same or different per contract |

The timestamp is metadata, not identity.

---

## Model references

`MODEL_REF` is an opaque typed reference resolved by the application layer.

### Supported syntaxes

| Syntax | Example | Resolver |
|---|---|---|
| Local path | `./checkpoints/best.ckpt` | Local checkpoint loader |
| Run artifact | `runs:/01J.../checkpoints/best.ckpt` | MLflow run artifact resolver |
| Registered model | `models:/arena-tem-v1@production` | MLflow model registry resolver |
| Content-addressed | `sha256:a18d...` | Content-addressable store resolver |

### Resolved model

```python
@dataclass(frozen=True)
class ResolvedModel:
    source_ref: str          # Original user-supplied reference
    local_path: Path         # Local filesystem path to the checkpoint
    digest: str              # SHA-256 hex digest of the checkpoint bytes
    family: str              # Model family string (e.g. "tem-v1")
    schema_version: str | None
    run_id: str | None       # MLflow run ID, if applicable
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | CLI usage or parsing error |
| `3` | Configuration or recipe error |
| `4` | Reference resolution error (model, dataset) |
| `5` | Compatibility or validation failure |
| `6` | Evaluation execution failure |
| `7` | Artifact commit or publication failure |
| `8` | Threshold or gate failure (future) |

---

## Evaluation versus validation gates

Metric threshold enforcement is explicitly modelled as a separate concern.

**Preferred**: `evaluate run` computes and persists results. A separate
downstream command performs acceptance:

```bash
ehp-sn evaluation check <ARTIFACT> --policy release-gate
```

**Acceptable compact design**: Add an optional gate to `evaluate run`:

```bash
ehp-sn evaluate run arena-tem-v1 --model best.ckpt --gate release
```

The artifacts are still committed, but the process exits with code `8` when
the gate fails.

---

## Canonical examples

```bash
# Standard execution
ehp-sn evaluate run arena-tem-v1 --model best.ckpt

# Explicit dataset and runtime placement
ehp-sn evaluate run arena-tem-v1 \
    --model runs:/01J.../checkpoints/best.ckpt \
    --dataset arena:v3@test \
    --device cuda

# Bounded evaluation
ehp-sn evaluate run mazehard-hrm-v1 \
    --model best.ckpt \
    --max-cases 32 \
    --output data/evaluations/mazehard-hrm-v1/ablations/032

# Resolve effective configuration (no execution)
ehp-sn evaluate plan arena-tem-v1 \
    --model best.ckpt \
    --format json

# CI preflight
ehp-sn evaluate validate arena-tem-v1 \
    --model best.ckpt

# Recipe discovery
ehp-sn evaluate recipes list
ehp-sn evaluate recipes show arena-tem-v1
```

---

## Related commands

| Command | Relationship |
|---|---|
| [`ehp-sn train run`](train.md) | Produces the checkpoints that `evaluate run` consumes |
| `ehp-sn evaluation show` | Inspect a completed evaluation artifact (separate concern) |
| `ehp-sn evaluation compare` | Compare results across evaluation artifacts (separate concern) |
| `ehp-sn figures render` | Generate publication/report figures from evaluation artifacts (separate concern) |
| `ehp-sn report build` | Produce report-data packages from evaluation artifacts (separate concern) |

---

## See also

- [Evaluation concepts](../usage/evaluation.md) — conceptual overview of
  evaluation regimes, recipes, and consumers
- [Checkpoints and Artifacts](../usage/checkpoints-and-artifacts.md) —
  artifact layout and lifecycle
- [Configuration reference](../configuration.md) — evaluation configuration
  schema and recipe format

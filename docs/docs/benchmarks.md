## Benchmarks

Benchmark entrypoints are thin wrappers in scripts/benchmarks.

## Available Wrappers

- scripts/benchmarks/mazehard-delib.py
- scripts/benchmarks/b0-mazehard.py
- scripts/benchmarks/arena-struct.py

Inspect surfaces with:

```bash
python scripts/benchmarks/mazehard-delib.py --help
python scripts/benchmarks/b0-mazehard.py --help
```

## Input Contracts

Benchmark wrappers consume benchmark configuration and artifact metadata from
TOML and JSON payloads, then emit normalized reports.

## Output Location

Canonical benchmark report output path:

- reports/benchmarks/<track>-<model>.json

Do not write benchmark artifacts under data/processed.

## Reporting Requirements

Repository-level reporting must preserve benchmark identity, split semantics,
seed treatment, and major caveats.

## Related Specs

- ../../spec/spec-benchmark-suite.md
- ../../spec/spec-benchmark-configuration-contracts.md
- ../../spec/spec-requirements.md

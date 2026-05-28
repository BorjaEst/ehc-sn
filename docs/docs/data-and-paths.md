## Data And Paths

Data and path constraints are normative in spec/spec-requirements.md.

## Canonical Data Pipeline

- data/raw
- data/interim
- data/processed

DataModules consume only processed data.

## Versioned Root Rules

Processed roots are immutable versioned leaves.

Two canonical classes:

- Shared substrate: data/processed/<shared-family>/v<integer>/
- Task corpus: data/processed/<task-name>/<corpus-name>/v<integer>/

## Reporting And Artifacts

Benchmark reports and manifests are not dataset contents.

Write benchmark outputs to:

- reports/benchmarks
- outputs

## Path Hygiene

- Avoid hard-coded absolute paths.
- Use CLI arguments or config values.

## Related Specs

- ../../spec/spec-requirements.md
- ../../spec/spec-data-contracts.md

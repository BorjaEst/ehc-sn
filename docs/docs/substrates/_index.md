# Substrates (Interim Data Generators)

Substrates are the intermediate, reusable parent artifacts that downstream
task builders consume. Each substrate family owns a set of task-neutral
channels — topology, observations, masks, regions — that describe a world
without task-specific protocol (trajectories, episodes, supervision).

## Pipeline Position

```text
Generators (substrates)         scripts/data-gen/
    │
    ▼
  data/raw/        pre-contract source specs / normalized staging
    │
    ▼
  data/interim/    immutable versioned shared substrate / layout dataset
    │  (manifest-bearing, versioned)
    ▼
  data/processed/  task corpus (consumed by DataModules)
```

Substrates live at the `data/interim/` layer. Task corpora live at
`data/processed/`. See [Data and Paths](../data-and-paths.md) for the
canonical path grammar.

## Registered Substrate Families

| Family       | Topology | Source                 | Path                                     | Downstream Tasks          |
| ------------ | -------- | ---------------------- | ---------------------------------------- | ------------------------- |
| `maze-nd`    | grid2d   | HuggingFace            | `data/interim/maze-nd/v<N>/`             | mazehard                  |
| `dungeongen` | grid2d   | procedural (local lib) | `data/interim/dungeongen/<preset>/v<N>/` | dungeon, arena, routebind |
| `dagflow`    | dag      | synthetic              | `data/interim/dagflow/<preset>/v<N>/`    | seqmaze, goaltrace        |
| `openfield`  | grid2d   | synthetic              | `data/interim/openfield/<preset>/v<N>/`  | arena, routebind          |

## Common Properties

- **Immutability**: Versioned roots (`v<N>`) must not be modified after
  creation. Rebuilding requires bumping the version integer.
- **Transactional builds**: Builders write to a temporary sibling directory
  (`.building-v<N>`) and only rename after validation succeeds.
- **Manifest**: Every version root contains a `manifest.json` with
  `dataset_class`, `family`, `version`, `channels`, `topology_kind`, and
  reproducibility parameters.
- **Determinism**: All builders accept a `--seed` and store stage parameters
  in the manifest for reproducibility.

## Building Substrates

Each family has a staged CLI under `scripts/data-gen/`:

```bash
# maze-nd: fetch from HuggingFace, normalize, build
python scripts/data-gen/build-maze-nd.py build

# dungeongen: generate topologies, materialize layout dataset
python scripts/data-gen/build-dungeongen.py build

# dagflow: generate DAG topologies
python scripts/data-gen/build-dagflow.py build --preset balanced --version 1

# openfield: generate grid layouts, assign sensory IDs
python scripts/data-gen/build-openfield.py build

```

See the per-family pages for detailed stage breakdowns and parameters.

## Related

- [Data Generation](../data-generation.md) — full pipeline overview
- [Data and Paths](../data-and-paths.md) — canonical path grammar
- [Spec: Data Contracts](../../spec/spec-data-contracts.md)

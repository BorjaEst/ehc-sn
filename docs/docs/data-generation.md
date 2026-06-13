## Data Generation

Data generation CLIs live in scripts/data-gen and use staged commands.

## Pipeline Classes

- Shared substrate pipelines
- Task corpus pipelines layered on shared substrates

Canonical data flow remains:

- data/raw
- data/interim
- data/processed

## Shared-Substrate Builders

### dungeongen

Script: scripts/data-gen/build-dungeongen.py

Stages:

- generate-topology
- materialize-layouts
- validate
- build-all

### maze-nd (MazeHard substrate)

Script: scripts/data-gen/build-maze-nd.py

Stages:

- fetch-raw
- normalize
- materialize-shared
- validate
- build-all

### mazehard

Script: scripts/data-gen/build-mazehard.py

Stages:

- materialize-task
- validate

Requires maze-nd shared substrate.

### numberline

Script: scripts/data-gen/build-numberline.py

Stages:

- materialize-shared
- validate
- build-all

## Task-Corpus Builders

### arena

Script: scripts/data-gen/build-arena.py

Stages:

- materialize-task
- validate

Requires layout dataset from dungeongen or openfield.

### dungeon

Script: scripts/data-gen/build-dungeon.py

Stages:

- materialize-task
- validate

Requires dungeongen shared substrate.

### countwalk

Script: scripts/data-gen/build-countwalk.py

Stages:

- materialize-task
- validate

Requires numberline shared substrate.

## Typical Build Order

Example for Arena or Dungeon:

```bash
python scripts/data-gen/build-dungeongen.py build-all
python scripts/data-gen/build-arena.py materialize-task ...
python scripts/data-gen/build-dungeon.py materialize-task ...
```

Example for Countwalk:

```bash
python scripts/data-gen/build-numberline.py build-all
python scripts/data-gen/build-countwalk.py materialize-task ...
```

Example for MazeHard:

```bash
python scripts/data-gen/build-maze-nd.py build-all
python scripts/data-gen/build-mazehard.py materialize-task ...
```

## Validation

Each data-gen CLI exposes a validate command for family/task checks and
manifest consistency.

## Related Specs

- ../../spec/spec-requirements.md
- ../../spec/spec-data-contracts.md

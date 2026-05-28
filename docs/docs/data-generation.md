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

- fetch-raw
- prepare-interim
- materialize-shared
- validate
- build-all

### maze-nd (MazeHard substrate)

Script: scripts/data-gen/build-mazehard.py

Stages:

- fetch-raw
- prepare-interim
- materialize-shared
- materialize-task
- validate
- build-all

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
- build-all

Requires dungeongen shared substrate.

### dungeon

Script: scripts/data-gen/build-dungeon.py

Stages:

- materialize-task
- validate
- build-all

Requires dungeongen shared substrate.

### countwalk

Script: scripts/data-gen/build-countwalk.py

Stages:

- materialize-task
- validate
- build-all

Requires numberline shared substrate.

## Typical Build Order

Example for Arena or Dungeon:

```bash
python scripts/data-gen/build-dungeongen.py build-all
python scripts/data-gen/build-arena.py build-all
python scripts/data-gen/build-dungeon.py build-all
```

Example for Countwalk:

```bash
python scripts/data-gen/build-numberline.py build-all
python scripts/data-gen/build-countwalk.py build-all
```

## Validation

Each data-gen CLI exposes a validate command for family/task checks and
manifest consistency.

## Related Specs

- ../../spec/spec-requirements.md
- ../../spec/spec-data-contracts.md

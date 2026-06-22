# numberline Shared Substrate

## Identity

| Property       | Value                                   |
| -------------- | --------------------------------------- |
| Family         | `numberline`                            |
| Topology kind  | `line1d`                                |
| Source         | synthetic (generated on-the-fly)        |
| Source ID      | `synthetic/numberline`                  |
| CLI script     | `scripts/data-gen/build-numberline.py`  |
| Builder module | `ehc_sn.data.substrate.numberline`      |
| Output path    | `data/processed/numberline/v<version>/` |
| Dataset class  | `shared_substrate`                      |

## Description

The numberline shared substrate is the simplest substrate in the family. It
represents N states on a bounded 1-D line (`0 .. n_states-1`) with no walls,
no sensory observations, and no 2-D geometry. All worlds share the same
transition graph; there is no world-local variation.

Numberline is synthetic — it materializes directly to `data/processed/` with
no raw download or interim staging step.

## Channels

| Channel      | Dtype | Shape       | Description                                      |
| ------------ | ----- | ----------- | ------------------------------------------------ |
| `state_ids`  | int32 | (n_states,) | Consecutive integer state identifiers.           |
| `valid_prev` | bool  | (n_states,) | `True` where PREV is legal (`False` at state 0). |
| `valid_next` | bool  | (n_states,) | `True` where NEXT is legal (`False` at last).    |

Every sample is structurally identical for a given `n_states`. The only
distinguishing attributes are the world index and the split assignment.

## Pipeline Stages

| Stage                | Command                                  | Description                              |
| -------------------- | ---------------------------------------- | ---------------------------------------- |
| `materialize-shared` | `build-numberline.py materialize-shared` | Synthesize the shared substrate.         |
| `validate`           | `build-numberline.py validate`           | Validate manifest and channel contracts. |
| `build-all`          | `build-numberline.py build-all`          | Run materialize-shared.                  |

There is no raw or interim stage — worlds are generated directly.

## Default Parameters

| Parameter    | Default | Description                          |
| ------------ | ------- | ------------------------------------ |
| `--version`  | 1       | Substrate version integer.           |
| `--n-states` | 200     | Number of states on the number line. |
| `--n-worlds` | 100     | Total world samples to materialise.  |
| `--seed`     | 42      | Deterministic base seed.             |

Split ratios: 70% train, 15% val, 15% test (rounded to integers with a
minimum of 1 per split).

## Usage Examples

```bash
# Default build
python scripts/data-gen/build-numberline.py build-all

# Custom parameters
python scripts/data-gen/build-numberline.py materialize-shared \
    --n-states 20 --n-worlds 200

# Validate an existing root
python scripts/data-gen/build-numberline.py validate data/processed/numberline/v2

# Build a CountWalk task corpus from this substrate
python scripts/data-gen/build-countwalk.py materialize-task \
    --parent-substrate data/processed/numberline/v2
```

## Downstream Consumers

| Task      | CLI Script                            | Parent Substrate Path             |
| --------- | ------------------------------------- | --------------------------------- |
| countwalk | `scripts/data-gen/build-countwalk.py` | `data/processed/numberline/v<N>/` |

## Related

- [Spec: Data Contracts §3.1](../../spec/spec-data-contracts.md)
- [CountWalk Task Builder](../../scripts/data-gen/build-countwalk.py)

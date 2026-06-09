"""CLI for building the Countwalk task corpus.

Countwalk consumes a parent NumberLine shared substrate and generates episodic
replay trajectories over it, organised into five evaluation buckets:

  Bucket 0 – ID           values 0..63,   horizon 1..8   (train + val + test)
  Bucket 1 – range_ood    values 64..99,  horizon 1..8   (test only)
  Bucket 2 – horizon_ood  values 0..63,   horizon 9..16  (test only)
  Bucket 3 – joint_ood    values 64..99,  horizon 9..16  (test only)
  Bucket 4 – stretch_ood  values 100..199, horizon 1..16 (test only)

Train and validation splits contain only bucket 0 (ID) episodes.
The test split contains all five buckets, each balanced across the four
protocol cells (digit/set cue × anchor-only/sparse-reanchor).

Stages
------
materialize-task  Build the Countwalk task corpus over a NumberLine substrate.
validate          Validate a Countwalk task-corpus version root (generic + family).
build-all         Convenience alias: materialize-task (requires substrate to exist).

Default paths
-------------
Shared substrate:  data/processed/numberline/v1
Task corpus:       data/processed/countwalk/default/v1

Prerequisites
-------------
A NumberLine shared substrate (n_states=200) must exist before running.
Build it first::

    python scripts/data-gen/build-numberline.py build-all

Examples
--------
Build the default Countwalk corpus::

    python build-countwalk.py build-all

With explicit versions and custom parameters::

    python build-countwalk.py materialize-task --shared-version 2 --version 2 --n-episodes-per-world 80

Validate an existing corpus::

    python build-countwalk.py validate
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.substrate.numberline import SHARED_FAMILY as NUMBERLINE_FAMILY
from ehc_sn.tasks.countwalk import (
    TASK_FAMILY,
    build_countwalk_task_corpus,
    validate_countwalk_task_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_SHARED_VERSION = 1
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_EPISODES_PER_WORLD = 40  # must be divisible by 4; benchmark default
_DEFAULT_ID_MAX_STEPS = 8  # horizon cap for ID and range-OOD buckets
_DEFAULT_OOD_MAX_STEPS = (
    16  # horizon cap for horizon-OOD, joint-OOD, stretch-OOD
)
_DEFAULT_SEED = 42

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
@app.command("materialize-task")
def materialize_task(
    shared_version: Annotated[
        int,
        typer.Option(
            "--shared-version", help="Parent numberline substrate version."
        ),
    ] = _DEFAULT_SHARED_VERSION,
    version: Annotated[
        int, typer.Option("--version", help="Task corpus version.")
    ] = _DEFAULT_TASK_VERSION,
    corpus: Annotated[
        str, typer.Option("--corpus", help="Corpus label.")
    ] = _DEFAULT_CORPUS,
    n_episodes_per_world: Annotated[
        int,
        typer.Option(
            "--n-episodes-per-world",
            help="Episodes per world per split per bucket. Must be a positive multiple of 4.",
        ),
    ] = _DEFAULT_N_EPISODES_PER_WORLD,
    id_max_steps: Annotated[
        int,
        typer.Option(
            "--id-max-steps",
            help="Max horizon for ID and range-OOD buckets (benchmark default: 8).",
        ),
    ] = _DEFAULT_ID_MAX_STEPS,
    ood_max_steps: Annotated[
        int,
        typer.Option(
            "--ood-max-steps",
            help="Max horizon for horizon-OOD, joint-OOD, and stretch-OOD buckets (benchmark default: 16).",
        ),
    ] = _DEFAULT_OOD_MAX_STEPS,
    seed: Annotated[
        int, typer.Option("--seed", help="Deterministic base seed.")
    ] = _DEFAULT_SEED,
    processed_root: Annotated[
        Path, typer.Option("--processed-root", help="Processed data root.")
    ] = Path("data/processed"),
) -> None:
    """Build the Countwalk task corpus over a NumberLine shared substrate."""
    parent_substrate = processed_root / NUMBERLINE_FAMILY / f"v{shared_version}"
    version_root = processed_root / TASK_FAMILY / corpus / f"v{version}"
    typer.echo(f"Building Countwalk task corpus → {version_root}")
    typer.echo(f"  Parent substrate: {parent_substrate}")
    build_countwalk_task_corpus(
        version_root,
        parent_substrate=parent_substrate,
        corpus=corpus,
        n_episodes_per_world=n_episodes_per_world,
        id_max_steps=id_max_steps,
        ood_max_steps=ood_max_steps,
        seed=seed,
    )
    typer.echo("Done.")


@app.command("validate")
def validate(
    root: Annotated[
        Path, typer.Argument(help="Countwalk task-corpus root to validate.")
    ],
) -> None:
    """Validate a Countwalk task-corpus version root (generic + Countwalk family validators)."""
    validate_countwalk_task_root(root.resolve())
    typer.echo(f"OK  {root}")


# build-all removed — this script has a single stage: materialize-task.

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()

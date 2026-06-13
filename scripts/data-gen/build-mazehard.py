"""Staged CLI for building the MazeHard task corpus.

MazeHard consumes a maze-nd shared substrate and creates a task corpus
with MazeHard-specific protocol channels.

This CLI does not fetch raw data, prepare interim artifacts, or build
the shared substrate.  Those stages belong in ``build-maze-nd.py``.

Stages
------
materialize-task     Build the MazeHard task corpus over a shared substrate.
validate             Validate a MazeHard task-corpus version root.

Default paths
-------------
Shared substrate:  <user-specified --parent-substrate>
Task corpus:       data/processed/mazehard/<corpus>/v<version>

Examples
--------
Build the MazeHard task corpus against a maze-nd shared substrate::

    python build-mazehard.py materialize-task \\
        --parent-substrate data/interim/maze-nd/v2

With custom sizes::

    python build-mazehard.py materialize-task \\
        --parent-substrate data/interim/maze-nd/v2 \\
        --n-train 4000 --n-val 500 --n-test 500 --seed 7

Validate an existing task corpus::

    python build-mazehard.py validate data/processed/mazehard/default/v1

Prerequisites
-------------
A maze-nd shared substrate (schema version >= 2) must exist before running.
Build it first::

    python scripts/data-gen/build-maze-nd.py build-all

Or, with explicit version::

    python scripts/data-gen/build-maze-nd.py build-all --version 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.tasks.mazehard import build_mazehard_task_corpus

# ---------------------------------------------------------------------------
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_TRAIN = 200
_DEFAULT_N_VAL = 40
_DEFAULT_N_TEST = 40
_DEFAULT_SEED = 42

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
@app.command("materialize-task")
def materialize_task(
    parent_substrate: Annotated[
        Path,
        typer.Option(
            "--parent-substrate",
            help="Path to the maze-nd shared substrate version root.",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option(
            "--corpus",
            help=f"Corpus label (default: '{_DEFAULT_CORPUS}').",
        ),
    ] = _DEFAULT_CORPUS,
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train",
            help=f"Number of training samples (default: {_DEFAULT_N_TRAIN}).",
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val",
            help=f"Number of validation samples (default: {_DEFAULT_N_VAL}).",
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option(
            "--n-test",
            help=f"Number of test samples (default: {_DEFAULT_N_TEST}).",
        ),
    ] = _DEFAULT_N_TEST,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help=f"Task corpus version integer (default: {_DEFAULT_TASK_VERSION}).",
        ),
    ] = _DEFAULT_TASK_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=f"Deterministic base seed (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
) -> None:
    """Build the MazeHard task corpus over a maze-nd shared substrate.

    Reads all channels (topology, mask_valid, start, goals, solution) from
    the parent substrate.  Writes a versioned, immutable task corpus.

    Requires a maze-nd shared substrate with shared_schema_version >= 2.
    """
    task_root = Path(f"data/processed/mazehard/{corpus}/v{version}")
    build_mazehard_task_corpus(
        task_root.resolve(),
        parent_substrate=parent_substrate.resolve(),
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )


# ---------------------------------------------------------------------------
@app.command("validate")
def validate(
    root: Annotated[
        Path, typer.Argument(help="MazeHard task-corpus root to validate.")
    ],
) -> None:
    """Validate a MazeHard task-corpus version root (generic + MazeHard family validators)."""
    from ehc_sn.tasks.mazehard import validate_mazehard_task_root

    validate_mazehard_task_root(root.resolve())
    typer.echo(f"OK  {root}")


if __name__ == "__main__":
    app()

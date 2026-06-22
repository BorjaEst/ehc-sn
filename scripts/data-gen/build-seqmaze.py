"""Staged CLI for building the SeqMaze task corpus.

SeqMaze consumes a dagflow shared substrate and creates a task corpus
with path-prediction and edge-lookup protocol channels.

This CLI does not generate graph topology.  That stage belongs in
``build-dagflow.py``.

Stages
------
build      Build the SeqMaze task corpus over a shared substrate.
validate   Validate a SeqMaze task-corpus version root.
inspect    Print a human-readable summary of a version root manifest.

Default paths
-------------
Parent layout dataset:  <user-specified --layout-root>
Task corpus:            data/processed/seqmaze/<corpus>/v<version>

Examples
--------
Build the SeqMaze task corpus against a dagflow layout dataset::

    python build-seqmaze.py build \\
        --layout-root data/interim/dagflow/sparse/v1

With custom sizes::

    python build-seqmaze.py build \\
        --layout-root data/interim/dagflow/sparse/v1 \\
        --n-max 64 --t-max 48 --n-train 8000 --n-val 1000 --n-test 1000 --seed 7

Validate an existing task corpus::

    python build-seqmaze.py validate data/processed/seqmaze/default/v1

Inspect an existing corpus::

    python build-seqmaze.py inspect data/processed/seqmaze/default/v1

Prerequisites
-------------
A dagflow shared substrate must exist before running.
Build it first::

    python scripts/data-gen/build-dagflow.py build --preset branching --version 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.tasks.seqmaze.builder import (
    TASK_FAMILY,
    build_seqmaze_task_corpus,
    validate_seqmaze_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_MAX = 45
_DEFAULT_T_MAX = 16
_DEFAULT_MAX_OUT_DEGREE = 4
_DEFAULT_N_TRAIN = 4000
_DEFAULT_N_VAL = 500
_DEFAULT_N_TEST = 500
_DEFAULT_SEED = 42

app = typer.Typer(add_completion=False, help="SeqMaze task corpus builder.")


# =============================================================================
@app.command("build")
def build(
    layout_root: Annotated[
        Path,
        typer.Option(
            "--layout-root",
            help="Path to dagflow layout dataset root (e.g. data/interim/dagflow/sparse/v1).",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option("--corpus", help="Corpus label (default: 'default')."),
    ] = _DEFAULT_CORPUS,
    n_max: Annotated[
        int,
        typer.Option(
            "--n-max", help="Maximum candidate nodes N (default: 32)."
        ),
    ] = _DEFAULT_N_MAX,
    t_max: Annotated[
        int,
        typer.Option("--t-max", help="Maximum path length T (default: 32)."),
    ] = _DEFAULT_T_MAX,
    max_out_degree: Annotated[
        int,
        typer.Option(
            "--max-out-degree", help="Maximum out-degree K (default: 4)."
        ),
    ] = _DEFAULT_MAX_OUT_DEGREE,
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train", help="Number of training samples (default: 4000)."
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val", help="Number of validation samples (default: 500)."
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option("--n-test", help="Number of test samples (default: 500)."),
    ] = _DEFAULT_N_TEST,
    version: Annotated[
        int,
        typer.Option(
            "--version", help="Task corpus version integer (default: 1)."
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic base seed (default: 42)."),
    ] = _DEFAULT_SEED,
) -> None:
    """Build the SeqMaze task corpus over a dagflow layout dataset."""
    root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    build_seqmaze_task_corpus(
        version_root=root.resolve(),
        layout_root=layout_root.resolve(),
        corpus=corpus,
        n_max=n_max,
        t_max=t_max,
        max_out_degree=max_out_degree,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )
    typer.echo(f"SeqMaze corpus built at {root.resolve()}")
    typer.echo(f"  Profile: N={n_max}, T={t_max}, K={max_out_degree}")
    typer.echo(f"  S={n_max + t_max}, V={n_max + 2}")


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to validate (e.g. data/processed/seqmaze/default/v1)."
        ),
    ],
) -> None:
    """Validate an existing versioned root's manifest and data."""
    manifest = validate_seqmaze_root(root.resolve())
    typer.echo(f"SeqMaze corpus at {root.resolve()} is valid.")
    typer.echo(f"  Profile: N={manifest['n_max']}, T={manifest['t_max']}")
    typer.echo(
        f"  S={manifest['n_max'] + manifest['t_max']}, V={manifest['path_vocab_size']}"
    )
    typer.echo(f"  Samples: {manifest['n_samples']}")


# =============================================================================
@app.command("inspect")
def inspect(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to inspect (e.g. data/processed/seqmaze/default/v1)."
        ),
    ],
) -> None:
    """Print a human-readable summary of a SeqMaze version root manifest."""
    manifest = validate_seqmaze_root(root.resolve())
    typer.echo(f"Root: {root}")
    typer.echo(f"  dataset_class : {manifest.get('dataset_class', '?')}")
    typer.echo(f"  task          : {manifest.get('task', '?')}")
    typer.echo(f"  corpus        : {manifest.get('corpus', '?')}")
    typer.echo(f"  version       : {manifest.get('version', '?')}")
    typer.echo(f"  channels      : {manifest.get('channels', [])}")
    typer.echo(f"  n_samples     : {manifest.get('n_samples', {})}")
    typer.echo(f"  Profile: N={manifest['n_max']}, T={manifest['t_max']}")
    typer.echo(
        f"  S={manifest['n_max'] + manifest['t_max']}, V={manifest['path_vocab_size']}"
    )


# =============================================================================
if __name__ == "__main__":
    app()

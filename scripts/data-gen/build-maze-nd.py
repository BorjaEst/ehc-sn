"""Staged CLI for the maze-nd shared-family pipeline.

This script is the sole CLI owner of all maze-nd raw, interim, and shared
substrate stages. Task-level materialization is owned by the respective
task CLIs (build-mazehard.py).

Stages
------
fetch-raw            Download raw maze-nd corpus from HuggingFace.
normalize            Normalize raw corpus to raw staging records.
materialize-shared   Build the maze-nd shared substrate (consumable by build-mazehard).
validate             Validate a maze-nd shared-substrate version root.
build-all            Convenience alias: fetch-raw -> normalize -> materialize-shared.

Default paths
-------------
Raw corpus:           data/external/maze_hard_augmented
Normalized staging:   data/raw/maze-nd
Shared substrate:     data/interim/maze-nd/v{version}

Examples
--------
Quick local build::

    python build-maze-nd.py build-all

Custom sizes::

    python build-maze-nd.py build-all \\
        --n-train 4000 --n-val 500 --n-test 500 --seed 7

Validate an existing shared substrate::

    python build-maze-nd.py validate data/interim/maze-nd/v2

Build a MazeHard task corpus from the shared substrate::

    python build-mazehard.py materialize-task \\
        --parent-substrate data/interim/maze-nd/v2
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.maze_nd import (
    build_shared_substrate,
    ensure_raw,
)
from ehc_sn.data.substrate.maze_nd import prepare_interim as _normalize
from ehc_sn.data.substrate.maze_nd import (
    validate_maze_nd_shared_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_INTERIM_ROOT = Path("data/interim/maze-nd")
_DEFAULT_RAW_ROOT = Path("data/raw/maze-nd")
_DEFAULT_EXTERNAL_ROOT = Path("data/external/maze_hard_augmented")
_DEFAULT_VERSION = 1
_DEFAULT_N_TRAIN = 1000
_DEFAULT_N_VAL = 40
_DEFAULT_N_TEST = 40
_DEFAULT_SEED = 42


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("fetch-raw")
def fetch_raw(
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root",
            help="Root path for raw corpus (default: data/external/maze_hard_augmented).",
        ),
    ] = _DEFAULT_EXTERNAL_ROOT,
) -> None:
    """Download raw maze-nd corpus from HuggingFace."""
    ensure_raw(raw_root.resolve())
    typer.echo(f"Raw corpus at {raw_root}")


# =============================================================================
@app.command("normalize")
def normalize(
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root",
            help="Root path for raw corpus "
            "(default: data/external/maze_hard_augmented).",
        ),
    ] = _DEFAULT_EXTERNAL_ROOT,
    normalize_root: Annotated[
        Path,
        typer.Option(
            "--normalize-root",
            help="Destination normalized staging root (default: data/raw/maze-nd).",
        ),
    ] = _DEFAULT_RAW_ROOT,
) -> None:
    """Normalize raw corpus to a deterministic normalized staging artifact.

    Writes one uncompressed JSONL file per split under *normalize_root*.
    """
    _normalize(raw_root.resolve(), normalize_root.resolve())
    typer.echo(f"Normalized staging written to {normalize_root}")


# =============================================================================
@app.command("materialize-shared")
def materialize_shared(
    interim_root: Annotated[
        Path,
        typer.Option(
            "--interim-root",
            help="Output root for shared substrate (default: data/interim/maze-nd).",
        ),
    ] = _DEFAULT_INTERIM_ROOT,
    normalize_root: Annotated[
        Path,
        typer.Option(
            "--normalize-root",
            help="Root path for normalized staging files (default: data/raw/maze-nd).",
        ),
    ] = _DEFAULT_RAW_ROOT,
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
            help=f"Version of the emitted shared substrate (default: {_DEFAULT_VERSION}).",
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=f"Deterministic base seed (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
) -> None:
    """Build the maze-nd shared substrate from normalized staging records.

    Writes a versioned, immutable shared substrate to
    ``{interim_root}/v{version}/``.
    """
    shared_root = interim_root / f"v{version}"
    build_shared_substrate(
        shared_root.resolve(),
        interim_root=normalize_root.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(help="Maze-nd version root to validate."),
    ],
) -> None:
    """Validate a maze-nd shared-substrate version root.

    Raises an error if the root is not a valid maze-nd shared substrate.
    """
    manifest = validate_maze_nd_shared_root(root.resolve())
    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    family        : {manifest['family']}")
    typer.echo(f"    version       : {manifest['version']}")
    typer.echo(f"    channels      : {manifest['channels']}")
    typer.echo(f"    n_samples     : {manifest['n_samples']}")
    typer.echo(
        f"    shared_schema_version : {manifest.get('shared_schema_version')}"
    )


# =============================================================================
@app.command("build-all")
def build_all(
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root",
            help="Root path for raw corpus (default: data/external/maze_hard_augmented).",
        ),
    ] = _DEFAULT_EXTERNAL_ROOT,
    normalize_root: Annotated[
        Path,
        typer.Option(
            "--normalize-root",
            help="Root path for normalized staging files (default: data/raw/maze-nd).",
        ),
    ] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[
        Path,
        typer.Option(
            "--interim-root",
            help="Output root for shared substrate (default: data/interim/maze-nd).",
        ),
    ] = _DEFAULT_INTERIM_ROOT,
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
            help=f"Version of the emitted shared substrate (default: {_DEFAULT_VERSION}).",
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=f"Deterministic base seed (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
) -> None:
    """Full pipeline: fetch-raw -> normalize -> materialize-shared."""
    fetch_raw(raw_root=raw_root)
    normalize(raw_root=raw_root, normalize_root=normalize_root)
    materialize_shared(
        normalize_root=normalize_root,
        interim_root=interim_root,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        version=version,
        seed=seed,
    )


# =============================================================================
if __name__ == "__main__":
    app()

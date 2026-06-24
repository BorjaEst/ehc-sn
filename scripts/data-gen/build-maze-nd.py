"""CLI for the maze-nd shared-family pipeline.

This script is the sole CLI owner of all maze-nd raw, staging, and shared
substrate stages. Task-level materialization is owned by the respective
task CLI (build-mazehard.py).

Commands
--------
build       Fetch raw corpus, normalize to staging, and build the
            maze-nd shared substrate.
validate    Validate a maze-nd shared-substrate version root.
inspect     Print a human-readable summary of a version root manifest.

Default paths
-------------
Raw corpus:           data/external/maze_hard_augmented
Normalized staging:   data/raw/maze-nd
Shared substrate:     data/interim/maze-nd/v{version}

Examples
--------
Quick local build with defaults::

    python build-maze-nd.py build

Custom sizes::

    python build-maze-nd.py build \\
        --n-train 4000 --n-val 500 --n-test 500 --seed 7

Validate an existing shared substrate::

    python build-maze-nd.py validate data/interim/maze-nd/v2

Inspect an existing root::

    python build-maze-nd.py inspect data/interim/maze-nd/v2

Build a MazeHard task corpus from the shared substrate::

    python build-mazehard.py build \\
        --substrate-root data/interim/maze-nd/v2
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.substrate.maze_nd import (
    build_shared_substrate,
    ensure_raw,
)
from ehc_sn.data.substrate.maze_nd import prepare_interim as _normalize
from ehc_sn.data.substrate.maze_nd import (
    validate_maze_nd_shared_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_EXTERNAL_ROOT = Path("data/external/maze_hard_augmented")
_DEFAULT_STAGING_ROOT = Path("data/raw/maze-nd")
_DEFAULT_INTERIM_ROOT = Path("data/interim/maze-nd")
_DEFAULT_VERSION = 1
_DEFAULT_N_TRAIN = 1000
_DEFAULT_N_VAL = 40
_DEFAULT_N_TEST = 40
_DEFAULT_SEED = 42


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(
    external_root: Annotated[
        Path,
        typer.Option(
            "--external-root",
            help="Root path for raw HuggingFace corpus "
            "(default: data/external/maze_hard_augmented).",
        ),
    ] = _DEFAULT_EXTERNAL_ROOT,
    staging_root: Annotated[
        Path,
        typer.Option(
            "--staging-root",
            help="Normalized staging root " "(default: data/raw/maze-nd).",
        ),
    ] = _DEFAULT_STAGING_ROOT,
    interim_root: Annotated[
        Path,
        typer.Option(
            "--interim-root",
            help="Output root for the shared substrate "
            "(default: data/interim/maze-nd).",
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
            help=f"Version of the emitted shared substrate "
            "(default: {_DEFAULT_VERSION}).",
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=f"Deterministic base seed (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Delete the existing version root before building, "
            "if present.",
        ),
    ] = False,
) -> None:
    """Fetch raw corpus, normalize to staging, and build the substrate.

    Downloads the HuggingFace raw corpus (if not already present),
    normalizes it to JSONL staging files, then writes a versioned,
    immutable shared substrate to ``{interim_root}/v{version}/``.
    """
    _validate_build_args(version, n_train, n_val, n_test)
    _check_dest_and_force(interim_root / f"v{version}", force)

    # Fetch raw HuggingFace corpus (skips if already present).
    ensure_raw(external_root.resolve())

    # Normalize raw records to staging JSONL files.
    _normalize(external_root.resolve(), staging_root.resolve())

    # Build the versioned shared substrate.
    build_shared_substrate(
        (interim_root / f"v{version}").resolve(),
        normalized_root=staging_root.resolve(),
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
        f"    artifact_schema_version : {manifest.get('artifact_schema_version')}"
    )


# =============================================================================
@app.command("inspect")
def inspect(
    root: Annotated[
        Path,
        typer.Argument(
            help="Maze-nd version root to inspect "
            "(e.g. data/interim/maze-nd/v1)."
        ),
    ],
) -> None:
    """Print a human-readable summary of a version root manifest."""
    manifest = validate_maze_nd_shared_root(root.resolve())

    typer.echo(f"Root: {root}")
    typer.echo(f"  dataset_class : {manifest.get('dataset_class', '?')}")
    typer.echo(f"  family        : {manifest.get('family', '?')}")
    typer.echo(f"  version       : {manifest.get('version', '?')}")
    typer.echo(f"  channels      : {manifest.get('channels', [])}")
    typer.echo(f"  n_samples     : {manifest.get('n_samples', {})}")
    typer.echo(
        f"  artifact_schema_version : "
        f"{manifest.get('artifact_schema_version', '?')}"
    )
    typer.echo(f"  topology_kind : {manifest.get('topology_kind', '?')}")
    typer.echo(f"  extent        : {manifest.get('extent', '?')}")
    typer.echo(f"  seed          : {manifest.get('seed', '?')}")
    typer.echo(f"  builder       : {manifest.get('builder', '?')}")


# =============================================================================
# Helpers
# =============================================================================


def _validate_build_args(
    version: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    """Validate common build arguments."""
    if version < 1:
        typer.echo("Error: --version must be ≥ 1.", err=True)
        raise typer.Exit(code=1)
    if n_train < 0 or n_val < 0 or n_test < 0:
        typer.echo("Error: split counts must be ≥ 0.", err=True)
        raise typer.Exit(code=1)


def _check_dest_and_force(dest: Path, force: bool) -> None:
    """Check destination existence and handle --force."""
    if dest.exists():
        if force:
            typer.echo(f"Warning: --force set; deleting existing root: {dest}")
            shutil.rmtree(dest)
        else:
            typer.echo(
                f"Error: version root already exists (dataset roots are "
                f"immutable): {dest}\n"
                f"Use --force to delete and rebuild, or bump --version.",
                err=True,
            )
            raise typer.Exit(code=1)


# =============================================================================
if __name__ == "__main__":
    app()

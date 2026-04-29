"""Staged CLI for building the MazeHard datasets.

Stages
------
fetch-raw            Download raw maze-nd corpus from HuggingFace.
prepare-interim      Normalize raw corpus to a deterministic interim artifact.
materialize-shared   Build the maze-nd shared substrate.
materialize-task     Build the MazeHard task corpus over a shared substrate.
validate             Validate an existing versioned root's manifest and data.
build-all            Convenience alias: all stages in DAG order.

Default paths
-------------
Shared substrate:  data/processed/maze-nd/v1
Task corpus:       data/processed/mazehard/default/v1
Raw corpus:        data/raw/huggingface/maze_hard_augmented
Interim:           data/interim/maze-nd

Examples
--------
Quick local build::

    python build-mazehard.py build-all

Custom sizes::

    python build-mazehard.py build-all \\
        --n-train 4000 --n-val 500 --n-test 500 --seed 7
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.build import validate_version_root
from ehc_sn.data.substrate.maze_nd import SHARED_FAMILY, build_shared_substrate, ensure_raw as _ensure_raw_corpus
from ehc_sn.data.substrate.maze_nd import prepare_interim as _prepare_mazehard_interim
from ehc_sn.tasks.mazehard import build_mazehard_task_corpus

# ---------------------------------------------------------------------------
_DEFAULT_RAW_ROOT = Path("data/raw/huggingface/maze_hard_augmented")
_DEFAULT_INTERIM_ROOT = Path("data/interim/maze-nd")
_DEFAULT_SHARED_VERSION = 1
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
@app.command("fetch-raw")
def fetch_raw(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = _DEFAULT_RAW_ROOT,
) -> None:
    """Download raw maze-nd corpus from HuggingFace."""
    _ensure_raw_corpus(raw_root.resolve())
    typer.echo(f"Raw corpus at {raw_root}")


# ---------------------------------------------------------------------------
@app.command("prepare-interim")
def prepare_interim(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
) -> None:
    """Normalize raw corpus to a deterministic interim artifact under data/interim/."""
    _prepare_mazehard_interim(raw_root.resolve(), interim_root.resolve())
    typer.echo(f"Interim written to {interim_root}")


# ---------------------------------------------------------------------------
@app.command("materialize-shared")
def materialize_shared(
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_SHARED_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the maze-nd shared substrate."""
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{version}")
    build_shared_substrate(
        shared_root.resolve(),
        interim_root=interim_root.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )


# ---------------------------------------------------------------------------
@app.command("materialize-task")
def materialize_task(
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
    corpus: Annotated[str, typer.Option("--corpus")] = _DEFAULT_CORPUS,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    shared_version: Annotated[int, typer.Option("--shared-version")] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the MazeHard task corpus from a shared substrate."""
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{shared_version}")
    task_root = Path(f"data/processed/mazehard/{corpus}/v{version}")
    build_mazehard_task_corpus(
        task_root.resolve(),
        parent_substrate=shared_root.resolve(),
        interim_root=interim_root.resolve(),
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )


# ---------------------------------------------------------------------------
@app.command("validate")
def validate(
    root: Annotated[Path, typer.Argument(help="Versioned root to validate.")],
) -> None:
    """Validate the manifest and data of a versioned root."""
    manifest = validate_version_root(root.resolve())
    if manifest["dataset_class"] == "task_corpus":
        from ehc_sn.tasks.mazehard import validate_mazehard_task_root

        validate_mazehard_task_root(root.resolve())
    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    family        : {manifest['family']}")
    typer.echo(f"    version       : {manifest['version']}")
    typer.echo(f"    channels      : {manifest['channels']}")
    typer.echo(f"    n_samples     : {manifest['n_samples']}")


# ---------------------------------------------------------------------------
@app.command("build-all")
def build_all(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
    corpus: Annotated[str, typer.Option("--corpus")] = _DEFAULT_CORPUS,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    shared_version: Annotated[int, typer.Option("--shared-version")] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Full pipeline: fetch-raw → prepare-interim → materialize-shared → materialize-task."""
    fetch_raw(raw_root=raw_root)
    prepare_interim(raw_root=raw_root, interim_root=interim_root)
    materialize_shared(
        interim_root=interim_root,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        version=shared_version,
        seed=seed,
    )
    materialize_task(
        interim_root=interim_root,
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        shared_version=shared_version,
        version=version,
        seed=seed,
    )


if __name__ == "__main__":
    app()

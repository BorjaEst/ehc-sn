"""Staged CLI for building the Arena task corpus.

Arena consumes the dungeongen shared substrate and adds Arena-specific
trajectory semantics on top.  This CLI owns only the Arena task corpus
slice; the shared dungeongen pipeline (raw, interim, substrate) is owned by
scripts/data-gen/build-dungeongen.py.

Stages
------
materialize-task  Build the Arena task corpus over a dungeongen shared substrate.
validate          Validate an Arena task-corpus version root.
build-all         Convenience alias: materialize-task (requires substrate to exist).

Default paths
-------------
Shared substrate:  data/processed/dungeongen/v1
Task corpus:       data/processed/arena/default/v1

Prerequisites
-------------
A dungeongen shared substrate must exist before running any command here.
Build it first::

    python scripts/data-gen/build-dungeongen.py build-all

Examples
--------
Build the Arena task corpus against the default shared substrate::

    python build-arena.py build-all

With an explicit shared-substrate version::

    python build-arena.py build-all --shared-version 2 --version 2

Custom trajectory length::

    python build-arena.py materialize-task --max-steps 80 --seed 7
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data._validator import validate_version_root
from ehc_sn.data.dungeon_builder import SHARED_FAMILY
from ehc_sn.tasks.arena.data import TASK_FAMILY as ARENA_TASK_FAMILY
from ehc_sn.tasks.arena.data import build_arena_task_corpus, validate_arena_task_root

# ---------------------------------------------------------------------------
_DEFAULT_SHARED_VERSION = 1
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"

app = typer.Typer(add_completion=False, help=__doc__)


def _require_shared_substrate(shared_root: Path) -> None:
    """Fail fast with an actionable error when the parent substrate is missing."""
    if not shared_root.exists():
        typer.echo(
            f"Error: parent shared substrate not found at {shared_root}.\n"
            "Build it first with:\n"
            "    python scripts/data-gen/build-dungeongen.py build-all\n"
            "or:\n"
            f"    python scripts/data-gen/build-dungeongen.py materialize-shared",
            err=True,
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
@app.command("materialize-task")
def materialize_task(
    corpus: Annotated[str, typer.Option("--corpus")] = _DEFAULT_CORPUS,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 50,
    shared_version: Annotated[int, typer.Option("--shared-version")] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the Arena task corpus over a dungeongen shared substrate."""
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{shared_version}")
    task_root = Path(f"data/processed/arena/{corpus}/v{version}")
    _require_shared_substrate(shared_root.resolve())
    build_arena_task_corpus(
        task_root.resolve(),
        parent_substrate=shared_root.resolve(),
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_steps=max_steps,
        seed=seed,
    )


# ---------------------------------------------------------------------------
@app.command("validate")
def validate(
    root: Annotated[Path, typer.Argument(help="Arena task-corpus root to validate.")],
) -> None:
    """Validate an Arena task-corpus version root.

    Raises an error if the root is not a valid Arena task_corpus.
    """
    manifest = validate_version_root(root.resolve())
    if manifest.get("dataset_class") != "task_corpus":
        typer.echo(
            f"Error: expected dataset_class 'task_corpus', "
            f"got '{manifest.get('dataset_class')}'. "
            "Use build-dungeongen.py to validate shared-substrate roots.",
            err=True,
        )
        raise typer.Exit(code=1)
    if manifest.get("task") != ARENA_TASK_FAMILY:
        typer.echo(
            f"Error: expected task '{ARENA_TASK_FAMILY}', got '{manifest.get('task')}'.",
            err=True,
        )
        raise typer.Exit(code=1)
    validate_arena_task_root(root.resolve())
    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    task          : {manifest['task']}")
    typer.echo(f"    version       : {manifest['version']}")
    typer.echo(f"    channels      : {manifest['channels']}")
    typer.echo(f"    n_samples     : {manifest['n_samples']}")


# ---------------------------------------------------------------------------
@app.command("build-all")
def build_all(
    corpus: Annotated[str, typer.Option("--corpus")] = _DEFAULT_CORPUS,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 50,
    shared_version: Annotated[int, typer.Option("--shared-version")] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the Arena task corpus (alias for materialize-task).

    Requires the parent dungeongen shared substrate to exist.  Build it first::

        python scripts/data-gen/build-dungeongen.py build-all
    """
    materialize_task(
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_steps=max_steps,
        shared_version=shared_version,
        version=version,
        seed=seed,
    )


if __name__ == "__main__":
    app()

"""Staged CLI for building the Dungeon task corpus.

Dungeon consumes the dungeongen shared substrate and adds Dungeon-specific
trajectory semantics on top.  This CLI owns only the Dungeon task corpus
slice; the shared dungeongen pipeline (raw, interim, substrate) is owned by
scripts/data-gen/build-dungeongen.py.

Stages
------
build      Build the Dungeon task corpus over a dungeongen shared substrate.
validate   Validate a Dungeon task-corpus version root.
inspect    Print a human-readable summary of a version root manifest.

Default paths
-------------
Shared substrate:  data/interim/dungeongen/v1
Task corpus:       data/processed/dungeon/default/v1

Prerequisites
-------------
A dungeongen shared substrate must exist before running any command here.
Build it first::

    python scripts/data-gen/build-dungeongen.py build-all

Examples
--------
Build the Dungeon task corpus against the default shared substrate::

    python build-dungeon.py build

With an explicit shared-substrate version::

    python build-dungeon.py build --shared-version 2 --version 2

Custom trajectory length::

    python build-dungeon.py build --max-steps 80 --seed 7

Validate an existing corpus::

    python build-dungeon.py validate data/processed/dungeon/default/v1

Inspect an existing corpus::

    python build-dungeon.py inspect data/processed/dungeon/default/v1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.dungeongen import SHARED_FAMILY
from ehc_sn.tasks.dungeon import TASK_FAMILY as DUNGEON_TASK_FAMILY
from ehc_sn.tasks.dungeon import (
    build_dungeon_task_corpus,
    validate_dungeon_task_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_SHARED_VERSION = 1
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(
    corpus: Annotated[str, typer.Option("--corpus")] = _DEFAULT_CORPUS,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 50,
    shared_version: Annotated[
        int, typer.Option("--shared-version")
    ] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the Dungeon task corpus over a dungeongen shared substrate."""
    shared_root = Path(f"data/interim/{SHARED_FAMILY}/v{shared_version}")
    task_root = Path(f"data/processed/dungeon/{corpus}/v{version}")
    _require_shared_substrate(shared_root.resolve())
    build_dungeon_task_corpus(
        task_root.resolve(),
        parent_substrate=shared_root.resolve(),
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_steps=max_steps,
        seed=seed,
    )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path, typer.Argument(help="Dungeon task-corpus root to validate.")
    ],
) -> None:
    """Validate a Dungeon task-corpus version root.

    Raises an error if the root is not a valid Dungeon task_corpus.
    """
    manifest = validate_dungeon_task_root(root.resolve())
    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    task          : {manifest['task']}")
    typer.echo(f"    version       : {manifest['version']}")
    typer.echo(f"    channels      : {manifest['channels']}")
    typer.echo(f"    n_samples     : {manifest['n_samples']}")


# =============================================================================
@app.command("inspect")
def inspect(
    root: Annotated[
        Path, typer.Argument(help="Dungeon task-corpus root to inspect.")
    ],
) -> None:
    """Print a human-readable summary of a Dungeon version root manifest."""
    manifest = validate_dungeon_task_root(root.resolve())
    typer.echo(f"Root: {root}")
    typer.echo(f"  dataset_class : {manifest.get('dataset_class', '?')}")
    typer.echo(f"  task          : {manifest.get('task', '?')}")
    typer.echo(f"  corpus        : {manifest.get('corpus', '?')}")
    typer.echo(f"  version       : {manifest.get('version', '?')}")
    typer.echo(f"  channels      : {manifest.get('channels', [])}")
    typer.echo(f"  n_samples     : {manifest.get('n_samples', {})}")


# =============================================================================
# Helpers
# =============================================================================


def _require_shared_substrate(shared_root: Path) -> None:
    """Fail fast with an actionable error when the parent substrate is missing."""
    if not shared_root.exists():
        typer.echo(
            f"Error: parent shared substrate not found at {shared_root}.\n"
            "Build it first with:\n"
            "    python scripts/data-gen/build-dungeongen.py build-all\n"
            "or:\n"
            f"    python scripts/data-gen/build-dungeongen.py materialize-layouts\n"
            "Requires generate-topology first.",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

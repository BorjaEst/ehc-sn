"""Staged CLI for building the Arena task corpus (v1, topology-free).

Arena consumes the dungeongen shared substrate and generates topology-free
episode trajectories. This CLI owns only the Arena task corpus slice; the
shared dungeongen pipeline (raw, interim, substrate) is owned by
scripts/data-gen/build-dungeongen.py.

Stages
------
materialize-task   Build an Arena task corpus over a dungeongen shared substrate.
validate           Validate an Arena task-corpus version root.
build-all          Convenience alias: materialize-task (requires substrate to exist).

Default paths
-------------
Shared substrate:  data/processed/dungeongen/v1
Task corpus:       data/processed/arena/default/v1

Documented recipes
------------------
Standard Arena recipe:
    --start-policy random_valid --walk-policy no_immediate_backtrack

Canonical-entrance uniform recipe:
    --start-policy canonical_entrance --walk-policy uniform

Arena v1 always materializes 250 steps per episode.

Prerequisites
-------------
A dungeongen shared substrate must exist before running any command here.
Build it first::

    python scripts/data-gen/build-dungeongen.py build-all

Examples
--------
Build the default Arena task corpus against the shared substrate::

    python build-arena.py build-all

Build the canonical-entrance uniform recipe into a descriptive corpus label::

    python build-arena.py materialize-task \
        --corpus canonical_entrance_uniform \
        --start-policy canonical_entrance \
        --walk-policy uniform
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.dungeongen import SHARED_FAMILY
from ehc_sn.tasks.arena import TASK_FAMILY as ARENA_TASK_FAMILY
from ehc_sn.tasks.arena import build_arena_task_corpus, validate_arena_task_root

# ---------------------------------------------------------------------------
_DEFAULT_SHARED_VERSION = 1
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_START_POLICY = "random_valid"
_DEFAULT_WALK_POLICY = "no_immediate_backtrack"

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
    start_policy: Annotated[str, typer.Option("--start-policy")] = _DEFAULT_START_POLICY,
    walk_policy: Annotated[str, typer.Option("--walk-policy")] = _DEFAULT_WALK_POLICY,
    train_parent_maps: Annotated[int, typer.Option("--train-parent-maps")] = 200,
    val_parent_maps: Annotated[int, typer.Option("--val-parent-maps")] = 40,
    test_parent_maps: Annotated[int, typer.Option("--test-parent-maps")] = 40,
    train_episodes_per_parent: Annotated[int, typer.Option("--train-episodes-per-parent")] = 100,
    val_episodes_per_parent: Annotated[int, typer.Option("--val-episodes-per-parent")] = 4,
    test_episodes_per_parent: Annotated[int, typer.Option("--test-episodes-per-parent")] = 1,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 250,
    shared_version: Annotated[int, typer.Option("--shared-version")] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the Arena task corpus (v1, topology-free) over a dungeongen shared substrate."""
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{shared_version}")
    task_root = Path(f"data/processed/arena/{corpus}/v{version}")
    _require_shared_substrate(shared_root.resolve())
    build_arena_task_corpus(
        task_root.resolve(),
        corpus=corpus,
        start_policy=start_policy,
        walk_policy=walk_policy,
        train_parent_maps=train_parent_maps,
        val_parent_maps=val_parent_maps,
        test_parent_maps=test_parent_maps,
        train_episodes_per_parent=train_episodes_per_parent,
        val_episodes_per_parent=val_episodes_per_parent,
        test_episodes_per_parent=test_episodes_per_parent,
        max_steps=max_steps,
        parent_substrate=shared_root.resolve(),
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
    typer.echo(f"    dataset_class          : {manifest['dataset_class']}")
    typer.echo(f"    task                   : {manifest['task']}")
    typer.echo(f"    task_protocol_version  : {manifest.get('task_protocol_version')}")
    typer.echo(f"    version                : {manifest['version']}")
    typer.echo(f"    channels               : {manifest['channels']}")
    typer.echo(f"    n_samples              : {manifest['n_samples']}")
    typer.echo(f"    observation_vocab_size : {manifest.get('observation_vocab_size')}")


# ---------------------------------------------------------------------------
@app.command("build-all")
def build_all(
    corpus: Annotated[str, typer.Option("--corpus")] = _DEFAULT_CORPUS,
    start_policy: Annotated[str, typer.Option("--start-policy")] = _DEFAULT_START_POLICY,
    walk_policy: Annotated[str, typer.Option("--walk-policy")] = _DEFAULT_WALK_POLICY,
    train_parent_maps: Annotated[int, typer.Option("--train-parent-maps")] = 200,
    val_parent_maps: Annotated[int, typer.Option("--val-parent-maps")] = 40,
    test_parent_maps: Annotated[int, typer.Option("--test-parent-maps")] = 40,
    train_episodes_per_parent: Annotated[int, typer.Option("--train-episodes-per-parent")] = 100,
    val_episodes_per_parent: Annotated[int, typer.Option("--val-episodes-per-parent")] = 4,
    test_episodes_per_parent: Annotated[int, typer.Option("--test-episodes-per-parent")] = 1,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 250,
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
        start_policy=start_policy,
        walk_policy=walk_policy,
        train_parent_maps=train_parent_maps,
        val_parent_maps=val_parent_maps,
        test_parent_maps=test_parent_maps,
        train_episodes_per_parent=train_episodes_per_parent,
        val_episodes_per_parent=val_episodes_per_parent,
        test_episodes_per_parent=test_episodes_per_parent,
        max_steps=max_steps,
        shared_version=shared_version,
        version=version,
        seed=seed,
    )


if __name__ == "__main__":
    app()

"""Staged CLI for building the Arena task corpus from layout datasets.

Arena consumes an interim layout dataset root and generates topology-free
episode trajectories.  Layout generation is owned by the respective layout
CLIs (build-dungeongen.py, build-openfield.py).

This CLI has no source-specific code.  It reads the layout manifest to
determine the action space, topology type, and observation vocabulary size,
then calls :func:`build_arena_task_corpus`.

Stages
------
materialize-task   Build an Arena task corpus over a layout dataset.
validate           Validate an Arena task-corpus version root.


Default paths
-------------
Parent layout dataset:  <user-specified --layout-root>
Task corpus:            data/processed/arena/<corpus>/v<version>

Documented recipes
------------------
Openfield square (TEM reproduction)::

    python build-arena.py materialize-task \\
        --layout-root data/interim/openfield/square/v1 \\
        --corpus openfield-square \\
        --walk-policy angle_bias

Standard dungeongen recipe::

    python build-arena.py materialize-task \\
        --layout-root data/interim/dungeongen/default/v1 \\
        --corpus dungeons \\
        --walk-policy no_backtrack

Examples
--------
Build from openfield square layouts::

    python build-arena.py materialize-task \\
        --layout-root data/interim/openfield/tem-square/v1 \\
        --corpus openfield-square

Build from dungeongen layouts::

    python build-arena.py materialize-task \\
        --layout-root data/interim/dungeongen/default/v1 \\
        --corpus dungeons
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.layout.io import load_layout_dataset
from ehc_sn.tasks.arena import (
    build_arena_task_corpus,
    validate_arena_task_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_WALK_POLICY = "angle_bias"
_DEFAULT_MAX_STEPS = 2000
_DEFAULT_N_EPISODES = 4
_DEFAULT_WALK_SEED = 45


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("materialize-task")
def materialize_task(  # ------------------------------------------------------
    layout_root: Annotated[
        Path,
        typer.Option(
            "--layout-root",
            help="Interim layout dataset root.",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option(
            "--corpus",
            help="Corpus name (e.g. 'openfield-square' or 'dungeons').",
        ),
    ] = _DEFAULT_CORPUS,
    walk_policy: Annotated[
        str,
        typer.Option(
            "--walk-policy",
            help="Walk policy for trajectory generation (default: angle_bias).",
        ),
    ] = _DEFAULT_WALK_POLICY,
    n_episodes: Annotated[
        int,
        typer.Option(
            "--n-episodes",
            help="Number of episodes per layout.",
        ),
    ] = _DEFAULT_N_EPISODES,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Maximum steps per episode.",
        ),
    ] = _DEFAULT_MAX_STEPS,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Version number for the generated task corpus (default: 1).",
        ),
    ] = _DEFAULT_TASK_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for trajectory generation (default: 45).",
        ),
    ] = _DEFAULT_WALK_SEED,
) -> None:
    """Build the Arena task corpus from an interim layout dataset.
    ...
    """
    if not layout_root.exists():
        typer.echo(
            f"Error: layout root not found at {layout_root.resolve()}.\n"
            "Build layouts first with:\n"
            "    python scripts/data-gen/build-openfield.py build-all\n"
            "or:\n"
            "    python scripts/data-gen/build-dungeongen.py build-all\n"
            "or:\n"
            "    python scripts/data-gen/build-dungeongen.py materialize-layouts",
            err=True,
        )
        raise typer.Exit(code=1)

    task_root = Path(f"data/processed/arena/{corpus}/v{version}")

    print(f"Loading layouts from {layout_root.resolve()}...")
    layouts = load_layout_dataset(layout_root.resolve())
    print(f"  {len(layouts)} layouts loaded.")

    build_arena_task_corpus(
        version_root=task_root.resolve(),
        layouts=layouts,
        corpus=corpus,
        walk_policy=walk_policy,
        n_episodes_per_layout=n_episodes,
        max_steps=max_steps,
        seed=seed,
    )


# =============================================================================
@app.command("validate")
def validate(  # --------------------------------------------------------------
    root: Annotated[
        Path, typer.Argument(help="Arena task-corpus root to validate.")
    ],
) -> None:
    """Validate an Arena task-corpus version root.

    Raises an error if the root is not a valid Arena task_corpus.
    """
    manifest = validate_arena_task_root(root.resolve())
    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class          : {manifest['dataset_class']}")
    typer.echo(f"    task                   : {manifest['task']}")
    typer.echo(
        f"    task_protocol_version  : {manifest.get('task_protocol_version')}"
    )
    typer.echo(f"    version                : {manifest['version']}")
    typer.echo(f"    channels               : {manifest['channels']}")
    typer.echo(f"    n_samples              : {manifest['n_samples']}")
    typer.echo(
        f"    observation_vocab_size : {manifest.get('observation_vocab_size')}"
    )


# build-all removed — this script has a single stage: materialize-task.

# =============================================================================
if __name__ == "__main__":
    app()

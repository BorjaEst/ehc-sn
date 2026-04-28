"""Staged CLI for building the Dungeon datasets.

Stages
------
fetch-raw           Generate raw dungeongen topologies (if not present).
prepare-interim     Copy raw topologies to a deterministic interim artifact.
materialize-shared  Build the dungeongen shared substrate.
materialize-task    Build the Dungeon task corpus over a shared substrate.
validate            Validate an existing versioned root's manifest and data.
build-all           Convenience alias: all stages in DAG order.

Default paths
-------------
Shared substrate:  data/processed/dungeongen/v1
Task corpus:       data/processed/dungeon/default/v1
Raw corpus:        data/raw/dungeongen
Interim:           data/interim/dungeongen

Examples
--------
Quick local build::

    python build-dungeon.py build-all

Custom sizes::

    python build-dungeon.py build-all \\
        --n-train 2000 --n-val 200 --n-test 200 \\
        --height 40 --width 40 --n-observations 8 --max-steps 80 --seed 7
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data._validator import validate_version_root
from ehc_sn.data.dungeon_builder import _RAW_SEED_OFFSET, SHARED_FAMILY, build_dungeongen_substrate, prepare_dungeongen_interim
from ehc_sn.data.dungeon_raw import ensure_raw_corpus
from ehc_sn.tasks.dungeon.data import build_dungeon_task_corpus

# ---------------------------------------------------------------------------
_DEFAULT_RAW_ROOT = Path("data/raw/dungeongen")
_DEFAULT_INTERIM_ROOT = Path("data/interim/dungeongen")
_DEFAULT_SHARED_VERSION = 1
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
@app.command("fetch-raw")
def fetch_raw(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = _DEFAULT_RAW_ROOT,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Generate raw dungeongen topologies (if not already present)."""
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        ensure_raw_corpus(raw_root, split, n, seed + _RAW_SEED_OFFSET[split])
    typer.echo(f"Raw corpus at {raw_root}")


# ---------------------------------------------------------------------------
@app.command("prepare-interim")
def prepare_interim(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
) -> None:
    """Copy raw topologies to a deterministic interim artifact under data/interim/."""
    prepare_dungeongen_interim(
        raw_root.resolve(),
        interim_root.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
    typer.echo(f"Interim written to {interim_root}")


# ---------------------------------------------------------------------------
@app.command("materialize-shared")
def materialize_shared(
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    height: Annotated[int, typer.Option("--height")] = 40,
    width: Annotated[int, typer.Option("--width")] = 40,
    n_observations: Annotated[int, typer.Option("--n-observations")] = 6,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_SHARED_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the dungeongen shared substrate."""
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{version}")
    build_dungeongen_substrate(
        shared_root.resolve(),
        interim_root=interim_root.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        n_observations=n_observations,
        seed=seed,
    )


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
    """Build the Dungeon task corpus from a shared substrate."""
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{shared_version}")
    task_root = Path(f"data/processed/dungeon/{corpus}/v{version}")
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


# ---------------------------------------------------------------------------
@app.command("validate")
def validate(
    root: Annotated[Path, typer.Argument(help="Versioned root to validate.")],
) -> None:
    """Validate the manifest and data of a versioned root."""
    manifest = validate_version_root(root.resolve())
    if manifest["dataset_class"] == "task_corpus":
        from ehc_sn.tasks.dungeon.data import validate_dungeon_task_root

        validate_dungeon_task_root(root.resolve())
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
    height: Annotated[int, typer.Option("--height")] = 40,
    width: Annotated[int, typer.Option("--width")] = 40,
    n_observations: Annotated[int, typer.Option("--n-observations")] = 6,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 50,
    shared_version: Annotated[int, typer.Option("--shared-version")] = _DEFAULT_SHARED_VERSION,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_TASK_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Full pipeline: fetch-raw → prepare-interim → materialize-shared → materialize-task."""
    fetch_raw(raw_root=raw_root, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
    prepare_interim(raw_root=raw_root, interim_root=interim_root, n_train=n_train, n_val=n_val, n_test=n_test)
    materialize_shared(
        interim_root=interim_root,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        n_observations=n_observations,
        version=shared_version,
        seed=seed,
    )
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

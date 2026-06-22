"""CLI for building dungeongen layout datasets.

Dungeongen generates procedurally varied 2-D grid topologies with random
sensory assignments.  The output is an interim layout dataset consumed by
task builders (build-arena.py, build-dungeon.py).

Commands
--------
build       Produce a complete dungeongen layout dataset.
validate    Validate a version root's manifest, channels, and data.
inspect     Print a human-readable summary of a version root manifest.


Default paths
-------------
Raw corpus:      data/raw/dungeongen
Interim:         data/interim/dungeongen
Layout dataset:  data/interim/dungeongen/{preset}/v{version}

Examples
--------
Quick local build::

    python build-dungeongen.py build

Custom version::

    python build-dungeongen.py build --version 2

Inspect an existing root::

    python build-dungeongen.py inspect data/interim/dungeongen/default/v1

"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.dungeongen import (
    SHARED_FAMILY,
    build_dungeongen_layouts,
)
from ehc_sn.data.substrate.dungeongen import ensure_raw as _ensure_raw
from ehc_sn.data.substrate.dungeongen import prepare_interim as _prepare_interim

# ---------------------------------------------------------------------------
_DEFAULT_RAW_ROOT = Path("data/raw/dungeongen")
_DEFAULT_INTERIM_ROOT = Path("data/interim/dungeongen")
_DEFAULT_VERSION = 1
_DEFAULT_PRESET = "default"
_DEFAULT_S_SIZE = 45
_DEFAULT_N_SENSORY_INSTANCES = 1
_DEFAULT_TOPOLOGY_SEED = 42
_DEFAULT_N_TRAIN = 250
_DEFAULT_N_VAL = 10
_DEFAULT_N_TEST = 10


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(  # -----------------------------------------------------------------
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Named source preset (currently only 'default' is supported).",
        ),
    ] = _DEFAULT_PRESET,
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
    pad_height: Annotated[
        int | None,
        typer.Option(
            "--pad-height",
            help="Uniform storage canvas height target for padding. "
            "When omitted, inferred from max natural extent. "
            "Does not affect dungeon generation.",
        ),
    ] = None,
    pad_width: Annotated[
        int | None,
        typer.Option(
            "--pad-width",
            help="Uniform storage canvas width target for padding. "
            "When omitted, inferred from max natural extent. "
            "Does not affect dungeon generation.",
        ),
    ] = None,
    s_size: Annotated[
        int,
        typer.Option(
            "--s-size",
            help="Sensory vocabulary size (deprecated; use --observation-vocabulary-size).",
        ),
    ] = _DEFAULT_S_SIZE,
    observation_vocabulary_size: Annotated[
        int | None,
        typer.Option(
            "--observation-vocabulary-size",
            help=f"Observation vocabulary size (default: {_DEFAULT_S_SIZE}).",
        ),
    ] = None,
    n_sensory_instances: Annotated[
        int,
        typer.Option(
            "--n-sensory-instances",
            help="Number of randomized sensory assignments per topology.",
        ),
    ] = _DEFAULT_N_SENSORY_INSTANCES,
    topology_seed: Annotated[
        int,
        typer.Option(
            "--topology-seed",
            help=f"Seed controlling topology generation (default: {_DEFAULT_TOPOLOGY_SEED}).",
        ),
    ] = _DEFAULT_TOPOLOGY_SEED,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help=f"Version of the emitted dataset (default: {_DEFAULT_VERSION}).",
        ),
    ] = _DEFAULT_VERSION,
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root",
            help=f"Root path for raw corpus (default: {_DEFAULT_RAW_ROOT}).",
        ),
    ] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[
        Path,
        typer.Option(
            "--interim-root",
            help=f"Root path for interim files (default: {_DEFAULT_INTERIM_ROOT}).",
        ),
    ] = _DEFAULT_INTERIM_ROOT,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Delete the existing version root before building, if present.",
        ),
    ] = False,
) -> None:
    """Produce a complete dungeongen layout dataset (topology + layouts).

    Chains the two internal stages into one command.
    """
    if observation_vocabulary_size is not None:
        s_size = observation_vocabulary_size
    elif s_size != _DEFAULT_S_SIZE:
        typer.echo(
            "Warning: --s-size is deprecated, use --observation-vocabulary-size instead.",
            err=True,
        )
    dest = interim_root / preset / f"v{version}"
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
    _generate_topology(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        topology_seed=topology_seed,
        version=version,
        raw_root=raw_root,
        interim_root=interim_root,
    )
    _materialize_layouts(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        pad_height=pad_height,
        pad_width=pad_width,
        s_size=s_size,
        n_sensory_instances=n_sensory_instances,
        topology_seed=topology_seed,
        version=version,
        interim_root=interim_root,
    )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(help="Dungeongen version root to validate."),
    ],
) -> None:
    """Validate a dungeongen shared-substrate or layout version root.

    Raises an error if the root is not a valid dungeongen dataset root.
    """
    manifest = validate_version_root(root.resolve())
    # source_spec is not yet accepted here because dungeongen does not
    # produce source_spec artifacts.
    if manifest.get("dataset_class") not in {
        "shared_substrate",
        "layout_dataset",
    }:
        typer.echo(
            f"Error: expected dataset_class 'shared_substrate' or 'layout_dataset', "
            f"got '{manifest.get('dataset_class')}'. "
            "Use the owning task CLI to validate task corpus roots.",
            err=True,
        )
        raise typer.Exit(code=1)
    if manifest.get("dataset_class") == "layout_dataset":
        pass  # family check not required for layout datasets
    elif manifest.get("family") != SHARED_FAMILY:
        typer.echo(
            f"Error: expected family '{SHARED_FAMILY}', got '{manifest.get('family')}'.",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    family        : {manifest['family']}")
    typer.echo(f"    version       : {manifest['version']}")
    typer.echo(f"    channels      : {manifest['channels']}")
    typer.echo(f"    n_samples     : {manifest['n_samples']}")


# =============================================================================
@app.command("inspect")
def inspect_command(  # -------------------------------------------------------
    root: Annotated[
        Path,
        typer.Argument(help="Dungeongen version root to inspect."),
    ],
) -> None:
    """Print a human-readable summary of a version root manifest."""
    manifest = validate_version_root(root.resolve())

    typer.echo(f"Root: {root}")
    typer.echo(f"  dataset_class : {manifest.get('dataset_class', '?')}")
    typer.echo(f"  family        : {manifest.get('family', '?')}")
    typer.echo(f"  version       : {manifest.get('version', '?')}")
    typer.echo(f"  channels      : {manifest.get('channels', [])}")
    typer.echo(f"  n_samples     : {manifest.get('n_samples', {})}")

    stage_params = manifest.get("stage_params", {})
    topology_type = stage_params.get("topology_type")
    if topology_type:
        typer.echo(f"  topology_type : {topology_type}")

    extent = manifest.get("extent")
    if extent:
        typer.echo(f"  extent        : {extent}")


# =============================================================================
# Helpers
# =============================================================================


def _generate_topology(  # ----------------------------------------------------
    preset: str = _DEFAULT_PRESET,
    n_train: int = _DEFAULT_N_TRAIN,
    n_val: int = _DEFAULT_N_VAL,
    n_test: int = _DEFAULT_N_TEST,
    topology_seed: int = _DEFAULT_TOPOLOGY_SEED,
    version: int = _DEFAULT_VERSION,
    raw_root: Path = _DEFAULT_RAW_ROOT,
    interim_root: Path = _DEFAULT_INTERIM_ROOT,
) -> None:
    """Internal stage: generate raw dungeongen corpus and prepare interim files."""
    raw_leaf = raw_root / preset / f"v{version}"
    _ensure_raw(
        raw_leaf,
        topology_seed,
        {"train": n_train, "val": n_val, "test": n_test},
    )
    print(f"Raw corpus at {raw_leaf}")
    interim_path = interim_root / preset / "interim" / f"v{version}"
    interim_path.mkdir(parents=True, exist_ok=True)
    _prepare_interim(
        raw_leaf.resolve(),
        interim_path.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
    print(f"Interim written to {interim_path}")


# ---------------------------------------------------------------------------
def _materialize_layouts(  # ---------------------------------------------------
    preset: str = _DEFAULT_PRESET,
    n_train: int = _DEFAULT_N_TRAIN,
    n_val: int = _DEFAULT_N_VAL,
    n_test: int = _DEFAULT_N_TEST,
    pad_height: int | None = None,
    pad_width: int | None = None,
    s_size: int = _DEFAULT_S_SIZE,
    observation_vocabulary_size: int | None = None,
    n_sensory_instances: int = _DEFAULT_N_SENSORY_INSTANCES,
    topology_seed: int = _DEFAULT_TOPOLOGY_SEED,
    version: int = _DEFAULT_VERSION,
    interim_root: Path = _DEFAULT_INTERIM_ROOT,
) -> None:
    """Internal stage: materialize layout datasets from interim files."""
    if observation_vocabulary_size is not None:
        s_size = observation_vocabulary_size
    interim_leaf = interim_root / preset / "interim" / f"v{version}"
    layout_leaf = interim_root / preset / f"v{version}"

    build_dungeongen_layouts(
        layout_leaf,
        interim_root=interim_leaf.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=pad_height,
        width=pad_width,
        s_size=s_size,
        topology_seed=topology_seed,
        preset=preset,
        n_sensory_instances=n_sensory_instances,
    )


# =============================================================================
if __name__ == "__main__":
    app()

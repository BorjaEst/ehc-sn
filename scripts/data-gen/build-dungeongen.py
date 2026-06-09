"""Staged CLI for the dungeongen shared-family pipeline.

This script is the sole CLI owner of all dungeongen raw, interim, and layout
stages.  Task-level materialization is owned by the respective task CLIs
(build-arena.py, build-dungeon.py).

Stages
------
generate-topology   Ensure raw snapshot exists + normalize to interim topology records.
materialize-layouts Build the dungeongen layout dataset (consumable by build-arena).
validate            Validate a dungeongen shared-substrate or layout version root.
build-all           Convenience alias: generate-topology -> materialize-layouts.

Default paths
-------------
Raw corpus:      data/raw/dungeongen
Interim:         data/interim/dungeongen
Layout dataset:  data/interim/dungeongen/{preset}/v{version}

Examples
--------
Quick local build (grid shape inferred from interim slice)::

    python build-dungeongen.py build-all

Custom version::

    python build-dungeongen.py build-all --version 2

With explicit grid shape override::

    python build-dungeongen.py build-all \\
        --n-train 2000 --n-val 200 --n-test 200 \\
        --height 48 --width 48 --s-size 8 --topology-seed 7

Build an Arena corpus from dungeongen layouts::

    python build-arena.py build-all \\
        --layout-root data/interim/dungeongen/default/v1 \\
        --corpus dungeons
"""

from __future__ import annotations

import json
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
_DEFAULT_N_TRAIN = 1000
_DEFAULT_N_VAL = 40
_DEFAULT_N_TEST = 40


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("generate-topology")
def generate_topology(  # ----------------------------------------------------
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
    height: Annotated[
        int | None,
        typer.Option(
            "--height", help="Target grid height. Inferred when omitted."
        ),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option(
            "--width", help="Target grid width. Inferred when omitted."
        ),
    ] = None,
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
            help=f"Version of the emitted interim dataset (default: {_DEFAULT_VERSION}).",
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
) -> None:
    """Ensure raw snapshot exists + normalize to interim topology records.

    If the raw corpus does not exist, generates it.  Then normalizes the raw
    topologies into per-split NPZ files under ``{interim_root}/{preset}/interim/v{version}/``.
    """
    _ensure_raw(
        raw_root,
        topology_seed,
        {"train": n_train, "val": n_val, "test": n_test},
    )
    print(f"Raw corpus at {raw_root}")
    interim_path = interim_root / preset / "interim" / f"v{version}"
    interim_path.mkdir(parents=True, exist_ok=True)
    _prepare_interim(
        raw_root.resolve(),
        interim_path.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
    print(f"Interim written to {interim_path}")


# ---------------------------------------------------------------------------
@app.command("materialize-layouts")
def materialize_layouts(  # ---------------------------------------------------
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
    height: Annotated[
        int | None,
        typer.Option(
            "--height", help="Target grid height. Inferred when omitted."
        ),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option(
            "--width", help="Target grid width. Inferred when omitted."
        ),
    ] = None,
    s_size: Annotated[
        int,
        typer.Option(
            "--s-size",
            help=f"Sensory vocabulary size (default: {_DEFAULT_S_SIZE}).",
        ),
    ] = _DEFAULT_S_SIZE,
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
    interim_root: Annotated[
        Path,
        typer.Option(
            "--interim-root",
            help=f"Root path for interim files (default: {_DEFAULT_INTERIM_ROOT}).",
        ),
    ] = _DEFAULT_INTERIM_ROOT,
) -> None:
    """Build the dungeongen layout dataset.

    Reads interim topology files, assigns sensory IDs, and writes
    :class:`SpatialLayout` records to
    ``{interim_root}/{preset}/v{version}/``.
    """
    interim_leaf = interim_root / preset / "interim" / f"v{version}"
    layout_leaf = interim_root / preset / f"v{version}"

    build_dungeongen_layouts(
        layout_leaf,
        interim_root=interim_leaf.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        s_size=s_size,
        topology_seed=topology_seed,
        preset=preset,
        n_sensory_instances=n_sensory_instances,
    )


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
@app.command("build-all")
def build_all(
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
    height: Annotated[
        int | None,
        typer.Option(
            "--height", help="Target grid height. Inferred when omitted."
        ),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option(
            "--width", help="Target grid width. Inferred when omitted."
        ),
    ] = None,
    s_size: Annotated[
        int,
        typer.Option(
            "--s-size",
            help=f"Sensory vocabulary size (default: {_DEFAULT_S_SIZE}).",
        ),
    ] = _DEFAULT_S_SIZE,
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
) -> None:
    """Full pipeline: generate-topology -> materialize-layouts.

    Grid shape is inferred from the interim slice unless --height/--width
    are given explicitly.
    """
    generate_topology(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        topology_seed=topology_seed,
        version=version,
        raw_root=raw_root,
        interim_root=interim_root,
    )
    materialize_layouts(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        s_size=s_size,
        n_sensory_instances=n_sensory_instances,
        topology_seed=topology_seed,
        version=version,
        interim_root=interim_root,
    )


if __name__ == "__main__":
    app()

"""Staged CLI for the dungeongen shared-family pipeline.

This script is the sole CLI owner of all dungeongen raw, interim, and shared-
substrate stages.  Task-level materialization is owned by the respective task
CLIs (build-arena.py, build-dungeon.py).

Stages
------
fetch-raw           Create or validate the canonical tar-sharded raw snapshot.
prepare-interim     Normalize the raw snapshot into deterministic per-split NPZ files.
materialize-shared  Build the dungeongen shared substrate version root.
validate            Validate a dungeongen shared-substrate root against manifest and schema.
build-all           Convenience alias: fetch-raw → prepare-interim → materialize-shared.

Default paths
-------------
Shared substrate:  data/processed/dungeongen/v1
Raw corpus:        data/raw/dungeongen
Interim:           data/interim/dungeongen

Examples
--------
Quick local build (grid shape inferred from interim slice)::

    python build-dungeongen.py build-all

Custom version::

    python build-dungeongen.py build-all --version 2

With explicit grid shape override::

    python build-dungeongen.py build-all \\
        --n-train 2000 --n-val 200 --n-test 200 \\
        --height 48 --width 48 --n-observations 8 --seed 7

Build task corpora against the shared substrate::

    python build-arena.py build-all --shared-version 1
    python build-dungeon.py build-all --shared-version 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.dungeongen import SHARED_FAMILY, build_shared_substrate
from ehc_sn.data.substrate.dungeongen import ensure_raw as _ensure_raw
from ehc_sn.data.substrate.dungeongen import prepare_interim as _prepare_interim

# ---------------------------------------------------------------------------
_DEFAULT_RAW_ROOT = Path("data/raw/dungeongen")
_DEFAULT_INTERIM_ROOT = Path("data/interim/dungeongen")
_DEFAULT_SHARED_VERSION = 1

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
    """Create the canonical tar-sharded raw snapshot for dungeongen (if not already present).

    If data/raw/dungeongen does not exist, generates the snapshot and writes manifest.json.
    If it exists and the manifest identity matches the request, this is a no-op.
    If it exists with a mismatched identity, exits with an actionable error.
    """
    _ensure_raw(raw_root, seed, {"train": n_train, "val": n_val, "test": n_test})
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
    """Normalize the raw snapshot into per-split NPZ files under data/interim/dungeongen/.

    Reads from the canonical tar-sharded raw snapshot and writes one NPZ file per split.
    The interim format is materially different from raw: no tar packaging, no per-sample
    file fan-out, padded arrays with height/width metadata for native-shape reconstruction.
    """
    _prepare_interim(
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
    height: Annotated[
        int | None,
        typer.Option("--height", help="Target grid height. Inferred from interim slice when omitted."),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option("--width", help="Target grid width. Inferred from interim slice when omitted."),
    ] = None,
    n_observations: Annotated[int, typer.Option("--n-observations")] = 6,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_SHARED_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Build the dungeongen shared substrate.

    When --height and --width are omitted, the processed grid shape is inferred
    as the maximum height and width across the selected interim slice.
    """
    shared_root = Path(f"data/processed/{SHARED_FAMILY}/v{version}")
    build_shared_substrate(
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
@app.command("validate")
def validate(
    root: Annotated[Path, typer.Argument(help="Dungeongen shared-substrate root to validate.")],
) -> None:
    """Validate a dungeongen shared-substrate version root.

    Raises an error if the root is not a valid dungeongen shared_substrate.
    """
    manifest = validate_version_root(root.resolve())
    if manifest.get("dataset_class") != "shared_substrate":
        typer.echo(
            f"Error: expected dataset_class 'shared_substrate', "
            f"got '{manifest.get('dataset_class')}'. "
            "Use the owning task CLI to validate task corpus roots.",
            err=True,
        )
        raise typer.Exit(code=1)
    if manifest.get("family") != SHARED_FAMILY:
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
    raw_root: Annotated[Path, typer.Option("--raw-root")] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[Path, typer.Option("--interim-root")] = _DEFAULT_INTERIM_ROOT,
    n_train: Annotated[int, typer.Option("--n-train")] = 200,
    n_val: Annotated[int, typer.Option("--n-val")] = 40,
    n_test: Annotated[int, typer.Option("--n-test")] = 40,
    height: Annotated[
        int | None,
        typer.Option("--height", help="Target grid height. Inferred from interim slice when omitted."),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option("--width", help="Target grid width. Inferred from interim slice when omitted."),
    ] = None,
    n_observations: Annotated[int, typer.Option("--n-observations")] = 6,
    version: Annotated[int, typer.Option("--version")] = _DEFAULT_SHARED_VERSION,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Shared-family pipeline: fetch-raw → prepare-interim → materialize-shared.

    Grid shape is inferred from the interim slice unless --height/--width are given explicitly.
    To build task corpora against this substrate run:

        python build-arena.py build-all --shared-version <version>
        python build-dungeon.py build-all --shared-version <version>
    """
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
        version=version,
        seed=seed,
    )


if __name__ == "__main__":
    app()
    app()

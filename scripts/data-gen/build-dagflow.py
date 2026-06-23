"""CLI for building dagflow layout interim assets.

Dagflow is a layout source (peer to openfield, dungeongen) that generates
random DAG topologies with permuted node order and remapped observation IDs.

The output is an interim layout dataset root consumed by downstream task
builders (build-seqmaze.py, build-goaltrace.py, build-routebind.py).

Commands
--------
build       Generate graph layouts and write a versioned layout dataset.
validate    Validate a version root's manifest, channels, and data.
inspect     Print a human-readable summary of a version root manifest.

Default paths
-------------
Layout dataset:    data/interim/dagflow/{preset}/v{version}

Prerequisites
-------------
None — layouts are generated procedurally.

Examples
--------
Quick local build with default preset::

    python build-dagflow.py build --preset routing --version 1

Small smoke test::

    python build-dagflow.py build --preset small --version 1

Canonical routing graph::

    python build-dagflow.py build --preset routing --version 1

Long-composition stress test::

    python build-dagflow.py build --preset chain16 --version 1

Custom density and span profile::

    python build-dagflow.py build \\
        --preset branching --version 1 \
        --extra-edge-density 0.20 --span-profile heavy

Validate an existing version root::

    python build-dagflow.py validate data/interim/dagflow/routing/v1

Inspect manifest::

    python build-dagflow.py inspect data/interim/dagflow/routing/v1
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.dagflow import (
    _SPAN_PROFILES,
    DAGFLOW_PRESETS,
    LAYOUT_FAMILY,
    build_dagflow_layouts,
    validate_dagflow_layout_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_ROOT = Path("data/interim/dagflow")
_DEFAULT_PRESET = "routing"
_DEFAULT_VERSION = 1
_DEFAULT_N_TRAIN = 4000
_DEFAULT_N_VAL = 500
_DEFAULT_N_TEST = 500
_DEFAULT_SEED = 42


_SPAN_PROFILE_HELP = (
    f"Rank-span distribution profile ({', '.join(sorted(_SPAN_PROFILES))}). "
    "Overrides the preset's profile when set."
)
_SPAN_PROFILES_VALUES = sorted(_SPAN_PROFILES)

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(  # -----------------------------------------------------------------
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help=f"Named generation preset ({', '.join(sorted(DAGFLOW_PRESETS))}).",
        ),
    ] = _DEFAULT_PRESET,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Substrate artifact version integer.  Must be ≥ 1.",
        ),
    ] = _DEFAULT_VERSION,
    n_max: Annotated[
        int,
        typer.Option(
            "--n-max",
            help="Number of candidate nodes N (overrides preset).  Must be ≥ 2.",
        ),
    ] = None,  # type: ignore[arg-type]
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train",
            help=f"Number of training graph artifacts (default: {_DEFAULT_N_TRAIN}).",
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val",
            help=f"Number of validation graph artifacts (default: {_DEFAULT_N_VAL}).",
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option(
            "--n-test",
            help=f"Number of test graph artifacts (default: {_DEFAULT_N_TEST}).",
        ),
    ] = _DEFAULT_N_TEST,
    extra_edge_density: Annotated[
        float | None,
        typer.Option(
            "--extra-edge-density",
            help="Fraction of possible extra forward edges beyond the backbone. "
            "0.0 = backbone only; 1.0 = complete forward DAG. Overrides preset.",
        ),
    ] = None,
    max_out_degree: Annotated[
        int | None,
        typer.Option(
            "--max-out-degree",
            help="Maximum out-degree K (overrides preset).  Must be ≥ 2.",
        ),
    ] = None,
    span_profile: Annotated[
        str | None,
        typer.Option(
            "--span-profile",
            help=_SPAN_PROFILE_HELP,
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=f"Deterministic base seed (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=f"Root path for interim files (default: {_DEFAULT_OUTPUT_ROOT}).",
        ),
    ] = _DEFAULT_OUTPUT_ROOT,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Delete the existing version root before building, if present.",
        ),
    ] = False,
) -> None:
    """Generate graph layouts and write a versioned layout dataset.

    Produces a dagflow layout dataset at ``{output_root}/{preset}/v{version}/``
    with all ``LAYOUT_CHANNELS`` written as per-split NPY files.

    The Hamiltonian backbone (0→1→…→N-1) is mandatory and non-configurable.
    All graphs have permuted public observation IDs (rank→obs_id bijection).

    Preset parameters (n_max, max_out_degree, extra_edge_density,
    span_profile) can be overridden individually via CLI flags.
    """
    if version < 1:
        typer.echo("Error: --version must be ≥ 1.", err=True)
        raise typer.Exit(code=1)
    if n_max is not None and n_max < 2:
        typer.echo("Error: --n-max must be ≥ 2.", err=True)
        raise typer.Exit(code=1)
    if max_out_degree is not None and max_out_degree < 2:
        typer.echo("Error: --max-out-degree must be ≥ 2.", err=True)
        raise typer.Exit(code=1)
    if extra_edge_density is not None and not (
        0.0 <= extra_edge_density <= 1.0
    ):
        typer.echo(
            "Error: --extra-edge-density must be in [0.0, 1.0].", err=True
        )
        raise typer.Exit(code=1)
    if span_profile is not None and span_profile not in _SPAN_PROFILES:
        typer.echo(
            f"Error: unknown --span-profile '{span_profile}'. "
            f"Choose from: {', '.join(_SPAN_PROFILES_VALUES)}.",
            err=True,
        )
        raise typer.Exit(code=1)
    if n_train < 0 or n_val < 0 or n_test < 0:
        typer.echo("Error: split counts must be ≥ 0.", err=True)
        raise typer.Exit(code=1)

    dest = _resolve_dest(output_root, preset, version)

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

    build_dagflow_layouts(
        version_root=dest,
        preset=preset,
        n_max=n_max,
        max_out_degree=max_out_degree,
        extra_edge_density=extra_edge_density,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        span_profile=span_profile,
    )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Dagflow version root to validate "
            "(e.g. data/interim/dagflow/routing/v1)."
        ),
    ],
) -> None:
    """Validate a dagflow version root's manifest, channels, and data.

    Performs structural validation (manifest fields, path grammar, split
    presence, channel files, sample counts) followed by per-sample semantic
    validation for layout_dataset roots.
    """
    manifest = validate_version_root(root.resolve())
    dc = manifest.get("dataset_class", "")

    if dc == "layout_dataset":
        manifest = validate_dagflow_layout_root(root.resolve())
        typer.echo(f"OK  {root}")
        typer.echo(f"    dataset_class : {manifest['dataset_class']}")
        typer.echo(f"    family        : {manifest['family']}")
        typer.echo(f"    version       : {manifest['version']}")
        typer.echo(f"    channels      : {manifest['channels']}")
        typer.echo(f"    n_samples     : {manifest['n_samples']}")
    elif dc == "source_spec":
        typer.echo(f"OK  {root}")
        typer.echo(f"    dataset_class : {manifest['dataset_class']}")
        typer.echo(f"    family        : {manifest['family']}")
        typer.echo(f"    version       : {manifest['version']}")
    else:
        typer.echo(
            f"Error: expected dataset_class 'layout_dataset' or 'source_spec', "
            f"got '{dc}'.",
            err=True,
        )
        raise typer.Exit(code=1)


# =============================================================================
@app.command("inspect")
def inspect(
    root: Annotated[
        Path,
        typer.Argument(
            help="Dagflow version root to inspect "
            "(e.g. data/interim/dagflow/routing/v1)."
        ),
    ],
) -> None:
    """Print a human-readable summary of a version root manifest."""
    manifest = validate_version_root(root.resolve())
    _print_manifest_summary(manifest)


# =============================================================================
# Helpers
# =============================================================================


def _resolve_dest(
    output_root: Path,
    preset: str,
    version: int,
) -> Path:
    """Return the destination version leaf path."""
    return output_root / preset / f"v{version}"


def _print_manifest_summary(manifest: dict) -> None:
    """Print a human-readable summary of a version root manifest."""
    typer.echo(f"  dataset_class  : {manifest.get('dataset_class', '?')}")
    typer.echo(f"  family         : {manifest.get('family', '?')}")
    typer.echo(f"  version        : {manifest.get('version', '?')}")
    typer.echo(f"  preset         : {manifest.get('preset', '?')}")
    typer.echo(f"  n_max          : {manifest.get('n_max', '?')}")
    typer.echo(f"  max_out_degree : {manifest.get('max_out_degree', '?')}")
    typer.echo(f"  target_edges   : {manifest.get('target_edges', '?')}")
    typer.echo(
        f"  extra_edge_density: " f"{manifest.get('extra_edge_density', '?')}"
    )
    typer.echo(f"  span_profile   : {manifest.get('span_profile', '?')}")
    typer.echo(f"  channels       : {manifest.get('channels', [])}")
    typer.echo(f"  n_samples      : {manifest.get('n_samples', {})}")
    typer.echo(f"  seed           : {manifest.get('seed', '?')}")
    typer.echo(
        f"  input_fingerprint: " f"{manifest.get('input_fingerprint', '?')}"
    )


if __name__ == "__main__":
    app()

"""CLI for building the NumberLine shared substrate.

NumberLine is a synthetic substrate: no raw download, no interim step.
The ``materialize-shared`` command generates the substrate directly.

Stages
------
materialize-shared  Synthesize the NumberLine shared substrate.
validate            Validate a NumberLine shared substrate version root.
build-all           Convenience alias: materialize-shared.

Default paths
-------------
Shared substrate:  data/processed/numberline/v1

Examples
--------
Build the default NumberLine substrate::

    python build-numberline.py build-all

With custom parameters::

    python build-numberline.py materialize-shared --n-states 20 --n-worlds 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.numberline import (
    SHARED_FAMILY,
    build_shared_substrate,
    validate_numberline_shared_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_VERSION = 1
_DEFAULT_N_STATES = 200
_DEFAULT_N_WORLDS = 100
_DEFAULT_SEED = 42

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("materialize-shared")
def materialize_shared(
    version: Annotated[
        int,
        typer.Option(
            "--version", help="Version integer for the substrate root."
        ),
    ] = _DEFAULT_VERSION,
    n_states: Annotated[
        int,
        typer.Option("--n-states", help="Number of states on the number line."),
    ] = _DEFAULT_N_STATES,
    n_worlds: Annotated[
        int,
        typer.Option(
            "--n-worlds", help="Total number of world samples to materialise."
        ),
    ] = _DEFAULT_N_WORLDS,
    seed: Annotated[
        int, typer.Option("--seed", help="Deterministic base seed.")
    ] = _DEFAULT_SEED,
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Processed data root.")
    ] = Path("data/processed"),
) -> None:
    """Synthesize the NumberLine shared substrate."""
    version_root = output_root / SHARED_FAMILY / f"v{version}"
    typer.echo(f"Building NumberLine shared substrate → {version_root}")
    build_shared_substrate(
        version_root,
        n_states=n_states,
        n_worlds=n_worlds,
        seed=seed,
    )
    typer.echo("Done.")


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(help="NumberLine shared-substrate root to validate."),
    ],
) -> None:
    """Validate a NumberLine shared-substrate version root."""
    validate_numberline_shared_root(root.resolve())
    typer.echo(f"OK  {root}")


# build-all removed — this script has a single stage: materialize-shared.

if __name__ == "__main__":
    app()

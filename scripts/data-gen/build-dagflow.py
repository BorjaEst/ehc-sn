"""Staged CLI for generating dagflow layout interim assets.

Dagflow is a layout source (peer to openfield, dungeongen) that generates
random DAG topologies with permuted node order and remapped observation IDs.

The output is an interim layout dataset root consumed by build-seqmaze.py.

Stages
------
generate-topology    Generate per-split source spec JSONL files (raw stage).
materialize-layouts  Expand source specs, assign structure, write layout dataset.
validate             Validate manifest.json and data of a layout version root.
build-all            Convenience alias: generate-topology -> materialize-layouts.

Default paths
-------------
Raw source specs:  data/raw/dagflow/{preset}/v{version}
Layout dataset:    data/interim/dagflow/{preset}/v{version}

Prerequisites
-------------
None — layouts are generated procedurally.

Examples
--------
Quick local build::

    python build-dagflow.py build-all

Custom sizes::

    python build-dagflow.py build-all \\
        --n-max 64 --t-max 48 --n-train 8000 --n-val 1000 --n-test 1000 --seed 7

Validate a dagflow layout dataset::

    python build-dagflow.py validate data/interim/dagflow/default/v1

Build a SeqMaze task corpus from dagflow layouts::

    python build-seqmaze.py materialize-task \\
        --layout-root data/interim/dagflow/default/v1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.substrate.dagflow import (
    LAYOUT_FAMILY,
    build_dagflow_layouts,
    validate_dagflow_layout_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_PRESET = "default"
_DEFAULT_RAW_ROOT = Path("data/raw/dagflow")
_DEFAULT_INTERIM_ROOT = Path("data/interim/dagflow")
_DEFAULT_VERSION = 1
_DEFAULT_N_MAX = 45
_DEFAULT_T_MAX = 16
_DEFAULT_MAX_OUT_DEGREE = 4
_DEFAULT_TARGET_EDGES = 105
_DEFAULT_FIXED_N_ACTUAL = True
_DEFAULT_MIN_EXTRA_EDGES = 1
_DEFAULT_N_TRAIN = 4000
_DEFAULT_N_VAL = 500
_DEFAULT_N_TEST = 500
_DEFAULT_SEED = 42

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("generate-topology")
def generate_topology(
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Named source preset (default: 'default').",
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
    topology_seed: Annotated[
        int,
        typer.Option(
            "--topology-seed",
            help=f"Seed controlling topology generation (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
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
            help=f"Root path for raw source specs (default: {_DEFAULT_RAW_ROOT}).",
        ),
    ] = _DEFAULT_RAW_ROOT,
) -> None:
    """Produce source-spec JSONL topology records (raw stage).

    Generates a reproducible source-spec corpus (per-split JSONL topology
    specs under ``{raw_root}/{preset}/v{version}/``).
    """
    raw_leaf = raw_root / preset / f"v{version}"
    if raw_leaf.exists():
        typer.echo(f"Source specs already exist at {raw_leaf}, skipping.")
        return

    import json

    raw_leaf.mkdir(parents=True, exist_ok=True)
    rng = __import__("numpy").random.SeedSequence(topology_seed)

    for split, n in [
        ("train", n_train),
        ("val", n_val),
        ("test", n_test),
    ]:
        (raw_leaf / split).mkdir(parents=True, exist_ok=True)
        specs = []
        split_rng = __import__("numpy").random.default_rng(rng.spawn(1)[0])
        for _ in range(n):
            sample_seed = int(split_rng.integers(0, 2**31))
            specs.append(
                {
                    "example_id": f"dagflow-{preset}-{split}-{sample_seed}",
                    "n_max": _DEFAULT_N_MAX,
                    "t_max": _DEFAULT_T_MAX,
                    "max_out_degree": _DEFAULT_MAX_OUT_DEGREE,
                    "seed_offset": sample_seed,
                    "split": split,
                }
            )
        spec_file = raw_leaf / split / "specs.jsonl"
        with spec_file.open("w") as fh:
            for s in specs:
                fh.write(json.dumps(s) + "\n")

    typer.echo(
        f"Source specs written to {raw_leaf}.  "
        "Run materialize-layouts to expand."
    )


# =============================================================================
@app.command("materialize-layouts")
def materialize_layouts(
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Named source preset (default: 'default').",
        ),
    ] = _DEFAULT_PRESET,
    n_max: Annotated[
        int,
        typer.Option(
            "--n-max",
            help=f"Maximum candidate nodes N (default: {_DEFAULT_N_MAX}).",
        ),
    ] = _DEFAULT_N_MAX,
    t_max: Annotated[
        int,
        typer.Option(
            "--t-max",
            help=f"Maximum path length T (default: {_DEFAULT_T_MAX}).",
        ),
    ] = _DEFAULT_T_MAX,
    max_out_degree: Annotated[
        int,
        typer.Option(
            "--max-out-degree",
            help=f"Maximum out-degree K (default: {_DEFAULT_MAX_OUT_DEGREE}).",
        ),
    ] = _DEFAULT_MAX_OUT_DEGREE,
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
    target_edges: Annotated[
        int,
        typer.Option(
            "--target-edges",
            help=f"Desired total edge count per graph "
            f"(default: {_DEFAULT_TARGET_EDGES}).",
        ),
    ] = _DEFAULT_TARGET_EDGES,
    fixed_n_actual: Annotated[
        bool,
        typer.Option(
            "--fixed-n-actual/--no-fixed-n-actual",
            help="When True, always use exactly n_max nodes "
            f"(default: {_DEFAULT_FIXED_N_ACTUAL}).",
        ),
    ] = _DEFAULT_FIXED_N_ACTUAL,
    min_extra_edges: Annotated[
        int,
        typer.Option(
            "--min-extra-edges",
            help="Minimum extra edges per non-terminal node beyond the "
            "Hamiltonian backbone "
            f"(default: {_DEFAULT_MIN_EXTRA_EDGES}).",
        ),
    ] = _DEFAULT_MIN_EXTRA_EDGES,
    topology_seed: Annotated[
        int,
        typer.Option(
            "--topology-seed",
            help=f"Seed controlling topology generation (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help=f"Version of the emitted layout dataset (default: {_DEFAULT_VERSION}).",
        ),
    ] = _DEFAULT_VERSION,
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root",
            help=f"Root path for raw source specs (default: {_DEFAULT_RAW_ROOT}).",
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
    """Expand source specs and write the dagflow layout dataset.

    Writes layout records to ``{interim_root}/{preset}/v{version}/``.
    """
    raw_leaf = raw_root / preset / f"v{version}"
    if not raw_leaf.exists():
        typer.echo(
            f"Error: source specs not found at {raw_leaf}. "
            f"Run generate-topology first.",
            err=True,
        )
        raise typer.Exit(code=1)

    layout_leaf = interim_root / preset / f"v{version}"
    build_dagflow_layouts(
        layout_leaf,
        preset=preset,
        n_max=n_max,
        t_max=t_max,
        max_out_degree=max_out_degree,
        target_edges=target_edges,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=topology_seed,
        fixed_n_actual=fixed_n_actual,
        min_extra_edges_per_node=min_extra_edges,
    )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(help="Dagflow version root to validate."),
    ],
) -> None:
    """Validate a dagflow version root's manifest and data.

    Supports both ``source_spec`` and ``layout_dataset`` roots.
    """
    manifest = validate_version_root(root.resolve())
    dc = manifest.get("dataset_class", "")
    if dc == "layout_dataset":
        manifest = validate_dagflow_layout_root(root.resolve())
    elif dc != "source_spec":
        typer.echo(
            f"Error: expected dataset_class 'layout_dataset' or 'source_spec', "
            f"got '{dc}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    family        : {manifest['family']}")
    typer.echo(f"    version       : {manifest['version']}")
    if dc == "layout_dataset":
        typer.echo(f"    channels      : {manifest['channels']}")
        typer.echo(f"    n_samples     : {manifest['n_samples']}")


# =============================================================================
@app.command("build-all")
def build_all(
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Named source preset (default: 'default').",
        ),
    ] = _DEFAULT_PRESET,
    n_max: Annotated[
        int,
        typer.Option(
            "--n-max",
            help=f"Maximum candidate nodes N (default: {_DEFAULT_N_MAX}).",
        ),
    ] = _DEFAULT_N_MAX,
    t_max: Annotated[
        int,
        typer.Option(
            "--t-max",
            help=f"Maximum path length T (default: {_DEFAULT_T_MAX}).",
        ),
    ] = _DEFAULT_T_MAX,
    max_out_degree: Annotated[
        int,
        typer.Option(
            "--max-out-degree",
            help=f"Maximum out-degree K (default: {_DEFAULT_MAX_OUT_DEGREE}).",
        ),
    ] = _DEFAULT_MAX_OUT_DEGREE,
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
    target_edges: Annotated[
        int,
        typer.Option(
            "--target-edges",
            help=f"Desired total edge count per graph "
            f"(default: {_DEFAULT_TARGET_EDGES}).",
        ),
    ] = _DEFAULT_TARGET_EDGES,
    fixed_n_actual: Annotated[
        bool,
        typer.Option(
            "--fixed-n-actual/--no-fixed-n-actual",
            help="When True, always use exactly n_max nodes "
            f"(default: {_DEFAULT_FIXED_N_ACTUAL}).",
        ),
    ] = _DEFAULT_FIXED_N_ACTUAL,
    min_extra_edges: Annotated[
        int,
        typer.Option(
            "--min-extra-edges",
            help="Minimum extra edges per non-terminal node beyond the "
            "Hamiltonian backbone "
            f"(default: {_DEFAULT_MIN_EXTRA_EDGES}).",
        ),
    ] = _DEFAULT_MIN_EXTRA_EDGES,
    topology_seed: Annotated[
        int,
        typer.Option(
            "--topology-seed",
            help=f"Seed controlling topology generation (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
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
            help=f"Root path for raw source specs (default: {_DEFAULT_RAW_ROOT}).",
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
    """Full pipeline: generate-topology -> materialize-layouts."""
    generate_topology(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        topology_seed=topology_seed,
        version=version,
        raw_root=raw_root,
    )
    materialize_layouts(
        preset=preset,
        n_max=n_max,
        t_max=t_max,
        max_out_degree=max_out_degree,
        target_edges=target_edges,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        topology_seed=topology_seed,
        fixed_n_actual=fixed_n_actual,
        min_extra_edges=min_extra_edges,
        version=version,
        raw_root=raw_root,
        interim_root=interim_root,
    )


if __name__ == "__main__":
    app()

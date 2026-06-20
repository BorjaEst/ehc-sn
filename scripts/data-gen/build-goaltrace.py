"""Staged CLI for building the Goaltrace task corpus.

Goaltrace consumes a dagflow shared substrate and creates a task corpus
with field-prediction protocol channels.

This CLI does not generate graph topology.  That stage belongs in
``build-dagflow.py``.

Stages
------
materialize-task     Build the Goaltrace task corpus over a shared substrate.
validate             Validate a Goaltrace task-corpus version root.

Default paths
-------------
Parent layout dataset:  <user-specified --layout-root>
Task corpus:            data/processed/goaltrace/<corpus>/v<version>

Examples
--------
Build the Goaltrace task corpus against a dagflow layout dataset::

    python build-goaltrace.py materialize-task \\
        --layout-root data/interim/dagflow/default/v1

With custom sizes::

    python build-goaltrace.py materialize-task \\
        --layout-root data/interim/dagflow/default/v1 \\
        --n-observations 64 --num-graphs 1 \\
        --oracle-semantics reliability --field-decay 0.8 \\
        --n-train 8000 --n-val 1000 --n-test 1000 --seed 7

With multiple DAGs::

    python build-goaltrace.py materialize-task \\
        --layout-root data/interim/dagflow/default/v1 \\
        --num-graphs 8 --n-train 4000

Validate an existing task corpus::

    python build-goaltrace.py validate data/processed/goaltrace/default/v1

Build a static-weight corpus (deterministic, contradiction-free)::

    python build-goaltrace.py materialize-task \\
        --layout-root data/interim/dagflow/default/v1 \\
        --static-weights \\
        --n-train 500 --n-val 250 --n-test 240 \\
        --distance-tau 8.0 \\
        --grid-width 20 --grid-height 30 \\
        --version 3

Prerequisites
-------------
A dagflow shared substrate must exist before running.
Build it first::

    python scripts/data-gen/build-dagflow.py build-all

For static-weight corpora, the dagflow substrate should be built with
``generate_hamiltonian_dag`` (the default after 2026-06-20).  Run::

    python scripts/data-gen/build-dagflow.py build-all \\
        --n-max 45 --max-out-degree 4 --version 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.tasks.goaltrace.builder import (
    TASK_FAMILY,
    build_goaltrace_task_corpus,
    validate_goaltrace_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_OBSERVATIONS = 45
_DEFAULT_NUM_GRAPHS = 1
_DEFAULT_ORACLE_SEMANTICS = "reliability"
_DEFAULT_FIELD_DECAY = 0.8
_DEFAULT_N_TRAIN = 500
_DEFAULT_N_VAL = 250
_DEFAULT_N_TEST = 240
_DEFAULT_SEED = 42
_DEFAULT_STATIC = True
_DEFAULT_MARGIN = 0.0
_DEFAULT_TAU = 8.0
_DEFAULT_DMAX = 0.0  # 0 means auto-compute from grid dimensions
_DEFAULT_GW = 20
_DEFAULT_GH = 30
_DEFAULT_GEOM_SEED = 42

app = typer.Typer(help="Goaltrace task corpus builder.")


# =============================================================================
@app.command("materialize-task")
def materialize_task(
    layout_root: Annotated[
        Path,
        typer.Option(
            "--layout-root",
            help="Path to dagflow layout dataset root "
            "(e.g. data/interim/dagflow/default/v1).",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option("--corpus", help="Corpus label (default: 'default')."),
    ] = _DEFAULT_CORPUS,
    n_observations: Annotated[
        int,
        typer.Option(
            "--n-observations",
            help="Maximum padded observation count N (default: 32).",
        ),
    ] = _DEFAULT_N_OBSERVATIONS,
    num_graphs: Annotated[
        int,
        typer.Option(
            "--num-graphs",
            help="Number of DAGs to sample from the layout pool "
            "(default: 1, all samples share one DAG).",
        ),
    ] = _DEFAULT_NUM_GRAPHS,
    oracle_semantics: Annotated[
        str,
        typer.Option(
            "--oracle-semantics",
            help="Weight semantics for path selection. "
            "One of: reliability, linear_cost, preference (default: reliability).",
        ),
    ] = _DEFAULT_ORACLE_SEMANTICS,
    field_decay: Annotated[
        float,
        typer.Option(
            "--field-decay",
            help="Field decay factor gamma in (0, 1) (default: 0.8).",
        ),
    ] = _DEFAULT_FIELD_DECAY,
    preference_step_penalty: Annotated[
        float,
        typer.Option(
            "--preference-step-penalty",
            help="Per-step penalty lambda for 'preference' semantics "
            "(default: 0.0).",
        ),
    ] = 0.0,
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train",
            help="Number of training samples (default: 4000).",
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val",
            help="Number of validation samples (default: 500).",
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option(
            "--n-test",
            help="Number of test samples (default: 500).",
        ),
    ] = _DEFAULT_N_TEST,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Task corpus version integer (default: 1).",
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Deterministic base seed (default: 42).",
        ),
    ] = _DEFAULT_SEED,
    # --- Static-weight pipeline flags ---
    static_weights: Annotated[
        bool,
        typer.Option(
            "--static-weights",
            help="Use static geometry-derived relational weights instead of "
            "per-sample random weights.  Requires a dagflow substrate built "
            "with the Hamiltonian backbone generator (default: false).",
        ),
    ] = _DEFAULT_STATIC,
    min_optimality_margin: Annotated[
        float,
        typer.Option(
            "--min-optimality-margin",
            help="Minimum cost margin between optimal and second-best path. "
            "Pairs below this threshold are rejected.  Only used with "
            "--static-weights (default: 0.0).",
        ),
    ] = _DEFAULT_MARGIN,
    distance_tau: Annotated[
        float,
        typer.Option(
            "--distance-tau",
            help="Temperature / distance scale for the spatial kernel. "
            "Only used with --static-weights (default: 8.0).",
        ),
    ] = _DEFAULT_TAU,
    distance_max: Annotated[
        float,
        typer.Option(
            "--distance-max",
            help="Maximum effective distance for the kernel.  Set to 0 to "
            "auto-compute from grid width+height.  Only used with "
            "--static-weights (default: 0 = auto).",
        ),
    ] = _DEFAULT_DMAX,
    grid_width: Annotated[
        int,
        typer.Option(
            "--grid-width",
            help="Hidden spatial grid width in cells.  Only used with "
            "--static-weights (default: 20).",
        ),
    ] = _DEFAULT_GW,
    grid_height: Annotated[
        int,
        typer.Option(
            "--grid-height",
            help="Hidden spatial grid height in cells.  Only used with "
            "--static-weights (default: 30).",
        ),
    ] = _DEFAULT_GH,
    geometry_seed: Annotated[
        int,
        typer.Option(
            "--geometry-seed",
            help="Seed for anchor placement on the spatial grid.  Only used "
            "with --static-weights (default: 42).",
        ),
    ] = _DEFAULT_GEOM_SEED,
) -> None:
    """Build the Goaltrace task corpus over a dagflow layout dataset."""
    # Translate distance_max=0 to None for the builder (auto-compute)
    parsed_distance_max = None if distance_max == 0.0 else distance_max

    root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    build_goaltrace_task_corpus(
        version_root=root.resolve(),
        layout_root=layout_root.resolve(),
        corpus=corpus,
        n_observations=n_observations,
        num_graphs=num_graphs,
        oracle_semantics=oracle_semantics,
        field_decay=field_decay,
        preference_step_penalty=preference_step_penalty,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        static_weights=static_weights,
        min_optimality_margin=min_optimality_margin,
        distance_tau=distance_tau,
        distance_max=parsed_distance_max,
        grid_width=grid_width,
        grid_height=grid_height,
        geometry_seed=geometry_seed,
    )
    print(f"Goaltrace corpus built at {root.resolve()}")
    print(f"  Profile: N={n_observations}, num_graphs={num_graphs}")
    print(f"  Oracle: {oracle_semantics}, field_decay={field_decay}")
    if static_weights:
        print(
            f"  Mode: static weights  tau={distance_tau}  "
            f"grid={grid_width}x{grid_height}  margin={min_optimality_margin}"
        )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Path to the goaltrace task corpus version root, "
            "e.g. data/processed/goaltrace/default/v1.",
        ),
    ],
) -> None:
    """Validate a goaltrace task-corpus version root."""
    root = root.resolve()
    manifest = validate_goaltrace_root(root)
    print(f"Goaltrace corpus valid: {root}")
    print(f"  Observations: {manifest.get('n_observations', '?')}")
    print(f"  Samples: {manifest.get('n_samples', {})}")
    print(f"  Oracle: {manifest.get('oracle_semantics', '?')}")
    print(f"  Field decay: {manifest.get('field_decay', '?')}")


# =============================================================================
if __name__ == "__main__":
    app()

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

Prerequisites
-------------
A dagflow shared substrate must exist before running.
Build it first::

    python scripts/data-gen/build-dagflow.py build-all
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
_DEFAULT_N_TRAIN = 4000
_DEFAULT_N_VAL = 500
_DEFAULT_N_TEST = 500
_DEFAULT_SEED = 42

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
) -> None:
    """Build the Goaltrace task corpus over a dagflow layout dataset."""
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
    )
    print(f"Goaltrace corpus built at {root.resolve()}")
    print(f"  Profile: N={n_observations}, num_graphs={num_graphs}")
    print(f"  Oracle: {oracle_semantics}, field_decay={field_decay}")


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

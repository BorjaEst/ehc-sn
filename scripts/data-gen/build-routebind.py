"""Staged CLI for building the Routebind task corpus.

Routebind consumes two parent artifacts:
1. A spatial topology interim (openfield or dungeongen) providing the
   complete visible spatial world, including observation identities.
2. A dagflow graph artifact providing the hidden semantic DAG over the
   same public observation vocabulary.

Routebind does not place, modify, or remap observations.  It reads
observation identities from the topology substrate verbatim.

This CLI does not generate topology layouts or DAGs.  Those stages belong
in ``build-openfield.py`` / ``build-dungeongen.py`` and ``build-dagflow.py``.

Stages
------
materialize-task     Build the Routebind task corpus over both parents.
validate             Validate a Routebind task-corpus version root.

Default paths
-------------
Topology dataset:       <user-specified --topology-root>
Dagflow dataset:        <user-specified --dagflow-root>
Task corpus:            data/processed/routebind/<corpus>/v<version>

Examples
--------
Build the Routebind task corpus with a dagflow graph::

    python build-routebind.py materialize-task \
        --topology-root data/interim/openfield/big-square/v1 \
        --dagflow-root data/interim/dagflow/default/v1 \
        --dagflow-graph-id dagflow-train-000001 \
        --n-queries-per-layout 10 --seed 42

Validate an existing task corpus::

    python build-routebind.py validate data/processed/routebind/default/v1

Prerequisites
-------------
An openfield (or dungeongen) layout dataset and a dagflow graph artifact
must exist before building.  Build them first::

    python scripts/data-gen/build-openfield.py build-all
    python scripts/data-gen/build-dagflow.py build-all
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.layout.io import load_layout_dataset
from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.routebind.builder import (
    TASK_FAMILY,
    build_routebind_task_corpus,
    validate_routebind_root,
)

# ---------------------------------------------------------------------------
_DEFAULT_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_FIELD_DECAY_SPATIAL = 0.9848
_DEFAULT_FIELD_DECAY_SEMANTIC = 0.8
_DEFAULT_MAX_ROUTE_LENGTH = 150
_DEFAULT_N_QUERIES_PER_LAYOUT = 10
_DEFAULT_SEED = 42
app = typer.Typer(help="Routebind task corpus builder.")


# =============================================================================
@app.command("materialize-task")
def materialize_task(
    topology_root: Annotated[
        Path,
        typer.Option(
            "--topology-root",
            help="Path to topology dataset root "
            "(e.g. data/interim/openfield/tem-square/v1).",
        ),
    ],
    dagflow_root: Annotated[
        Path,
        typer.Option(
            "--dagflow-root",
            help="Path to dagflow dataset root "
            "(e.g. data/interim/dagflow/default/v1).",
        ),
    ],
    dagflow_graph_id: Annotated[
        str,
        typer.Option(
            "--dagflow-graph-id",
            help="Stable artifact ID of the single DAG within dagflow-root "
            "(e.g. dagflow-default-v1-train-000042).",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option("--corpus", help="Corpus label (default: 'default')."),
    ] = _DEFAULT_CORPUS,
    field_decay_spatial: Annotated[
        float,
        typer.Option(
            "--field-decay-spatial",
            help="Spatial field decay factor (default: 0.9848).",
        ),
    ] = _DEFAULT_FIELD_DECAY_SPATIAL,
    field_decay_semantic: Annotated[
        float,
        typer.Option(
            "--field-decay-semantic",
            help="Semantic field decay factor (default: 0.8).",
        ),
    ] = _DEFAULT_FIELD_DECAY_SEMANTIC,
    max_supported_route_length: Annotated[
        int,
        typer.Option(
            "--max-supported-route-length",
            help="Maximum route length for terminal activation check "
            "(default: 150).",
        ),
    ] = _DEFAULT_MAX_ROUTE_LENGTH,
    n_queries_per_layout: Annotated[
        int,
        typer.Option(
            "--n-queries-per-layout",
            help="Number of start/goal queries per layout (default: 10).",
        ),
    ] = _DEFAULT_N_QUERIES_PER_LAYOUT,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Task corpus version integer (default: 1).",
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic base seed (default: 42)."),
    ] = _DEFAULT_SEED,
) -> None:
    """Build the Routebind task corpus over spatial topology + dagflow graph."""
    root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    topology_root_resolved = topology_root.resolve()
    dagflow_root_resolved = dagflow_root.resolve()

    if not topology_root_resolved.exists():
        typer.echo(
            f"Error: topology root not found at {topology_root_resolved}.\n"
            "Build topology first with:\n"
            "    python scripts/data-gen/build-openfield.py build-all\n"
            "or:\n"
            "    python scripts/data-gen/build-dungeongen.py build-all",
            err=True,
        )
        raise typer.Exit(code=1)

    if not dagflow_root_resolved.exists():
        typer.echo(
            f"Error: dagflow root not found at {dagflow_root_resolved}.\n"
            "Build dagflow first with:\n"
            "    python scripts/data-gen/build-dagflow.py build-all",
            err=True,
        )
        raise typer.Exit(code=1)

    topology_manifest = read_manifest(topology_root_resolved)
    layouts = load_layout_dataset(topology_root_resolved)

    version_root = root.resolve()
    build_routebind_task_corpus(
        version_root=version_root,
        layouts=layouts,
        topology_manifest=topology_manifest,
        dagflow_root=dagflow_root_resolved,
        dagflow_graph_id=dagflow_graph_id,
        corpus=corpus,
        field_decay_spatial=field_decay_spatial,
        field_decay_semantic=field_decay_semantic,
        max_supported_route_length=max_supported_route_length,
        n_queries_per_layout=n_queries_per_layout,
        seed=seed,
    )
    print(f"Routebind corpus built at {version_root}")
    print(f"  Topology: {topology_root_resolved}")
    print(f"  DAG graph: {dagflow_graph_id} @ {dagflow_root_resolved}")
    print(f"  Queries per layout: {n_queries_per_layout}")


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to validate "
            "(e.g. data/processed/routebind/default/v1)."
        ),
    ],
) -> None:
    """Validate an existing versioned root's manifest and data."""
    manifest = validate_routebind_root(root.resolve())
    print(f"Routebind corpus at {root.resolve()} is valid.")
    print(
        f"  Profile: N_obs={manifest.get('n_observations', '?')}, "
        f"Grid={manifest.get('grid_height', '?')}x{manifest.get('grid_width', '?')}"
    )
    print(f"  Samples: {manifest['n_samples']}")


# =============================================================================
if __name__ == "__main__":
    app()

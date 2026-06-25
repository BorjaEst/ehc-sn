#!/usr/bin/env python3
"""Canonical generic evaluation CLI — run, inspect, and benchmark.

Commands:

    run
        Execute a checkpoint against a fully-specified experiment config.
        Produces one evaluation artifact.

    inspect
        Read a completed evaluation artifact and display structural and
        field-level metadata.  Does not construct a model or provider.

    benchmark
        Aggregate evaluation runs.  Not yet implemented.

Usage::

    python scripts/evaluation/run_eval.py run \\
        --config config/evaluation/hrm-v1-mazehard.toml \\
        --checkpoint path/to/best.ckpt \\
        --output artifacts/eval/example

    python scripts/evaluation/run_eval.py inspect \\
        artifacts/eval/example \\
        --case 0 --list-fields

All reusable evaluation logic lives under ``src/ehc_sn.eval`` and
``src/ehc_sn.experiments``.  This script owns only CLI parsing and
terminal output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.eval.configuration import (
    EvaluationConfigurationError,
    load_evaluation_experiment,
)
from ehc_sn.eval.inspection import (
    EvaluationArtifactError,
    format_inspection_text,
    inspect_evaluation_artifact,
)
from ehc_sn.eval.offline import run_offline_eval
from ehc_sn.experiments._infra import EvaluationRunRequest

# ── Application ─────────────────────────────────────────────────────────────

app = typer.Typer(
    name="eval",
    help="Run, inspect, and benchmark evaluation artifacts.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


# ── run ─────────────────────────────────────────────────────────────────────


@app.command()
def run(  # -------------------------------------------------------------------
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Typed evaluation experiment TOML.",
        ),
    ],
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Model checkpoint to evaluate.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Destination for the evaluation artifact.",
        ),
    ],
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Execution device (e.g. cpu, cuda).",
        ),
    ] = "cpu",
    max_batches: Annotated[
        int | None,
        typer.Option(
            "--max-batches",
            min=1,
            help="Optional batch limit for diagnostic runs.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace an existing incomplete or completed output.",
        ),
    ] = False,
) -> None:
    """Execute a checkpoint against a fully-specified evaluation experiment."""
    # ── Load config ─────────────────────────────────────────────────────────
    try:
        loaded = load_evaluation_experiment(config)
    except EvaluationConfigurationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    # ── Build request ───────────────────────────────────────────────────────
    request = EvaluationRunRequest(
        checkpoint_path=checkpoint,
        output_dir=output,
        device=device,
        max_batches=max_batches or 0,
        overwrite=overwrite,
    )

    # ── Execute ─────────────────────────────────────────────────────────────
    try:
        artifact_path = run_offline_eval(
            experiment=loaded.experiment,
            request=request,
        )
    except Exception as exc:
        typer.echo(f"Execution failed: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    # ── Report ──────────────────────────────────────────────────────────────
    typer.echo("")
    typer.echo("Evaluation completed")
    typer.echo(f"  Experiment:  {loaded.experiment_id}")
    if loaded.experiment.identity:
        typer.echo(f"  Task:        {loaded.experiment.identity.task}")
    typer.echo(f"  Checkpoint:  {checkpoint}")
    typer.echo(f"  Artifact:    {artifact_path}")


# ── inspect ─────────────────────────────────────────────────────────────────


@app.command()
def inspect(  # ---------------------------------------------------------------
    artifact: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Completed evaluation artifact directory.",
        ),
    ],
    case: Annotated[
        int | None,
        typer.Option(
            "--case",
            min=0,
            help="Inspect one case by artifact case index.",
        ),
    ] = None,
    list_cases: Annotated[
        bool,
        typer.Option(
            "--list-cases",
            help="List available cases without loading dense arrays.",
        ),
    ] = False,
    list_fields: Annotated[
        bool,
        typer.Option(
            "--list-fields",
            help="Show captured fields and their stored array metadata.",
        ),
    ] = False,
    show_manifest: Annotated[
        bool,
        typer.Option(
            "--show-manifest",
            help="Include raw manifest contents.",
        ),
    ] = False,
) -> None:
    """Inspect a completed evaluation artifact."""
    try:
        result = inspect_evaluation_artifact(
            artifact,
            case_index=case,
            list_cases=list_cases,
            list_fields=list_fields,
            include_manifest=show_manifest,
        )
    except EvaluationArtifactError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    text = format_inspection_text(result)
    typer.echo(text)


# ── benchmark ───────────────────────────────────────────────────────────────


@app.command()
def benchmark() -> None:
    """Aggregate evaluation runs into a benchmark artifact."""
    typer.echo(
        "The benchmark command is not implemented yet.",
        err=True,
    )
    raise typer.Exit(code=2)


# ── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()

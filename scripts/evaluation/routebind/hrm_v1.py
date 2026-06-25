#!/usr/bin/env python3
"""Evaluate HRM-V1 on routebind — convenience alias around ``run_eval.py run``."""

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

DEFAULT_CONFIG = Path("config/evaluation/hrm-v1-routebind.toml")
DEFAULT_OUTPUT = Path("artifacts/eval/hrm-v1-routebind")

app = typer.Typer(no_args_is_help=True, help="Evaluate HRM-V1 on routebind.")


@app.command()
def run(
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
    ] = DEFAULT_OUTPUT,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Evaluation experiment TOML.",
        ),
    ] = DEFAULT_CONFIG,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    max_batches: Annotated[
        int | None, typer.Option("--max-batches", min=1)
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Execute a checkpoint against the routebind HRM-V1 experiment."""
    try:
        loaded = load_evaluation_experiment(
            config,
            expected_experiment_id="hrm-v1-routebind",
        )
    except EvaluationConfigurationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    request = EvaluationRunRequest(
        checkpoint_path=checkpoint,
        output_dir=output,
        device=device,
        max_batches=max_batches or 0,
        overwrite=overwrite,
    )
    try:
        artifact_path = run_offline_eval(
            experiment=loaded.experiment,
            request=request,
        )
    except Exception as exc:
        typer.echo(f"Execution failed: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    typer.echo(f"\nEvaluation completed\n  Experiment:  {loaded.experiment_id}")
    typer.echo(f"  Artifact:    {artifact_path}")


@app.command()
def inspect(
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
    case: Annotated[int | None, typer.Option("--case", min=0)] = None,
    list_cases: Annotated[bool, typer.Option("--list-cases")] = False,
    list_fields: Annotated[bool, typer.Option("--list-fields")] = False,
    show_manifest: Annotated[bool, typer.Option("--show-manifest")] = False,
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
    typer.echo(format_inspection_text(result))


@app.command()
def benchmark() -> None:
    """Aggregate evaluation runs (not implemented)."""
    typer.echo("The benchmark command is not implemented yet.", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

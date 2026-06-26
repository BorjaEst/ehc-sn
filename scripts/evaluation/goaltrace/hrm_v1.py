#!/usr/bin/env python3
"""Evaluate HRM-V1 on goaltrace — convenience alias around ``run_eval.py run``."""

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

DEFAULT_CONFIG = Path("config/evaluation/hrm-v1-goaltrace.toml")
DEFAULT_GALLERY_DIR = Path("artifacts/inspection/hrm-v1-goaltrace")
DEFAULT_EVAL_ARTIFACT_DIR = Path("artifacts/eval/hrm-v1-goaltrace")

app = typer.Typer(no_args_is_help=True, help="Evaluate HRM-V1 on goaltrace.")


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
    ] = DEFAULT_EVAL_ARTIFACT_DIR,
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
    """Execute a checkpoint against the goaltrace HRM-V1 experiment."""
    try:
        loaded = load_evaluation_experiment(
            config,
            expected_experiment_id="hrm-v1-goaltrace",
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
    ] = DEFAULT_EVAL_ARTIFACT_DIR,
    case: Annotated[int | None, typer.Option("--case", min=0)] = None,
    list_cases: Annotated[bool, typer.Option("--list-cases")] = False,
    list_fields: Annotated[bool, typer.Option("--list-fields")] = False,
    show_manifest: Annotated[bool, typer.Option("--show-manifest")] = False,
    gallery: Annotated[
        bool,
        typer.Option("--gallery", help="Render evaluation figure images."),
    ] = False,
    gallery_output: Annotated[
        Path,
        typer.Option(
            "--gallery-output",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Destination directory for the inspection artifact.",
        ),
    ] = DEFAULT_GALLERY_DIR,
    max_gallery_samples: Annotated[
        int,
        typer.Option(
            "--max-gallery-samples",
            min=1,
            help="Number of evaluation batches to render.",
        ),
    ] = 8,
    gallery_roles: Annotated[
        str,
        typer.Option(
            "--gallery-roles",
            help="Comma-separated figure role names.",
        ),
    ] = "prediction_reasoning",
) -> None:
    """Inspect a completed evaluation artifact.

    When ``--gallery`` is set, renders evaluation figure images and writes
    them to ``--gallery-output``.  The library never loads the checkpoint
    or executes the model.
    """
    if gallery:
        gallery_output.mkdir(parents=True, exist_ok=True)

    try:
        result = inspect_evaluation_artifact(
            artifact,
            case_index=case,
            list_cases=list_cases,
            list_fields=list_fields,
            include_manifest=show_manifest,
            gallery=gallery,
            gallery_output=gallery_output,
            max_gallery_samples=max_gallery_samples,
            gallery_roles=tuple(
                r.strip() for r in gallery_roles.split(",") if r.strip()
            ),
        )
    except EvaluationArtifactError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    text = format_inspection_text(result)
    typer.echo(text)

    # Gallery summary
    if gallery and result.gallery is not None:
        typer.echo("")
        typer.echo(f"  Gallery output: {result.gallery.output_root}")
        typer.echo(
            f"  Rendered roles: {', '.join(result.gallery.rendered_roles)}"
        )
        successful = sum(1 for img in result.gallery.images if img.success)
        typer.echo(f"  Images written: {successful}")
        if result.gallery.images and successful < len(result.gallery.images):
            typer.echo(
                f"  Failed images:  {len(result.gallery.images) - successful}",
                err=True,
            )


@app.command()
def benchmark() -> None:
    """Aggregate evaluation runs (not implemented)."""
    typer.echo("The benchmark command is not implemented yet.", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

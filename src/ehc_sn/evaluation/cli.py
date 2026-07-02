"""Single ``evaluation`` command and shared ``inspect`` subcommand.

This module exports ``evaluation_app`` — a Typer app registered as
``evaluation`` in the top-level ``ehp`` CLI.

Usage::

    ehp evaluation arena-tem-v1 --model ./checkpoints/best.ckpt
    ehp evaluation run --config examples/evaluation/invocation.toml
    ehp evaluation inspect artifacts/evaluation/arena-tem-v1-20260627/
"""

from __future__ import annotations

import json as _json
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

from ehc_sn.evaluation.artifact_models import (
    DatasetProvenance,
    EvaluationIdentity,
    ModelProvenance,
    write_evaluation_manifest,
)
from ehc_sn.evaluation.artifacts import _MANIFEST_FILENAME
from ehc_sn.evaluation.configuration import (
    EvaluationConfigurationError,
    _recipe_defaults,
    build_experiment_from_invocation,
    invocation_to_provenance,
    load_evaluation_experiment,
    resolve_evaluation_invocation,
)
from ehc_sn.evaluation.inspection import (
    EvaluationArtifactError,
    format_inspection_text,
    inspect_evaluation_artifact,
)
from ehc_sn.evaluation.invocation import (
    EvaluationExecutionRequest,
    LocalModelArtifactResolver,
    merge_evaluation_config,
)
from ehc_sn.evaluation.model_ref import ModelRef
from ehc_sn.evaluation.offline import run_offline_eval
from ehc_sn.evaluation.recipes import (
    EvaluationAlias,
    list_recipes,
    resolve_recipe,
)
from ehc_sn.evaluation.tracking import MLflowEvaluationRecorder

# =============================================================================
# Helpers
# =============================================================================

_DEFAULT_GALLERY_DIR = Path("artifacts/inspection")

_RECIPE_ONLY_TOML_KEYS = frozenset(
    {
        "task",
        "model_family",
        "primary_metric",
        "required_capabilities",
    }
)
"""Keys that belong to recipe TOMLs and prove a file is a recipe definition, not an invocation."""


_MINIMAL_INVOCATION_TOML = """schema_version = 1
alias = "arena-tem-v1"

[model]
uri = "./checkpoints/best.ckpt"
"""


def _read_config_toml(path: Path) -> dict[str, object]:
    """Read and parse a TOML config file, exiting on error."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        typer.echo(f"Error reading {path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _reject_recipe_toml(cfg: dict[str, object], path: Path) -> None:
    """Exit with a clear error if *cfg* appears to be a recipe TOML.

    Recipe TOMLs contain keys like ``task``, ``model_family``,
    ``primary_metric``, ``required_capabilities``, or a ``[defaults]``
    section.  These are not valid in an invocation file.
    """
    recipe_only_fields = sorted(_RECIPE_ONLY_TOML_KEYS & cfg.keys())
    if not recipe_only_fields:
        return

    typer.echo(
        f"{path} is an evaluation recipe definition, "
        f"not an evaluation invocation.\n\n"
        f"Recipe-only fields found: {', '.join(recipe_only_fields)}\n\n"
        "Recipe definitions are selected automatically by alias:\n"
        "  ehp-sn evaluation run <alias> --model <checkpoint>\n\n"
        "Example:\n"
        "  ehp-sn evaluation run arena-tem-v1 "
        "--model ./checkpoints/best.ckpt\n\n"
        "To use --config, provide an invocation TOML.  Example:\n\n"
        f"{_MINIMAL_INVOCATION_TOML}",
        err=True,
    )
    raise typer.Exit(code=2)


def _parse_tags(tag_list: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` tag strings into a dict."""
    tags: dict[str, str] = {}
    if tag_list is None:
        return tags
    for item in tag_list:
        if "=" not in item:
            raise typer.BadParameter(
                f"Tag must be in KEY=VALUE form, got {item!r}."
            )
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def _run_evaluation(
    recipe: Any,
    model_ref: ModelRef,
    split: str | None,
    cases: int | None,
    seed: int | None,
    device: str | None,
    config_path: Path | None,
    figure_names: tuple[str, ...],
    inspect_enabled: bool | None,
    user_tags: dict[str, str],
    experiment: str,
    tracking_uri: str | None,
    run_name: str | None,
) -> Any:
    """Resolve and execute one evaluation from a recipe.

    Config loading goes through ``load_evaluation_experiment``: recipe/
    Pydantic defaults → TOML file values → CLI overrides.

    Returns a dict with outcome information.
    """
    from ehc_sn.analysis import (
        EvaluationRegistries,
        FigurePlanError,
        compile_figure_evaluation_plan,
        register_builtin_analysis_specs,
    )
    from ehc_sn.figures import REGISTRY as figure_registry
    from ehc_sn.figures import list_figures

    # ---- Resolve model artifact ---------------------------------------------
    resolver = LocalModelArtifactResolver()
    resolved_model = resolver.resolve(model_ref)

    # ---- Build CLI overrides ------------------------------------------------
    cli_overrides: dict[str, object] = {"model": {"uri": model_ref.value}}
    if split is not None:
        cli_overrides.setdefault("cases", {})["split"] = split  # type: ignore
    if cases is not None:
        cli_overrides.setdefault("cases", {})["count"] = cases  # type: ignore
    if seed is not None:
        cli_overrides.setdefault("cases", {})["seed"] = seed  # type: ignore
    if device is not None:
        cli_overrides["runtime"] = {"device": device}
    if inspect_enabled is not None:
        cli_overrides["inspection"] = {"enabled": inspect_enabled}
    # Tracking is always enabled — no boolean toggle needed.

    # ---- Load config → build experiment (single canonical path) -------------
    if config_path is not None:
        loaded = load_evaluation_experiment(
            config_path,
            expected_alias=recipe.config.alias,
            model=resolved_model,
            cli_overrides=cli_overrides,
        )
    else:
        # No config file — merge recipe TOML defaults + CLI overrides directly.
        recipe_defaults = _recipe_defaults(recipe)
        invocation = merge_evaluation_config(
            recipe_defaults=recipe_defaults,
            cli_overrides=cli_overrides,
        )
        resolved = resolve_evaluation_invocation(
            recipe=recipe,
            invocation=invocation,
            model=resolved_model,
        )
        loaded = build_experiment_from_invocation(
            resolved=resolved,
        )

    # ---- Resolve figure names -----------------------------------------------
    resolved_figures = list(figure_names) if figure_names else []
    inspect_effective = (
        inspect_enabled
        if inspect_enabled is not None
        else loaded.resolved.inspection.enabled
    )
    if not resolved_figures and inspect_effective:
        resolved_figures = list(
            loaded.resolved.inspection.figures
            if loaded.resolved.inspection.figures
            else []
        )

    # Compile figure plan if figures requested.
    plan = None
    if resolved_figures:
        list_figures()
        registries = EvaluationRegistries(figures=figure_registry)
        register_builtin_analysis_specs(registries)
        try:
            plan = compile_figure_evaluation_plan(
                experiment=loaded.experiment,
                figure_names=resolved_figures,
                registries=registries,
            )
        except FigurePlanError as exc:
            raise EvaluationConfigurationError(
                f"Invalid figure evaluation plan: {exc}"
            ) from exc

    # ---- Build execution request --------------------------------------------
    output_dir = Path(f"artifacts/evaluation/{loaded.resolved.alias.value}")
    execution_request = EvaluationExecutionRequest(
        experiment=loaded.experiment,
        model_artifact=loaded.resolved.model,
        output_dir=output_dir,
        device=loaded.resolved.runtime.device,
        max_batches=0,
        overwrite=False,
        no_reuse=False,
        precision=loaded.resolved.runtime.precision,
    )

    # ---- Build tags ---------------------------------------------------------
    tags: dict[str, str] = {
        "ehp.alias": loaded.resolved.alias.value,
        "ehp.task": loaded.resolved.recipe.config.task,
        "ehp.model_family": loaded.resolved.recipe.config.model_family.value,
        "ehp.primary_metric": loaded.resolved.recipe.config.primary_metric
        or "",
        "ehp.model.requested_uri": model_ref.value,
        **user_tags,
    }

    # ---- Recorder (always MLflow, never null) -----------------------------
    recorder = MLflowEvaluationRecorder(
        tracking_uri=tracking_uri,
        experiment_name=(loaded.resolved.tracking.experiment or experiment),
        run_name=run_name or loaded.resolved.tracking.run_name,
        tags=tags,
    )

    # ---- Execute ------------------------------------------------------------
    with recorder:
        artifact_path = run_offline_eval(
            experiment=loaded.experiment,
            execution=execution_request,
            plan=plan,
        )

        # Write resolved-invocation.toml artifact.
        resolved_dict = invocation_to_provenance(loaded.resolved)
        import tomli_w as _tomli_w

        resolved_toml = _tomli_w.dumps(resolved_dict)
        resolved_path = artifact_path / "resolved-invocation.toml"
        resolved_path.write_text(resolved_toml, encoding="utf-8")
        recorder.log_artifact(str(resolved_path))

        # Write user-supplied invocation TOML if provided.
        if config_path is not None:
            invocation_path = artifact_path / "invocation.toml"
            invocation_path.write_text(
                config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            recorder.log_artifact(str(invocation_path))

        # Write execution-manifest.json.
        exec_manifest = {
            "model": {
                "requested_uri": loaded.resolved.model.requested_uri,
                "resolved_source": loaded.resolved.model.resolved_source,
            },
            "invocation_alias": loaded.resolved.alias.value,
            "split": loaded.resolved.cases.split,
            "capture_profile": loaded.resolved.capture.profile,
        }
        exec_manifest_path = artifact_path / "execution-manifest.json"
        exec_manifest_path.write_text(
            _json.dumps(exec_manifest, indent=2),
            encoding="utf-8",
        )
        recorder.log_artifact(str(exec_manifest_path))

        # Write evaluation-manifest.json.
        eval_identity = EvaluationIdentity(
            evaluation_id=(
                str(recorder.run_id)
                if recorder.run_id
                else f"evaluation-{loaded.resolved.alias.value}"
            ),
            alias=loaded.resolved.alias.value,
            task=loaded.resolved.recipe.config.task,
            model_family=loaded.resolved.recipe.config.model_family.value,
        )
        model_prov = ModelProvenance(
            uri=loaded.resolved.model.requested_uri,
            resolved_uri=(
                loaded.resolved.model.resolved_source
                if loaded.resolved.model.resolved_source
                else None
            ),
            digest=None,
        )
        dataset_prov = DatasetProvenance(
            uri=loaded.resolved.cases.split,
            split=loaded.resolved.cases.split,
            digest=None,
        )
        # The regime manifest is at the artifact root itself (single regime).
        regimes = {
            loaded.experiment.regime_id or "diagnostic": _MANIFEST_FILENAME
        }
        write_evaluation_manifest(
            artifact_path,
            identity=eval_identity,
            model=model_prov,
            dataset=dataset_prov,
            regimes=regimes,
        )
        recorder.log_artifact(str(artifact_path / "evaluation-manifest.json"))

        # Log parameters (compact scalars only).
        recorder.log_params(
            {
                "cases.split": loaded.resolved.cases.split,
                "cases.case_count": str(len(loaded.resolved.cases.case_ids)),
                "cases.seed": str(loaded.resolved.recipe.config.cases.seed),
                "runtime.batch_size": str(loaded.resolved.runtime.batch_size),
                "runtime.workers": str(loaded.resolved.runtime.workers),
                "runtime.precision": (loaded.resolved.runtime.precision.value),
                "runtime.deterministic": str(
                    loaded.resolved.runtime.deterministic
                ),
                "capture.profile": str(loaded.resolved.capture.profile),
                "capture.max_cases": str(
                    loaded.resolved.capture.max_cases or ""
                ),
                "inspection.enabled": str(loaded.resolved.inspection.enabled),
            }
        )

        # Log evaluation options if present.
        eval_options = loaded.resolved.evaluation
        if hasattr(eval_options, "model_dump"):
            for k, v in eval_options.model_dump(exclude_none=True).items():
                if isinstance(v, (str, int, float, bool)):
                    recorder.log_params({f"evaluation.{k}": str(v)})

    outcome = {
        "artifact_path": artifact_path,
        "primary_metric": loaded.resolved.recipe.config.primary_metric,
    }
    outcome["mlflow_run_uri"] = recorder.run_id

    return outcome


# =============================================================================
# Commands
# =============================================================================

evaluation_app = typer.Typer(
    name="evaluation",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@evaluation_app.command()
def run(
    recipe: Annotated[
        str | None,
        typer.Argument(
            help="Registered evaluation recipe ID "
            "(e.g. arena-tem-v1). "
            "Run without arguments to list supported recipes.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Local model checkpoint path or MLflow model URI.",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            help="Override the recipe's default dataset split.",
        ),
    ] = None,
    cases: Annotated[
        int | None,
        typer.Option(
            "--count",
            min=1,
            help="Maximum evaluation cases.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(min=0),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            help="Runtime device (auto, cpu, cuda, cuda:N).",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Evaluation invocation TOML; explicit CLI options override it.",
        ),
    ] = None,
    figure: Annotated[
        list[str] | None,
        typer.Option(
            "--figure",
            help="Restrict generated figures; repeatable.",
        ),
    ] = None,
    no_inspect: Annotated[
        bool | None,
        typer.Option(
            "--no-inspect",
            help="Disable inspection artifact generation.",
        ),
    ] = None,
    experiment: Annotated[
        str,
        typer.Option(
            envvar="MLFLOW_EXPERIMENT_NAME",
            help="MLflow experiment name.",
        ),
    ] = "ehp-evaluation",
    run_name: Annotated[
        str | None,
        typer.Option(
            "--run-name",
            help="Human-readable MLflow run name.",
        ),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            help="Additional MLflow tag in KEY=VALUE form; repeatable.",
        ),
    ] = None,
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            envvar="MLFLOW_TRACKING_URI",
            help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI, "
            "then the local EHP SQLite tracking store.",
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Show full Python traceback on unexpected failures.",
        ),
    ] = False,
) -> None:
    """Evaluate a checkpoint against a registered task--model recipe.

    Two invocation forms are supported:

        ehp-sn evaluation run RECIPE --model MODEL [OPTIONS]
        ehp-sn evaluation run --config INVOCATION.toml [OPTIONS]

    ALIAS selects the recipe (e.g. arena-tem-v1).  --model provides a
    local checkpoint path or MLflow model URI.  When --config supplies
    both alias and model, neither positional ALIAS nor --model is required.

    Resolution precedence: recipe defaults < invocation TOML < CLI flags.
    Run without arguments to list supported aliases.
    """
    # ---- Resolve alias and config -----------------------------------------
    alias_id: EvaluationAlias | None = None
    effective_config: Path | None = config

    if recipe is None:
        if config is not None:
            cfg = _read_config_toml(config)
            _reject_recipe_toml(cfg, config)
            alias_value = cfg.get("alias")
            if not alias_value or not isinstance(alias_value, str):
                typer.echo(
                    f"Error: config {config} must declare 'alias'.", err=True
                )
                raise typer.Exit(code=2)
            recipe = alias_value.strip()
            effective_config = config
        else:
            typer.echo("Supported evaluation aliases:")
            for r in list_recipes():
                typer.echo(f"  {r.value}")
            raise typer.Exit(code=0)

    if model is None and effective_config is not None:
        cfg = _read_config_toml(effective_config)
        model_uri = (
            cfg.get("model", {}).get("uri")
            if isinstance(cfg.get("model"), dict)
            else None
        )
        if model_uri:
            model = model_uri

    if model is None:
        typer.echo(
            "Error: --model is required when a recipe is provided.", err=True
        )
        raise typer.Exit(code=2)

    # Resolve and validate alias.
    try:
        alias_id = EvaluationAlias(recipe)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    # Alias conflict: if both CLI and config provide an alias, they must agree.
    if config is not None and recipe is not None:
        cfg = _read_config_toml(config)
        _reject_recipe_toml(cfg, config)
        file_alias_str = cfg.get("alias")
        if file_alias_str and isinstance(file_alias_str, str):
            try:
                file_alias_id = EvaluationAlias(file_alias_str.strip())
                if alias_id != file_alias_id:
                    typer.echo(
                        f"CLI alias {alias_id.value!r} conflicts with "
                        f"invocation alias {file_alias_id.value!r}.",
                        err=True,
                    )
                    raise typer.Exit(code=2)
            except ValueError:
                pass  # Let the alias validation handle invalid IDs.

    recipe = resolve_recipe(alias_id)

    model_ref = ModelRef(model)
    user_tags = _parse_tags(tag)

    # Compute effective boolean overrides.
    inspect_effective: bool | None = (
        not no_inspect if no_inspect is not None else None
    )
    try:
        outcome = _run_evaluation(
            recipe=recipe,
            model_ref=model_ref,
            split=split,
            cases=cases,
            seed=seed,
            device=device,
            config_path=effective_config,
            figure_names=tuple(figure or ()),
            inspect_enabled=inspect_effective,
            user_tags=user_tags,
            experiment=experiment,
            tracking_uri=tracking_uri,
            run_name=run_name,
        )
    except typer.Exit:
        raise
    except EvaluationConfigurationError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except Exception as exc:
        if debug:
            raise
        typer.echo(
            f"Execution failed: {exc}\n"
            "Re-run with --debug for the full traceback.",
            err=True,
        )
        raise typer.Exit(code=5) from exc

    typer.echo("")
    typer.echo(f"  Alias:          {recipe.config.alias}")
    typer.echo(f"  Task:           {recipe.config.task}")
    typer.echo(f"  Model family:   {recipe.config.model_family.value}")
    typer.echo(f"  Primary metric: {recipe.config.primary_metric or 'N/A'}")
    typer.echo(f"  Artifact:       {outcome['artifact_path']}")
    typer.echo("status: completed")
    if outcome.get("mlflow_run_uri"):
        typer.echo(f"run: {outcome['mlflow_run_uri']}")


@evaluation_app.command()
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
    ] = ...,
    case: Annotated[
        int | None,
        typer.Option("--case", min=0),
    ] = None,
    list_cases: Annotated[
        bool,
        typer.Option("--list-cases"),
    ] = False,
    list_fields: Annotated[
        bool,
        typer.Option("--list-fields"),
    ] = False,
    show_manifest: Annotated[
        bool,
        typer.Option("--show-manifest"),
    ] = False,
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
        ),
    ] = _DEFAULT_GALLERY_DIR,
    max_gallery_samples: Annotated[
        int,
        typer.Option("--max-gallery-samples", min=1),
    ] = 4,
    gallery_roles: Annotated[
        str,
        typer.Option(
            "--gallery-roles",
            help="Comma-separated figure role names.  Empty (default) = "
            "auto-detect from recipe [inspection].figures.",
        ),
    ] = "",
) -> None:
    """Inspect a completed evaluation artifact."""
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
        if result.gallery.diagnostics:
            typer.echo("  Diagnostics:", err=True)
            from ehc_sn.evaluation.inspection import FigureRejectionCode

            # Group diagnostics by code.
            by_code: dict[str, list[str]] = {}
            for d in result.gallery.diagnostics:
                by_code.setdefault(d.code.value, []).append(
                    f"{d.figure_key}: {d.message}"
                )
            for code_val in sorted(by_code):
                typer.echo(f"    [{code_val}]")
                for line in by_code[code_val]:
                    typer.echo(f"      {line}")

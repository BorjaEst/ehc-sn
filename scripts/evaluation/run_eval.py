#!/usr/bin/env python3
"""CLI entry point for offline evaluation — checkpoint → eval artifact v3.

Usage::

    python scripts/evaluation/run_eval.py \\
        --config config/evaluation/hrm-v1-mazehard.toml \\
        --checkpoint runs/hrm_v1/version_x/checkpoints/best.ckpt \\
        --output artifacts/eval/hrm_v1/mazehard_test \\
        --device cuda

The TOML config fully specifies the experiment: model, execution, provider,
regime, and capture settings.  The CLI only receives run-instance parameters.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.eval.offline import run_offline_eval
from ehc_sn.experiments._infra import (
    EvaluationRunRequest,
    resolve_evaluation_experiment,
)

# ---------------------------------------------------------------------------
# CLI settings
# ---------------------------------------------------------------------------


class EvalArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """CLI settings for offline evaluation — checkpoint → eval artifact v3."""

    model_config = SettingsConfigDict(extra="forbid")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """CLI overrides all other sources."""
        extra = [
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]
        return CliSettingsSource(settings_cls), *extra

    config: Path = Field(
        ...,
        description="Path to a TOML config defining the full evaluation experiment.",
    )
    checkpoint: Path = Field(
        ..., description="Path to a weights-only checkpoint file."
    )
    output: Path = Field(
        ..., description="Target directory for the persisted eval artifact."
    )
    device: str = Field(
        default="cpu",
        description="PyTorch device string (e.g. cpu, cuda).",
    )
    max_batches: int = Field(
        default=0,
        ge=0,
        description="Maximum provider batches. 0 = all.",
    )


def main() -> None:
    settings = EvalArguments()

    # ---- Resolve experiment from config -------------------------------------
    experiment = resolve_evaluation_experiment(settings.config)

    # ---- Build run request --------------------------------------------------
    request = EvaluationRunRequest(
        checkpoint_path=settings.checkpoint,
        output_dir=settings.output,
        device=settings.device,
        max_batches=settings.max_batches,
    )

    output_path = run_offline_eval(
        experiment=experiment,
        request=request,
    )
    print(f"Eval artifact written to: {output_path!s}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

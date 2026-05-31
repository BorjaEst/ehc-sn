#!/usr/bin/env python3
"""CLI entry point for offline evaluation — checkpoint → eval artifact v3.

Usage::

    python scripts/evaluation/run_eval.py \\
        --model-family hrm-v2 \\
        --checkpoint runs/hrm_v2/version_x/checkpoints/best.ckpt \\
        --config configs/training/hrm_v2_debug.toml \\
        --task mazehard \\
        --provider-ref ehc_sn.tasks.mazehard.providers.MazeHardReplayProvider \\
        --provider-settings '{"dataset_path": "data/processed/mazehard", "split": "val", "n_cases": 2}' \\
        --regime-id mazehard_test \\
        --regime-kind diagnostic \\
        --output artifacts/eval/hrm_v2/mazehard_test \\
        --device cpu

Wires existing ``ehc_sn.eval.offline.run_offline_eval`` — no eval execution
or artifact-production logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.eval.offline import run_offline_eval

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

    model_family: str = Field(
        ...,
        min_length=1,
        description="Canonical model-family identifier (e.g. hrm-v2, tem-v2).",
    )
    checkpoint: Path = Field(
        ..., description="Path to a weights-only checkpoint file."
    )
    config: Path = Field(
        ...,
        description="Path to a TOML config for the model family's executor.",
    )
    task: str = Field(
        ...,
        min_length=1,
        description="Canonical task identifier (e.g. mazehard, arena).",
    )
    provider_ref: str = Field(
        ...,
        min_length=1,
        description="Dotted import path to an EvaluationSourceProvider class.",
    )
    provider_settings: dict[str, object] = Field(
        default_factory=dict,
        description="JSON-encoded provider settings dict.",
    )
    regime_id: str = Field(
        ...,
        min_length=1,
        description="Unique regime identifier for the artifact manifest.",
    )
    regime_kind: Literal["diagnostic", "benchmark"] = Field(
        default="diagnostic",
        description="Regime classification.",
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
    trace_keys: list[str] | None = Field(
        default=None,
        description="Comma-separated trace key strings (e.g. act/halted,pred/solution_overlay).",
    )

    @field_validator("provider_settings", mode="before")
    @classmethod
    def _parse_provider_settings(cls, v: object) -> dict[str, object]:
        """Parse a JSON string into a dict."""
        if isinstance(v, str):
            result = json.loads(v)
            if not isinstance(result, dict):
                raise ValueError(
                    "--provider-settings must be a JSON object (dict), "
                    f"got {type(result).__name__}."
                )
            return result
        return v  # type: ignore[return-value]

    @field_validator("trace_keys", mode="before")
    @classmethod
    def _parse_trace_keys(cls, v: object) -> list[str] | None:
        """Parse a comma-separated string into a list."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v  # type: ignore[return-value]


def main() -> None:
    settings = EvalArguments()

    provider_settings: dict[str, object] = settings.provider_settings
    trace_keys: list[str] | None = settings.trace_keys

    output_path = run_offline_eval(
        model_family=settings.model_family,
        executor_config_path=settings.config,
        checkpoint_path=settings.checkpoint,
        task=settings.task,
        provider_ref=settings.provider_ref,
        provider_settings=provider_settings,
        regime_id=settings.regime_id,
        regime_kind=settings.regime_kind,
        output_dir=settings.output,
        device=settings.device,
        max_batches=settings.max_batches,
        trace_keys=trace_keys,
    )
    print(f"Eval artifact written to: {output_path!s}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

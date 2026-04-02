"""Entrypoint for the B0 MazeHard bridge benchmark."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn import benchmark, models

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("BENCHMARK_B0_CONFIGURATION_PATH", "config/benchmark-b0.hrm-v1.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, populate_by_name=True, cli_implicit_flags=True):
    """CLI arguments for the B0 benchmark entrypoint."""

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        """Place CLI arguments ahead of all other settings sources."""
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    benchmark_config: Path = Field(
        ...,
        description="Path to the B0 benchmark wrapper TOML file.",
    )
    model_kind: str = Field(
        default="hrm_v1",
        description="The B0 model kind to evaluate. Supported values: 'hrm_v1'.",
    )
    model_config: Path = Field(
        ...,
        description="Path to the B0 model configuration TOML file that specifies the model.",
    )
    model_checkpoint: Path | None = Field(
        default=None,
        description="Optional checkpoint override used for evaluation.",
    )

    seed: int | None = Field(
        default=None,
        description="Optional seed override. When omitted, use canonical benchmark seeds.",
    )
    compute_budget: int | None = Field(
        default=None,
        description="Optional compute-budget override. When omitted, use the wrapper default behavior.",
    )
    output_dir: Path | None = Field(
        default=None,
        description="Optional output directory override for benchmark artifacts.",
    )


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    # Load defaults from TOML, then parse settings.
    # CLI arguments override TOML values; Pydantic defaults fill in anything missing.
    defaults = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**defaults)

    # TODO: Load model, dataset, and evaluation configuration based ...
    model = models...
    benchmark = benchmark.B0BridgeBenchmark(...)

    manifest = ...
    print(
        f"Wrote {len(manifest['artifact_paths'])} B0 result artifacts "
        f"to {manifest['output_root']}"
    )  # fmt: skip

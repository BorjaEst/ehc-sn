"""Entrypoint for the M0 Episodic Memory Bridge."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.benchmarks.m0 import M0Benchmark, M0BenchmarkConfig, build_m0_agent

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("BENCHMARK_M0_CONFIGURATION_PATH", "config/benchmark-m0.tem-v1.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, populate_by_name=True, cli_implicit_flags=True):
    """CLI arguments for the M0 Episodic Memory Bridge entrypoint."""

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        """Place CLI arguments ahead of all other settings sources."""
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    benchmark_config: Path = Field(
        ...,
        description="Path to the M0 Episodic Memory Bridge wrapper TOML file.",
    )
    model_kind: str = Field(
        default="tem_v1",
        description="The M0 bridge model kind to evaluate. Supported values include 'tem_v1' and registered bridge-compatible variants.",
    )
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the model.",
    )
    model_checkpoint: Path | None = Field(
        default=None,
        description="Optional checkpoint override used for evaluation.",
    )

    seed: int | None = Field(
        default=None,
        description="Optional seed override. When omitted, use canonical benchmark seeds.",
    )
    device: str = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
        description="The device on which to run the benchmark evaluation.",
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

    # Resolve benchmark configuration
    benchmark_config = M0BenchmarkConfig.from_config(settings.benchmark_config)
    # Sensible that CLI can override seed and output_dir
    run_overrides = {}
    if settings.seed is not None:
        run_overrides["seed"] = settings.seed
    if settings.output_dir is not None:
        run_overrides["output_dir"] = settings.output_dir
    if run_overrides:
        benchmark_config = benchmark_config.model_copy(update=run_overrides)

    # Resolve the model-agent ...
    agent = build_m0_agent(
        model_kind=settings.model_kind,
        model_config_path=settings.model_config_path,
        checkpoint_path=settings.model_checkpoint,
        device=settings.device,
    )

    # Run the benchmark and write artifacts
    manifest = M0Benchmark(benchmark_config).evaluate(agent)
    print(
        f"Wrote {len(manifest.artifact_paths)} M0 result artifacts "
        f"to {manifest.output_root}"
    )  # fmt: skip

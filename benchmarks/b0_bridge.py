"""Entrypoint for the B0 MazeHard bridge benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.benchmark import b0
from ehc_sn.runtimes.benchmark import resolve_b0_runner

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

    config_path: Path = Field(
        default=Path(CONFIGURATION_PATH),
        alias="config-path",
        description="Path to the B0 benchmark wrapper TOML file.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional seed override. When omitted, use canonical benchmark seeds.",
    )
    compute_budget: int | None = Field(
        default=None,
        alias="compute-budget",
        description="Optional compute-budget override. When omitted, use the wrapper default behavior.",
    )
    output_root: Path | None = Field(
        default=None,
        alias="output-root",
        description="Optional output directory override for benchmark artifacts.",
    )
    checkpoint_path: Path | None = Field(
        default=None,
        alias="checkpoint-path",
        description="Optional checkpoint override used for evaluation.",
    )


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """Run the B0 benchmark wrapper with optional dry-run scheduling output."""
    # Load the wrapper configuration and build jobs
    settings = RunArguments()
    wrapper, _ = b0.load_b0_wrapper_config(
        config_path=settings.config_path,
    )

    # Load the dataset partitions and build jobs
    partitions = b0.load_b0_partitions(
        dataset_root=wrapper.benchmark.dataset_path,
        hard_subset_manifest_path=wrapper.benchmark.hard_subset_manifest,
    )
    jobs = b0.build_b0_jobs(
        partitions=partitions,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
        benchmark_config=wrapper.benchmark,
    )

    # Execute the jobs and write results
    manifest = b0.execute_b0_jobs(
        config_path=settings.config_path,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
        output_root=settings.output_root,
        checkpoint_path=settings.checkpoint_path,
        runner=resolve_b0_runner(wrapper.benchmark.model_kind),
    )
    print(
        f"Wrote {len(manifest['artifact_paths'])} B0 result artifacts "
        f"to {manifest['output_root']}"
    )  # fmt: skip

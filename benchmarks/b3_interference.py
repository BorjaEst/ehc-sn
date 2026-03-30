"""Thin CLI wrapper for the B3 interference-and-control benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.benchmark import b3
from ehc_sn.benchmark.b3 import ALL_SCHEDULE_TYPES_SELECTOR

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("BENCHMARK_B3_CONFIGURATION_PATH", "config/benchmarks-b3.hrm-v1.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, populate_by_name=True, cli_implicit_flags=True):
    """Arguments for the B3 benchmark wrapper."""

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    config_path: Path = Field(
        default=Path(CONFIGURATION_PATH),
        alias="config-path",
        description="",
    )
    seed: int | None = Field(
        default=None,
        description="",
    )
    compute_budget: int | None = Field(
        default=None,
        alias="compute-budget",
        description="",
    )
    manifest_path: Path | None = Field(
        default=None,
        alias="manifest-path",
        description="",
    )
    schedule_type: str = Field(
        default=ALL_SCHEDULE_TYPES_SELECTOR,
        alias="schedule-type",
        description="",
    )
    output_root: Path | None = Field(
        default=None,
        alias="output-root",
        description="",
    )


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """Run the B3 benchmark wrapper with optional dry-run scheduling output."""
    # Load the wrapper configuration and build jobs
    settings = RunArguments()
    wrapper, resolved_config_path, _ = b3.load_b3_wrapper_config(settings.config_path)

    # Load the dataset partitions and build jobs
    jobs = b3.build_b3_jobs(
        wrapper=wrapper,
        schedule_type_selector=settings.schedule_type,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
    )
    # Execute the jobs and write results
    manifest = b3.execute_b3_jobs(
        config_path=settings.config_path,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
        manifest_path=settings.manifest_path,
        schedule_type_selector=settings.schedule_type,
        output_root=settings.output_root,
        build_adapter_fn=b3.build_adapter,
    )
    print(
        f"Wrote {len(manifest['artifact_paths'])} B3 artifacts "
        f"to {manifest['output_root']}"
    )  # fmt: skip

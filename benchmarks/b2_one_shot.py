"""Thin CLI wrapper for the B2 one-shot goal-relocation benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.benchmark import b2
from ehc_sn.benchmark.b2 import ALL_LAYOUT_MODES_SELECTOR

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("BENCHMARK_B2_CONFIGURATION_PATH", "config/benchmarks-b2.hrm-v1.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, populate_by_name=True, cli_implicit_flags=True):
    """Arguments for the B2 benchmark wrapper."""

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
    layout_mode: str = Field(
        default=ALL_LAYOUT_MODES_SELECTOR,
        alias="layout-mode",
        description="",
    )
    output_root: Path | None = Field(
        default=None,
        alias="output-root",
        description="",
    )
    failure_test_mode: bool = Field(
        default=False,
        alias="failure-test-mode",
        description="",
    )


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """Run the B2 benchmark wrapper with optional dry-run scheduling output."""
    # Load the wrapper configuration and build jobs
    settings = RunArguments()
    wrapper, _, _ = b2.load_b2_wrapper_config(
        settings.config_path,
    )

    # Load the dataset partitions and build jobs
    jobs = b2.build_b2_jobs(
        wrapper=wrapper,
        layout_mode_selector=settings.layout_mode,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
    )

    # Execute the jobs and write results
    manifest = b2.execute_b2_jobs(
        config_path=settings.config_path,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
        manifest_path=settings.manifest_path,
        layout_mode_selector=settings.layout_mode,
        output_root=settings.output_root,
        failure_test_mode=settings.failure_test_mode,
        build_adapter_fn=b2.build_adapter,
    )
    print(
        f"Wrote {len(manifest['artifact_paths'])} B2 artifacts "
        f"to {manifest['output_root']}"
    )  # fmt: skip

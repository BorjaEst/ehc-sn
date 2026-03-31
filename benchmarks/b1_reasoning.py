"""Thin CLI wrapper for the B1 dungeon reasoning benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.benchmark import b1
from ehc_sn.benchmark.b1 import CORPUS_ID
from ehc_sn.policies.benchmark_scripted import build_scripted_policy

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("BENCHMARK_B1_CONFIGURATION_PATH", "config/benchmarks-b1.hrm-v1.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, populate_by_name=True, cli_implicit_flags=True):
    """Arguments for the B1 benchmark wrapper."""

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    config_path: Path = Field(
        default=Path(CONFIGURATION_PATH),
        alias="config-path",
        description="Path to the B1 benchmark wrapper TOML file.",
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
    manifest_path: Path | None = Field(
        default=None,
        alias="manifest-path",
        description="Optional manifest override for resuming or post-hoc analysis.",
    )
    corpus_selector: str = Field(
        default=CORPUS_ID,
        alias="corpus-selector",
        description=(
            "Corpus selector string. Determines which corpus partition to use for job scheduling. "
            "When set to 'all', use all corpora and append corpus_id to job names."
        ),
    )
    output_root: Path | None = Field(
        default=None,
        alias="output-root",
        description="Optional output directory override for benchmark artifacts.",
    )


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """Run the B1 benchmark wrapper with optional dry-run scheduling output."""
    # Load the wrapper configuration and build jobs
    settings = RunArguments()
    wrapper, _, _ = b1.load_b1_wrapper_config(
        config_path=settings.config_path,
    )

    # Load the dataset partitions and build jobs
    jobs = b1.build_b1_jobs(
        wrapper=wrapper,
        corpus_selector=settings.corpus_selector,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
    )

    # Execute the jobs and write results
    manifest = b1.execute_b1_jobs(
        config_path=settings.config_path,
        seed=settings.seed,
        compute_budget=settings.compute_budget,
        manifest_path=settings.manifest_path,
        corpus_selector=settings.corpus_selector,
        output_root=settings.output_root,
        policy_factory=build_scripted_policy,
    )
    print(
        f"Wrote {len(manifest['artifact_paths'])} B1 artifacts "
        f"to {manifest['output_root']}"
    )  # fmt: skip

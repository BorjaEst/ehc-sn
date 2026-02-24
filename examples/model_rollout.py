#!/usr/bin/env python3
"""Generate TEM model rollout figures with spatial rate maps.

This script demonstrates the complete figure generation pipeline for TEM
model rollouts, including spatial rate maps that combine model activity
with position information.

Examples:
    # Default settings (interactive plots).
    python examples/model_rollout_figures.py

    # Run headless (do not show or save figures).
    python examples/model_rollout_figures.py --show_plots false --save_plots false

    # Use a trained checkpoint
    python examples/model_rollout_figures.py --checkpoint models/nice_model.1
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn import data, figures
from ehc_sn.data.datamodule import DataConfig
from ehc_sn.diagnostics.trace_collectors import collect_rollout_trace_tree
from ehc_sn.figures.registry import FigureContext
from ehc_sn.model import Model as TEMModel
from ehc_sn.model import TEMConfig

logger = logging.getLogger(__file__)


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """CLI configuration for model rollout figure generation."""

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # Data configuration (leaves)
    data_seed: int = Field(
        default=42,
        description="Random seed for data sampling.",
    )
    space: SpaceContractSettings = Field(
        default_factory=SpaceContractSettings,
        description="Space contract: observation and action space dimensions.",
    )
    env: EnvironmentSettings = Field(
        default_factory=EnvironmentSettings,
        description="Environment generation settings.",
    )
    policy: EnvSamplingSettings = Field(
        default_factory=EnvSamplingSettings,
        description="Data generation policies (exploration + shiny).",
    )
    walk: CurriculumSettings = Field(
        default_factory=CurriculumSettings,
        description="Walk length curriculum settings.",
    )
    max_rollout_steps: int = Field(
        default=400,
        ge=10,
        description="Maximum rollout steps to collect.",
    )

    # Model configuration
    autoencoder: AutoencoderSettings = Field(
        default_factory=AutoencoderSettings,
        description="Autoencoder (observation embedding) settings.",
    )
    lec: LECSettings = Field(
        default_factory=LECSettings,
        description="LEC (sensory features) settings.",
    )
    lec_proj: LECProjectionSettings = Field(
        default_factory=LECProjectionSettings,
        description="LEC projection settings.",
    )
    mec: MECSettings = Field(
        default_factory=MECSettings,
        description="MEC (abstract location) settings.",
    )
    mec_proj: MECProjectionSettings = Field(
        default_factory=MECProjectionSettings,
        description="MEC projection settings.",
    )
    hpc: HPCSettings = Field(
        default_factory=HPCSettings,
        description="HPC (grounded location memory) settings.",
    )

    # Checkpoint
    checkpoint: Path | None = Field(
        default=Path("models/tem-g3ovc2-step=20000.ckpt"),
        description="Path to model checkpoint (optional). If None, uses random init.",
    )

    # Output
    output_dir: Path = Field(
        default=Path("outputs/model_rollout"),
        description="Directory for saving plots",
    )
    show_plots: bool = Field(
        default=False,
        description="Display plots interactively",
    )
    save_plots: bool = Field(
        default=True,
        description="Save plots to output directory",
    )

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output_dir if it does not exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def iterator(self) -> RolloutStreamSettings:
        """Create the aggregate RolloutStreamSettings consumed by the DataModule."""
        eval_settings = EvalSettings(val_steps=1, val_seed=self.data_seed)
        rollout_settings = RolloutSettings(batch_size=1, n_rollout=self.max_rollout_steps)
        return RolloutStreamSettings(eval=eval_settings, rollout=rollout_settings)

    @property
    def data(self) -> DataConfig:
        """Create the aggregate DataConfig consumed by the DataModule."""
        return DataConfig.model_validate(self, from_attributes=True)

    @property
    def model(self) -> TEMConfig:
        """Create the aggregate TEMConfig for model construction."""
        return TEMConfig.model_validate(self, from_attributes=True)


# ==============================================================================
# Main Experiment
# ==============================================================================
def main() -> None:
    """Run the model rollout figure generation example.

    The flow is:
    1) Parse configuration from CLI.
    2) Build a DataModule and TEM model.
    3) Load checkpoint if provided, otherwise use random init.
    4) Collect a rollout trace combining model outputs and spatial info.
    5) Generate figures including spatial rate maps.
    6) Optionally save/show the figures.
    """

    # Parse all settings from CLI and environment.
    args = ExampleArguments()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=args.log_level.upper())

    print("=" * 80)
    print("TEM Model Rollout Figure Generation")
    print("=" * 80)
    print(f" - Log level: {args.log_level.upper()}")
    print(f" - Checkpoint: {args.checkpoint or 'None (random init)'}")
    print(f" - Output directory: {args.output_dir}")
    print(f" - Save plots: {args.save_plots}")
    print(f" - Show plots: {args.show_plots}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Instantiate DataModule and build runtime objects.
    # ------------------------------------------------------------------
    datamodule = data.DataModule(args.data)
    datamodule.setup(None)  # Enable all configured splits.

    print("Step 1: DataModule setup complete.")
    print()

    # ------------------------------------------------------------------
    # Step 2: Initialize TEM model.
    # ------------------------------------------------------------------
    model = TEMModel(args.model)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        tem_sd = {k.removeprefix("tem."): v for k, v in state_dict.items() if k.startswith("tem.")}
        model.load_state_dict(tem_sd, strict=False)
        print("Checkpoint loaded.")
    else:
        print("Using random initialization (no checkpoint provided).")

    model.eval()  # Set to evaluation mode
    print("Step 2: TEM model initialized.")
    print()

    # ------------------------------------------------------------------
    # Step 3: Collect rollout trace with spatial alignment.
    # ------------------------------------------------------------------
    dataset = datamodule.get_dataset("test")
    trace = collect_rollout_trace_tree(
        batch=datamodule.sample_batch(split="test"),
        environments=dataset.environments,
        model=model,
        stop=args.max_rollout_steps,
        meta={"split": "test"},
    )

    print("Step 3: Collecting rollout trace complete.")
    print(f" - Batch size: {int(trace.batch_size or 0)}")
    print(f" - Time steps: {trace.length}")
    print(f" - Environments: {len(trace.get_environments())}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Generate diagnostic visualizations.
    # ------------------------------------------------------------------

    FREQUENCY_INDEX = 0  # Select frequency index for multi-scale signals
    extras = {"lec": model.lec}
    ctx = FigureContext(env_idx=0, freq_idx=FREQUENCY_INDEX, split_name="validate", extras=extras)
    figs: list[tuple[str, plt.Figure]] = [
        (f"01.1_lec_overview.png", figures.lec_overview.plot(trace, ctx)),
        (f"01.2_mec_overview.png", figures.mec_overview.plot(trace, ctx)),
        (f"01.3_hpc_overview.png", figures.hpc_overview.plot(trace, ctx)),
        (f"02.1_feature_cells.png", figures.feature_cells.plot(trace, ctx)),
        (f"02.2_grid_cells.png", figures.grid_cells.plot(trace, ctx)),
        (f"02.3_place_cells.png", figures.place_cells.plot(trace, ctx)),
    ]

    print(f"Step 4: Generated {len(figs)} figure(s).")
    print()

    # ------------------------------------------------------------------
    # Step 5: Save and/or show plots.
    # ------------------------------------------------------------------
    if args.save_plots:
        for filename, fig in figs:
            fig.savefig(args.output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {args.output_dir}")
    plt.show() if args.show_plots else plt.close("all")

    print("Example completed.")
    print("=" * 80)


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()

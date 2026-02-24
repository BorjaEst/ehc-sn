#!/usr/bin/env python3
"""Generate TEM data via CLI and visualize results.

This script is a runnable, end-to-end example of the `ehc_sn.data` stack:

- Load and validate an environment.
- Configure policies (random, distance-based, Q-learning) and optional shiny
    object sampling.
- Sample agent walks under different curricula.
- Integrate the resulting streams with a Lightning-style `DataModule`.
- Produce quick diagnostic figures for sanity-checking the generated data.

Examples:
        # Default settings (interactive plots).
        python examples/data_datamodule.py

        # Run headless (do not show or save figures).
    python examples/data_datamodule.py --show_plots false --save_plots false

        # Override nested settings.
        python examples/data_datamodule.py --env.width 10 --env.height 10 \
                --policy.type distance --policy.beta 2.0
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn import data, figures
from ehc_sn.data.datamodule import DataConfig
from ehc_sn.diagnostics.trace_collectors import collect_world_trace_tree
from ehc_sn.figures.registry import FigureContext

logger = logging.getLogger(__file__)


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """CLI configuration for this example.

    This settings object is intentionally composed of the same "leaf" settings
    used by the library so the example stays close to real training runs.
    """

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # Data configuration (leaves)
    space: SpaceContractSettings = Field(
        default_factory=SpaceContractSettings,
        description="Space contract: observation and action space dimensions.",
    )
    env: EnvironmentSettings = Field(
        default_factory=EnvironmentSettings,
        description="Environment generation settings.",
    )
    iterator: RolloutStreamSettings = Field(
        default_factory=RolloutStreamSettings,
        description="Iterator protocol settings (rollout chunking + eval protocol).",
    )
    policy: EnvSamplingSettings = Field(
        default_factory=EnvSamplingSettings,
        description="Data generation policies (exploration + shiny).",
    )
    walk: CurriculumSettings = Field(
        default_factory=CurriculumSettings,
        description="Walk length curriculum settings.",
    )

    # Output
    output_dir: Path = Field(
        default=Path("outputs/data_generation"),
        description="Directory for saving plots",
    )
    show_plots: bool = Field(
        default=True,
        description="Display plots interactively",
    )
    save_plots: bool = Field(
        default=True,
        description="Save plots to output directory",
    )

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create `output_dir` if it does not exist.

        Args:
            v: Output directory path.

        Returns:
            The same path, after ensuring it exists.
        """
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def data(self) -> DataConfig:
        """Create the aggregate `DataConfig` consumed by the `DataModule`.

        The example keeps the same structure as training code: the `walk`
        curriculum is shared with the trainer to ensure a single source of truth.

        Returns:
            A validated `DataConfig` instance.
        """
        return DataConfig.model_validate(self, from_attributes=True)


# ==============================================================================
# Main Experiment
# ==============================================================================
def main() -> None:
    """Run the data generation example.

    The flow is:
    1) Parse configuration from CLI.
    2) Build a `DataModule` and sample a single batch.
    3) Generate a few diagnostic figures.
    4) Optionally save/show the figures.
    """

    # Parse all settings from CLI and environment.
    # Pydantic Settings parses `sys.argv` when `cli_parse_args=True`.
    args = ExampleArguments()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=args.log_level.upper())

    print("=" * 80); print("TEM DataModule Example"); print("=" * 80)  # fmt: skip
    print(f" - Log level: {args.log_level.upper()}")
    print(f" - Output directory: { args.output_dir}")
    print(f" - Save plots: { args.save_plots}")
    print(f" - Show plots: { args.show_plots}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Instantiate `DataModule` and build runtime objects.
    # ------------------------------------------------------------------
    datamodule = data.DataModule(args.data)
    datamodule.setup(None)  # `None` enables all configured splits.

    print("Step 1: Setup complete (environment + generators).")
    print()

    # ------------------------------------------------------------------
    # Step 2: Fetch a single batch and print key tensor shapes.
    # ------------------------------------------------------------------
    walk, visited = datamodule.sample_batch("validate")
    locations, observation_0, action_0 = walk[0]

    print("Step 2: Sampled one validation batch (time-major).")
    print(f" - Number of locations: {len(locations)}")
    print(f" - Observation[0] shape: {observation_0.shape}")
    print(f" - Action[0] len: {len(action_0)}")
    print(f" - Visited len: {len(visited)}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Generate diagnostic visualizations.
    # ------------------------------------------------------------------
    dataset = datamodule.get_dataset("validate")
    environments = dataset.environments
    trace = collect_world_trace_tree(walk, environments, visited, meta={"split": "validate"})
    ctx = FigureContext(env_idx=0, split_name="validate")

    figs: list[tuple[str, plt.Figure]] = [
        ("01_environment_layout.png", figures.environment.layout.plot(trace, ctx)),
        ("02_walk_trajectories.png", figures.walk.trajectories.plot(trace, ctx)),
        ("03_walk_statistics.png", figures.walk.statistics.plot(trace, ctx)),
        ("04_split_statistics.png", figures.split.statistics.plot(trace, ctx)),
    ]

    print(f"Step 3: Generated {len(figs)} figure(s).")
    print()

    # ------------------------------------------------------------------
    # Step 4: Save and/or show plots.
    # ------------------------------------------------------------------
    if args.save_plots:
        for filename, fig in figs:
            fig.savefig(args.output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {args.output_dir}")
    plt.show() if args.show_plots else plt.close("all")

    print("Example completed."); print("=" * 80)  # fmt: skip


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()

"""Visualize a processed maze sample.

This example script:

1) Parses CLI arguments (dataset path, sample index, split, output path).
2) Loads a single sample from a processed dataset (no transforms applied).
3) Renders a multi-panel figure showing all available channels grouped by
   semantic role: Navigation, Structure, Perception.
4) Displays the figure interactively or saves to file.

Typical usage:

```bash
# Dungeon dataset (Structure + Perception panels)
python examples/data_visualization.py --dataset_path data/processed/dungeon --split train --idx 0

# Maze-hard dataset (Navigation panel)
python examples/data_visualization.py --dataset_path data/processed/mazehard --split train --idx 42

# Save to file instead of showing
python examples/data_visualization.py --dataset_path data/processed/dungeon --split train --output sample.pdf
```
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.data.datasets import MazeDataset
from ehc_sn.data.index import MazeIndexEntry, filter_index, read_index
from ehc_sn.figures.modules.dataset import plot

NAME = __file__.split("/")[-1].replace(".py", "")
logger = logging.getLogger(NAME)


def resolve_dataset_split(  # ---------------------------------------------------------------------
    dataset_path: Path, split: str | None,
) -> tuple[list[MazeIndexEntry], Path, str]:  # fmt: skip
    """Resolve root-level processed data metadata to one concrete split directory."""
    if not (dataset_path / "index.jsonl").is_file():
        if (dataset_path / "dataset.json").is_file() and (dataset_path.parent / "index.jsonl").is_file():
            raise SystemExit(
                f"'{dataset_path}' is a split directory, not a dataset root. "
                f"Pass '{dataset_path.parent}' and set --split={dataset_path.name}."
            )
        raise SystemExit(f"Missing dataset index: {dataset_path / 'index.jsonl'}")

    entries = read_index(dataset_path / "index.jsonl")
    if split is not None:
        entries = filter_index(entries, split=split)
    if not entries:
        raise SystemExit(f"No entries found in {dataset_path / 'index.jsonl'}")

    splits = {entry.split for entry in entries}
    if len(splits) != 1:
        available = ", ".join(sorted(splits))
        raise SystemExit(f"Visualization requires exactly one split. Pass --split with one of: {available}.")

    resolved_split = next(iter(splits))
    return entries, dataset_path / resolved_split, resolved_split


# =================================================================================================
# Configuration
# =================================================================================================
class ExampleArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """CLI arguments for data visualization.

    Notes:
        - All arguments are provided via CLI flags.
        - ``--dataset_path`` is required; the rest have sensible defaults.
    """

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        """Define settings precedence.

        Order:
            1) CLI args
            2) Explicit init kwargs
            3) Environment variables / dotenv
            4) Secret files
        """
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ---------------------------------------------------------------------------------------------
    # Data settings
    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset root (contains index.jsonl and per-split channel arrays).",
    )
    idx: int = Field(
        default=0,
        description="Sample index to visualize.",
    )
    split: str | None = Field(
        default=None,
        description="Filter to a specific split (train/val/test).",
    )

    # ---------------------------------------------------------------------------------------------
    # Output settings
    output: Path | None = Field(
        default=None,
        description="Save figure to file instead of showing interactively.",
    )


# =================================================================================================
# Main
# =================================================================================================
def main() -> None:
    """Load a processed sample and render a multi-panel diagnostic figure."""

    # Step 0: Parse settings from CLI.
    args = ExampleArguments()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=args.log_level.upper())

    print("=" * 80)
    print("Processed Data Visualization")
    print("=" * 80)
    print(f" - Dataset: {args.dataset_path}")
    print(f" - Sample index: {args.idx}")
    print(f" - Split filter: {args.split or 'all'}")
    print(f" - Output: {args.output or 'interactive'}")
    print()

    # ---------------------------------------------------------------------------------------------
    # Step 1: Load index and optionally filter by split.
    # ---------------------------------------------------------------------------------------------
    entries, split_dir, resolved_split = resolve_dataset_split(args.dataset_path, args.split)

    print(f"Step 1/3: Loaded {len(entries)} index entries.")
    print(f" - Resolved split: {resolved_split}")
    print()

    # ---------------------------------------------------------------------------------------------
    # Step 2: Load raw sample (no transforms → original channels preserved).
    # ---------------------------------------------------------------------------------------------
    ds = MazeDataset(entries, split_dir, transform=None)
    if args.idx >= len(ds):
        raise SystemExit(f"Index {args.idx} out of range (dataset has {len(ds)} samples)")

    sample = ds[args.idx]
    entry = entries[args.idx]
    print(f"Step 2/3: Sample {args.idx} loaded.")
    print(f" - source={entry.source}, shape={entry.shape}, channels={entry.channels}")
    print()

    # ---------------------------------------------------------------------------------------------
    # Step 3: Render and save/show figure.
    # ---------------------------------------------------------------------------------------------
    fig = plot(sample)

    if args.output:
        if args.output.suffix == ".png":
            fig.savefig(args.output, dpi=200)
        else:
            fig.savefig(args.output)
        print(f"Step 3/3: Saved to {args.output}")
    else:
        matplotlib.use("TkAgg")
        plt.show()
        print("Step 3/3: Figure displayed.")

    plt.close(fig)

    print()
    print("Done.")
    print("=" * 80)


# =================================================================================================
# Main Entry Point
# =================================================================================================
if __name__ == "__main__":
    main()

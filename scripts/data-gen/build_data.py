#!/usr/bin/env python3
import json
from pathlib import Path

import typer

from mazes import DataProcessConfig, convert_subset

app = typer.Typer()


@app.command()
def generate_mazes(  # ----------------------------------------------------------------------------
):  # fmt: skip
    """Use Maze-nd to generate raw maze data."""


@app.command()
def process_mazes(  # -----------------------------------------------------------------------------
):  # fmt: skip
    """Convert the raw maze into ready to use train and test datasets."""


@app.command()
def generate_dungeon(  # --------------------------------------------------------------------------
):  # fmt: skip
    """Use dungeongen to generate raw dungeons maps."""


@app.command()
def add_markers(  # -------------------------------------------------------------------------------
):  # fmt: skip
    """Add markers to the raw maze and dungeon data to indicate the start and end points."""


@app.command()
def process_dungeons(  # --------------------------------------------------------------------------
):  # fmt: skip
    """Convert the raw dungeons into ready to use train and test datasets."""


@app.command()
def generate(  # ----------------------------------------------------------------------------------
):  # fmt: skip
    """Run all the steps to generate the maze or dungeon data."""


if __name__ == "__main__":
    app()
if __name__ == "__main__":
    app()

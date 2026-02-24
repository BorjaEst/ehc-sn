#!/usr/bin/env python3
from pathlib import Path
import json
import typer
from ehc_sn.data.maze_dataset import DataProcessConfig, convert_subset

app = typer.Typer()

@app.command()
def generate(  # ----------------------------------------------------------------------------------
    config_json: Path = None, input_dir: Path = None, output_dir: Path = None,
    subset: str = "train",
):  # fmt: skip
    cfg = DataProcessConfig(**json.loads(config_json.read_text())) if config_json else DataProcessConfig()
    convert_subset(config=cfg,
                   input_dir=str(input_dir) if input_dir else None,
                   output_dir=str(output_dir) if output_dir else None,
                   subset=subset)
    typer.echo("Dataset generation finished.")

if __name__ == "__main__":
    app()
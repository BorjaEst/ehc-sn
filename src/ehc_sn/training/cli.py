from __future__ import annotations

import json as _json
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

train_app = typer.Typer(
    name="train",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

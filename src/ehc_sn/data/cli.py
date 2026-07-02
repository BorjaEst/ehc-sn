from __future__ import annotations

import json as _json
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

data_app = typer.Typer(
    name="data",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

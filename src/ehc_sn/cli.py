"""Top-level ``ehp`` CLI entry point.

Registered in ``pyproject.toml``::

    [project.scripts]
    ehp = "ehp_sn.cli:app"

Usage::

    ehp-sn data build openfield --preset tem-square
    ehp-sn tasks build arena --layout-root data/interim/tem-square/v1
    ehp-sn train run arena-tem-v1 --config config/training/arena-tem-v1.toml
    ehp-sn evaluation run arena-tem-v1 --model artifacts/models/arena-tem-v1/2024-06-05_15-30-00
    ehp-sn report prepare config/reporting/arena-tem-diagnostic.toml -o reports/arena-tem/data
"""

from __future__ import annotations

import typer

from ehc_sn.data.cli import data_app
from ehc_sn.evaluation.cli import evaluation_app
from ehc_sn.reporting.cli import report_app
from ehc_sn.tasks.cli import tasks_app
from ehc_sn.training.cli import train_app

app = typer.Typer(
    name="ehp",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


app.add_typer(
    data_app,
    name="data",
    help="Build and manage data generation pipelines.",
)

app.add_typer(
    tasks_app,
    name="tasks",
    help="Build and manage task generation pipelines.",
)

app.add_typer(
    train_app,
    name="train",
    help="Run and inspect model training.",
)

app.add_typer(
    evaluation_app,
    name="evaluation",
    help="Run and inspect model evaluations.",
)

app.add_typer(
    report_app,
    name="report",
    help="Prepare, validate, and inspect report-data packages.",
)


if __name__ == "__main__":
    app()

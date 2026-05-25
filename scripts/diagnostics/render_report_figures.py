"""Render post-hoc report figures from one persisted eval regime run directory."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from ehc_sn.eval.offline_report import (
    OfflineReportRenderSettings,
    render_report_figures_from_run,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline report figure rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/figures/report-regime.toml"),
        help="Path to TOML settings for one offline report rendering run.",
    )
    return parser.parse_args()


def main() -> None:
    """Load TOML settings and render report figures from saved artifacts."""
    args = parse_args()
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    settings = OfflineReportRenderSettings.model_validate(raw)
    summary = render_report_figures_from_run(settings)
    result = {
        "config": str(args.config),
        "run_dir": str(settings.run_dir),
        "output_dir": str(
            settings.output_dir or (settings.run_dir / "report_figures")
        ),
        "n_entries": len(settings.entries),
        **summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

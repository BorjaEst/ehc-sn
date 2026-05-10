"""Export TensorBoard scalar logs to JSONL and an optional summary JSON.

This script accepts a single event file, a run directory, an experiment
directory, or a broader log root. Reusable export logic lives in
``ehc_sn.logging.tensorboard``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ehc_sn.logging.tensorboard import export_scalars


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the TensorBoard scalar exporter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("logs"),
        help="Event file, run directory, experiment directory, or log root to scan.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/diagnostics/tensorboard-scalars.jsonl"),
        help="Destination JSONL file for raw scalar events.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/diagnostics/tensorboard-scalars-summary.json"),
        help="Destination JSON file for compact per-tag summaries.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip writing the compact summary JSON.",
    )
    parser.add_argument(
        "--summary-points",
        type=int,
        default=16,
        help="Maximum sampled points retained per tag in the summary JSON.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Repeatable exact-match scalar tag filter. Omit to export all scalar tags.",
    )
    parser.add_argument(
        "--no-hparams",
        action="store_true",
        help="Exclude hparams.yaml metadata from the summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the TensorBoard scalar export CLI."""

    args = parse_args()
    summary_path = None if args.skip_summary else args.summary_path
    report = export_scalars(
        args.input_path,
        args.output_path,
        summary_path=summary_path,
        tags=args.tag,
        sample_points=args.summary_points,
        include_hparams=not args.no_hparams,
    )
    result = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "summary_path": None if summary_path is None else str(summary_path),
        "n_runs": report.n_runs,
        "n_tags": report.n_tags,
        "n_points": report.n_points,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

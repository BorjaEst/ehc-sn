"""Staged CLI for building the SeqMaze task corpus.

SeqMaze consumes a dagflow shared substrate and creates a task corpus
with path-prediction and edge-lookup protocol channels.

This CLI does not generate graph topology.  That stage belongs in
``build-dagflow.py``.

Stages
------
build      Build the SeqMaze task corpus over a shared substrate.
validate   Validate a SeqMaze task-corpus version root.
inspect    Print a human-readable summary of a version root manifest.

Default paths
-------------
Parent layout dataset:  <user-specified --layout-root>
Task corpus:            data/processed/seqmaze/<corpus>/v<version>

Examples
--------
Build the SeqMaze task corpus against a dagflow layout dataset::

    python build-seqmaze.py build \\
        --layout-root data/interim/dagflow/sparse/v1

With custom sizes::

    python build-seqmaze.py build \\
        --layout-root data/interim/dagflow/sparse/v1 \\
        --n-max 64 --t-max 48 --n-train 8000 --n-val 1000 --n-test 1000 --seed 7

Validate an existing task corpus::

    python build-seqmaze.py validate data/processed/seqmaze/default/v1

Inspect an existing corpus::

    python build-seqmaze.py inspect data/processed/seqmaze/default/v1

Prerequisites
-------------
A dagflow shared substrate must exist before running.
Build it first::

    python scripts/data-gen/build-dagflow.py build --preset branching --version 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.manifest import read_manifest
from ehc_sn.figures import FigureContext, render
from ehc_sn.tasks.seqmaze import (
    TASK_FAMILY,
    build_seqmaze_task_corpus,
    validate_seqmaze_root,
)
from ehc_sn.tasks.seqmaze.corpus import load_sample, load_split_arrays
from ehc_sn.tasks.seqmaze.diagnostics import compute_corpus_statistics
from ehc_sn.tasks.seqmaze.inspection import prepare_sample_inspection
from ehc_sn.tasks.seqmaze.validation import (
    SeqMazeValidationIssue,
    validate_all_samples,
)

# ---------------------------------------------------------------------------
_DEFAULT_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_MAX = 45
_DEFAULT_T_MAX = 32
_DEFAULT_MAX_OUT_DEGREE = 4
_DEFAULT_N_TRAIN = 4000
_DEFAULT_N_VAL = 500
_DEFAULT_N_TEST = 500
_DEFAULT_SEED = 42
_DEFAULT_OUTPUT_DIR = Path("outputs/seqmaze-validation")

app = typer.Typer(add_completion=False, help="SeqMaze task corpus builder.")


# =============================================================================
@app.command("build")
def build(
    layout_root: Annotated[
        Path,
        typer.Option(
            "--layout-root",
            help="Path to dagflow layout dataset root "
            "(e.g. data/interim/dagflow/sparse/v1).",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option("--corpus", help="Corpus label (default: 'default')."),
    ] = _DEFAULT_CORPUS,
    n_max: Annotated[
        int,
        typer.Option(
            "--n-max",
            help="Maximum candidate nodes N (default: 32).",
        ),
    ] = _DEFAULT_N_MAX,
    t_max: Annotated[
        int,
        typer.Option(
            "--t-max",
            help="Maximum path length T (default: 32).",
        ),
    ] = _DEFAULT_T_MAX,
    max_out_degree: Annotated[
        int,
        typer.Option(
            "--max-out-degree", help="Maximum out-degree K (default: 4)."
        ),
    ] = _DEFAULT_MAX_OUT_DEGREE,
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train", help="Number of training samples (default: 4000)."
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val", help="Number of validation samples (default: 500)."
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option("--n-test", help="Number of test samples (default: 500)."),
    ] = _DEFAULT_N_TEST,
    version: Annotated[
        int,
        typer.Option(
            "--version", help="Task corpus version integer (default: 1)."
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic base seed (default: 42)."),
    ] = _DEFAULT_SEED,
) -> None:
    """Build the SeqMaze task corpus over a dagflow layout dataset."""
    root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    build_seqmaze_task_corpus(
        version_root=root.resolve(),
        layout_root=layout_root.resolve(),
        corpus=corpus,
        n_max=n_max,
        t_max=t_max,
        max_out_degree=max_out_degree,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )
    typer.echo(f"SeqMaze corpus built at {root.resolve()}")
    typer.echo(f"  Profile: N={n_max}, T={t_max}, K={max_out_degree}")
    typer.echo(f"  S={n_max + t_max}, V={n_max + 2}")


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to validate "
            "(e.g. data/processed/seqmaze/default/v1)."
        ),
    ],
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="Validate a single split only (default: all).",
        ),
    ] = None,
    max_samples: Annotated[
        int,
        typer.Option(
            "--max-samples",
            help="Max samples to check per split (default: all).",
        ),
    ] = -1,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Write report bundle to this directory.",
        ),
    ] = _DEFAULT_OUTPUT_DIR,
) -> None:
    """Validate a SeqMaze task-corpus root with structured reports."""
    root = root.resolve()
    manifest = validate_seqmaze_root(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    targets = [split] if split else all_splits

    issues, sample_counts = validate_all_samples(
        root, splits=targets, max_samples=max_samples
    )

    stats = compute_corpus_statistics(root, manifest, targets, max_samples)
    stats["corpus"] = manifest.get("corpus", "?")
    stats["version"] = manifest.get("version", "?")

    # Write report bundle
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_validation_bundle(issues, stats, sample_counts, output_dir)

    errors = [i for i in issues if i.severity == "ERROR"]
    n_err = len(errors)
    typer.echo(f"SeqMaze corpus validation: {root}")
    typer.echo(f"  Validation report: {output_dir / 'validation_report.txt'}")
    typer.echo(f"  Diagnostics:       {output_dir / 'diagnostics.txt'}")
    typer.echo(f"  Summary:           {output_dir / 'summary.txt'}")
    typer.echo(f"Validated {sum(sample_counts.values())} samples")
    typer.echo(
        f"  Errors: {n_err}  Warnings: "
        f"{len([i for i in issues if i.severity == 'WARNING'])}"
    )
    if n_err > 0:
        typer.echo("\n  First 5 errors:")
        for e in errors[:5]:
            typer.echo(
                f"    [{e.split}/{e.sample_index}] {e.code}: {e.message}"
            )

    raise typer.Exit(code=1 if n_err > 0 else 0)


# =============================================================================
@app.command("inspect")
def inspect(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to inspect "
            "(e.g. data/processed/seqmaze/default/v1)."
        ),
    ],
    summary: Annotated[
        bool,
        typer.Option("--summary", help="Display corpus summary."),
    ] = False,
    split: Annotated[
        str,
        typer.Option("--split", help="Split name for sample inspection."),
    ] = "train",
    sample_index: Annotated[
        int | None,
        typer.Option("--sample-index", help="Sample index to inspect."),
    ] = None,
    sample_figure: Annotated[
        bool,
        typer.Option(
            "--sample-figure",
            help="Render seqmaze_task_overview for the selected sample.",
            show_default=False,
        ),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("--show", help="Display figures interactively."),
    ] = False,
    gallery: Annotated[
        int,
        typer.Option(
            "--gallery",
            help="Number of overview figures to produce.",
            show_default=False,
        ),
    ] = 0,
    selection: Annotated[
        str,
        typer.Option(
            "--selection",
            help="Sample selection policy: random, stratified.",
        ),
    ] = "random",
    figure_seed: Annotated[
        int | None,
        typer.Option("--figure-seed", help="RNG seed for figure selection."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Write inspection JSON to path."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Output directory for figures.",
        ),
    ] = Path("outputs/seqmaze-inspection"),
) -> None:
    """Inspect a SeqMaze corpus — metadata, samples, diagnostics, figures."""
    root = root.resolve()
    manifest = validate_seqmaze_root(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    n_max = manifest.get("n_max", 45)
    t_max = manifest.get("t_max", 32)

    # ── Summary ────────────────────────────────────────────────────────────
    if summary:
        stats = compute_corpus_statistics(root, manifest, all_splits)
        typer.echo("=" * 72)
        typer.echo("SeqMaze corpus summary")
        typer.echo("=" * 72)
        typer.echo(f"  Corpus: {manifest.get('corpus', '?')}")
        typer.echo(f"  Version: {manifest.get('version', '?')}")
        typer.echo(f"  N_max: {n_max}")
        typer.echo(f"  T_max: {t_max}")
        typer.echo(f"  Path vocab: {manifest.get('path_vocab_size', '?')}")
        typer.echo("")
        typer.echo("Samples:")
        for s, c in stats.get("per_split", {}).items():
            typer.echo(f"  {s}: {c}")
        typer.echo("")
        for split_name in all_splits:
            pl = stats.get("path_length", {}).get(split_name, {})
            if pl.get("count", 0) > 0:
                typer.echo(
                    f"  {split_name}: path length min={pl['min']:.0f} "
                    f"median={pl['median']:.0f} max={pl['max']:.0f}"
                )
            na = stats.get("n_actual", {}).get(split_name, {})
            if na.get("count", 0) > 0:
                typer.echo(
                    f"  {split_name}: n_actual min={na['min']:.0f} "
                    f"median={na['median']:.0f} max={na['max']:.0f}"
                )

    # ── Single sample inspection ─────────────────────────────────────────
    if sample_index is not None:
        sample = load_sample(root, split, sample_index)
        ins = prepare_sample_inspection(
            sample, split=split, index=sample_index, n_max=n_max, t_max=t_max
        )
        typer.echo(f"\nSample {sample_index} ({split}):")
        typer.echo(f"  N_actual: {ins.n_actual}  N_max: {ins.n_max}")
        typer.echo(f"  Path length: {ins.path_length}  T_max: {ins.t_max}")
        typer.echo(
            f"  Start: obs {ins.start_obs_id} "
            f"(candidate {ins.start_candidate_index})"
        )
        typer.echo(
            f"  Goal: obs {ins.goal_obs_id} "
            f"(candidate {ins.goal_candidate_index})"
        )
        if ins.target_path:
            path_str = " -> ".join(str(n) for n in ins.target_path)
            typer.echo(f"  Target path (candidate indices): {path_str}")
        if ins.warnings:
            typer.echo(f"  Warnings ({len(ins.warnings)}):")
            for w in ins.warnings:
                typer.echo(f"    {w}")

        if sample_figure:
            typer.echo(
                "  (Figure rendering not yet registered for seqmaze; "
                "skipping.)"
            )

    # ── Sample gallery ────────────────────────────────────────────────────
    if gallery > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for g_idx in range(gallery):
            try:
                sample = load_sample(root, split, g_idx)
                ins = prepare_sample_inspection(
                    sample,
                    split=split,
                    index=g_idx,
                    n_max=n_max,
                    t_max=t_max,
                )
                typer.echo(f"  Sample {g_idx}: {ins.sample_id}")
            except (IndexError, FileNotFoundError) as e:
                typer.echo(f"  Sample {g_idx}: skipped ({e})")

    if show:
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except ImportError:
            pass

    if json_out is not None:
        stats = compute_corpus_statistics(root, manifest, all_splits)
        import json as _json

        _json.dump(stats, json_out.open("w"), indent=2, default=str)


# =============================================================================
# Report bundle helpers
# =============================================================================


def _write_validation_bundle(
    issues: list,
    stats: dict,
    sample_counts: dict[str, int],
    output_dir: Path,
) -> None:
    """Write validation report, diagnostics, and summary to *output_dir*."""
    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARNING"]

    # Validation report
    report_path = output_dir / "validation_report.txt"
    with report_path.open("w") as f:
        f.write(f"SeqMaze corpus validation: {stats.get('corpus', '?')}\n")
        f.write(f"  Version: {stats.get('version', '?')}\n")
        f.write(f"  Validated splits: {list(sample_counts.keys())}\n")
        f.write(f"  Errors: {len(errors)}\n")
        f.write(f"  Warnings: {len(warnings)}\n")
        for i in issues:
            f.write(
                f"  [{i.split}/{i.sample_index}] " f"{i.code}: {i.message}\n"
            )

    # Diagnostics
    diag_path = output_dir / "diagnostics.txt"
    with diag_path.open("w") as f:
        f.write("Corpus diagnostics\n")
        f.write("==================\n")
        for split_name in sample_counts:
            f.write(f"\n{split_name}:\n")
            pl = stats.get("path_length", {}).get(split_name, {})
            if pl:
                f.write(f"  path_length: {pl}\n")
            na = stats.get("n_actual", {}).get(split_name, {})
            if na:
                f.write(f"  n_actual: {na}\n")

    # Summary
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Corpus: {stats.get('corpus', '?')}\n")
        f.write(f"Version: {stats.get('version', '?')}\n")
        f.write(f"N_max: {stats.get('n_max', '?')}\n")
        f.write(f"T_max: {stats.get('t_max', '?')}\n")
        f.write(f"Total samples: {sum(sample_counts.values())}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Warnings: {len(warnings)}\n")


# =============================================================================
if __name__ == "__main__":
    app()

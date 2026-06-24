"""Staged CLI for building the MazeHard task corpus.

MazeHard consumes a maze-nd shared substrate and creates a task corpus
with MazeHard-specific protocol channels.

This CLI does not fetch raw data, prepare interim artifacts, or build
the shared substrate.  Those stages belong in ``build-maze-nd.py``.

Stages
------
build      Build the MazeHard task corpus over a shared substrate.
validate   Validate a MazeHard task-corpus version root.
inspect    Print a human-readable summary of a version root manifest.

Default paths
-------------
Shared substrate:  <user-specified --substrate-root>
Task corpus:       data/processed/mazehard/<corpus>/v<version>

Examples
--------
Build the MazeHard task corpus against a maze-nd shared substrate::

    python build-mazehard.py build \\
        --substrate-root data/interim/maze-nd/v2

With custom sizes::

    python build-mazehard.py build \\
        --substrate-root data/interim/maze-nd/v2 \\
        --n-train 4000 --n-val 500 --n-test 500 --seed 7

Validate an existing task corpus::

    python build-mazehard.py validate data/processed/mazehard/default/v1

Inspect an existing corpus::

    python build-mazehard.py inspect data/processed/mazehard/default/v1

Prerequisites
-------------
A maze-nd shared substrate (artifact_schema_version = 1) must exist before running.
Build it first::

    python scripts/data-gen/build-maze-nd.py build

Or, with explicit version::

    python scripts/data-gen/build-maze-nd.py build --version 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from ehc_sn.data.manifest import read_manifest
from ehc_sn.figures import FigureContext, render
from ehc_sn.tasks.mazehard import (
    MAZEHARD_TASK_CHANNELS,
    TASK_FAMILY,
    build_mazehard_task_corpus,
    validate_mazehard_root,
)
from ehc_sn.tasks.mazehard.corpus import load_sample, load_split_arrays
from ehc_sn.tasks.mazehard.diagnostics import compute_corpus_statistics
from ehc_sn.tasks.mazehard.inspection import prepare_sample_inspection

# ---------------------------------------------------------------------------
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_TRAIN = 1000
_DEFAULT_N_VAL = 40
_DEFAULT_N_TEST = 40
_DEFAULT_SEED = 42
_DEFAULT_OUTPUT_DIR = Path("outputs/mazehard-validation")

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(
    substrate_root: Annotated[
        Path,
        typer.Option(
            "--substrate-root",
            help="Path to the maze-nd shared substrate version root.",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option(
            "--corpus",
            help=f"Corpus label (default: '{_DEFAULT_CORPUS}').",
        ),
    ] = _DEFAULT_CORPUS,
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train",
            help=f"Number of training samples (default: {_DEFAULT_N_TRAIN}).",
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val",
            help=f"Number of validation samples (default: {_DEFAULT_N_VAL}).",
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option(
            "--n-test",
            help=f"Number of test samples (default: {_DEFAULT_N_TEST}).",
        ),
    ] = _DEFAULT_N_TEST,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help=f"Task corpus version integer (default: {_DEFAULT_TASK_VERSION}).",
        ),
    ] = _DEFAULT_TASK_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=f"Deterministic base seed (default: {_DEFAULT_SEED}).",
        ),
    ] = _DEFAULT_SEED,
) -> None:
    """Build the MazeHard task corpus over a maze-nd shared substrate.

    Reads all channels (topology, mask_valid, start, goals, solution) from
    the parent substrate.  Writes a versioned, immutable task corpus.

    Requires a maze-nd shared substrate with artifact_schema_version = 1.
    """
    task_root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    build_mazehard_task_corpus(
        task_root.resolve(),
        substrate_root=substrate_root.resolve(),
        corpus=corpus,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )
    typer.echo(f"MazeHard corpus built at {task_root.resolve()}")
    typer.echo(f"  Samples: train={n_train}, val={n_val}, test={n_test}")


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(help="MazeHard task-corpus root to validate."),
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
    """Validate a MazeHard task-corpus root with structured reports."""
    from ehc_sn.reporting.mazehard import write_validation_bundle
    from ehc_sn.tasks.mazehard.validation import validate_all_samples

    root = root.resolve()
    manifest = validate_mazehard_root(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    targets = [split] if split else all_splits

    issues, sample_counts = validate_all_samples(
        root, splits=targets, max_samples=max_samples
    )

    stats = compute_corpus_statistics(root, manifest, targets, max_samples)
    stats["corpus"] = manifest.get("corpus", "?")
    stats["version"] = manifest.get("version", "?")

    paths = write_validation_bundle(
        issues, stats, sample_counts, output_dir=output_dir
    )

    errors = [i for i in issues if i.severity == "ERROR"]
    n_err = len(errors)
    typer.echo(f"MazeHard corpus validation: {root}")
    typer.echo(f"  Validation report: {paths['validation']}")
    typer.echo(f"  Diagnostics:       {paths['diagnostics']}")
    typer.echo(f"  Summary:           {paths['summary']}")
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
        typer.Argument(help="MazeHard task-corpus root to inspect."),
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
            help="Render mazehard_task_layout for the selected sample.",
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
    ] = Path("outputs/mazehard-inspection"),
) -> None:
    """Inspect a MazeHard corpus — metadata, samples, diagnostics, figures."""
    root = root.resolve()
    manifest = validate_mazehard_root(root)
    all_splits = list(manifest.get("n_samples", {}).keys())

    # ── Summary ────────────────────────────────────────────────────────────
    if summary:
        stats = compute_corpus_statistics(root, manifest, all_splits)
        typer.echo("=" * 72)
        typer.echo("MazeHard corpus summary")
        typer.echo("=" * 72)
        typer.echo(f"  Corpus: {manifest.get('corpus', '?')}")
        typer.echo(f"  Version: {manifest.get('version', '?')}")
        typer.echo(f"  Channels: {manifest.get('channels', [])}")
        typer.echo(f"  Extent: {manifest.get('extent', '?')}")
        typer.echo("")
        typer.echo("Samples:")
        for s, c in stats.get("per_split", {}).items():
            typer.echo(f"  {s}: {c}")
        typer.echo("")
        for split_name in all_splits:
            wd = stats.get("wall_density", {}).get(split_name, {})
            if wd.get("count", 0) > 0:
                typer.echo(
                    f"  {split_name}: wall density min={wd['min']:.3f} "
                    f"median={wd['median']:.3f} max={wd['max']:.3f}"
                )
            sl = stats.get("solution_length", {}).get(split_name, {})
            if sl.get("count", 0) > 0:
                typer.echo(
                    f"  {split_name}: solution length min={sl['min']:.0f} "
                    f"median={sl['median']:.0f} max={sl['max']:.0f}"
                )

    # ── Single sample inspection ─────────────────────────────────────────
    if sample_index is not None:
        sample = load_sample(root, split, sample_index)
        ins = prepare_sample_inspection(sample, split=split, index=sample_index)
        typer.echo(f"\nSample {sample_index} ({split}):")
        typer.echo(f"  Grid: {ins.grid_shape[0]}x{ins.grid_shape[1]}")
        typer.echo(f"  Wall density: {ins.wall_density:.1%}")
        typer.echo(f"  Solution length: {ins.solution_length}")
        typer.echo(f"  Start cells: {ins.n_start_cells}")
        typer.echo(f"  Goal cells: {ins.n_goal_cells}")
        if ins.warnings:
            typer.echo(f"  Warnings ({len(ins.warnings)}):")
            for w in ins.warnings:
                typer.echo(f"    {w}")

        if sample_figure:
            output_dir.mkdir(parents=True, exist_ok=True)
            fpath = _render_overview_figure(
                sample, output_dir, split, sample_index
            )
            typer.echo(f"  Figure: {fpath}")

    # ── Sample gallery ────────────────────────────────────────────────────
    if gallery > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for g_idx in range(gallery):
            fpath = _render_overview_figure(
                None, output_dir, split, g_idx, root=root
            )
            typer.echo(f"  Figure: {fpath}")

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
# Figure helpers
# =============================================================================


def _build_trace_for_sample(
    sample: dict[str, np.ndarray],
    *,
    case_id: str = "mazehard-sample",
) -> "TraceTree":
    """Build a minimal TraceTree with mazehard meta keys from a corpus sample.

    Converts corpus channels (topology, start, goals, solution) into the
    ``input_ids`` / ``target/solution_overlay`` meta keys that the
    ``mazehard_task_layout`` figure selector expects.

    Args:
        sample: Channel dict for one corpus sample.
        case_id: Case identifier (e.g. ``"train/0"``).

    Returns:
        TraceTree with mazehard meta keys populated.
    """
    from ehc_sn.tasks.mazehard.runtime import coerce_maze_hard_batch
    from ehc_sn.traces.keys import (
        MAZEHARD_META_KEY_CASE_ID,
        MAZEHARD_META_KEY_GT_OVERLAY,
        MAZEHARD_META_KEY_INPUT_IDS,
    )
    from ehc_sn.traces.trace_tree import TraceTree

    batch = coerce_maze_hard_batch(sample)
    input_ids = batch["input_ids"].numpy().squeeze()
    labels = batch["labels"].numpy().squeeze()

    # Build overlay: 1 where labels != input_ids (PATH overlay).
    gt_overlay = (labels != input_ids).astype(np.float32)

    # Attach meta keys at the exact paths the figure selector expects,
    # mirroring ``build_mazehard_hrm_trace_meta`` in the HRM adapter.
    gt_root, gt_leaf = MAZEHARD_META_KEY_GT_OVERLAY.split("/", maxsplit=1)
    meta: dict[str, object] = {
        gt_root: {
            gt_leaf: np.expand_dims(gt_overlay, 0),
        },
        MAZEHARD_META_KEY_INPUT_IDS: np.expand_dims(input_ids, 0),
    }
    # case_id is optional metadata; store under its own namespace.
    case_root, case_leaf = MAZEHARD_META_KEY_CASE_ID.split("/", maxsplit=1)
    if case_root not in meta:
        meta[case_root] = {}
    meta[case_root][case_leaf] = case_id  # type: ignore[index]

    trace = TraceTree()
    trace.attach_meta(meta)
    return trace


def _render_overview_figure(
    sample: dict[str, np.ndarray] | None,
    output_dir: Path,
    split: str,
    index: int,
    *,
    root: Path | None = None,
) -> Path:
    """Render mazehard_task_layout for one sample via the registry.

    Args:
        sample: Channel dict for one corpus sample.  If None, loads from
            *root* using *split* and *index*.
        output_dir: Output directory.
        split: Dataset split label.
        index: Sample index within the split.
        root: Versioned corpus root (required when *sample* is None).

    Returns:
        Path to the saved figure.
    """
    if sample is None:
        if root is None:
            raise ValueError("root required when sample is None")
        sample = load_sample(root, split, index)
    case_id = f"{split}/{index}"
    kb = _build_trace_for_sample(sample, case_id=case_id)
    ctx = FigureContext(sample_idx=0)
    fig = render("mazehard_task_layout", kb, ctx)
    fname = f"mazehard_task_layout_{split}_{index}.png"
    fpath = output_dir / fname
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return fpath


if __name__ == "__main__":
    app()

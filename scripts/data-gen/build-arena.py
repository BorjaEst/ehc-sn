"""CLI for building the Arena task corpus from layout datasets.

Arena consumes an interim layout dataset root and generates topology-free
episode trajectories.  Layout generation is owned by the respective layout
CLIs (build-dungeongen.py, build-openfield.py).

Commands
--------
build       Build an Arena task corpus over a layout dataset.
validate    Validate an Arena task-corpus version root.
inspect     Print a human-readable summary of a version root manifest.


Default paths
-------------
Parent layout dataset:  <user-specified --layout-root>
Task corpus:            data/processed/arena/<corpus>/v<version>

Examples
--------
Build from openfield square layouts::

    python build-arena.py build \\
        --layout-root data/interim/openfield/tem-square/v1 \\
        --corpus openfield-square

Build from dungeongen layouts::

    python build-arena.py build \\
        --layout-root data/interim/dungeongen/default/v1 \\
        --corpus dungeons

Inspect an existing corpus::

    python build-arena.py inspect data/processed/arena/openfield-square/v1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from ehc_sn.data.layout.io import load_layout_dataset
from ehc_sn.figures import FigureContext, render
from ehc_sn.reporting.arena import (
    format_validation_summary,
    serialize_validation_result,
    write_validation_bundle,
)
from ehc_sn.tasks.arena import (
    build_arena_task_corpus,
    validate_arena_task_root,
)
from ehc_sn.tasks.arena.corpus import load_sample, load_split_arrays
from ehc_sn.tasks.arena.diagnostics import compute_corpus_statistics
from ehc_sn.tasks.arena.inspection import prepare_sample_inspection
from ehc_sn.tasks.arena.validation import (
    validate_all_samples,
)
from ehc_sn.traces.keys import (
    ARENA_TRACE_KEY_OBSERVATION_IDS,
    ARENA_TRACE_KEY_REVISIT_MASK,
    ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS,
    ARENA_TRACE_KEY_WALL_MASK,
)
from ehc_sn.traces.trace_tree import TraceTree

# ---------------------------------------------------------------------------
_DEFAULT_TASK_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_WALK_POLICY = "angle_bias"
_DEFAULT_MAX_STEPS = 2000
_DEFAULT_N_EPISODES = 4
_DEFAULT_WALK_SEED = 45


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(  # -----------------------------------------------------------------
    layout_root: Annotated[
        Path,
        typer.Option(
            "--layout-root",
            help="Interim layout dataset root.",
        ),
    ],
    corpus: Annotated[
        str,
        typer.Option(
            "--corpus",
            help="Corpus name (e.g. 'openfield-square' or 'dungeons').",
        ),
    ] = _DEFAULT_CORPUS,
    walk_policy: Annotated[
        str,
        typer.Option(
            "--walk-policy",
            help="Walk policy for trajectory generation (default: angle_bias).",
        ),
    ] = _DEFAULT_WALK_POLICY,
    n_episodes: Annotated[
        int,
        typer.Option(
            "--n-episodes",
            help="Number of episodes per layout.",
        ),
    ] = _DEFAULT_N_EPISODES,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Maximum steps per episode.",
        ),
    ] = _DEFAULT_MAX_STEPS,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Version number for the generated task corpus (default: 1).",
        ),
    ] = _DEFAULT_TASK_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for trajectory generation (default: 45).",
        ),
    ] = _DEFAULT_WALK_SEED,
) -> None:
    """Build the Arena task corpus from an interim layout dataset."""
    if not layout_root.exists():
        typer.echo(
            f"Error: layout root not found at {layout_root.resolve()}.\n"
            "Build layouts first with:\n"
            "    python scripts/data-gen/build-openfield.py build\n"
            "or:\n"
            "    python scripts/data-gen/build-dungeongen.py build",
            err=True,
        )
        raise typer.Exit(code=1)

    task_root = Path(f"data/processed/arena/{corpus}/v{version}")

    print(f"Loading layouts from {layout_root.resolve()}...")
    layouts = load_layout_dataset(layout_root.resolve())
    print(f"  {len(layouts)} layouts loaded.")

    build_arena_task_corpus(
        version_root=task_root.resolve(),
        layouts=layouts,
        corpus=corpus,
        walk_policy=walk_policy,
        n_episodes_per_layout=n_episodes,
        max_steps=max_steps,
        seed=seed,
    )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path, typer.Argument(help="Arena task-corpus root to validate.")
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
    ] = Path("outputs/arena-validation"),
) -> None:
    """Validate an Arena task-corpus root with structured reports."""
    from ehc_sn.data.manifest import read_manifest

    root = root.resolve()
    manifest = read_manifest(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    splits = [split] if split else all_splits

    issues, sample_counts = validate_all_samples(
        root, splits=splits, max_samples=max_samples
    )

    stats = compute_corpus_statistics(root, manifest, splits, max_samples)
    stats["corpus"] = manifest.get("corpus", "?")
    stats["version"] = manifest.get("version", "?")

    paths = write_validation_bundle(
        issues, stats, sample_counts, output_dir=output_dir
    )
    typer.echo(f"Arena corpus validation: {root}")
    typer.echo(f"  Validation report: {paths['validation']}")
    typer.echo(f"  Diagnostics:       {paths['diagnostics']}")
    typer.echo(f"  Summary:           {paths['summary']}")

    errors = [i for i in issues if i.severity == "ERROR"]
    n_err = len(errors)
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
def inspect_command(
    root: Annotated[
        Path, typer.Argument(help="Arena task-corpus root to inspect.")
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
            help="Render task_overview_arena for the selected sample.",
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
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Output directory for figures.",
        ),
    ] = Path("outputs/arena-inspection"),
) -> None:
    """Inspect an Arena corpus — metadata, samples, diagnostics, figures."""
    from ehc_sn.data.manifest import read_manifest

    root = root.resolve()
    manifest = read_manifest(root)
    all_splits = list(manifest.get("n_samples", {}).keys())

    # ── Summary ────────────────────────────────────────────────────────────
    if summary:
        stats = compute_corpus_statistics(root, manifest, all_splits)
        typer.echo("=" * 72)
        typer.echo("Arena corpus summary")
        typer.echo("=" * 72)
        typer.echo(f"  Corpus: {manifest.get('corpus', '?')}")
        typer.echo(f"  Version: {manifest.get('version', '?')}")
        typer.echo("")
        typer.echo("Samples:")
        for s, c in stats.get("per_split", {}).items():
            typer.echo(f"  {s}: {c}")
        typer.echo("")
        for split_name in all_splits:
            el = stats.get("episode_length", {}).get(split_name, {})
            if el.get("count", 0) > 0:
                typer.echo(
                    f"  {split_name}: episode length min={el['min']:.0f} "
                    f"median={el['median']:.0f} max={el['max']:.0f}"
                )
            rr = stats.get("revisit_rate", {}).get(split_name, {})
            if rr.get("count", 0) > 0:
                typer.echo(
                    f"  {split_name}: revisit rate min={rr['min']:.3f} "
                    f"median={rr['median']:.3f} max={rr['max']:.3f}"
                )

    # ── Single sample inspection ─────────────────────────────────────────
    if sample_index is not None:
        sample = load_sample(root, split, sample_index)
        ins = prepare_sample_inspection(sample, split=split, index=sample_index)
        typer.echo(f"\nSample {sample_index} ({split}):")
        typer.echo(f"  Grid: {ins.grid_shape[0]}x{ins.grid_shape[1]}")
        typer.echo(f"  Episode length: {ins.episode_length}")
        typer.echo(f"  Revisits: {ins.revisit_count} ({ins.revisit_rate:.1%})")
        typer.echo(
            f"  Start: ({ins.start_position[0]}, {ins.start_position[1]})"
        )
        typer.echo(f"  Wall density: {ins.wall_density:.1%}")
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


# =============================================================================
# Figure helpers
# =============================================================================


def _build_trace_for_sample(
    sample: dict[str, np.ndarray],
) -> TraceTree:
    """Build a minimal TraceTree with arena trace keys from a corpus sample.

    The task_overview_arena figure selector reads from trace keys
    (``trace.get()``), not meta keys.  We use ``attach_dense`` to inject
    them.

    We add a batch dimension so the selector's single-sample slice works
    with the same indexing logic used for full evaluation traces.

    Args:
        sample: Channel dict for one corpus sample.

    Returns:
        TraceTree with arena trace keys populated.
    """
    trace = TraceTree()

    wall_mask = np.asarray(sample["topology"], dtype=bool)
    obs_ids = np.asarray(sample["observations"], dtype=np.int32)
    traj_len = int(np.asarray(sample["trajectory_length"]).flat[0])
    rows = np.asarray(sample["trajectory_row"])[:traj_len]
    cols = np.asarray(sample["trajectory_col"])[:traj_len]

    # Infer grid shape from wall_mask
    H, W = wall_mask.shape

    # Row-major location IDs
    traj_locs = (rows * W + cols).astype(np.int32)
    revisit = np.asarray(sample["trajectory_is_revisit"])[:traj_len]

    # Attach as dense trace arrays WITHOUT extra batch dim —
    # the figure selector expects (H,W) for wall_mask, (H,W) for obs_ids,
    # (T,) for trajectory_locations, (T,) for revisit_mask.
    trace.attach_dense(ARENA_TRACE_KEY_WALL_MASK, wall_mask)
    trace.attach_dense(ARENA_TRACE_KEY_OBSERVATION_IDS, obs_ids)
    trace.attach_dense(ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS, traj_locs)
    trace.attach_dense(ARENA_TRACE_KEY_REVISIT_MASK, revisit)

    return trace


def _render_overview_figure(
    sample: dict[str, np.ndarray] | None,
    output_dir: Path,
    split: str,
    index: int,
    *,
    root: Path | None = None,
) -> Path:
    """Render task_overview_arena for one sample via the registry.

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
    kb = _build_trace_for_sample(sample)
    ctx = FigureContext(sample_idx=0)
    fig = render("task_overview_arena", kb, ctx)
    fname = f"task_overview_arena_{split}_{index}.png"
    fpath = output_dir / fname
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return fpath


# =============================================================================
if __name__ == "__main__":
    app()

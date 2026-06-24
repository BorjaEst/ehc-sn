"""Staged CLI for building the Goaltrace task corpus.

Goaltrace consumes a dagflow shared substrate and creates a task corpus
with field-prediction protocol channels.

This CLI does not generate graph topology.  That stage belongs in
``build-dagflow.py``.

Stages
------
build      Build the Goaltrace task corpus over a shared substrate.
validate   Validate a Goaltrace task-corpus version root.
inspect    Print a human-readable summary of a version root manifest.

Default paths
-------------
Parent layout dataset:  <user-specified --layout-root>
Task corpus:            data/processed/goaltrace/<corpus>/v<version>

Examples
--------
Build the Goaltrace task corpus against a dagflow layout dataset::

    python build-goaltrace.py build \\
        --layout-root data/interim/dagflow/sparse/v1

With custom sizes::

    python build-goaltrace.py build \\
        --layout-root data/interim/dagflow/sparse/v1 \\
        --n-observations 64 --num-graphs 1 \\
        --oracle-semantics reliability --field-decay 0.8 \\
        --n-train 8000 --n-val 1000 --n-test 1000 --seed 7

With multiple DAGs::

    python build-goaltrace.py build \\
        --layout-root data/interim/dagflow/sparse/v1 \\
        --num-graphs 8 --n-train 4000

Validate an existing task corpus::

    python build-goaltrace.py validate data/processed/goaltrace/default/v1

Inspect an existing corpus::

    python build-goaltrace.py inspect data/processed/goaltrace/default/v1

Build a static-weight corpus (deterministic, contradiction-free)::

    python build-goaltrace.py build \\
        --layout-root data/interim/dagflow/sparse/v1 \\
        --static-weights \\
        --n-train 500 --n-val 250 --n-test 240 \\
        --distance-tau 8.0 \\
        --grid-width 20 --grid-height 30 \\
        --version 3

Prerequisites
-------------
A dagflow shared substrate must exist before running.
Build it first::

    python scripts/data-gen/build-dagflow.py build --preset branching --version 1

For static-weight corpora, a larger dagflow substrate can be built with::

    python scripts/data-gen/build-dagflow.py build --preset branching \
        --n-max 45 --max-out-degree 4 --version 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from ehc_sn.figures import FigureContext, render
from ehc_sn.reporting.goaltrace import (
    format_validation_summary,
    serialize_validation_result,
    write_validation_bundle,
)
from ehc_sn.tasks.goaltrace import (
    TASK_FAMILY,
    build_goaltrace_task_corpus,
)
from ehc_sn.tasks.goaltrace.corpus import load_sample, load_split_arrays
from ehc_sn.tasks.goaltrace.diagnostics import (
    compute_corpus_statistics,
    select_samples,
)
from ehc_sn.tasks.goaltrace.inspection import prepare_sample_inspection
from ehc_sn.tasks.goaltrace.validation import validate_all_samples
from ehc_sn.traces.keys import (
    GOALTRACE_META_KEY_CURRENT_FLAG,
    GOALTRACE_META_KEY_GOAL_FLAG,
    GOALTRACE_META_KEY_NODE_MASK,
    GOALTRACE_META_KEY_OBSERVATION_ID,
    GOALTRACE_META_KEY_SUCCESSOR_INDICES,
    GOALTRACE_META_KEY_SUCCESSOR_MASK,
    GOALTRACE_META_KEY_TARGET_FIELD,
    GOALTRACE_META_KEY_WEIGHT,
)
from ehc_sn.traces.trace_tree import TraceTree

# ---------------------------------------------------------------------------
_DEFAULT_VERSION = 1
_DEFAULT_CORPUS = "default"
_DEFAULT_N_OBSERVATIONS = 45
_DEFAULT_NUM_GRAPHS = 1
_DEFAULT_ORACLE_SEMANTICS = "reliability"
_DEFAULT_FIELD_DECAY = 0.8
_DEFAULT_N_TRAIN = 500
_DEFAULT_N_VAL = 250
_DEFAULT_N_TEST = 240
_DEFAULT_SEED = 42
_DEFAULT_STATIC = True
_DEFAULT_MARGIN = 0.0
_DEFAULT_TAU = 8.0
_DEFAULT_DMAX = 0.0  # 0 means auto-compute from grid dimensions
_DEFAULT_GW = 20
_DEFAULT_GH = 30
_DEFAULT_GEOM_SEED = 42

app = typer.Typer(add_completion=False, help="Goaltrace task corpus builder.")


# =============================================================================
@app.command("build")
def build(
    ctx: typer.Context,
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
    n_observations: Annotated[
        int,
        typer.Option(
            "--n-observations",
            help="Maximum padded observation count N (default: 32).",
        ),
    ] = _DEFAULT_N_OBSERVATIONS,
    dagflow_graph_id: Annotated[
        str | None,
        typer.Option(
            "--dagflow-graph-id",
            help="Stable artifact ID of a single DAG within layout-root. "
            "When set, overrides --num-graphs and uses exactly this graph. "
            "(default: None, uses --num-graphs).",
        ),
    ] = None,
    num_graphs: Annotated[
        int,
        typer.Option(
            "--num-graphs",
            help="Number of DAGs to sample from the layout pool "
            "(default: 1, all samples share one DAG).  Deprecated when "
            "--dagflow-graph-id is provided.",
        ),
    ] = _DEFAULT_NUM_GRAPHS,
    oracle_semantics: Annotated[
        str,
        typer.Option(
            "--oracle-semantics",
            help="Weight semantics for path selection. "
            "One of: reliability, linear_cost, preference (default: reliability).",
        ),
    ] = _DEFAULT_ORACLE_SEMANTICS,
    field_decay: Annotated[
        float,
        typer.Option(
            "--field-decay",
            help="Field decay factor gamma in (0, 1) (default: 0.8).",
        ),
    ] = _DEFAULT_FIELD_DECAY,
    preference_step_penalty: Annotated[
        float,
        typer.Option(
            "--preference-step-penalty",
            help="Per-step penalty lambda for 'preference' semantics "
            "(default: 0.0).",
        ),
    ] = 0.0,
    n_train: Annotated[
        int,
        typer.Option(
            "--n-train",
            help="Number of training samples (default: 4000).",
        ),
    ] = _DEFAULT_N_TRAIN,
    n_val: Annotated[
        int,
        typer.Option(
            "--n-val",
            help="Number of validation samples (default: 500).",
        ),
    ] = _DEFAULT_N_VAL,
    n_test: Annotated[
        int,
        typer.Option(
            "--n-test",
            help="Number of test samples (default: 500).",
        ),
    ] = _DEFAULT_N_TEST,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Task corpus version integer (default: 1).",
        ),
    ] = _DEFAULT_VERSION,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Deterministic base seed (default: 42).",
        ),
    ] = _DEFAULT_SEED,
    # --- Static-weight pipeline flags ---
    static_weights: Annotated[
        bool,
        typer.Option(
            "--static-weights",
            help="Use static geometry-derived relational weights instead of "
            "per-sample random weights.  Requires a dagflow substrate built "
            "with the Hamiltonian backbone generator (default: false).",
        ),
    ] = _DEFAULT_STATIC,
    min_optimality_margin: Annotated[
        float,
        typer.Option(
            "--min-optimality-margin",
            help="Minimum cost margin between optimal and second-best path. "
            "Pairs below this threshold are rejected.  Only used with "
            "--static-weights (default: 0.0).",
        ),
    ] = _DEFAULT_MARGIN,
    distance_tau: Annotated[
        float,
        typer.Option(
            "--distance-tau",
            help="Temperature / distance scale for the spatial kernel. "
            "Only used with --static-weights (default: 8.0).",
        ),
    ] = _DEFAULT_TAU,
    distance_max: Annotated[
        float,
        typer.Option(
            "--distance-max",
            help="Maximum effective distance for the kernel.  Set to 0 to "
            "auto-compute from grid width+height.  Only used with "
            "--static-weights (default: 0 = auto).",
        ),
    ] = _DEFAULT_DMAX,
    grid_width: Annotated[
        int,
        typer.Option(
            "--grid-width",
            help="Hidden spatial grid width in cells.  Only used with "
            "--static-weights (default: 20).",
        ),
    ] = _DEFAULT_GW,
    grid_height: Annotated[
        int,
        typer.Option(
            "--grid-height",
            help="Hidden spatial grid height in cells.  Only used with "
            "--static-weights (default: 30).",
        ),
    ] = _DEFAULT_GH,
    geometry_seed: Annotated[
        int,
        typer.Option(
            "--geometry-seed",
            help="Seed for anchor placement on the spatial grid.  Only used "
            "with --static-weights (default: 42).",
        ),
    ] = _DEFAULT_GEOM_SEED,
) -> None:
    """Build the Goaltrace task corpus over a dagflow layout dataset."""
    # Translate distance_max=0 to None for the builder (auto-compute)
    parsed_distance_max = None if distance_max == 0.0 else distance_max

    root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    build_goaltrace_task_corpus(
        version_root=root.resolve(),
        layout_root=layout_root.resolve(),
        corpus=corpus,
        n_observations=n_observations,
        num_graphs=num_graphs,
        dagflow_graph_id=dagflow_graph_id,
        oracle_semantics=oracle_semantics,
        field_decay=field_decay,
        preference_step_penalty=preference_step_penalty,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        static_weights=static_weights,
        min_optimality_margin=min_optimality_margin,
        distance_tau=distance_tau,
        distance_max=parsed_distance_max,
        grid_width=grid_width,
        grid_height=grid_height,
        geometry_seed=geometry_seed,
    )
    typer.echo(f"Goaltrace corpus built at {root.resolve()}")
    typer.echo(f"  Profile: N={n_observations}, num_graphs={num_graphs}")
    typer.echo(f"  Oracle: {oracle_semantics}, field_decay={field_decay}")
    if static_weights:
        typer.echo(
            f"  Mode: static weights  tau={distance_tau}  "
            f"grid={grid_width}x{grid_height}  margin={min_optimality_margin}"
        )


# =============================================================================
@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Path to the goaltrace task corpus version root, "
            "e.g. data/processed/goaltrace/default/v1.",
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
    ] = Path("outputs/goaltrace-validation"),
) -> None:
    """Validate a goaltrace task-corpus version root with structured reports."""
    from ehc_sn.data.manifest import read_manifest

    root = root.resolve()
    manifest = read_manifest(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    splits = [split] if split else all_splits

    # Run validation
    issues, sample_counts = validate_all_samples(
        root, splits=splits, max_samples=max_samples
    )

    # Statistics
    stats = compute_corpus_statistics(root, manifest, splits, max_samples)
    stats["n_observations"] = manifest.get("n_observations", 0)
    stats["corpus"] = manifest.get("corpus", "?")
    stats["version"] = manifest.get("version", "?")
    stats["field_decay"] = manifest.get("field_decay", 0.8)

    # Reports
    paths = write_validation_bundle(
        issues, stats, sample_counts, output_dir=output_dir
    )
    typer.echo(f"Goaltrace corpus validation: {root}")
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
def inspect(
    root: Annotated[
        Path,
        typer.Argument(
            help="Path to the goaltrace task corpus version root, "
            "e.g. data/processed/goaltrace/default/v1.",
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
            help="Render task_overview_goaltrace for the selected sample.",
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
            help="Sample selection policy: random, stratified, "
            "longest_path, shortest_path.",
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
    ] = Path("outputs/goaltrace-inspection"),
) -> None:
    """Inspect a Goaltrace corpus — metadata, samples, diagnostics, figures."""
    from ehc_sn.data.manifest import read_manifest

    root = root.resolve()
    manifest = read_manifest(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    n_obs = manifest.get("n_observations", 0)
    field_decay = manifest.get("field_decay", 0.8)

    # ── Summary ────────────────────────────────────────────────────────────
    if summary:
        stats = compute_corpus_statistics(root, manifest, all_splits)
        typer.echo("=" * 72)
        typer.echo("Goaltrace corpus summary")
        typer.echo("=" * 72)
        typer.echo(f"  Corpus: {manifest.get('corpus', '?')}")
        typer.echo(f"  Version: {manifest.get('version', '?')}")
        typer.echo(f"  Observations: {n_obs}")
        typer.echo(f"  Field decay: {field_decay}")
        typer.echo("")
        typer.echo("Samples:")
        for s, c in stats.get("per_split", {}).items():
            typer.echo(f"  {s}: {c}")
        typer.echo("")
        for split_name in all_splits:
            rp = stats.get("reachable_pairs", {}).get(split_name, {})
            if rp:
                typer.echo(
                    f"  {split_name}: reachable pairs "
                    f"{rp.get('reachable', 0)}/{rp.get('total', 0)}"
                )
            dc = stats.get("decay_consistency", {}).get(split_name, {})
            if dc:
                typer.echo(
                    f"  {split_name}: decay consistent "
                    f"{dc.get('passes', 0)}/{dc.get('total', 0)}"
                )
            iu = stats.get("input_uniqueness", {}).get(split_name, {})
            if iu:
                typer.echo(
                    f"  {split_name}: unique keys "
                    f"{iu.get('unique_keys', 0)}, "
                    f"contradictions {iu.get('contradictions', 0)}"
                )
            bl = stats.get("baselines", {}).get(split_name, {})
            if bl:
                typer.echo(
                    f"  {split_name}: baseline current_only "
                    f"{bl.get('current_only', 0):.4f}"
                )

    # ── Single sample inspection ─────────────────────────────────────────
    if sample_index is not None:
        sample = load_sample(root, split, sample_index)
        ins = prepare_sample_inspection(
            sample, split=split, index=sample_index, field_decay=field_decay
        )
        typer.echo(f"\nSample {sample_index} ({split}):")
        typer.echo(
            f"  Current: obs {ins.current_observation_id} "
            f"(idx {ins.current_idx})"
        )
        typer.echo(
            f"  Goal: obs {ins.goal_observation_id} " f"(idx {ins.goal_idx})"
        )
        typer.echo(f"  Valid nodes: {ins.n_valid}  Padded: {ins.n_padded}")
        typer.echo(f"  Path length: {ins.path_length} hops")
        if ins.optimal_path:
            path_str = " -> ".join(str(n) for n in ins.optimal_path)
            typer.echo(f"  Optimal path: {path_str}")
        typer.echo(f"  Current target: {ins.current_target:.4f}")
        typer.echo(f"  Goal target: {ins.goal_target:.4f}")
        typer.echo(f"  Decay consistent: {ins.decay_consistent}")
        if ins.weight_stats:
            ws = ins.weight_stats
            typer.echo(
                f"  Edge weights: min={ws['min']:.3f} "
                f"mean={ws['mean']:.3f} max={ws['max']:.3f}"
            )
        if ins.warnings:
            typer.echo(f"  Warnings ({len(ins.warnings)}):")
            for w in ins.warnings:
                typer.echo(f"    {w}")

        if sample_figure:
            output_dir.mkdir(parents=True, exist_ok=True)
            fpath = _render_overview_figure(
                sample,
                output_dir,
                split,
                sample_index,
                manifest=manifest,
            )
            typer.echo(f"  Figure: {fpath}")

    # ── Sample gallery ────────────────────────────────────────────────────
    if gallery > 0:
        selected = select_samples(
            root, all_splits, policy=selection, n=gallery, seed=figure_seed
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for s, idx in selected:
            sample = load_sample(root, s, idx)
            fpath = _render_overview_figure(
                sample, output_dir, s, idx, manifest=manifest
            )
            typer.echo(f"  Figure: {fpath}")

    if json_out is not None:
        stats = compute_corpus_statistics(root, manifest, all_splits)
        import json as _json

        _json.dump(stats, json_out.open("w"), indent=2, default=str)

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
    *,
    n_observations: int | None = None,
) -> TraceTree:
    """Build a minimal TraceTree with goaltrace meta keys from a corpus sample.

    Args:
        sample: Channel dict for one corpus sample.
        n_observations: Corpus-wide observation vocabulary size.

    Returns:
        TraceTree with goaltrace meta keys populated.
    """
    trace = TraceTree()
    oid = np.asarray(sample["observation_id"], dtype=np.int32)
    w = np.asarray(sample["weight"], dtype=np.float32)
    cf = np.asarray(sample["current_flag"], dtype=bool)
    gf = np.asarray(sample["goal_flag"], dtype=bool)
    nm = np.asarray(sample["node_mask"], dtype=bool)
    tf = np.asarray(sample["target_field"], dtype=np.float32)
    si = np.asarray(
        sample.get("successor_indices", np.zeros_like(oid)), dtype=np.int32
    )
    sm = np.asarray(sample.get("successor_mask", np.zeros_like(si)), dtype=bool)

    meta: dict[str, object] = {
        "goaltrace": {
            GOALTRACE_META_KEY_OBSERVATION_ID.split("/")[-1]: np.expand_dims(
                oid, 0
            ),
            GOALTRACE_META_KEY_WEIGHT.split("/")[-1]: np.expand_dims(w, 0),
            GOALTRACE_META_KEY_CURRENT_FLAG.split("/")[-1]: np.expand_dims(
                cf, 0
            ),
            GOALTRACE_META_KEY_GOAL_FLAG.split("/")[-1]: np.expand_dims(gf, 0),
            GOALTRACE_META_KEY_NODE_MASK.split("/")[-1]: np.expand_dims(nm, 0),
            GOALTRACE_META_KEY_TARGET_FIELD.split("/")[-1]: np.expand_dims(
                tf, 0
            ),
            GOALTRACE_META_KEY_SUCCESSOR_INDICES.split("/")[-1]: np.expand_dims(
                si, 0
            ),
            GOALTRACE_META_KEY_SUCCESSOR_MASK.split("/")[-1]: np.expand_dims(
                sm, 0
            ),
        }
    }
    if n_observations is not None:
        meta["goaltrace"]["n_observations"] = n_observations  # type: ignore[index]

    trace.attach_meta(meta)
    return trace


def _render_overview_figure(
    sample: dict[str, np.ndarray] | None,
    output_dir: Path,
    split: str,
    index: int,
    *,
    manifest: dict[str, object] | None = None,
    root: Path | None = None,
) -> Path:
    """Render task_overview_goaltrace for one sample via the registry.

    Args:
        sample: Channel dict for one corpus sample.  If None, loads from
            *root* using *split* and *index*.
        output_dir: Output directory.
        split: Dataset split label.
        index: Sample index within the split.
        manifest: Optional corpus manifest for authoritative metadata.
        root: Versioned corpus root (required when *sample* is None).

    Returns:
        Path to the saved figure.
    """
    if sample is None:
        if root is None:
            raise ValueError("root required when sample is None")
        sample = load_sample(root, split, index)
    kb = _build_trace_for_sample(
        sample,
        n_observations=manifest.get("n_observations") if manifest else None,  # type: ignore[arg-type]
    )
    ctx = FigureContext(sample_idx=0)
    fig = render("task_overview_goaltrace", kb, ctx)
    fname = f"task_overview_goaltrace_{split}_{index}.png"
    fpath = output_dir / fname
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return fpath


# =============================================================================
if __name__ == "__main__":
    app()

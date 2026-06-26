"""Single Routebind task-family CLI — build, validate, inspect.

Commands
--------
build               Materialize a Routebind corpus from topology and dagflow parents.
validate            Verify an existing corpus against structural, semantic, and corpus contracts.
inspect             Examine corpus metadata, samples, diagnostics, and visualizations.
materialize-task    Deprecated alias for ``build``.

Usage
-----
::

    python build-routebind.py build \
        --topology-root data/interim/dungeongen/routebind-30/v1 \
        --dagflow-root data/interim/dagflow/routing/v1 \
        --dagflow-graph-id dagflow-routing-v1-train-000000
    python build-routebind.py validate data/processed/routebind/default/v1
    python build-routebind.py inspect data/processed/routebind/default/v1 --summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from ehc_sn.data.layout.io import load_layout_dataset
from ehc_sn.data.manifest import read_manifest
from ehc_sn.data.substrate.dagflow import validate_dagflow_layout_sample
from ehc_sn.data.substrate.reader import find_artifact_by_id
from ehc_sn.figures import FigureContext, render
from ehc_sn.reporting.routebind import (
    format_validation_summary,
    serialize_validation_result,
    write_validation_bundle,
)
from ehc_sn.tasks.routebind import (
    ROUTEBIND_PRESETS,
    TASK_FAMILY,
    build_routebind_task_corpus,
    resolve_preset,
)
from ehc_sn.tasks.routebind.corpus import load_sample, load_split_arrays
from ehc_sn.tasks.routebind.diagnostics import (
    compute_corpus_statistics,
    select_samples,
)
from ehc_sn.tasks.routebind.inspection import prepare_sample_inspection
from ehc_sn.tasks.routebind.validation import (
    OracleValidationContext,
    ValidationIssue,
    check_oracle_optimal_subgraph,
    validate_corpus_root,
    validate_stored_sample,
    validate_support_channels,
)
from ehc_sn.traces.keys import (
    ROUTEBIND_META_KEY_CELL_TYPE,
    ROUTEBIND_META_KEY_GOAL_FLAG,
    ROUTEBIND_META_KEY_OBSERVATION_ID,
    ROUTEBIND_META_KEY_START_FLAG,
    ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
)
from ehc_sn.traces.trace_tree import TraceTree

# ---------------------------------------------------------------------------
_DEFAULT_PRESET = "balanced"
_DEFAULT_CORPUS = "default"
_DEFAULT_VERSION = 1
_DEFAULT_STORAGE_HEIGHT = 30
_DEFAULT_STORAGE_WIDTH = 30
_DEFAULT_FIELD_DECAY_SPATIAL = 0.9848
_DEFAULT_FIELD_DECAY_SEMANTIC = 0.8
_DEFAULT_MAX_ROUTE_LENGTH = 150
_DEFAULT_N_QUERIES_PER_LAYOUT = 10
_DEFAULT_SEED = 42
_DEFAULT_OUTPUT_DIR = Path("outputs/routebind-validation")

app = typer.Typer(add_completion=False, help="Routebind task corpus toolchain.")


# =============================================================================
# Helper: emit validation results and exit
# =============================================================================


def _emit_validation_results(
    all_issues: list,
    sample_counts: dict[str, int],
    output_dir: Path,
    json_out: Path | None = None,
    summary_out: Path | None = None,
    stats: dict | None = None,
) -> int:
    """Write validation bundle, print summary, return exit code."""
    from ehc_sn.reporting.routebind import (
        format_validation_summary,
        serialize_validation_result,
        write_validation_bundle,
    )

    if stats is None:
        stats = {
            "per_split": sample_counts,
            "route_length": {},
            "semantic_length": {},
        }
    paths = write_validation_bundle(
        all_issues,
        stats,
        sample_counts,
        output_dir=output_dir,
    )
    typer.echo(f"  Validation report: {paths['validation']}")
    typer.echo(f"  Diagnostics:       {paths['diagnostics']}")
    typer.echo(f"  Summary:           {paths['summary']}")
    if json_out is not None:
        json_out.write_text(
            json.dumps(
                serialize_validation_result(all_issues, sample_counts), indent=2
            )
        )
    if summary_out is not None:
        summary_out.write_text(format_validation_summary(stats, all_issues))

    errors = [i for i in all_issues if i.severity == "ERROR"]
    n_err = len(errors)
    typer.echo(f"Validated {sum(sample_counts.values())} samples")
    typer.echo(
        f"  Errors: {n_err}  Warnings: "
        f"{len([i for i in all_issues if i.severity == 'WARNING'])}"
    )
    if n_err > 0:
        typer.echo("\n  First 5 errors:")
        for e in errors[:5]:
            typer.echo(
                f"    [{e.split}/{e.sample_index}] {e.code}: {e.message}"
            )
    return 1 if n_err > 0 else 0


# =============================================================================
@app.command("build")
def build(
    topology_root: Annotated[
        Path,
        typer.Option(
            "--topology-root",
            help="Path to topology dataset root "
            "(e.g. data/interim/openfield/tem-square/v1).",
        ),
    ],
    dagflow_root: Annotated[
        Path,
        typer.Option(
            "--dagflow-root",
            help="Path to dagflow dataset root "
            "(e.g. data/interim/dagflow/routing/v1).",
        ),
    ],
    dagflow_graph_id: Annotated[
        str,
        typer.Option(
            "--dagflow-graph-id",
            help="Stable artifact ID of the single DAG within dagflow-root "
            "(e.g. dagflow-default-v1-train-000042).",
        ),
    ],
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help=f"Named Routebind preset ({', '.join(sorted(ROUTEBIND_PRESETS))})."
            " (default: 'balanced').",
        ),
    ] = _DEFAULT_PRESET,
    corpus: Annotated[
        str,
        typer.Option("--corpus", help="Corpus label (default: 'default')."),
    ] = _DEFAULT_CORPUS,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help="Task corpus version integer (default: 1).",
        ),
    ] = _DEFAULT_VERSION,
    field_decay_spatial: Annotated[
        float,
        typer.Option(
            "--field-decay-spatial",
            help="Spatial field decay factor (default: 0.9848).",
        ),
    ] = _DEFAULT_FIELD_DECAY_SPATIAL,
    field_decay_semantic: Annotated[
        float,
        typer.Option(
            "--field-decay-semantic",
            help="Semantic field decay factor (default: 0.8).",
        ),
    ] = _DEFAULT_FIELD_DECAY_SEMANTIC,
    max_supported_route_length: Annotated[
        int,
        typer.Option(
            "--max-supported-route-length",
            help="Maximum route length for terminal activation check "
            "(default: 150).",
        ),
    ] = _DEFAULT_MAX_ROUTE_LENGTH,
    n_queries_per_layout: Annotated[
        int,
        typer.Option(
            "--n-queries-per-layout",
            help="Number of start/goal queries per layout (default: 10).",
        ),
    ] = _DEFAULT_N_QUERIES_PER_LAYOUT,
    storage_height: Annotated[
        int,
        typer.Option(
            "--storage-height",
            help="Storage canvas height in cells (default: 30).",
        ),
    ] = _DEFAULT_STORAGE_HEIGHT,
    storage_width: Annotated[
        int,
        typer.Option(
            "--storage-width",
            help="Storage canvas width in cells (default: 30).",
        ),
    ] = _DEFAULT_STORAGE_WIDTH,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic base seed (default: 42)."),
    ] = _DEFAULT_SEED,
    min_route_length: Annotated[
        int | None,
        typer.Option(
            "--min-route-length",
            help="Override hard minimum route length.",
        ),
    ] = None,
    max_route_length_override: Annotated[
        int | None,
        typer.Option(
            "--max-route-length",
            help="Override hard maximum route length.",
        ),
    ] = None,
    attempt_budget: Annotated[
        int | None,
        typer.Option(
            "--attempt-budget",
            help="Override preset's attempt budget.",
        ),
    ] = None,
    target_samples: Annotated[
        int | None,
        typer.Option(
            "--target-samples",
            help="Explicit corpus-level sample target "
            "(default: n_queries_per_layout * n_layouts).",
        ),
    ] = None,
) -> None:
    """Build the Routebind task corpus over spatial topology + dagflow graph."""
    root = Path(f"data/processed/{TASK_FAMILY}/{corpus}/v{version}")
    topology_root_resolved = topology_root.resolve()
    dagflow_root_resolved = dagflow_root.resolve()

    if not topology_root_resolved.exists():
        typer.echo(
            f"Error: topology root not found at {topology_root_resolved}.\n"
            "Build topology first with:\n"
            "    python scripts/data-gen/build-openfield.py build\n"
            "or:\n"
            "    python scripts/data-gen/build-dungeongen.py build",
            err=True,
        )
        raise typer.Exit(code=1)

    if not dagflow_root_resolved.exists():
        typer.echo(
            f"Error: dagflow root not found at {dagflow_root_resolved}.\n"
            "Build dagflow first with:\n"
            "    python scripts/data-gen/build-dagflow.py build --preset routing --version 1",
            err=True,
        )
        raise typer.Exit(code=1)

    topology_manifest = read_manifest(topology_root_resolved)
    layouts = load_layout_dataset(topology_root_resolved)

    version_root = root.resolve()

    overrides: dict = {}
    if min_route_length is not None:
        overrides["hard_min_semantic_edges"] = min_route_length
    if max_route_length_override is not None:
        overrides["hard_max_semantic_edges"] = max_route_length_override
    if attempt_budget is not None:
        overrides["attempt_budget"] = attempt_budget
    profile = resolve_preset(preset, overrides or None)

    build_routebind_task_corpus(
        version_root=version_root,
        layouts=layouts,
        topology_manifest=topology_manifest,
        dagflow_root=dagflow_root_resolved,
        dagflow_graph_id=dagflow_graph_id,
        corpus=corpus,
        storage_height=storage_height,
        storage_width=storage_width,
        field_decay_spatial=field_decay_spatial,
        field_decay_semantic=field_decay_semantic,
        max_supported_route_length=max_supported_route_length,
        n_queries_per_layout=n_queries_per_layout,
        seed=seed,
        preset=profile,
        target_samples_total=target_samples,
    )
    typer.echo(f"Routebind corpus built at {version_root}")
    typer.echo(f"  Topology: {topology_root_resolved}")
    typer.echo(f"  DAG graph: {dagflow_graph_id} @ {dagflow_root_resolved}")
    typer.echo(f"  Preset: {preset}")


# =============================================================================
# Validate
# =============================================================================


@app.command("validate")
def validate(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to validate "
            "(e.g. data/processed/routebind/default/v1)."
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
    max_root_samples: Annotated[
        int,
        typer.Option(
            "--max-root-samples",
            help="Max samples for root-level structural scan (default: all).",
        ),
    ] = -1,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json-out",
            help="Write validation JSON to this path.",
        ),
    ] = None,
    summary_out: Annotated[
        Path | None,
        typer.Option(
            "--summary-out",
            help="Write text summary to this path.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Write report bundle to this directory.",
        ),
    ] = _DEFAULT_OUTPUT_DIR,
) -> None:
    """Validate an existing Routebind task corpus against declared contracts.

    Runs Layer 1 (structural support/depth algebra) and Layer 2
    (Bellman-optimal product-state recomputation) on all samples.
    """
    _run_validate(
        root,
        splits=[split] if split else None,
        max_samples=max_samples,
        max_root_samples=max_root_samples,
        json_out=json_out,
        summary_out=summary_out,
        output_dir=output_dir,
    )


def _run_validate(
    root: Path,
    splits: list[str] | None = None,
    max_samples: int = -1,
    max_root_samples: int = -1,
    json_out: Path | None = None,
    summary_out: Path | None = None,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> int:
    """Shared validation logic for ``validate`` and ``build --validate-after-build``.

    Applies the canonical Routebind optimal-subgraph validation:
    - Layer 1: structural support/depth/field algebra on every sample.
    - Layer 2: Bellman-optimal product-state recomputation on every sample
      (requires ``oracle_ctx`` populated from the manifest's DAG reference).
    """
    root = root.resolve()

    # Load manifest
    try:
        manifest = read_manifest(root)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1

    # Determine splits
    all_splits = list(manifest.get("n_samples", {}).keys())
    if splits is None:
        splits = all_splits

    n_obs = manifest.get("n_observations", 0)
    topo_vocab_size = manifest.get(
        "topology_observation_vocabulary_size", n_obs
    )
    gs = manifest.get("field_decay_spatial", 0.9848)
    gw = manifest.get("field_decay_semantic", 0.8)
    canvas_width = manifest.get("canvas_width", None)

    # Determine S
    S = manifest.get("num_spatial_slots", 0) or manifest.get("n_states", 0)
    if S <= 0:
        ext = manifest.get("storage_extent") or manifest.get("extent", [])
        if len(ext) == 2:
            S = int(ext[0]) * int(ext[1])
        else:
            for split in splits:
                arrays = load_split_arrays(root, split)
                if arrays is not None and "cell_type" in arrays:
                    S = int(arrays["cell_type"].shape[1])
                    break
    if S <= 0:
        typer.echo(
            "Error: cannot determine corpus spatial dimension S.", err=True
        )
        raise typer.Exit(code=1)

    # Corpus-level validation
    _, corpus_issues = validate_corpus_root(
        root, max_root_samples=max_root_samples
    )
    all_issues = list(corpus_issues)

    # ── Target semantics check ────────────────────────────────────────
    stage_params = manifest.get("stage_params", {})
    target_semantics = stage_params.get("target_semantics", "")
    target_schema_version = stage_params.get("target_schema_version", 0)
    expected_semantics = "optimal_subgraph_support"
    expected_schema_version = 1

    if target_semantics != expected_semantics:
        all_issues.append(
            ValidationIssue(
                severity="ERROR",
                code="target_semantics_manifest_mismatch",
                message=(
                    f"Target semantics {target_semantics!r} "
                    f"(version {target_schema_version}) "
                    f"does not match expected "
                    f"{expected_semantics!r} (v{expected_schema_version}). "
                    f"This validator supports only the optimal-subgraph contract."
                ),
            )
        )
        # Abort — cannot validate with wrong contract
        return _emit_validation_results(
            all_issues, {}, output_dir, json_out, summary_out
        )

    # ── Load semantic DAG for oracle recomputation ────────────────────
    oracle_ctx: OracleValidationContext | None = None
    parents = manifest.get("parents", {})
    sem_ref = parents.get("semantic_graph", {})
    dag_root_str = sem_ref.get("root", "")
    dag_graph_id = sem_ref.get("artifact_id", "")
    if dag_root_str and dag_graph_id:
        try:
            dag_root = Path(dag_root_str)
            if not dag_root.is_absolute():
                dag_root = (
                    root.parent.parent.parent.parent / dag_root
                ).resolve()
            dagflow_manifest = read_manifest(dag_root)
            dagflow_max_out_degree = dagflow_manifest.get("max_out_degree", 4)
            _, dagflow_sample = find_artifact_by_id(dag_root, dag_graph_id)
            validate_dagflow_layout_sample(dagflow_sample)
            n_actual = int(dagflow_sample["node_mask"].sum())
            pub_adjacency: list[list[int]] = [[] for _ in range(n_actual)]
            for rank in range(n_actual):
                src_pub = int(dagflow_sample["node_obs_id"][rank])
                for k in range(dagflow_max_out_degree):
                    if dagflow_sample["successor_mask"][rank, k]:
                        succ_pub = int(
                            dagflow_sample["successor_indices"][rank, k]
                        )
                        if 0 <= succ_pub < n_actual:
                            pub_adjacency[src_pub].append(succ_pub)

            from ehc_sn.tasks.routebind.oracle import _build_dag_csr

            max_out = max((len(s) for s in pub_adjacency), default=0)
            pub_succ_mask = np.zeros((n_actual, max_out), dtype=bool)
            pub_succ_indices = np.full((n_actual, max_out), -1, dtype=np.int32)
            for src_pub in range(n_actual):
                for ki, dst_pub in enumerate(pub_adjacency[src_pub]):
                    pub_succ_mask[src_pub, ki] = True
                    pub_succ_indices[src_pub, ki] = np.int32(dst_pub)

            pred_offsets, pred_nodes = _build_dag_csr(pub_adjacency, n_actual)

            oracle_ctx = OracleValidationContext(
                physical_neighbors=np.empty((0, 4), dtype=np.int32),
                node_at_position=np.empty(0, dtype=np.int32),
                pred_offsets=pred_offsets,
                pred_nodes=pred_nodes,
                pub_succ_mask=pub_succ_mask,
                pub_succ_indices=pub_succ_indices,
                n_slots=S,
                n_obs=n_actual,
            )
        except Exception as e:
            all_issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="dag_parent_unavailable",
                    message=f"Cannot load semantic DAG parent: {e}",
                )
            )
    else:
        all_issues.append(
            ValidationIssue(
                severity="INFO",
                code="dag_parent_not_configured",
                message="No semantic_graph parent reference found in manifest",
            )
        )

    # Pre-allocate oracle workspace
    n_obs_valid = (
        n_obs
        if n_obs > 0
        else (oracle_ctx.n_obs if oracle_ctx is not None else 1)
    )
    n_states = S * n_obs_valid
    _dist = np.full(n_states, np.iinfo(np.int32).max, dtype=np.int32)
    _pkind = np.zeros(n_states, dtype=np.int8)
    _pnext = np.full(n_states, -1, dtype=np.int32)
    _optct = np.zeros(n_states, dtype=np.uint8)
    _deque = np.zeros(2 * n_states, dtype=np.int32)
    _oracle_workspace = dict(
        distance=_dist,
        policy_kind=_pkind,
        policy_next=_pnext,
        opt_count=_optct,
        deque_buf=_deque,
    )

    # ── Per-sample validation ─────────────────────────────────────────
    sample_counts: dict[str, int] = {}
    for split in splits:
        arrays = load_split_arrays(root, split)
        if arrays is None:
            sample_counts[split] = 0
            continue
        n = next(iter(arrays.values())).shape[0]
        sample_counts[split] = n
        n_check = min(n, max_samples) if max_samples > 0 else n

        for idx in range(n_check):
            sample = {
                ch: arrays[ch][idx]
                for ch in manifest.get("channels", [])
                if ch in arrays
            }

            struct_issues = validate_stored_sample(
                sample,
                n_obs=n_obs,
                topo_vocab_size=topo_vocab_size,
                S=S,
                gamma_space=gs,
                gamma_semantic=gw,
                grid_width=canvas_width,
            )
            for si in struct_issues:
                si.split = split
                si.sample_index = idx
            all_issues.extend(struct_issues)

            # Layer 1: structural support/depth/field algebra
            support_issues = validate_support_channels(
                sample,
                S=S,
                gamma_space=gs,
                gamma_semantic=gw,
                split=split,
                idx=idx,
            )
            for si in support_issues:
                si.split = split
                si.sample_index = idx
            all_issues.extend(support_issues)

            # Layer 2: optional oracle recomputation
            if (
                oracle_ctx is not None
                and oracle_ctx.physical_neighbors.size > 0
            ):
                oracle_issues = check_oracle_optimal_subgraph(
                    sample,
                    oracle_ctx,
                    gamma_space=gs,
                    gamma_semantic=gw,
                    split=split,
                    idx=idx,
                    _workspace=_oracle_workspace,
                )
                for oi in oracle_issues:
                    oi.split = split
                    oi.sample_index = idx
                all_issues.extend(oracle_issues)

    # ── Statistics and reports ────────────────────────────────────────
    stats = compute_corpus_statistics(
        root, manifest, splits, max_samples, grid_width=canvas_width
    )
    return _emit_validation_results(
        all_issues, sample_counts, output_dir, json_out, summary_out, stats
    )


# =============================================================================
# Inspect
# =============================================================================


@app.command("inspect")
def inspect(
    root: Annotated[
        Path,
        typer.Argument(
            help="Versioned root to inspect "
            "(e.g. data/processed/routebind/default/v1)."
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
            help="Render task_overview_routebind for the selected sample.",
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
            "longest_route, shortest_route, most_waypoints, "
            "most_goal_occurrences, largest_spatial_detour.",
        ),
    ] = "stratified",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Output directory for figures and reports.",
        ),
    ] = Path("outputs/routebind-inspection"),
    figure_seed: Annotated[
        int | None,
        typer.Option("--figure-seed", help="RNG seed for figure selection."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Write inspection JSON to path."),
    ] = None,
    show_dag_distances: Annotated[
        bool,
        typer.Option(
            "--show-dag-distances",
            help="Print DAG semantic distance distribution from corpus manifest.",
        ),
    ] = False,
) -> None:
    """Inspect a Routebind corpus — metadata, samples, diagnostics, figures."""
    root = root.resolve()

    try:
        manifest = read_manifest(root)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    all_splits = list(manifest.get("n_samples", {}).keys())

    # ── DAG distance distribution ─────────────────────────────────────
    if show_dag_distances:
        stage = manifest.get("stage_params", {})
        diam = stage.get("semantic_distance_diameter")
        if diam is None:
            typer.echo(
                "DAG distance distribution not available — "
                "this corpus predates semantic-distance precomputation."
            )
            raise typer.Exit(code=0)
        n_obs = manifest.get("stage_params", {}).get("n_observations", "?")
        typer.echo(
            f"DAG semantic distance distribution\n"
            f"  observation vocabulary: {n_obs} nodes\n"
            f"  diameter: {diam} edges\n"
            f"  (Full histogram can be regenerated from the original\n"
            f"   dagflow artifact via ``build-dagflow.py inspect``)."
        )
        raise typer.Exit(code=0)

    # ── Summary ────────────────────────────────────────────────────────────
    if summary:
        cw = manifest.get("canvas_width", None)
        stats = compute_corpus_statistics(
            root, manifest, all_splits, grid_width=cw
        )
        typer.echo("=" * 72)
        typer.echo("Routebind corpus summary")
        typer.echo("=" * 72)
        for k in ("task", "corpus", "version", "n_observations", "grid"):
            if k in stats:
                typer.echo(f"  {k}: {stats[k]}")
        typer.echo("")
        typer.echo("Samples:")
        for s, c in stats.get("per_split", {}).items():
            typer.echo(f"  {s}: {c}")
        typer.echo("")
        rl = stats.get("route_length", {})
        if rl.get("count", 0) > 0:
            typer.echo(
                f"  Route length: min={rl['min']:.0f} "
                f"median={rl['median']:.0f} max={rl['max']:.0f}"
            )
        sl = stats.get("semantic_length", {})
        if sl.get("count", 0) > 0:
            typer.echo(
                f"  Semantic length: min={sl['min']:.0f} "
                f"median={sl['median']:.0f} max={sl['max']:.0f}"
            )

    # ── Single sample inspection ─────────────────────────────────────────
    if sample_index is not None:
        sample = load_sample(root, split, sample_index)
        cw_inspect = manifest.get("canvas_width", None)
        ins = prepare_sample_inspection(
            sample, split=split, index=sample_index, grid_width=cw_inspect
        )
        start_obs = (
            int(ins.observation_id[ins.start_position])
            if ins.start_position >= 0
            else -1
        )
        typer.echo(f"\nSample {sample_index} ({split}):")
        typer.echo(f"  Start: pos {ins.start_position} / obs {start_obs}")
        typer.echo(f"  Goal obs: {ins.goal_observation}")
        typer.echo(f"  Goal occurrences: {len(ins.goal_positions)}")
        typer.echo(f"  Route length: {ins.route_length}")
        typer.echo(f"  Waypoints: {ins.semantic_length}")
        obs_chain = " → ".join(str(obs) for _, obs, _ in ins.waypoint_events)
        if obs_chain:
            typer.echo(f"  Waypoint seq: {obs_chain}")
        typer.echo(f"  Next direction: {ins.next_direction}")
        typer.echo(f"  Next observation: {ins.next_observation}")
        typer.echo(f"  Wall density: {ins.wall_density:.1%}")
        if ins.warnings:
            typer.echo(f"  Warnings ({len(ins.warnings)}):")
            for w in ins.warnings:
                typer.echo(f"    {w}")

        if sample_figure:
            output_dir.mkdir(parents=True, exist_ok=True)
            fpath = _render_overview_figure(
                sample, output_dir, split, sample_index, manifest=manifest
            )
            typer.echo(f"  Figure: {fpath}")

    # ── Sample gallery ────────────────────────────────────────────────────
    if gallery > 0:
        cw = manifest.get("canvas_width", None)
        selected = select_samples(
            root,
            all_splits,
            policy=selection,
            n=gallery,
            seed=figure_seed,
            grid_width=cw,
        )
        for s, idx in selected:
            sample = load_sample(root, s, idx)
            output_dir.mkdir(parents=True, exist_ok=True)
            fpath = _render_overview_figure(
                sample, output_dir, s, idx, manifest=manifest
            )
            typer.echo(f"  Figure: {fpath}")

    if show:
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except ImportError:
            pass

    if json_out is not None:
        cw = manifest.get("canvas_width", None)
        stats = compute_corpus_statistics(
            root, manifest, all_splits, grid_width=cw
        )
        json_out.write_text(json.dumps(stats, indent=2, default=str))


# =============================================================================
# Figure helpers
# =============================================================================


def _build_trace_for_sample(
    sample: dict[str, np.ndarray],
    *,
    n_observations: int | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> TraceTree:
    """Build a minimal TraceTree with routebind meta keys from a corpus sample.

    Args:
        sample: Channel dict for one corpus sample.
        n_observations: Corpus-wide observation vocabulary size.
        canvas_width: Grid width in cells.
        canvas_height: Grid height in cells.
    """
    trace = TraceTree()
    ct = np.asarray(sample["cell_type"], dtype=np.int32)
    oid = np.asarray(sample["observation_id"], dtype=np.int32)
    sf = np.asarray(sample["start_flag"], dtype=bool)
    gf = np.asarray(sample["goal_flag"], dtype=bool)
    tf = np.asarray(sample["target_trajectory"], dtype=np.float32)
    tw = np.asarray(sample["target_waypoint"], dtype=np.float32)
    cm = np.asarray(sample["spatial_mask"], dtype=bool)

    meta: dict[str, object] = {
        "routebind": {
            "cell_type": np.expand_dims(ct, 0),
            "observation_id": np.expand_dims(oid, 0),
            "start_flag": np.expand_dims(sf, 0),
            "goal_flag": np.expand_dims(gf, 0),
            "target_trajectory": np.expand_dims(tf, 0),
            "target_waypoint": np.expand_dims(tw, 0),
            "cell_mask": np.expand_dims(cm, 0),
        }
    }

    # Attach waypoint support channels for direct waypoint extraction
    # (figure selector falls back to greedy-route extraction when absent).
    wp_sup = sample.get("waypoint_support")
    wp_sd = sample.get("waypoint_semantic_depth")
    if wp_sup is not None:
        meta["routebind"]["waypoint_support"] = np.expand_dims(  # type: ignore[index]
            np.asarray(wp_sup, dtype=bool), 0
        )
    if wp_sd is not None:
        meta["routebind"]["waypoint_semantic_depth"] = np.expand_dims(  # type: ignore[index]
            np.asarray(wp_sd, dtype=np.int16), 0
        )
    if n_observations is not None:
        meta["routebind"]["n_observations"] = n_observations  # type: ignore[index]
    if canvas_width is not None:
        meta["routebind"]["canvas_width"] = canvas_width  # type: ignore[index]
    if canvas_height is not None:
        meta["routebind"]["canvas_height"] = canvas_height  # type: ignore[index]

    trace.attach_meta(meta)
    return trace


def _render_overview_figure(
    sample: dict[str, np.ndarray],
    output_dir: Path,
    split: str,
    index: int,
    *,
    manifest: dict[str, object] | None = None,
) -> Path:
    """Render task_overview_routebind for one sample via the registry.

    Args:
        sample: Channel dict for one corpus sample.
        output_dir: Output directory.
        split: Dataset split label.
        index: Sample index within the split.
        manifest: Optional corpus manifest; when provided, canonical
            task-schema metadata (n_observations, canvas dimensions)
            are attached to the trace so the figure selector uses
            authoritative values.
    """
    kb = _build_trace_for_sample(
        sample,
        n_observations=manifest.get("n_observations") if manifest else None,  # type: ignore[arg-type]
        canvas_width=manifest.get("canvas_width") if manifest else None,  # type: ignore[arg-type]
        canvas_height=manifest.get("canvas_height") if manifest else None,  # type: ignore[arg-type]
    )
    ctx = FigureContext(sample_idx=0)
    fig = render("task_overview_routebind", kb, ctx)
    fname = f"task_overview_routebind_{split}_{index}.png"
    fpath = output_dir / fname
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return fpath


# =============================================================================
if __name__ == "__main__":
    app()

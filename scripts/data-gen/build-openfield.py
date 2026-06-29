"""CLI for building openfield layout interim assets.

Openfield generates square and rectangle grid worlds with random sensory
assignments.  The output is an interim layout dataset consumed by task
builders (build-arena.py, build-routebind.py).

Commands
--------
build       Produce a complete openfield layout dataset.
validate    Validate a version root's manifest, channels, and data.
inspect     Print a human-readable summary of a version root manifest.


Default paths
-------------
Raw source specs:  data/raw/openfield/{preset}/v{version}
Layout dataset:    data/interim/openfield/{preset}/v{version}

Prerequisites
-------------
None — layouts are generated procedurally.

Examples
--------
Default faithful TEM reproduction (16 square grids)::

    python build-openfield.py build

Rectangle grids::

    python build-openfield.py build --preset tem-rectangle

Small smoke test::

    python build-openfield.py build --preset small

Custom widths::

    python build-openfield.py build --widths 10 --widths 10 --widths 11

Inspect an existing root::

    python build-openfield.py inspect data/interim/openfield/tem-square/v1

"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer

from ehc_sn.data.layout import SpatialLayout
from ehc_sn.data.layout.io import write_layout_dataset
from ehc_sn.data.layout.observation_placement import (
    ObservationPlacementConfig,
)
from ehc_sn.data.layout.openfield import (
    OPENFIELD_PRESETS,
    enrich_layout_with_sensory,
)
from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.data.source.openfield import (
    expand_openfield_source_specs,
    generate_openfield_source_specs,
)

# ---------------------------------------------------------------------------
_DEFAULT_RAW_ROOT = Path("data/raw/openfield")
_DEFAULT_INTERIM_ROOT = Path("data/interim/openfield")
_DEFAULT_VERSION = 1
_DEFAULT_PRESET = "tem-square"
_DEFAULT_S_SIZE = 45
_DEFAULT_N_SENSORY_INSTANCES = 1
_DEFAULT_TOPOLOGY_SEED = 42
_DEFAULT_N_TRAIN = 250
_DEFAULT_N_VAL = 10
_DEFAULT_N_TEST = 10


app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
@app.command("build")
def build(  # -----------------------------------------------------------------
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help=f"Named source preset ({', '.join(OPENFIELD_PRESETS)}).",
        ),
    ] = _DEFAULT_PRESET,
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
    height: Annotated[
        int | None,
        typer.Option("--height", help="Grid height override."),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option("--width", help="Grid width override."),
    ] = None,
    widths: Annotated[
        list[int] | None,
        typer.Option(
            "--widths", help="Grid widths (repeat for multiple values)."
        ),
    ] = None,
    s_size: Annotated[
        int,
        typer.Option(
            "--s-size",
            help="Sensory vocabulary size (deprecated; use --observation-vocabulary-size).",
        ),
    ] = _DEFAULT_S_SIZE,
    observation_vocabulary_size: Annotated[
        int | None,
        typer.Option(
            "--observation-vocabulary-size",
            help=f"Observation vocabulary size (default: {_DEFAULT_S_SIZE}).",
        ),
    ] = None,
    n_sensory_instances: Annotated[
        int,
        typer.Option(
            "--n-sensory-instances",
            help="Number of randomized sensory assignments per topology.",
        ),
    ] = _DEFAULT_N_SENSORY_INSTANCES,
    topology_seed: Annotated[
        int,
        typer.Option(
            "--topology-seed",
            help=f"Seed controlling topology generation (default: {_DEFAULT_TOPOLOGY_SEED}).",
        ),
    ] = _DEFAULT_TOPOLOGY_SEED,
    version: Annotated[
        int,
        typer.Option(
            "--version",
            help=f"Version of the emitted dataset (default: {_DEFAULT_VERSION}).",
        ),
    ] = _DEFAULT_VERSION,
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root",
            help=f"Root path for raw source specs (default: {_DEFAULT_RAW_ROOT}).",
        ),
    ] = _DEFAULT_RAW_ROOT,
    interim_root: Annotated[
        Path,
        typer.Option(
            "--interim-root",
            help=f"Root path for interim files (default: {_DEFAULT_INTERIM_ROOT}).",
        ),
    ] = _DEFAULT_INTERIM_ROOT,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Delete the existing version root before building, if present.",
        ),
    ] = False,
    observation_policy: Annotated[
        str,
        typer.Option(
            "--observation-policy",
            help="Observation placement policy: dense_uniform, exactly_one, or bounded.",
        ),
    ] = "dense_uniform",
    observation_max_occurrences: Annotated[
        int,
        typer.Option(
            "--observation-max-occurrences",
            help="Max occurrences per obs ID (for bounded policy).",
        ),
    ] = 1,
) -> None:
    """Produce a complete openfield layout dataset (topology + layouts).

    Chains the two internal stages into one command.
    """
    if observation_vocabulary_size is not None:
        s_size = observation_vocabulary_size
    elif s_size != _DEFAULT_S_SIZE:
        typer.echo(
            "Warning: --s-size is deprecated, use --observation-vocabulary-size instead.",
            err=True,
        )

    dest = interim_root / preset / f"v{version}"
    if dest.exists():
        if force:
            typer.echo(f"Warning: --force set; deleting existing root: {dest}")
            shutil.rmtree(dest)
        else:
            typer.echo(
                f"Error: version root already exists (dataset roots are "
                f"immutable): {dest}\n"
                f"Use --force to delete and rebuild, or bump --version.",
                err=True,
            )
            raise typer.Exit(code=1)

    _generate_topology(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        widths=widths,
        topology_seed=topology_seed,
        version=version,
        raw_root=raw_root,
    )
    _materialize_layouts(
        preset=preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        s_size=s_size,
        n_sensory_instances=n_sensory_instances,
        topology_seed=topology_seed,
        version=version,
        raw_root=raw_root,
        interim_root=interim_root,
        observation_policy=observation_policy,
        observation_max_occurrences=observation_max_occurrences,
    )


# =============================================================================
@app.command("validate")
def validate(  # --------------------------------------------------------------
    root: Annotated[
        Path,
        typer.Argument(help="Openfield version root to validate."),
    ],
) -> None:
    """Validate manifest.json, index.jsonl, NPZ files, and dataset_class constraints.

    Supports both ``source_spec`` and ``layout_dataset`` roots.
    """
    manifest = validate_version_root(root.resolve())

    typer.echo(f"OK  {root}")
    typer.echo(f"    dataset_class : {manifest['dataset_class']}")
    typer.echo(f"    family        : {manifest['family']}")
    typer.echo(f"    version       : {manifest['version']}")
    if manifest["dataset_class"] == "source_spec":
        typer.echo(f"    preset        : {manifest.get('preset', '')}")
        typer.echo(
            f"    spec_schema   : {manifest.get('spec_schema_version', '')}"
        )
    typer.echo(f"    n_samples     : {manifest['n_samples']}")


# =============================================================================
@app.command("inspect")
def inspect_command(  # -------------------------------------------------------
    root: Annotated[
        Path,
        typer.Argument(help="Openfield version root to inspect."),
    ],
) -> None:
    """Print a human-readable summary of a version root manifest."""
    manifest = validate_version_root(root.resolve())

    typer.echo(f"Root: {root}")
    typer.echo(f"  dataset_class : {manifest.get('dataset_class', '?')}")
    typer.echo(f"  family        : {manifest.get('family', '?')}")
    typer.echo(f"  version       : {manifest.get('version', '?')}")
    typer.echo(f"  preset        : {manifest.get('preset', '?')}")
    typer.echo(f"  n_samples     : {manifest.get('n_samples', {})}")
    typer.echo(f"  channels      : {manifest.get('channels', [])}")

    if manifest.get("dataset_class") == "source_spec":
        typer.echo(
            f"  spec_schema   : {manifest.get('spec_schema_version', '')}"
        )

    stage_params = manifest.get("stage_params", {})
    topology_type = stage_params.get("topology_type")
    if topology_type:
        typer.echo(f"  topology_type : {topology_type}")

    extent = manifest.get("extent")
    if extent:
        typer.echo(f"  extent        : {extent}")


# =============================================================================
# Helpers
# =============================================================================


def _resolve_preset(preset: str) -> dict:
    try:
        return dict(OPENFIELD_PRESETS[preset])
    except KeyError:
        typer.echo(
            f"Error: unknown preset '{preset}'. "
            f"Choose from: {', '.join(OPENFIELD_PRESETS)}.",
            err=True,
        )
        raise typer.Exit(code=1)


def _check_widths_conflict(
    widths: list[int] | None, height: int | None, width: int | None
) -> None:
    """Exit with error when --widths is combined with --height or --width."""
    if widths is not None and (height is not None or width is not None):
        typer.echo(
            "Error: --widths cannot be combined with --height or --width. "
            "Use --widths for batch-shape mode, or --height/--width for a single shape.",
            err=True,
        )
        raise typer.Exit(code=1)


def _apply_shape_overrides(
    cfg: dict,
    height: int | None,
    width: int | None,
    widths: list[int] | None,
) -> dict:
    """Return a copy of *cfg* with shape fields overridden by CLI args.

    - ``--widths`` replaces ``cfg["widths"]`` entirely (batch mode).
    - ``--height`` and/or ``--width`` create a single-element widths list.
    - When neither is given, returns an unmodified copy.
    """
    result = dict(cfg)
    if widths is not None:
        result["widths"] = list(widths)
        result.pop("heights", None)
    elif height is not None or width is not None:
        w = width if width is not None else cfg["widths"][0]
        h = (
            height
            if height is not None
            else (cfg.get("heights") or cfg["widths"])[0]
        )
        result["widths"] = [w]
        result["heights"] = [h]
    return result


# =============================================================================
def _generate_topology(  # ----------------------------------------------------
    preset: str = _DEFAULT_PRESET,
    n_train: int = _DEFAULT_N_TRAIN,
    n_val: int = _DEFAULT_N_VAL,
    n_test: int = _DEFAULT_N_TEST,
    height: int | None = None,
    width: int | None = None,
    widths: list[int] | None = None,
    topology_seed: int = _DEFAULT_TOPOLOGY_SEED,
    version: int = _DEFAULT_VERSION,
    raw_root: Path = _DEFAULT_RAW_ROOT,
) -> None:
    """Internal stage: produce source-spec JSONL topology records (raw stage)."""
    _resolve_preset(preset)
    _check_widths_conflict(widths, height, width)

    cfg = _apply_shape_overrides(
        OPENFIELD_PRESETS[preset], height, width, widths
    )

    raw_leaf = raw_root / preset / f"v{version}"
    if raw_leaf.exists():
        print(f"Source specs already exist at {raw_leaf}, skipping.")
        return

    print(
        f"Generating source specs for preset '{preset}' "
        f"({n_train} train + {n_val} val + {n_test} test records)..."
    )
    generate_openfield_source_specs(
        preset,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        topology_seed=topology_seed,
        version=version,
        raw_root=raw_root,
        _overridden_widths=cfg.get("widths"),
        _overridden_heights=cfg.get("heights"),
    )

    print("Done — source specs written.")


# =============================================================================
def _materialize_layouts(  # ---------------------------------------------------
    preset: str = _DEFAULT_PRESET,
    n_train: int = _DEFAULT_N_TRAIN,
    n_val: int = _DEFAULT_N_VAL,
    n_test: int = _DEFAULT_N_TEST,
    s_size: int = _DEFAULT_S_SIZE,
    observation_vocabulary_size: int | None = None,
    n_sensory_instances: int = _DEFAULT_N_SENSORY_INSTANCES,
    topology_seed: int = _DEFAULT_TOPOLOGY_SEED,
    version: int = _DEFAULT_VERSION,
    raw_root: Path = _DEFAULT_RAW_ROOT,
    interim_root: Path = _DEFAULT_INTERIM_ROOT,
    observation_policy: str = "dense_uniform",
    observation_max_occurrences: int = 1,
) -> None:
    """Internal stage: expand source specs, assign sensory IDs, write SpatialLayout records."""
    if observation_vocabulary_size is not None:
        s_size = observation_vocabulary_size
    _resolve_preset(preset)

    raw_leaf = raw_root / preset / f"v{version}"
    if not raw_leaf.exists():
        typer.echo(
            f"Error: source specs not found at {raw_leaf}. "
            f"Run generate-topology first.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Expand all source specs into topology-only layouts once.
    all_topo = expand_openfield_source_specs(
        raw_root,
        preset=preset,
        version=version,
        topology_seed=topology_seed,
    )
    # Build lookup: example_id -> layout.
    topo_by_id: dict[str, SpatialLayout] = {}
    for tl in all_topo:
        lid = tl["layout_id"]
        if lid.endswith("-topo-only"):
            topo_by_id[lid[: -len("-topo-only")]] = tl
        else:
            topo_by_id[lid] = tl

    layouts: list[SpatialLayout] = []

    for split_name, split_n in [
        ("train", n_train),
        ("val", n_val),
        ("test", n_test),
    ]:
        if split_n == 0:
            continue
        spec_file = raw_leaf / split_name / "specs.jsonl"
        if not spec_file.exists():
            raise FileNotFoundError(
                f"Source spec file not found: {spec_file}. "
                f"Run generate-topology with --n-{split_name} >= {split_n}."
            )

        spec_lines = [
            l.strip()
            for l in spec_file.read_text().strip().split("\n")
            if l.strip()
        ]
        if len(spec_lines) < split_n:
            raise ValueError(
                f"Source spec pool for '{split_name}' has {len(spec_lines)} records, "
                f"but --n-{split_name}={split_n}. "
                f"Run generate-topology with a larger --n-{split_name}."
            )

        selected = spec_lines[:split_n]
        for spec_line in selected:
            spec = json.loads(spec_line)
            tl = topo_by_id.get(spec["example_id"])
            if tl is None:
                typer.echo(
                    f"Error: no topology layout found for "
                    f"example_id={spec['example_id']!r}.",
                    err=True,
                )
                raise typer.Exit(code=1)
            for inst_idx in range(n_sensory_instances):
                sensory_seed = topology_seed + spec["seed_offset"] + inst_idx
                obs_placement_cfg = None
                if observation_policy != "dense_uniform":
                    obs_placement_cfg = ObservationPlacementConfig(
                        policy=observation_policy,
                        max_occurrences=observation_max_occurrences,
                    )
                enriched = enrich_layout_with_sensory(
                    tl,
                    observation_vocabulary_size=s_size,
                    observation_seed=sensory_seed,
                    observation_placement=obs_placement_cfg,
                )
                # split is preserved by enrich_layout_with_sensory.
                layouts.append(enriched)

    if not layouts:
        typer.echo(
            "Error: no layouts produced (all split counts are zero?).", err=True
        )
        raise typer.Exit(code=1)

    cfg = OPENFIELD_PRESETS[preset]
    topology_type = cfg["topology_type"]
    materialized_leaf = interim_root / preset / f"v{version}"

    # Derive declared extent from the preset configuration.
    widths_list: list[int] = cfg.get("widths", [30])
    heights_list: list[int] | None = cfg.get("heights")
    extent_w = max(widths_list)
    if heights_list:
        extent_h = max(heights_list)
    else:
        extent_h = extent_w  # square: height == width

    write_layout_dataset(
        layouts,
        materialized_leaf,
        topology_type=topology_type,
        s_size=s_size,
        topology_seed=topology_seed,
        version=version,
        layout_family="openfield",
        preset=preset,
        extent=[extent_h, extent_w],
    )
    print("Done.")


# =============================================================================
if __name__ == "__main__":
    app()

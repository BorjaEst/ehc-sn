"""TensorBoard logger configuration and scalar export utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Optional

from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field
from tensorboard.backend.event_processing import event_accumulator

EVENT_FILE_PREFIX = "events.out.tfevents"
_SIZE_GUIDANCE = {event_accumulator.SCALARS: 0}


# =============================================================================
class LoggerSettings(BaseModel, extra="forbid"):
    """Logging settings for TensorBoard logger."""

    save_dir: Path = Field(
        default=Path("./logs"),
        description="Directory to save logs (default: ./logs).",
    )
    name: Optional[str] = Field(
        default=None,
        description="Experiment name for logger.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version/run identifier (auto-increments if None).",
    )
    log_graph: bool = Field(
        default=False,
        description="Log model graph to TensorBoard.",
    )
    prefix: str = Field(
        default="",
        description="Prefix for all logged metrics.",
    )


# =============================================================================
class Logger(TensorBoardLogger):
    """Custom TensorBoard logger that accepts LoggerSettings."""

    def __init__(self, settings: LoggerSettings):
        super().__init__(**settings.model_dump())


# =============================================================================
@dataclass(frozen=True)
class TensorBoardRun:
    """Resolved TensorBoard run metadata."""

    experiment: str | None
    run: str
    run_dir: Path
    relative_run_dir: Path
    hparams_path: Path | None


# =============================================================================
@dataclass(frozen=True)
class TensorBoardExportReport:
    """Counts returned after an export completes."""

    n_runs: int
    n_tags: int
    n_points: int


# =============================================================================
def discover_tensorboard_runs(  # ---------------------------------------------
    search_root: Path,
) -> list[TensorBoardRun]:
    """Return all TensorBoard runs reachable from ``search_root``."""

    normalized_root = _normalize_search_root(search_root)
    if _is_run_dir(normalized_root):
        run_dirs = [normalized_root]
    else:
        run_dirs = sorted({event_file.parent for event_file in normalized_root.rglob(f"{EVENT_FILE_PREFIX}*")})

    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard event files were found beneath '{normalized_root}'.")

    runs: list[TensorBoardRun] = []
    for run_dir in run_dirs:
        relative_run_dir = _relative_run_dir(normalized_root, run_dir)
        hparams_path = run_dir / "hparams.yaml"
        runs.append(
            TensorBoardRun(
                experiment=_infer_experiment_name(normalized_root, run_dir, relative_run_dir),
                run=run_dir.name,
                run_dir=run_dir,
                relative_run_dir=relative_run_dir,
                hparams_path=hparams_path if hparams_path.is_file() else None,
            )
        )
    return runs


# =============================================================================
def export_tensorboard_scalars(  # --------------------------------------------
    search_root: Path,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    tags: Collection[str] | None = None,
    sample_points: int = 16,
    include_hparams: bool = True,
) -> TensorBoardExportReport:
    """Export TensorBoard scalar series to JSONL plus an optional summary JSON."""

    if sample_points < 1:
        raise ValueError("sample_points must be at least 1.")

    runs = discover_tensorboard_runs(search_root)
    requested_tags = set(tags) if tags is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tags = 0
    total_points = 0
    summary_runs: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as output_file:
        for run in runs:
            accumulator = event_accumulator.EventAccumulator(str(run.run_dir), size_guidance=_SIZE_GUIDANCE)
            accumulator.Reload()
            scalar_tags = sorted(accumulator.Tags().get("scalars", []))
            if requested_tags is not None:
                scalar_tags = [tag for tag in scalar_tags if tag in requested_tags]

            run_summary = {
                "experiment": run.experiment,
                "run": run.run,
                "run_dir": str(run.run_dir),
                "relative_run_dir": str(run.relative_run_dir),
                "hparams_path": None if run.hparams_path is None else str(run.hparams_path),
                "scalars": [],
            }
            run_point_count = 0

            for tag in scalar_tags:
                events = accumulator.Scalars(tag)
                if not events:
                    continue

                total_tags += 1
                total_points += len(events)
                run_point_count += len(events)

                for event in events:
                    row = {
                        "experiment": run.experiment,
                        "run": run.run,
                        "run_dir": str(run.run_dir),
                        "relative_run_dir": str(run.relative_run_dir),
                        "tag": tag,
                        "step": int(event.step),
                        "value": float(event.value),
                        "wall_time": float(event.wall_time),
                    }
                    output_file.write(json.dumps(row, sort_keys=True) + "\n")

                run_summary["scalars"].append(_summarize_events(tag, events, sample_points=sample_points))

            if summary_path is not None:
                run_summary["n_tags"] = len(run_summary["scalars"])
                run_summary["n_points"] = run_point_count
                if include_hparams:
                    run_summary["hparams"] = _read_hparams(run.hparams_path)
                summary_runs.append(run_summary)

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_document = {
            "source_root": str(_normalize_search_root(search_root)),
            "n_runs": len(runs),
            "n_tags": total_tags,
            "n_points": total_points,
            "runs": summary_runs,
        }
        summary_path.write_text(json.dumps(summary_document, indent=2, sort_keys=True), encoding="utf-8")

    return TensorBoardExportReport(n_runs=len(runs), n_tags=total_tags, n_points=total_points)


# =============================================================================
def _normalize_search_root(  # ------------------------------------------------
    search_root: Path,
) -> Path:
    """Return a directory search root from a supported input path."""

    resolved = search_root.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"TensorBoard log path '{resolved}' does not exist.")
    if resolved.is_file():
        if resolved.name.startswith(EVENT_FILE_PREFIX):
            return resolved.parent
        raise ValueError(f"Unsupported file input '{resolved}'. Expected an event file or a directory.")
    return resolved


# =============================================================================
def _is_run_dir(  # -----------------------------------------------------------
    path: Path,
) -> bool:
    """Return ``True`` when ``path`` contains TensorBoard event files."""

    return path.is_dir() and any(child.is_file() and child.name.startswith(EVENT_FILE_PREFIX) for child in path.iterdir())


# =============================================================================
def _relative_run_dir(  # -----------------------------------------------------
    search_root: Path,
    run_dir: Path,
) -> Path:
    """Return a stable run-relative path for export metadata."""

    if search_root == run_dir:
        return Path(run_dir.name)
    return run_dir.relative_to(search_root)


# =============================================================================
def _infer_experiment_name(  # ------------------------------------------------
    search_root: Path,
    run_dir: Path,
    relative_run_dir: Path,
) -> str | None:
    """Infer a useful experiment label from the scanned directory layout."""

    parts = relative_run_dir.parts
    if len(parts) >= 2:
        return parts[0]
    if search_root == run_dir:
        return run_dir.parent.name or None
    return search_root.name or None


# =============================================================================
def _summarize_events(  # -----------------------------------------------------
    tag: str,
    events: list[Any],
    *,
    sample_points: int,
) -> dict[str, Any]:
    """Build a compact summary for a single scalar series."""

    values = [float(event.value) for event in events]
    points = [{"step": int(event.step), "value": float(event.value)} for event in events]
    return {
        "tag": tag,
        "n_points": len(events),
        "first_step": points[0]["step"],
        "last_step": points[-1]["step"],
        "last_value": points[-1]["value"],
        "min_value": min(values),
        "max_value": max(values),
        "sample": _sample_points(points, sample_points),
    }


# =============================================================================
def _sample_points(  # --------------------------------------------------------
    points: list[dict[str, float | int]],
    sample_points: int,
) -> list[dict[str, float | int]]:
    """Return at most ``sample_points`` evenly spaced points from ``points``."""

    if len(points) <= sample_points:
        return points
    if sample_points == 1:
        return [points[-1]]

    last_index = len(points) - 1
    indices = {round(index * last_index / (sample_points - 1)) for index in range(sample_points)}
    return [points[index] for index in sorted(indices)]


# =============================================================================
def _read_hparams(  # ---------------------------------------------------------
    hparams_path: Path | None,
) -> Any | None:
    """Return best-effort parsed Lightning hparams metadata."""

    if hparams_path is None:
        return None

    contents = hparams_path.read_text(encoding="utf-8").strip()
    if not contents:
        return {}

    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return {"_raw_yaml": contents}

    parsed = yaml.safe_load(contents)
    return {} if parsed is None else parsed


# =============================================================================
__all__ = [
    "LoggerSettings",
    "Logger",
    "TensorBoardRun",
    "TensorBoardExportReport",
    "discover_tensorboard_runs",
    "export_tensorboard_scalars",
]

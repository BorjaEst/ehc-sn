"""Collect and persist one eval-owned figure bundle from checkpoint + provider."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ehc_sn.eval.figure_bundle import (
    FigureBundleExecutorArtifact,
    collect_regime_figure_bundle_from_artifact,
)


class FigureBundleCollectionSettings(BaseModel, extra="forbid"):
    """TOML settings for one figure-bundle collection run."""

    artifact: FigureBundleExecutorArtifact = Field(
        ...,
        description="Typed executor artifact metadata for family-aware loading.",
    )
    provider_ref: str = Field(
        ...,
        min_length=1,
        description="Dotted provider class path using provider_settings kwargs.",
    )
    provider_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword settings forwarded to the provider constructor.",
    )
    run_dir: Path = Field(
        ...,
        description="Output run directory for bundle artifacts.",
    )
    regime_id: str = Field(
        default="figure_collection",
        min_length=1,
        description="Identifier stored in the persisted bundle.",
    )
    regime_kind: Literal["diagnostic", "benchmark"] = Field(
        default="diagnostic",
        description="Top-level regime namespace kind.",
    )
    max_batches: int = Field(
        default=0,
        ge=0,
        description="Max provider batches; 0 means provider default/all.",
    )
    trace_keys: list[str] = Field(
        default_factory=list,
        description="Optional explicit semantic trace keys to request.",
    )
    figures: list[str] = Field(
        default_factory=list,
        description="Optional figure names used to infer required trace keys.",
    )
    trigger_kind: str = Field(
        default="manual",
        description="Run trigger label persisted in the bundle manifest.",
    )
    epoch: int = Field(default=0, ge=0, description="Persisted epoch metadata.")
    step: int = Field(default=0, ge=0, description="Persisted step metadata.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for figure-bundle collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML settings for one collection run.",
    )
    return parser.parse_args()


def main() -> None:
    """Load TOML settings and collect one figure-ready persisted bundle."""
    args = parse_args()
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    settings = FigureBundleCollectionSettings.model_validate(raw)

    regime_result = collect_regime_figure_bundle_from_artifact(
        artifact=settings.artifact,
        provider_ref=settings.provider_ref,
        provider_settings=settings.provider_settings,
        run_dir=settings.run_dir,
        regime_id=settings.regime_id,
        regime_kind=settings.regime_kind,
        max_batches=settings.max_batches,
        trace_keys=settings.trace_keys,
        figure_names=settings.figures,
        trigger_kind=settings.trigger_kind,
        epoch=settings.epoch,
        step=settings.step,
        write_legacy_compat=True,
    )

    summary = {
        "run_dir": str(settings.run_dir),
        "regime_id": settings.regime_id,
        "regime_kind": settings.regime_kind,
        "n_cases": len(regime_result.case_results),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

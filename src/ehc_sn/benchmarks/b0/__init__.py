"""B0 MazeHard bridge benchmark package."""

from pathlib import Path

from ehc_sn.benchmarks._bindings import resolve_binding
from ehc_sn.benchmarks._capabilities import BatchPredicts
from ehc_sn.benchmarks.b0.evaluator import B0Benchmark, B0BenchmarkConfig
from ehc_sn.benchmarks.b0.manifest import B0Manifest


# =================================================================================================
def build_b0_predictor(  # ------------------------------------------------------------------------
    model_kind: str,
    *,
    model_config_path: Path,
    checkpoint_path: Path | None,
    compute_budget: int,
    device: str,
) -> BatchPredicts:
    """Return a predictor instance for the given B0 benchmark model kind."""
    factory = resolve_binding(
        model_family=model_kind,
        capability="batch_prediction",
    )
    return factory(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        compute_budget=compute_budget,
        device=device,
    )


# =================================================================================================
__all__ = [
    "B0Benchmark", "B0BenchmarkConfig", "B0Manifest", "build_b0_predictor",
]  # fmt: skip

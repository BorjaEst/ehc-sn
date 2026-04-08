"""M0 Episodic Memory Bridge benchmark package."""

from pathlib import Path

from ehc_sn.benchmarks._bindings import resolve_binding
from ehc_sn.benchmarks._capabilities import EpisodicMemoryAgent
from ehc_sn.benchmarks.m0.evaluator import M0Benchmark, M0BenchmarkConfig
from ehc_sn.benchmarks.m0.manifest import M0Manifest


# =================================================================================================
def build_m0_agent(  # ----------------------------------------------------------------------------
    model_kind: str,
    *,
    model_config_path: Path,
    checkpoint_path: Path | None,
    device: str,
) -> EpisodicMemoryAgent:
    factory = resolve_binding(
        model_family=model_kind,
        capability="episodic_memory",
    )
    return factory(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )


# =================================================================================================
__all__ = [
    "M0Benchmark", "M0BenchmarkConfig", "M0Manifest", "build_m0_agent",
]  # fmt: skip

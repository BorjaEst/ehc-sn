"""TEM benchmark binding family registration surface."""

from ehc_sn.benchmarks._bindings.registry import register_binding
from ehc_sn.benchmarks._bindings.tem.episodic_memory import (
	build_tem_v1_episodic_memory,
	build_tem_v2_episodic_memory,
)

register_binding(
	model_family="tem_v1",
	capability="episodic_memory",
	factory=build_tem_v1_episodic_memory,
)

register_binding(
	model_family="tem_v2",
	capability="episodic_memory",
	factory=build_tem_v2_episodic_memory,
)

__all__ = ["build_tem_v1_episodic_memory", "build_tem_v2_episodic_memory"]

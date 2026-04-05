"""HRM benchmark binding family registration surface."""

from ehc_sn.benchmarks._bindings.hrm.batch_prediction import (
    build_hrm_v1_batch_prediction,
    build_hrm_v2_batch_prediction,
)
from ehc_sn.benchmarks._bindings.registry import register_binding

register_binding(
    model_family="hrm_v1",
    capability="batch_prediction",
    factory=build_hrm_v1_batch_prediction,
)

register_binding(
    model_family="hrm_v2",
    capability="batch_prediction",
    factory=build_hrm_v2_batch_prediction,
)

__all__ = ["build_hrm_v1_batch_prediction", "build_hrm_v2_batch_prediction"]

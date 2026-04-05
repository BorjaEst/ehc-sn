"""Internal model-aware benchmark bindings."""

from ehc_sn.benchmarks._bindings.bootstrap import bootstrap_bindings
from ehc_sn.benchmarks._bindings.registry import BindingFactory, resolve_binding

bootstrap_bindings()

__all__ = ["BindingFactory", "resolve_binding"]

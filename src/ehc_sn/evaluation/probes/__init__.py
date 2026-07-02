"""Evaluation probes — compact derived evidence computed from dense traces.

Probes are the middle layer between raw dense traces and report figures.
They consume hidden-state tensors transiently during evaluation and output
small summary arrays (NPZ + JSON) that can be rendered without loading the
full traces.
"""

from ehc_sn.evaluation.probes.working_memory import (
    ProbeArtifact,
    compute_and_persist_probes_for_artifact,
    compute_pfc_path_memory_probe,
    compute_pfc_path_memory_probe_from_artifact,
    load_probe_artifact,
    save_probe_artifact,
)

__all__ = [
    "ProbeArtifact",
    "compute_and_persist_probes_for_artifact",
    "compute_pfc_path_memory_probe",
    "compute_pfc_path_memory_probe_from_artifact",
    "load_probe_artifact",
    "save_probe_artifact",
]

"""Public exports for the TEM hippocampal memory subsystem.

The package exposes the concrete HPC implementations, their configuration
models, recurrent state carriers, phase-transition payloads, and typed
read-operator contracts used by TEM model code.
"""

from ehc_sn.modules.hpc._base import HPCState, HPCTransition, HPCTransitionResult, SensoryRead, SensoryReadResult, WritePayload
from ehc_sn.modules.hpc.modules import HPCAttention, HPCAttentionSettings, HPCAttractor, HPCAttractorSettings
from ehc_sn.modules.hpc.query_policy import CueRead, ReadCues, TargetRead

# =================================================================================================
__all__ = [
    "HPCAttention", "HPCAttentionSettings", "HPCAttractor", "HPCAttractorSettings",
    "HPCState", "HPCTransition", "HPCTransitionResult",
    "SensoryRead", "SensoryReadResult", "WritePayload", "ReadCues", "CueRead", "TargetRead",
]  # fmt: skip

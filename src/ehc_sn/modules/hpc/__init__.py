from __future__ import annotations

"""HPC (hippocampus) memory and retrieval.

This component exposes two explicit hippocampal memory implementations used by
the TEM family:

- ``HPCAttractor``: dense Hebbian memory with iterative attractor retrieval.
- ``HPCAttention``: explicit episodic memory with masked softmax retrieval.

Both classes share a common TEM-facing contract defined by ``HPCBase`` and the
state/step dataclasses exported from this package.
"""

from ehc_sn.modules.hpc._base import (
    HPCSensoryStepInput,
    HPCSensoryStepOutput,
    HPCState,
    HPCStepInput,
    HPCStepOutput,
)
from ehc_sn.modules.hpc.attention import HPCAttention, HPCAttentionSettings
from ehc_sn.modules.hpc.attractor import HPCAttractor, HPCAttractorSettings

__all__ = [
    "HPCAttractor",
    "HPCAttractorSettings",
    "HPCAttention",
    "HPCAttentionSettings",
    "HPCSensoryStepInput",
    "HPCSensoryStepOutput",
    "HPCState",
    "HPCStepInput",
    "HPCStepOutput",
]

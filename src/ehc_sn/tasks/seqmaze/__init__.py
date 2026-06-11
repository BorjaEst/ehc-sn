"""SeqMaze task family — transition-graph edge-lookup probe.

SeqMaze tests whether a model can reason about sequences and transitions
without any spatial grounding.  Phase 0 is the edge-lookup probe that
validates successor-index embedding viability.

Phase 0 surface:

- :mod:`~ehc_sn.tasks.seqmaze.contracts` — task input/output contracts.
- :mod:`~ehc_sn.tasks.seqmaze.evaluation` — edge-prediction score report.
- :mod:`~ehc_sn.tasks.seqmaze.runtime` — batch key constants and extraction helpers.
- :mod:`~ehc_sn.tasks.seqmaze._graph_utils` — DAG generation and permutation utilities.
- :mod:`~ehc_sn.tasks.seqmaze._data` — on-the-fly probe dataset.
"""

from .contracts import (
    SeqMazeProbeInput,
    SeqMazeProbeOutput,
    SeqMazeProbeTargets,
)
from .evaluation import SeqMazeProbeScoreReport
from .runtime import (
    SEQUENCE_BATCH_KEYS,
    extract_seqmaze_probe_input,
    extract_seqmaze_probe_targets,
)

__all__ = [
    "SeqMazeProbeInput",
    "SeqMazeProbeTargets",
    "SeqMazeProbeOutput",
    "SeqMazeProbeScoreReport",
    "SEQUENCE_BATCH_KEYS",
    "extract_seqmaze_probe_input",
    "extract_seqmaze_probe_targets",
]

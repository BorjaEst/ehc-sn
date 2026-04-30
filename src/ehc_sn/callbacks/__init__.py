"""Public callback surface for EHC-SN.

Provides the two stable Lightning callbacks:

- :class:`EvaluationRegimesCallback` / :class:`EvaluationRegimesCallbackSettings` —
  named out-of-band evaluation regimes, separate from fit-path validation.
- :class:`FiguresCallback` / :class:`FigureCallbackSettings` —
  periodic figure generation, optionally backed by a named regime artifact.
"""

from ehc_sn.callbacks.eval_regimes import EvaluationRegimesCallback, EvaluationRegimesCallbackSettings
from ehc_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback

__all__ = [
    "EvaluationRegimesCallback",
    "EvaluationRegimesCallbackSettings",
    "FigureCallbackSettings",
    "FiguresCallback",
]

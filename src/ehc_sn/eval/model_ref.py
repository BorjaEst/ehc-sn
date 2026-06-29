"""Model reference value type — lightweight URI/path abstraction.

A ``ModelRef`` wraps a user-supplied string and classifies it by scheme
so downstream code (model loader, checkpoint resolver) can dispatch
appropriately.

Currently only the ``local`` scheme is implemented; ``mlflow-model`` and
``mlflow-run-artifact`` raise ``NotImplementedError`` at resolution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRef:
    """A model artifact reference with scheme-based dispatch.

    Attributes
    ----------
    value:
        The raw user-supplied string — a local path or a URI.
    """

    value: str

    @property
    def scheme(self) -> str:
        """Detect the reference scheme from the value prefix.

        Returns ``"mlflow-model"`` for ``models:/`` URIs,
        ``"mlflow-run-artifact"`` for ``runs:/`` URIs,
        ``"local"`` for everything else.
        """
        if self.value.startswith("models:/"):
            return "mlflow-model"
        if self.value.startswith("runs:/"):
            return "mlflow-run-artifact"
        return "local"

    def resolve_checkpoint_path(self) -> Path:
        """Resolve this reference to an absolute checkpoint path.

        For the ``local`` scheme the value is treated as a filesystem path.
        MLflow schemes raise ``NotImplementedError`` until the model-artifact
        loader contract is designed.
        """
        if self.scheme == "local":
            return Path(self.value).resolve()
        raise NotImplementedError(
            f"Model ref scheme {self.scheme!r} is not yet supported. "
            f"Use a local file path for now."
        )


__all__ = [
    "ModelRef",
]

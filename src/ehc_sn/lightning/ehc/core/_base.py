"""Private shared helpers for the EHC v1 unified training surface."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Sequence, Union

import torch
from pydantic import BaseModel, Field, model_validator
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationTraceRequest,
)
from ehc_sn.models.ehc.ehc_v1 import EHCModelV1, ModelSettingsV1
from ehc_sn.types import Batch

# EHC v1 unified training surface.  See EHCV1TrainingModel for details.
_SPATIAL_CORE_NAMES = ("lec", "mec", "hpc", "lec_to_hpc", "mec_to_hpc")
"""EHCModelV1 submodule name prefixes for the spatial core, which includes all
"""

_CONTROLLER_BRIDGE_NAMES = ("pfc_to_hpc", "hpc_to_pfc")
"""EHCModelV1 submodule name prefixes for the controller bridge, which includes 
all modules directly bridging pfc and hpc.
"""


# Mapping from public semantic group name to module-name prefixes on EHCModelV1.
_GROUP_TO_PREFIXES: dict[str, tuple[str, ...]] = {
    "spatial_core": _SPATIAL_CORE_NAMES,
    "controller_bridge": _CONTROLLER_BRIDGE_NAMES,
    "controller_heads": ("pfc.estimator", "str"),
}
"""Public semantic group names and their corresponding EHCModelV1 submodule 
prefixes.
"""

VALID_INIT_GROUPS: frozenset[str] = frozenset(_GROUP_TO_PREFIXES)
"""Set of valid semantic group names for weight initialization."""


# =============================================================================
class EHCMode(str, Enum):
    spatial_pretrain = "spatial_pretrain"
    reason_pretrain = "reason_pretrain"


# =============================================================================
class EHCRegime(Protocol):
    """Protocol for EHC v1 training regimes, which are discriminated by EHCMode."""

    def setup(  # -------------------------------------------------------------
        self,
        stage: str | None,
    ) -> None:
        """Setup method called by Lightning at the start of fit/validate/test/predict."""

    def configure_optimizers(  # ----------------------------------------------
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """Configure optimizers and schedulers for the regime."""

    def on_train_epoch_start(  # ----------------------------------------------
        self,
    ) -> None:
        """Hook called by Lightning at the start of each training epoch."""

    def on_validation_epoch_start(  # -----------------------------------------
        self,
    ) -> None:
        """Hook called by Lightning at the start of each validation epoch."""

    def training_step(  # -----------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict:
        """Training step method called by Lightning for each training batch."""

    def validation_step(  # ---------------------------------------------------
        self,
        batch: Batch,
        batch_idx: int,
    ) -> dict:
        """Validation step method called by Lightning for each validation batch."""

    def on_validation_epoch_end(  # -------------------------------------------
        self,
    ) -> None:
        """Hook called by Lightning at the end of each validation epoch."""

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider-owned replay evaluation case."""


# =============================================================================
def resolve_spatial_core_ids(  # ----------------------------------------------
    model: EHCModelV1,
) -> set[int]:
    """Return parameter IDs for the spatial core group."""
    return {
        id(p)
        for name in _SPATIAL_CORE_NAMES
        for p in getattr(model, name).parameters()
    }


# =============================================================================
def resolve_controller_bridge_ids(  # -----------------------------------------
    model: EHCModelV1,
) -> set[int]:
    """Return parameter IDs for the controller bridge group (pfc_to_hpc, hpc_to_pfc)."""
    return {
        id(p)
        for name in _CONTROLLER_BRIDGE_NAMES
        for p in getattr(model, name).parameters()
    }


# =============================================================================
def resolve_controller_heads_ids(  # ------------------------------------------
    model: EHCModelV1,
) -> set[int]:
    """Return parameter IDs for the controller heads group (pfc.estimator, str)."""
    return {id(p) for p in model.pfc.estimator.parameters()} | {
        id(p) for p in model.str.parameters()
    }


# =============================================================================
def freeze_params(  # ---------------------------------------------------------
    model: EHCModelV1,
    *names: str,
) -> None:
    """Set requires_grad=False on the named model submodules."""
    for name in names:
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)


# =============================================================================
def load_weights_from_checkpoint(  # ------------------------------------------
    model: EHCModelV1,
    checkpoint_path: str | Path,
    groups: Sequence[str],
) -> list[str]:
    """Hydrate named semantic-group weights from a checkpoint into *model*.

    Loads only the parameter subsets for the specified groups.  Optimizer,
    scheduler, and trainer-progress state in the checkpoint are not touched.
    This is the separate weight-initialization contract, distinct from
    ``Trainer.fit(ckpt_path=...)`` which restores full training state.

    Lightning checkpoint keys are expected to be prefixed with ``"model."``
    (e.g. ``"model.lec.weight"``); that prefix is stripped before matching.
    Raw model checkpoints without the prefix are also supported.

    Args:
        model: Target ``EHCModelV1`` to hydrate in place.
        checkpoint_path: Path to a Lightning or raw-model checkpoint.
        groups: Non-empty sequence of semantic group names.  Valid values:
            ``"spatial_core"``, ``"controller_bridge"``, ``"controller_heads"``.

    Returns:
        Sorted list of model state-dict keys that were loaded from the checkpoint.

    Raises:
        ValueError: If *groups* is empty, contains unknown names, or no keys
            matched the requested groups in the checkpoint.
    """
    if not groups:
        raise ValueError("groups must be non-empty.")
    unknown = [g for g in groups if g not in _GROUP_TO_PREFIXES]
    if unknown:
        raise ValueError(
            f"Unknown semantic groups: {unknown!r}. "
            f"Valid groups: {sorted(_GROUP_TO_PREFIXES)}."
        )

    prefixes = tuple(p for g in groups for p in _GROUP_TO_PREFIXES[g])

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd: dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    # Strip Lightning's "model." wrapper prefix so keys match EHCModelV1 directly.
    stripped: dict = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in raw_sd.items()
    }

    filtered = {
        k: v
        for k, v in stripped.items()
        if any(k == p or k.startswith(f"{p}.") for p in prefixes)
    }
    if not filtered:
        raise ValueError(
            f"No keys matched for groups {list(groups)!r} "
            f"in checkpoint {str(checkpoint_path)!r}. "
            f"Available top-level prefixes: "
            f"{sorted({k.split('.')[0] for k in stripped})!r}."
        )

    model.load_state_dict(filtered, strict=False)
    return sorted(filtered)


# =============================================================================
__all__ = [
    "EHCMode",
    "EHCRegime",
    "VALID_INIT_GROUPS",
    "resolve_spatial_core_ids",
    "resolve_controller_bridge_ids",
    "resolve_controller_heads_ids",
    "freeze_params",
    "load_weights_from_checkpoint",
]

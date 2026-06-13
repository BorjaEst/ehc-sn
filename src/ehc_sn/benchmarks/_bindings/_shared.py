"""Shared internal helpers for model-comparison benchmark bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ehc_sn.benchmarks.contracts import ArtifactManifest, TrackRecipe
from ehc_sn.models.ehp.ehp_v1 import EHCModelV1
from ehc_sn.models.ehp.ehp_v1 import ModelSettingsV1 as EHCModelSettingsV1
from ehc_sn.models.hrm.hrm_v1 import HRModelV1
from ehc_sn.models.hrm.hrm_v1 import ModelSettingsV1 as HRMModelSettingsV1
from ehc_sn.models.hrm.hrm_v2 import HRModelV2
from ehc_sn.models.hrm.hrm_v2 import ModelSettingsV2 as HRMModelSettingsV2
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1 as TEMModelSettingsV1
from ehc_sn.models.tem.tem_v1 import TEMModelV1
from ehc_sn.models.tem.tem_v2 import ModelSettingsV2 as TEMModelSettingsV2
from ehc_sn.models.tem.tem_v2 import TEMModelV2
from ehc_sn.training.ehp import load_weights_from_checkpoint as load_ehc_weights
from ehc_sn.training.hrm import load_weights_from_checkpoint as load_hrm_weights
from ehc_sn.training.tem import load_weights_from_checkpoint as load_tem_weights


# =============================================================================
def normalize_model_family(model_family: str) -> str:
    """Normalize a model family id used in benchmark recipe/manifests."""
    return model_family.strip().lower()


# =============================================================================
def binding_config(recipe: TrackRecipe, model_family: str) -> dict[str, Any]:
    """Return one model-family binding subtable as a mutable mapping."""
    try:
        raw = recipe.bindings[model_family].model_dump()
    except KeyError as exc:
        raise ValueError(
            "TrackRecipe.bindings has no entry for model_family "
            f"{model_family!r}."
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            "TrackRecipe.bindings entries must decode to mapping objects."
        )
    return raw


# =============================================================================
def load_frozen_model(
    model_family: str,
    manifest: ArtifactManifest,
) -> tuple[object, Callable[..., list[str]], list[str]]:
    """Instantiate model core, hydrate init-only weights, and freeze it."""
    model_config_path = Path(manifest.model_config_path)
    checkpoint_path = Path(manifest.checkpoint_path)

    if model_family == "tem-v1":
        settings = TEMModelSettingsV1.from_config(model_config_path)
        model = TEMModelV1(settings)
        loaded_keys = load_tem_weights(model, checkpoint_path, ("all",))
        freeze_model(model)
        return model, load_tem_weights, loaded_keys

    if model_family == "tem-v2":
        settings = TEMModelSettingsV2.from_config(model_config_path)
        model = TEMModelV2(settings)
        loaded_keys = load_tem_weights(model, checkpoint_path, ("all",))
        freeze_model(model)
        return model, load_tem_weights, loaded_keys

    if model_family == "hrm-v1":
        settings = HRMModelSettingsV1.from_config(model_config_path)
        model = HRModelV1(settings)
        loaded_keys = load_hrm_weights(model, checkpoint_path, ("all",))
        freeze_model(model)
        return model, load_hrm_weights, loaded_keys

    if model_family == "hrm-v2":
        settings = HRMModelSettingsV2.from_config(model_config_path)
        model = HRModelV2(settings)
        loaded_keys = load_hrm_weights(model, checkpoint_path, ("all",))
        freeze_model(model)
        return model, load_hrm_weights, loaded_keys

    if model_family == "ehp-v1":
        settings = EHCModelSettingsV1.from_config(model_config_path)
        model = EHCModelV1(settings)
        loaded_keys = load_ehc_weights(
            model,
            checkpoint_path,
            ("spatial_core", "controller_bridge", "controller_heads"),
        )
        freeze_model(model)
        return model, load_ehc_weights, loaded_keys

    raise ValueError(f"Unsupported model family: {model_family!r}.")


# =============================================================================
def freeze_model(model: object) -> None:
    """Freeze all model parameters and set eval mode when available."""
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad_(False)
    if hasattr(model, "eval"):
        model.eval()


# =============================================================================
def bridge_only_parameter_groups(
    model: object,
    bridge: object,
) -> tuple[dict[str, Any], ...]:
    """Return bridge-only optimizer groups excluding frozen core parameters."""
    if not hasattr(model, "parameters") or not hasattr(bridge, "parameters"):
        return ()
    core_ids = {id(param) for param in model.parameters()}
    bridge_only = [
        param
        for param in bridge.parameters()
        if param.requires_grad and id(param) not in core_ids
    ]
    if not bridge_only:
        return ()
    return ({"name": "bridge", "params": bridge_only},)


# =============================================================================
__all__ = [
    "binding_config",
    "bridge_only_parameter_groups",
    "freeze_model",
    "load_frozen_model",
    "normalize_model_family",
]

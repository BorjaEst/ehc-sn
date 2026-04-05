"""HRM model-loading helpers for benchmark bindings."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import torch

from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettings_V1
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettings_V2


def _resolve_checkpoint_state_dict(  # ------------------------------------------------------------
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load and normalize one checkpoint payload to a plain HRM state dict."""
    loaded = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(loaded).__name__}.")

    state_dict = loaded["state_dict"] if "state_dict" in loaded else loaded
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a valid state_dict mapping.")

    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}

    return state_dict


def load_hrm_v1_model(  # ------------------------------------------------------------------------
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> HRModelV1:
    """Instantiate HRM v1 and optionally hydrate it from a checkpoint."""
    resolved_model_config = Path(model_config_path)
    config = ModelSettings_V1.model_validate(tomllib.load(resolved_model_config.open("rb")))
    model = HRModelV1(config)

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        state_dict = _resolve_checkpoint_state_dict(resolved_checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to load HRM v1 checkpoint from {resolved_checkpoint}. "
                "The checkpoint is not compatible with the current HRModelV1 surface."
            ) from exc

    return model.to(device).eval()


def load_hrm_v2_model(  # ------------------------------------------------------------------------
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> HRModelV2:
    """Instantiate HRM v2 and optionally hydrate it from a checkpoint."""
    resolved_model_config = Path(model_config_path)
    config = ModelSettings_V2.model_validate(tomllib.load(resolved_model_config.open("rb")))
    model = HRModelV2(config)

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        state_dict = _resolve_checkpoint_state_dict(resolved_checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to load HRM v2 checkpoint from {resolved_checkpoint}. "
                "The checkpoint is not compatible with the current HRModelV2 surface."
            ) from exc

    return model.to(device).eval()


__all__ = ["load_hrm_v1_model", "load_hrm_v2_model"]

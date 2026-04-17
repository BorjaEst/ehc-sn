"""TEM model-loading helpers for benchmark bindings."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import torch

from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1
from ehc_sn.models.tem.tem_v2 import ModelSettingsV2, TEMModelV2


def _resolve_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    """Load and normalize one checkpoint payload to a plain TEM state dict."""
    loaded = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(loaded).__name__}.")

    state_dict = loaded["state_dict"] if "state_dict" in loaded else loaded
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a valid state_dict mapping.")

    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}

    return state_dict


def load_tem_v1_model(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> TEMModelV1:
    """Instantiate TEM v1 and optionally hydrate it from a checkpoint."""
    resolved_model_config = Path(model_config_path)
    config = ModelSettingsV1.model_validate(tomllib.load(resolved_model_config.open("rb")))
    model = TEMModelV1(config)

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        state_dict = _resolve_checkpoint_state_dict(resolved_checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to load TEM v1 checkpoint from {resolved_checkpoint}. "
                "The checkpoint is not compatible with the current TEMModelV1 surface."
            ) from exc

    return model.to(device).eval()


def load_tem_v2_model(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> TEMModelV2:
    """Instantiate TEM v2 and optionally hydrate it from a checkpoint."""
    resolved_model_config = Path(model_config_path)
    config = ModelSettingsV2.model_validate(tomllib.load(resolved_model_config.open("rb")))
    model = TEMModelV2(config)

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        state_dict = _resolve_checkpoint_state_dict(resolved_checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to load TEM v2 checkpoint from {resolved_checkpoint}. "
                "The checkpoint is not compatible with the current TEMModelV2 surface."
            ) from exc

    return model.to(device).eval()


__all__ = ["load_tem_v1_model", "load_tem_v2_model"]

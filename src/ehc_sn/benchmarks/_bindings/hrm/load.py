"""HRM model-loading helpers for benchmark bindings."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import torch

from ehc_sn.adapters.maze_hard.bridges.hrm.hrm_v1 import MazeHardHRMV1AdapterSettings, MazeHardHRMV1BridgeAdapter
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2


def _resolve_checkpoint_state_dict(  # ------------------------------------------------------------
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load one checkpoint payload to a plain state-dict mapping."""
    loaded = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(loaded).__name__}.")

    state_dict = loaded["state_dict"] if "state_dict" in loaded else loaded
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a valid state_dict mapping.")

    return state_dict


def _extract_prefixed_state_dict(state_dict: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    """Return a state dict filtered to keys that start with one prefix."""
    filtered = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
    return filtered or None


def _load_hrm_v1_adapter_settings(model_config_path: str | Path) -> MazeHardHRMV1AdapterSettings:
    """Resolve the minimal MazeHard bridge settings required at benchmark time."""
    raw_config = tomllib.load(Path(model_config_path).open("rb"))
    if "vocab_size" not in raw_config:
        raise ValueError(
            "HRM v1 benchmark binding requires model config to provide top-level 'vocab_size' "
            "until adapter settings are split into an explicit benchmark composition config."
        )
    return MazeHardHRMV1AdapterSettings(
        vocab_size=int(raw_config["vocab_size"]),
        encoder_kind=str(raw_config.get("encoder_kind", "learned")),
    )


def load_hrm_v1_model(  # ------------------------------------------------------------------------
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> HRModelV1:
    """Instantiate HRM v1 and optionally hydrate it from a checkpoint."""
    config = ModelSettingsV1.from_config(model_config_path)
    model = HRModelV1(config)

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        state_dict = _resolve_checkpoint_state_dict(resolved_checkpoint)
        state_dict = _extract_prefixed_state_dict(state_dict, "model.") or state_dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to load HRM v1 checkpoint from {resolved_checkpoint}. "
                "The checkpoint is not compatible with the current HRModelV1 surface."
            ) from exc

    return model.to(device).eval()


def load_hrm_v1_bridge_adapter(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> MazeHardHRMV1BridgeAdapter:
    """Instantiate the benchmark-time HRM v1 bridge adapter and optional checkpoint."""
    model = load_hrm_v1_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return (
        MazeHardHRMV1BridgeAdapter(
            model,
            _load_hrm_v1_adapter_settings(model_config_path),
        )
        .to(device)
        .eval()
    )


def load_hrm_v2_model(  # ------------------------------------------------------------------------
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> HRModelV2:
    """Instantiate HRM v2 and optionally hydrate it from a checkpoint."""
    resolved_model_config = Path(model_config_path)
    config = ModelSettingsV2.model_validate(tomllib.load(resolved_model_config.open("rb")))
    model = HRModelV2(config)

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        state_dict = _resolve_checkpoint_state_dict(resolved_checkpoint)
        state_dict = _extract_prefixed_state_dict(state_dict, "model.") or state_dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to load HRM v2 checkpoint from {resolved_checkpoint}. "
                "The checkpoint is not compatible with the current HRModelV2 surface."
            ) from exc

    return model.to(device).eval()


__all__ = ["load_hrm_v1_bridge_adapter", "load_hrm_v1_model", "load_hrm_v2_model"]

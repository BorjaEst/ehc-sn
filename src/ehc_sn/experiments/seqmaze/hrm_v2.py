"""SeqMaze + HRM v2 experiment (supervised path-prediction).

Composition:
    model: HRModelV2
    adapter: SeqMazeHRMV2BridgeAdapter
    module: SeqMazeModule (pure supervised)
"""

from __future__ import annotations

from pathlib import Path

from ehc_sn.adapters.hrm import SeqMazeAdapterSettings
from ehc_sn.lightning.modules.seqmaze import SeqMazeModule
from ehc_sn.models.hrm.hrm_v2 import ModelSettingsV2


def build_experiment(config: dict) -> SeqMazeModule:
    """Parse config and return an instantiated SeqMazeModule.

    Args:
        config: Flat dict with keys:
            - model_config_path: str
            - n_max: int
            - t_max: int
            - k_max: int
            - hidden_size: int
            - edge_encoding: str
            - lr: float

    Returns:
        Instantiated SeqMazeModule.
    """
    model_settings = ModelSettingsV2.from_config(
        Path(config["model_config_path"])
    )
    adapter_settings = SeqMazeAdapterSettings(
        n_max=config["n_max"],
        t_max=config["t_max"],
        k_max=config["k_max"],
        hidden_size=config.get("hidden_size", model_settings.pfc.hidden_size),
        edge_encoding=config.get("edge_encoding", "successor_index_embedding"),
        share_path_position_embeddings=config.get(
            "share_path_position_embeddings", True
        ),
    )
    return SeqMazeModule(
        model_settings=model_settings,
        adapter_settings=adapter_settings,
        lr=config.get("lr", 1e-3),
    )


__all__ = ["build_experiment"]

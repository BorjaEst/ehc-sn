"""TEM episodic-memory benchmark binding module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from ehc_sn.benchmarks._bindings.tem.load import load_tem_v1_model, load_tem_v2_model
from ehc_sn.benchmarks._capabilities.episodic_memory import (
    CueFamily,
    EpisodicMemoryAgent,
    EpisodicMemoryQuery,
    EpisodicMemoryReadout,
    EpisodicMemoryStep,
)
from ehc_sn.models.tem.tem_v1 import TEMModelV1
from ehc_sn.models.tem.tem_v1 import TEMState as TEMStateV1
from ehc_sn.models.tem.tem_v2 import TEMModelV2
from ehc_sn.models.tem.tem_v2 import TEMState as TEMStateV2

CueKey: TypeAlias = tuple[CueFamily, int]
TEMModel: TypeAlias = TEMModelV1 | TEMModelV2
TEMRolloutState: TypeAlias = TEMStateV1 | TEMStateV2


@dataclass
class TEMEpisodicMemoryState:
    """Benchmark-time TEM state plus nonparametric latent readout banks."""

    tem_state: TEMRolloutState | None = None
    current_code: Tensor | None = None
    location_codes: dict[int, Tensor] = field(default_factory=dict)
    cue_codes: dict[CueKey, Tensor] = field(default_factory=dict)


class TEMEpisodicMemoryAdapter(EpisodicMemoryAgent):
    """Benchmark-time episodic-memory adapter for TEM model families.

    The adapter keeps the pure TEM recurrent state as the source of memory
    dynamics and maintains a lightweight nonparametric latent bank that maps
    TEM place codes back to benchmark-visible location ids and cue ids.
    """

    def __init__(self, model: TEMModel) -> None:
        self._model = model.eval()

    @property
    def model(self) -> TEMModel:
        """Return the wrapped pure TEM model."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Return the device on which the wrapped TEM model runs."""
        return next(self._model.parameters()).device

    def reset_state(self) -> TEMEpisodicMemoryState:
        """Return an empty benchmark-time TEM adapter state."""
        return TEMEpisodicMemoryState()

    def ingest_step(self, step: EpisodicMemoryStep, state: TEMEpisodicMemoryState) -> TEMEpisodicMemoryState:
        """Advance the TEM state and refresh the latent readout banks."""
        model_state = self._prepare_tem_state(step, state.tem_state)
        model_inputs = self._build_model_inputs(step)

        with torch.no_grad():
            model_state, _, _, _, place = self._model(model_inputs, model_state)

        place_post, _, _ = place
        current_code = self._flatten_multiscale_code(place_post)
        location_codes = dict(state.location_codes)
        cue_codes = dict(state.cue_codes)

        location_id = self._tensor_scalar(step.location_id)
        if location_id is not None:
            self._merge_code(location_codes, location_id, current_code)

        observation_id = self._tensor_scalar(step.observation_id)
        if observation_id is not None:
            self._merge_code(cue_codes, ("observation", observation_id), current_code)

        landmark_id = self._tensor_scalar(step.landmark_id)
        if landmark_id is not None and landmark_id > 0:
            self._merge_code(cue_codes, ("landmark", landmark_id), current_code)

        return TEMEpisodicMemoryState(
            tem_state=model_state,
            current_code=current_code,
            location_codes=location_codes,
            cue_codes=cue_codes,
        )

    def readout(self, query: EpisodicMemoryQuery, state: TEMEpisodicMemoryState) -> EpisodicMemoryReadout:
        """Return one benchmark-visible readout from the latent banks."""
        if query.kind == "current_location":
            location_id, score = self._nearest_location(state.current_code, state.location_codes)
            return EpisodicMemoryReadout(
                location_id=location_id,
                diagnostics={"bank_size": len(state.location_codes), "score": score},
            )

        cue_key = (query.cue_family, int(query.cue_id))
        cue_code = state.cue_codes.get(cue_key)
        location_id, score = self._nearest_location(cue_code, state.location_codes)
        return EpisodicMemoryReadout(
            location_id=location_id,
            diagnostics={
                "bank_size": len(state.location_codes),
                "cue_key": cue_key,
                "score": score,
            },
        )

    def _prepare_tem_state(
        self,
        step: EpisodicMemoryStep,
        state: TEMRolloutState | None,
    ) -> TEMRolloutState:
        """Return the TEM recurrent state to use for the next benchmark step."""
        if state is None:
            return self._model.init_state(batch_size=1, memory=None, device=self.device)

        if not bool(step.episode_start.view(-1)[0].item()):
            return state

        return self._model.init_state(batch_size=1, memory=state.hpc.memory, device=self.device)

    def _build_model_inputs(self, step: EpisodicMemoryStep) -> dict[str, Tensor]:
        """Move one benchmark step payload onto the wrapped model device."""
        model_inputs = {
            "observation": step.observation.to(device=self.device, dtype=torch.float32),
            "previous_action": step.previous_action.to(device=self.device, dtype=torch.int64),
            "episode_start": step.episode_start.to(device=self.device, dtype=torch.bool),
        }
        if step.landmark_id is not None:
            model_inputs["landmark_id"] = step.landmark_id.to(device=self.device, dtype=torch.int64)
        return model_inputs

    @staticmethod
    def _flatten_multiscale_code(code: list[Tensor]) -> Tensor:
        """Return one CPU resident flattened TEM place code."""
        flat = [part.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1) for part in code]
        return torch.cat(flat, dim=-1).squeeze(0)

    @staticmethod
    def _merge_code(bank: dict[int | CueKey, Tensor], key: int | CueKey, code: Tensor) -> None:
        """Update one latent bank entry by running mean over repeated visits."""
        previous = bank.get(key)
        bank[key] = code.clone() if previous is None else 0.5 * (previous + code)

    @staticmethod
    def _nearest_location(query: Tensor | None, bank: dict[int, Tensor]) -> tuple[int | None, float | None]:
        """Return the nearest stored location id for the provided latent code."""
        if query is None or not bank:
            return None, None

        normalized_query = F.normalize(query.view(1, -1), dim=-1)
        best_location = None
        best_score = None
        for location_id, candidate in bank.items():
            score = float(torch.sum(normalized_query * F.normalize(candidate.view(1, -1), dim=-1)).item())
            if best_score is None or score > best_score:
                best_location = location_id
                best_score = score
        return best_location, best_score

    @staticmethod
    def _tensor_scalar(value: Tensor | None) -> int | None:
        """Return one Python int from a single-item tensor, when present."""
        if value is None:
            return None
        return int(value.view(-1)[0].item())


def build_tem_v1_episodic_memory(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> TEMEpisodicMemoryAdapter:
    """Build the benchmark-time TEM v1 episodic-memory adapter."""
    model = load_tem_v1_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return TEMEpisodicMemoryAdapter(model)


def build_tem_v2_episodic_memory(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> TEMEpisodicMemoryAdapter:
    """Build the benchmark-time TEM v2 episodic-memory adapter."""
    model = load_tem_v2_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return TEMEpisodicMemoryAdapter(model)


__all__ = [
    "TEMEpisodicMemoryAdapter",
    "TEMEpisodicMemoryState",
    "build_tem_v1_episodic_memory",
    "build_tem_v2_episodic_memory",
]

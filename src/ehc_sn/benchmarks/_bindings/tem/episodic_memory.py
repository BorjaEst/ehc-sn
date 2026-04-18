"""TEM episodic-memory benchmark binding module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, Union

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
from ehc_sn.models.tem.tem_v1 import TEMInputV1, TEMModelV1, TEMStateV1
from ehc_sn.models.tem.tem_v2 import TEMInputV2, TEMModelV2, TEMStateV2
from ehc_sn.modules.autoencoder import TwoHotEncoder
from ehc_sn.types import MultiScaleCode

CueKey: TypeAlias = tuple[CueFamily, int]
TEMModel: TypeAlias = TEMModelV1 | TEMModelV2
TEMRolloutState: TypeAlias = TEMStateV1 | TEMStateV2
_TEMInput: TypeAlias = Union[TEMInputV1, TEMInputV2]


class _TwoHotReplicatedEncoder:
    """Adapter-side TwoHot encoder that produces a replicated MultiScaleCode.

    Encodes a flat observation tensor into a list of ``n_freq`` identical
    per-band tensors of shape ``(B, feature_dim)``.  This mirrors the default
    navigation bridge encoder strategy (``kind="two_hot"``,
    ``layout="replicated"``).
    """

    def __init__(self, observation_dim: int, feature_dim: int, n_freq: int, device: torch.device) -> None:
        self._encoder = TwoHotEncoder(observation_dim, feature_dim).to(device)
        self._n_freq = n_freq

    def encode(self, obs: Tensor) -> MultiScaleCode:
        """Return ``n_freq`` distinct encoded bands from a flat observation."""
        code = self._encoder(obs)
        return [code.clone() for _ in range(self._n_freq)]


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
        self._sensory_encoder: _TwoHotReplicatedEncoder | None = None

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
        tem_input = self._build_tem_input(step)

        with torch.no_grad():
            output, model_state = self._model(tem_input, model_state)

        current_code = self._extract_place_code(output.place_codes.inference)
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

    def _build_tem_input(self, step: EpisodicMemoryStep) -> _TEMInput:
        """Build a model-native TEMInput from one benchmark step.

        The observation is encoded into multiscale sensory codes using the
        adapter-owned TwoHot replicated encoder. The encoder is initialized
        lazily on first call, inferring ``observation_dim`` from the step.
        """
        obs = step.observation.to(device=self.device, dtype=torch.float32)
        if self._sensory_encoder is None:
            self._sensory_encoder = _TwoHotReplicatedEncoder(
                observation_dim=obs.shape[-1],
                feature_dim=self._model.config.lec.feature_dim,
                n_freq=self._model.lec.n_freq,
                device=self.device,
            )
        sensory_codes: MultiScaleCode = self._sensory_encoder.encode(obs)
        previous_action = step.previous_action.to(device=self.device, dtype=torch.int64)
        episode_start = step.episode_start.to(device=self.device, dtype=torch.bool)
        landmark_id = step.landmark_id.to(device=self.device, dtype=torch.int64) if step.landmark_id is not None else None
        if isinstance(self._model, TEMModelV1):
            return TEMInputV1(
                sensory_codes=sensory_codes,
                previous_action=previous_action,
                episode_start=episode_start,
                landmark_id=landmark_id,
            )
        return TEMInputV2(
            sensory_codes=sensory_codes,
            previous_action=previous_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )

    @staticmethod
    def _extract_place_code(code: MultiScaleCode) -> Tensor:
        """Return one CPU-resident flattened TEM place code from a multiscale pathway."""
        return torch.cat(code, dim=-1).detach().to(device="cpu", dtype=torch.float32).reshape(-1)

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
    """Build the benchmark-time TEM v1 episodic-memory adapter.

    The sensory encoder is initialized lazily on the first ``ingest_step`` call,
    inferring ``observation_dim`` from the first benchmark observation.

    Args:
        model_config_path: Path to the TEM v1 model TOML config.
        checkpoint_path: Optional checkpoint to hydrate the model.
        device: Target device string.
    """
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
    """Build the benchmark-time TEM v2 episodic-memory adapter.

    The sensory encoder is initialized lazily on the first ``ingest_step`` call,
    inferring ``observation_dim`` from the first benchmark observation.

    Args:
        model_config_path: Path to the TEM v2 model TOML config.
        checkpoint_path: Optional checkpoint to hydrate the model.
        device: Target device string.
    """
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

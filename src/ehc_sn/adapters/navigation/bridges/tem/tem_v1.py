"""Navigation plus TEM v1 bridge implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes
from ehc_sn.models.tem.tem_v1 import TEMInputV1, TEMModelV1, TEMOutputV1, TEMStateV1
from ehc_sn.modules.autoencoder import MLPDecoder, TwoHotEncoder
from ehc_sn.tasks.navigation.contracts import NavigationTaskOutput
from ehc_sn.tasks.navigation.runtime import coerce_navigation_step_input
from ehc_sn.types import Batch, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class NavigationEncoderConfig(BaseModel, extra="forbid"):
    """Encoder strategy config for the navigation-to-TEM sensory pathway.

    The adapter owns observation packing; LEC owns model-native multiscale dynamics.

    Attributes:
        kind: Encoder family.
            ``"two_hot"`` uses a fixed two-hot lookup table (default, no learnable
            parameters). ``"mlp"`` and ``"identity"`` are reserved for future use
            and currently raise ``ValueError`` at construction time.
        layout: Band-packing strategy.
            ``"replicated"`` (default) passes the same encoded vector to every
            frequency band. ``"per_band"`` is reserved; with ``"two_hot"`` it
            would produce identical output to ``"replicated"`` and is therefore
            not supported \u2014 raises ``ValueError`` at construction time.
    """

    kind: Literal["two_hot", "mlp", "identity"] = Field(
        default="two_hot",
        description="Observation encoder family.",
    )
    layout: Literal["replicated", "per_band"] = Field(
        default="replicated",
        description="Band-packing strategy: 'replicated' or 'per_band'.",
    )


# =============================================================================
class NavigationDecoderConfig(BaseModel, extra="forbid"):
    """Decoder strategy config for the TEM-to-navigation output pathway.

    Attributes:
        kind: Decoder family.
            ``"single_scale"`` (default) reads one HPC frequency band selected by
            ``prediction_freq``.
            ``"concat_all_scales"`` concatenates all HPC bands before decoding.
        prediction_freq: HPC band index for ``"single_scale"`` decoding.
            Must be ``0`` (and is ignored) when ``kind="concat_all_scales"``.
    """

    kind: Literal["single_scale", "concat_all_scales"] = Field(
        default="single_scale",
        description="Decoder input policy.",
    )
    prediction_freq: int = Field(
        default=0,
        ge=0,
        description="HPC band index for single_scale decoding. Must be 0 when kind='concat_all_scales'.",
    )

    @model_validator(mode="after")
    def _check_concat_has_no_freq(self) -> "NavigationDecoderConfig":
        if self.kind == "concat_all_scales" and self.prediction_freq != 0:
            raise ValueError(
                "NavigationDecoderConfig: prediction_freq must be 0 (unused) when "
                f"kind='concat_all_scales', got prediction_freq={self.prediction_freq}."
            )
        return self


# =============================================================================
class NavigationTEMV1AdapterSettings(BaseModel, extra="forbid"):
    """Task-side Navigation settings required to bind the TEM v1 core."""

    observation_dim: int = Field(
        ...,
        description="Dimensionality of the navigation task observations (must match environment.observation_dim).",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions (must match environment.action_count).",
    )
    encoder: NavigationEncoderConfig = Field(
        default_factory=NavigationEncoderConfig,
        description="Observation encoder strategy. Adapter-owned; not part of LEC or TEM model config.",
    )
    decoder: NavigationDecoderConfig = Field(
        default_factory=NavigationDecoderConfig,
        description="Observation decoder strategy. Adapter-owned; not part of HPC or TEM model config.",
    )


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV1Diagnostics(DetachMixin):
    """TEM-family diagnostic surface for controller and objective consumption.

    Carries all three observation-logit pathways and the raw latent bundles
    needed by the TEM objective. Kept separate from the canonical task surface
    so the controller can remain model-agnostic.

    Attributes:
        obs_logits: Three observation-logit tensors ``(inference, retrieved, ancestral)``,
            each of shape ``(B, obs_dim)``.
        grid_codes: MEC grid codes ``(g_post, g_prior)`` for grid-transition latent loss.
        place_codes: Full named TEM place-code bundle for latent-relation assembly.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: GridCodes
    place_codes: PlaceCodes


# =============================================================================
@dataclass(frozen=True)
class NavigationTEMV1BridgeOutput(DetachMixin):
    """Split bridge output: canonical task surface plus TEM-family diagnostics.

    Attributes:
        task: Canonical :class:`~ehc_sn.tasks.navigation.contracts.NavigationTaskOutput`
            carrying the inference-pathway observation logits.
        tem: TEM-family diagnostic bundle consumed by the controller/objective.
    """

    task: NavigationTaskOutput
    tem: NavigationTEMV1Diagnostics


# =============================================================================
class NavigationInputsEncoder(nn.Module):
    """Encodes navigation step data into a :class:`TEMInputV1` payload.

    Produces a :class:`MultiScaleCode` sensory payload so the flat/multiscale
    boundary lives in the adapter, not inside LEC.

    Only ``kind="two_hot"`` with ``layout="replicated"`` is currently
    implemented. Other combinations raise ``ValueError`` at construction time.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        n_freq: int,
        encoder_config: NavigationEncoderConfig,
    ) -> None:
        super().__init__()
        if encoder_config.kind != "two_hot":
            raise ValueError(
                f"NavigationInputsEncoder: encoder kind {encoder_config.kind!r} is not yet "
                "implemented. Only 'two_hot' is supported in this release."
            )
        if encoder_config.layout != "replicated":
            raise ValueError(
                f"NavigationInputsEncoder: layout {encoder_config.layout!r} is not supported "
                "with kind='two_hot'. The two-hot table is a fixed lookup and cannot produce "
                "per-band distinct content. Use layout='replicated'."
            )
        self.encoder = TwoHotEncoder(observation_dim, feature_dim)
        self._n_freq = n_freq

    def forward(  # -----------------------------------------------------------
        self,
        batch: dict[str, Tensor],
    ) -> TEMInputV1:
        """Encode pre-extracted navigation step data into a TEM v1 input payload."""
        task_input = coerce_navigation_step_input(batch)
        code = self.encoder(task_input.observation)
        sensory_codes: MultiScaleCode = [code.clone() for _ in range(self._n_freq)]
        return TEMInputV1(
            sensory_codes=sensory_codes,
            previous_action=task_input.previous_action,
            episode_start=task_input.episode_start,
            landmark_id=task_input.landmark_id,
        )


# =============================================================================
class NavigationOutputsDecoder(nn.Module):
    """Decodes a :class:`TEMOutputV1` into a :class:`NavigationTEMV1BridgeOutput`.

    When ``single_freq`` is an int, uses only that HPC band (paper-fidelity
    single-scale decoder). When ``None``, concatenates all bands.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        single_freq: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = MLPDecoder(latent_dim, observation_dim)
        self._obs_dim = observation_dim
        self._single_freq = single_freq

    def forward(  # -----------------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> NavigationTEMV1BridgeOutput:
        """Decode all three place pathways and return the split task + TEM surfaces."""
        gc = model_output.grid_codes
        pc = model_output.place_codes

        obs_inference = self.decoder(_select_code(pc.inference, self._single_freq))
        obs_retrieved = self.decoder(_select_code(pc.retrieved, self._single_freq)) if pc.retrieved is not None else obs_inference.new_zeros(obs_inference.shape[0], self._obs_dim)  # fmt: skip
        obs_ancestral = self.decoder(_select_code(pc.ancestral, self._single_freq))
        ol = (obs_inference, obs_retrieved, obs_ancestral)

        task = NavigationTaskOutput(obs_logits=obs_inference)
        tem = NavigationTEMV1Diagnostics(obs_logits=ol, grid_codes=gc, place_codes=pc)
        return NavigationTEMV1BridgeOutput(task=task, tem=tem)


# =============================================================================
class NavigationTEMV1BridgeAdapter(nn.Module):
    """Navigation plus TEM v1 bridge adapter.

    Explicit task-to-model transformations follow the adapter-interface spec:
    :meth:`prepare_inputs` encodes a navigation step batch into a model-native
    payload; :meth:`prepare_outputs` decodes a model output into the split
    task + TEM surfaces.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        model: TEMModelV1,
        config: NavigationTEMV1AdapterSettings,
    ) -> None:
        """Initialize the navigation plus TEM v1 bridge adapter."""
        super().__init__()
        self._config = config
        self.model = model
        self.encoder = _build_encoder(model, config)
        self.decoder = _build_decoder(model, config)

    @property
    def config(self) -> NavigationTEMV1AdapterSettings:
        """The navigation plus TEM v1 bridge adapter settings."""
        return self._config

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> TEMStateV1:
        """Create a fresh TEM recurrent state for one rollout batch."""
        return self.model.init_state(batch_size, device=device)

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: TEMStateV1,
    ) -> TEMStateV1:
        """Reset halted rows of the TEM recurrent state."""
        return self.model.reset_state(reset_flag, state)

    def prepare_inputs(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> TEMInputV1:
        """Encode a pre-extracted navigation step dict into a model-native :class:`TEMInputV1`."""
        return self.encoder(batch)

    def prepare_outputs(  # ---------------------------------------------------
        self,
        model_output: TEMOutputV1,
    ) -> NavigationTEMV1BridgeOutput:
        """Decode a TEM model output into the split task + TEM bridge surfaces."""
        return self.decoder(model_output)

    def forward(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: TEMStateV1 | None = None,
    ) -> tuple[TEMStateV1, NavigationTEMV1BridgeOutput]:
        """Run a forward pass of the TEM v1 bridge adapter on a Navigation task batch."""
        inputs = self.prepare_inputs(batch)
        model_output, next_state = self.model(inputs, state=state)
        bridge_output = self.prepare_outputs(model_output)
        return next_state, bridge_output


# =============================================================================
def _build_encoder(  # --------------------------------------------------------
    model: TEMModelV1,
    config: NavigationTEMV1AdapterSettings,
) -> NavigationInputsEncoder:
    """Construct the observation encoder front-end for the v1 bridge."""
    return NavigationInputsEncoder(
        observation_dim=config.observation_dim,
        feature_dim=model.config.lec.feature_dim,
        n_freq=model.lec.n_freq,
        encoder_config=config.encoder,
    )


def _build_decoder(  # --------------------------------------------------------
    model: TEMModelV1,
    config: NavigationTEMV1AdapterSettings,
) -> NavigationOutputsDecoder:
    """Construct the observation decoder back-end for the v1 bridge.

    Policy is adapter-owned: ``single_scale`` uses one HPC band (paper-fidelity
    stream 1 / prediction_freq=0); ``concat_all_scales`` concatenates all bands.
    """
    hpc_shape = model.config.hpc.shape
    n_freq = len(hpc_shape)
    if config.decoder.kind == "single_scale":
        freq = config.decoder.prediction_freq
        if not (0 <= freq < n_freq):
            raise ValueError(f"prediction_freq={freq} is out of range for hpc.shape with {n_freq} bands (valid: 0..{n_freq - 1}).")
        return NavigationOutputsDecoder(
            observation_dim=config.observation_dim,
            latent_dim=hpc_shape[freq],
            single_freq=freq,
        )
    # concat_all_scales
    return NavigationOutputsDecoder(
        observation_dim=config.observation_dim,
        latent_dim=sum(hpc_shape),
    )


def _select_code(  # ----------------------------------------------------------
    code: Tensor | Sequence[Tensor],
    single_freq: int | None,
) -> Tensor:
    """Return a ``(B, D)`` tensor for decoder use.

    When ``single_freq`` is an int, returns the single band at that index
    (paper-fidelity stream 1 / prediction_freq=0 decoder). When ``None``,
    concatenates all bands.

    Raises ``ValueError`` if ``single_freq`` is set but ``code`` is already a
    flat ``Tensor``; band selection on a pre-concatenated tensor is ambiguous.
    """
    if isinstance(code, Tensor):
        if single_freq is not None:
            raise ValueError(
                "single_scale decoder received a pre-concatenated flat Tensor; "
                "expected a sequence of per-band tensors so that band "
                f"prediction_freq={single_freq} can be selected unambiguously."
            )
        return code
    if len(code) == 0:
        raise ValueError("TEM latent code sequences must not be empty.")
    if single_freq is not None:
        return code[single_freq]
    return torch.cat(tuple(code), dim=1)


# =============================================================================
__all__ = [
    "NavigationDecoderConfig",
    "NavigationEncoderConfig",
    "NavigationInputsEncoder",
    "NavigationOutputsDecoder",
    "NavigationTEMV1Diagnostics",
    "NavigationTEMV1AdapterSettings",
    "NavigationTEMV1BridgeAdapter",
    "NavigationTEMV1BridgeOutput",
]
